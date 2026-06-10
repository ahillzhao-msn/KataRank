"""
KataRank — Multi-Task Model
=============================
KataRankModel: Dual-View encoder + DualRatingHead + DualOrdinalRankHead.

Architecture:
  - Uses DualViewSetTransformer (causal masks, segmented pooling)
  - Separate B/W ordinal rank heads, 29 classes (20k → 9d)
  - OrdinalLogisticHead: thresholds init to linspace(-2.5, 2.5, 28)
  - forward() returns plain dict; no ability/style heads
  - save/load includes architecture version tag
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

from katarank.model.dual_view import DualViewSetTransformer


# ─── Ordinal Rank Head v3 ─────────────────────────────────────────────────────

class OrdinalLogisticHead(nn.Module):
    """
    29-class ordinal logistic regression head.
    Classes: 0=20k, 1=19k, ..., 19=1k, 20=1d, ..., 28=9d.

    Thresholds initialised to linspace(-2.5, 2.5) so that, before any
    HumanSL supervision, rank buckets are evenly spread across the
    normalised strength axis.  Training with rank_confidence-weighted CE
    will pull each threshold to the correct segment boundary.

    At inference, no external signal is needed: the thresholds encode the
    learned segment boundaries permanently.
    """

    NUM_RANKS = 29   # 20k(0) … 9d(28)
    RANK_NAMES = [
        '20k','19k','18k','17k','16k','15k','14k','13k','12k','11k',
        '10k','9k','8k','7k','6k','5k','4k','3k','2k','1k',
        '1d','2d','3d','4d','5d','6d','7d','8d','9d',
    ]

    def __init__(self, input_dim: int, n_classes: int = NUM_RANKS):
        super().__init__()
        self.n_classes = n_classes
        self.linear = nn.Linear(input_dim, 1)
        self.thresholds = nn.Parameter(
            torch.linspace(-2.5, 2.5, n_classes - 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, input_dim)
        Returns:
            (batch, n_classes) probability distribution over rank classes
        """
        theta = self.linear(z).squeeze(-1)                       # (batch,)
        cum = torch.sigmoid(
            self.thresholds.unsqueeze(0) - theta.unsqueeze(1)   # (batch, n-1)
        )
        probs = torch.zeros(z.size(0), self.n_classes, device=z.device)
        probs[:, 0]  = cum[:, 0]
        probs[:, 1:-1] = cum[:, 1:] - cum[:, :-1]
        probs[:, -1] = 1.0 - cum[:, -1]
        return probs.clamp(min=1e-8, max=1.0 - 1e-8)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """Returns (batch,) most likely rank index."""
        return self.forward(z).argmax(dim=-1)

    def theta(self, z: torch.Tensor) -> torch.Tensor:
        """Raw scalar projection — useful for inspecting the strength axis."""
        return self.linear(z).squeeze(-1)


# ─── KataRank v3 Model ────────────────────────────────────────────────────────

class KataRankModel(nn.Module):
    """
    KataRank v3 full model.

    Architecture:
        Input (N_b+N_w, input_dim)
          → DualViewSetTransformer  (causal cross-attn, segmented pool)
          → z  (hidden_dim,)
          → head_proj  Linear(hidden → head_dim) + ReLU
               ├── black_rating  Linear(head_dim → 1)   ← MSE vs norm(meanLogPrior_B)
               ├── white_rating  Linear(head_dim → 1)   ← MSE vs norm(meanLogPrior_W)
               ├── rank_head_b   OrdinalLogisticHead  ← CE vs humanRankIdx_B  (training only)
               └── rank_head_w   OrdinalLogisticHead  ← CE vs humanRankIdx_W  (training only)

    Inference:
        Only b_rating / w_rating and rank_probs_b / rank_probs_w are used.
        No HumanSL is needed at inference time; OrdinalHead thresholds
        encode learned segment boundaries.
    """

    VERSION = '3.0'

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_inducing: int = 16,
        encoder_depth: int = 2,
        cross_depth: int = 1,
        dropout: float = 0.1,
        n_rank_classes: int = OrdinalLogisticHead.NUM_RANKS,
        player_dim: int = 7,
    ):
        super().__init__()
        self.player_dim = player_dim
        self._cfg = dict(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, num_inducing=num_inducing,
            encoder_depth=encoder_depth, cross_depth=cross_depth,
            dropout=dropout, n_rank_classes=n_rank_classes,
            player_dim=player_dim,
        )

        self.encoder = DualViewSetTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_inducing=num_inducing,
            encoder_depth=encoder_depth,
            cross_depth=cross_depth,
            dropout=dropout,
        )

        head_dim = hidden_dim // 2
        self.head_proj = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.black_rating = nn.Linear(head_dim, 1)
        self.white_rating = nn.Linear(head_dim, 1)

        self.rank_head_b = OrdinalLogisticHead(head_dim, n_rank_classes)
        self.rank_head_w = OrdinalLogisticHead(head_dim, n_rank_classes)

    def forward(
        self,
        x: torch.Tensor,
        xlens: Optional[List[int]] = None,
        player_dim: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x:         (N_total, input_dim) packed move features
            xlens:     per-game sequence lengths; None = single game
            player_dim: override self.player_dim if needed

        Returns dict:
            'b_rating':     (batch,)       Black continuous strength
            'w_rating':     (batch,)       White continuous strength
            'rank_probs_b': (batch, 29)    Black rank distribution
            'rank_probs_w': (batch, 29)    White rank distribution
        """
        pdim = player_dim if player_dim is not None else self.player_dim
        z = self.encoder(x, xlens, pdim)          # (batch, hidden_dim)
        h = self.head_proj(z)                      # (batch, head_dim)

        return {
            'b_rating':     self.black_rating(h).squeeze(-1),
            'w_rating':     self.white_rating(h).squeeze(-1),
            'rank_probs_b': self.rank_head_b(h),
            'rank_probs_w': self.rank_head_w(h),
        }

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            'version':     self.VERSION,
            'type':        'KataRankModel',
            'config':      self._cfg,
            'model_state': self.state_dict(),
        }, path)

    @staticmethod
    def load(path: str, device: str = 'cpu') -> 'KataRankModel':
        data = torch.load(path, map_location=device)
        assert data.get('type') == 'KataRankModel', \
            f"Expected KataRankModel, got {data.get('type')}"
        model = KataRankModel(**data['config'])
        model.load_state_dict(data['model_state'])
        return model.to(device)

    def get_config(self) -> dict:
        return dict(self._cfg)

    # ── convenience ──────────────────────────────────────────────────────────

    def predict_rank(self, x: torch.Tensor,
                     xlens: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """Inference helper: returns rank indices + continuous ratings."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x, xlens)
        return {
            'b_rating':  out['b_rating'],
            'w_rating':  out['w_rating'],
            'rank_b':    out['rank_probs_b'].argmax(dim=-1),
            'rank_w':    out['rank_probs_w'].argmax(dim=-1),
            'rank_name_b': OrdinalLogisticHead.RANK_NAMES[
                out['rank_probs_b'].argmax(dim=-1).item()],
            'rank_name_w': OrdinalLogisticHead.RANK_NAMES[
                out['rank_probs_w'].argmax(dim=-1).item()],
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
