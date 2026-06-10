"""
KataRank — Multi-Task Learning Heads
======================================
Multiple output heads on top of the shared Dual-View Set Transformer encoder.

Heads:
  1. KataRankScoreHead    — Continuous rating regression (scalar)
  2. AbilityDimensionHead — Multi-dimensional ability scores (opening, midgame, etc.)
  3. StyleClassification  — Playing style classifier (aggressive, territorial, etc.)
  4. OrdinalRankHead      — Ordinal rank prediction (from go-analyzer)
  5. DualRatingHead       — Predict both Black and White ratings simultaneously
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

from model.set_transformer import (
    bradley_terry_score, scale_rating, unbatch,
)
from model.dual_view import (
    DualViewSetTransformer, SharedDualViewSetTransformer,
)


# ─── Score / Rating Heads ─────────────────────────────────────────────────────

class KataRankScoreHead(nn.Module):
    """
    Continuous rating regression head.

    Predicts a scalar rating score from the fused representation.
    This is the primary KataRank output.
    """

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, input_dim) fused representation

        Returns:
            (batch,) predicted rating in normalized scale
        """
        return self.mlp(z).squeeze(-1)


class DualRatingHead(nn.Module):
    """
    Predict both Black and White ratings from the same representation.

    Useful when training on game-level outcomes where we know both
    players' ratings.
    """

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.black_head = nn.Linear(hidden_dim, 1)
        self.white_head = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch, input_dim)

        Returns:
            (batch,) black_rating, (batch,) white_rating
        """
        h = self.shared(z)
        return self.black_head(h).squeeze(-1), self.white_head(h).squeeze(-1)


# ─── Ability Dimension Heads ─────────────────────────────────────────────────

class AbilityDimensionHead(nn.Module):
    """
    Multi-dimensional ability score predictor.

    Each dimension outputs a score in [0, 1] via sigmoid, representing
    the player's proficiency in that specific aspect of Go.

    Dimensions:
        - Opening (布局)
        - Middle game (中盘)
        - Endgame (官子)
        - Overall vision (大局观)
        - Calculation depth (计算深度)
        - Risk control (风险控制)
    """

    NAMES = [
        'opening',           # 布局能力
        'middle_game',       # 中盘战斗力
        'endgame',           # 官子水平
        'overall_vision',    # 大局观
        'calculation_depth', # 计算深度
        'risk_control',      # 风险控制
    ]

    def __init__(self, input_dim: int, num_dimensions: int = 6,
                 hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        self.num_dimensions = num_dimensions
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_dimensions),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, input_dim)

        Returns:
            (batch, num_dimensions) scores in [0, 1]
        """
        return torch.sigmoid(self.mlp(z))

    def get_dimension_names(self) -> List[str]:
        return self.NAMES[:self.num_dimensions]


# ─── Style Classification ────────────────────────────────────────────────────

class StyleClassificationHead(nn.Module):
    """
    Playing style classifier.

    Categories:
        - Territorial (实地派)
        - Influence (外势派)
        - Balanced (均衡派)
        - Aggressive (战斗派)
        - Flexible (灵活派)
    """

    NAMES = ['territorial', 'influence', 'balanced', 'aggressive', 'flexible']

    def __init__(self, input_dim: int, num_styles: int = 5,
                 hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        self.num_styles = num_styles
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_styles),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, input_dim)

        Returns:
            (batch, num_styles) logits (for cross-entropy loss)
        """
        return self.mlp(z)

    def predict(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predicted class and confidence."""
        logits = self.forward(z)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        conf = probs.max(dim=-1).values
        return pred, conf


# ─── Ordinal Regression Head (from go-analyzer) ──────────────────────────────

class OrdinalLogisticHead(nn.Module):
    """
    Ordinal logistic regression head for rank classification.

    Instead of predicting a continuous rating, predicts which rank
    category the player belongs to using ordinal thresholds.

    This is useful for backward compatibility with go-analyzer's
    rank prediction and for human-readable output.
    """

    def __init__(self, input_dim: int, n_classes: int = 9):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        # Initialize thresholds evenly spaced
        self.thresholds = nn.Parameter(torch.linspace(-1.5, 1.5, n_classes - 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, input_dim)

        Returns:
            (batch, n_classes) probability distribution over rank classes
        """
        theta = self.linear(z).squeeze(-1)  # (batch,)
        cum_probs = torch.sigmoid(
            self.thresholds.unsqueeze(0) - theta.unsqueeze(1)
        )  # (batch, n_classes - 1)

        n_classes = len(self.thresholds) + 1
        probs = torch.zeros(z.size(0), n_classes, device=z.device)
        probs[:, 0] = cum_probs[:, 0]
        for k in range(1, n_classes - 1):
            probs[:, k] = cum_probs[:, k] - cum_probs[:, k - 1]
        probs[:, n_classes - 1] = 1.0 - cum_probs[:, -1]
        probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)

        return probs

    def predict_rank(self, z: torch.Tensor) -> torch.Tensor:
        """Get most likely rank class."""
        probs = self.forward(z)
        return probs.argmax(dim=-1)

    def predict_theta(self, z: torch.Tensor) -> torch.Tensor:
        """Get raw ordinal logit (before thresholds)."""
        return self.linear(z).squeeze(-1)


# ─── Full Multi-Task Model ────────────────────────────────────────────────────

class KataRankMultiTaskModel(nn.Module):
    """
    Full multi-task model: Dual-View encoder + multiple output heads.

    This is the complete KataRank architecture as designed in DESIGN_V1.md:
      - Dual-view Set Transformer for encoding
      - Multi-task heads for comprehensive assessment
      - Configurable which heads are active
    """

    def __init__(
        self,
        input_dim: int = 256,          # Trunk feature dimension
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_inducing: int = 16,
        encoder_depth: int = 2,
        cross_depth: int = 1,
        dropout: float = 0.1,
        pooling: str = 'attention',
        enable_score: bool = True,              # KataRank score
        enable_dual_rating: bool = False,       # Black + White rating pair
        enable_abilities: bool = True,          # Multi-dimensional abilities
        num_ability_dims: int = 6,
        enable_style: bool = True,              # Style classification
        num_styles: int = 5,
        enable_ordinal: bool = False,           # Ordinal rank classification
        num_ranks: int = 9,
        shared_encoder: bool = False,           # Share B/W encoder weights
    ):
        super().__init__()

        # Encoder
        encoder_cls = (
            SharedDualViewSetTransformer if shared_encoder
            else DualViewSetTransformer
        )
        self.encoder = DualViewSetTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_inducing=num_inducing,
            encoder_depth=encoder_depth,
            cross_depth=cross_depth,
            dropout=dropout,
            pooling=pooling,
        )

        # Output heads
        self.enable_score = enable_score
        self.enable_dual_rating = enable_dual_rating
        self.enable_abilities = enable_abilities
        self.enable_style = enable_style
        self.enable_ordinal = enable_ordinal

        if enable_score:
            self.score_head = KataRankScoreHead(hidden_dim)

        if enable_dual_rating:
            self.dual_rating_head = DualRatingHead(hidden_dim)

        if enable_abilities:
            self.ability_head = AbilityDimensionHead(hidden_dim, num_ability_dims)

        if enable_style:
            self.style_head = StyleClassificationHead(hidden_dim, num_styles)

        if enable_ordinal:
            self.ordinal_head = OrdinalLogisticHead(hidden_dim, num_ranks)

    def forward(
        self,
        x: torch.Tensor, xlens: Optional[List[int]] = None,
        player_dim: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning dict of all active head outputs.

        Args:
            x: (total_seq, input_dim) packed batch
            xlens: batch structure
            player_dim: which column has player color (-1 = interleaved)

        Returns:
            dict with keys matching enabled heads:
              'score': (batch,) rating scores
              'black_rating', 'white_rating': (batch,) dual ratings
              'abilities': (batch, num_dims) ability scores
              'style_logits': (batch, num_styles) style logits
              'rank_probs': (batch, num_ranks) rank probabilities
        """
        z = self.encoder(x, xlens, player_dim)

        outputs = {}
        if self.enable_score:
            outputs['score'] = self.score_head(z)
        if self.enable_dual_rating:
            outputs['black_rating'], outputs['white_rating'] = self.dual_rating_head(z)
        if self.enable_abilities:
            outputs['abilities'] = self.ability_head(z)
        if self.enable_style:
            outputs['style_logits'] = self.style_head(z)
        if self.enable_ordinal:
            outputs['rank_probs'] = self.ordinal_head(z)

        return outputs

    def forward_single(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate a single game (no batching).

        Args:
            x: (seq_len, input_dim) move features

        Returns:
            dict of predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x.unsqueeze(0) if x.dim() == 2 else x,
                                xlens=[x.shape[0]] if x.dim() == 2 else None)

    def get_config(self) -> dict:
        """Return model configuration for serialization."""
        return {
            'input_dim': self.encoder.input_dim,
            'hidden_dim': self.encoder.hidden_dim,
            'num_heads': self.encoder.cross_attn.blocks[0].mab_bw.mha.num_heads if self.encoder.cross_attn.blocks else 4,
            'num_inducing': self.encoder.encoder_b.blocks[0].num_inducing if self.encoder.encoder_b.blocks else 16,
            'encoder_depth': len(self.encoder.encoder_b.blocks),
            'cross_depth': len(self.encoder.cross_attn.blocks),
            'dropout': 0.1,
            'pooling': 'attention',
            'enable_score': self.enable_score,
            'enable_dual_rating': self.enable_dual_rating,
            'enable_abilities': self.enable_abilities,
            'num_ability_dims': self.ability_head.num_dimensions if self.enable_abilities else 6,
            'enable_style': self.enable_style,
            'num_styles': self.style_head.num_styles if self.enable_style else 5,
            'enable_ordinal': self.enable_ordinal,
            'num_ranks': len(self.ordinal_head.thresholds) + 1 if self.enable_ordinal else 9,
            'shared_encoder': False,
        }

    def save(self, path: str):
        """Save model weights and config."""
        torch.save({
            'model_state': self.state_dict(),
            'config': self.get_config(),
            'type': 'KataRankMultiTaskModel',
            'ability_names': (
                self.ability_head.get_dimension_names()
                if self.enable_abilities else []
            ),
        }, path)

    @staticmethod
    def load(path: str, device: str = 'cpu') -> 'KataRankMultiTaskModel':
        """Load model from file."""
        data = torch.load(path, map_location=device)
        cfg = data['config']
        model = KataRankMultiTaskModel(
            input_dim=cfg['input_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_heads=cfg.get('num_heads', 4),
            num_inducing=cfg.get('num_inducing', 16),
            encoder_depth=cfg.get('encoder_depth', 2),
            cross_depth=cfg.get('cross_depth', 1),
            dropout=cfg.get('dropout', 0.1),
            pooling=cfg.get('pooling', 'attention'),
            enable_score=cfg.get('enable_score', True),
            enable_dual_rating=cfg.get('enable_dual_rating', False),
            enable_abilities=cfg.get('enable_abilities', True),
            num_ability_dims=cfg.get('num_ability_dims', 6),
            enable_style=cfg.get('enable_style', True),
            num_styles=cfg.get('num_styles', 5),
            enable_ordinal=cfg.get('enable_ordinal', False),
            num_ranks=cfg.get('num_ranks', 9),
            shared_encoder=cfg.get('shared_encoder', False),
        )
        model.load_state_dict(data['model_state'])
        return model


# ─── Backward compatibility adapter ──────────────────────────────────────────

class StrengthNetAdapter(nn.Module):
    """
    Adapter that makes KataRankMultiTaskModel compatible with the
    legacy StrengthNet interface (single rating output).

    This allows drop-in replacement in existing training scripts.
    """

    def __init__(self, model: KataRankMultiTaskModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor,
                xlens: Optional[List[int]] = None) -> torch.Tensor:
        """Legacy interface: returns (batch,) ratings."""
        outputs = self.model(x, xlens)
        if 'score' in outputs:
            return outputs['score']
        elif 'black_rating' in outputs:
            return outputs['black_rating']
        raise ValueError("No compatible output head found")

    def save(self, path: str):
        self.model.save(path)

    @staticmethod
    def load(path: str, device: str = 'cpu') -> 'StrengthNetAdapter':
        model = KataRankMultiTaskModel.load(path, device)
        return StrengthNetAdapter(model)
