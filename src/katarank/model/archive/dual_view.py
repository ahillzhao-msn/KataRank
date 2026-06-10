"""
KataRank — Dual-View Set Transformer
=======================================
The core innovation: dual-view architecture that processes black and white
moves in separate Set Transformer streams, then cross-attends between them.

This merges:
  - go-strength-model: permutation-invariant Set Transformer on deep trunk features
  - go-analyzer: black/white separated streams + cross-attention

Architecture:
    Black moves (N_b, C) ─→ SetEncoder_b ──┐
                                             ├── CrossMAB ──→ Pool ─→ Heads
    White moves (N_w, C) ─→ SetEncoder_w ──┘

The cross-attention uses Set Transformer's MAB mechanism:
  MAB(encoded_black, encoded_white) to model inter-player influence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict

from model.set_transformer import (
    MultiHeadAttentionBlock, SetEncoder, unbatch,
    bradley_terry_score, scale_rating,
)


# ─── Cross-Attention Module ───────────────────────────────────────────────────

class CrossMAB(nn.Module):
    """
    Cross Set Attention Block: MAB(X, Y) where X attends to Y.

    Unlike self-attention, the query comes from one stream and key/value
    from the other. This models "how does Black's play influence White's
    assessment" and vice versa.

    Supports bidirectional: both B→W and W→B directions.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mab_bw = MultiHeadAttentionBlock(dim, num_heads, dropout)
        self.mab_wb = MultiHeadAttentionBlock(dim, num_heads, dropout)

    def forward(
        self,
        h_b: torch.Tensor, h_w: torch.Tensor,
        blens: Optional[List[int]], wlens: Optional[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_b: (total_b, dim) Black stream encoding (packed batch)
            h_w: (total_w, dim) White stream encoding (packed batch)
            blens: batch structure for Black
            wlens: batch structure for White

        Returns:
            h_b_out: Black encoding updated with White context
            h_w_out: White encoding updated with Black context
        """
        # Black attends to White
        h_b_out = self.mab_bw(h_b, h_w, blens, wlens)

        # White attends to Black
        h_w_out = self.mab_wb(h_w, h_b, wlens, blens)

        return h_b_out, h_w_out


class BidirectionalCrossMAB(nn.Module):
    """
    Stacked bidirectional cross-attention with residual connections.

    This allows multiple rounds of "looking at the opponent's play"
    to refine the assessment, similar to how a human reviewer considers
    both players' perspectives.
    """

    def __init__(self, dim: int, num_heads: int = 4,
                 depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossMAB(dim, num_heads, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        h_b: torch.Tensor, h_w: torch.Tensor,
        blens: Optional[List[int]], wlens: Optional[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            h_b_delta, h_w_delta = block(h_b, h_w, blens, wlens)
            h_b = self.norm(h_b + h_b_delta)
            h_w = self.norm(h_w + h_w_delta)
        return h_b, h_w


# ─── Stream Pooling ───────────────────────────────────────────────────────────

class StreamPooling(nn.Module):
    """
    Pool each stream independently and fuse.

    Uses attention pooling (PMA) per stream, then concatenates
    or adds the pooled representations.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pool = MultiHeadAttentionBlock(dim, num_heads, dropout)
        # Learnable seed vectors for pooling
        self.seed_b = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.seed_w = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h_b: torch.Tensor, h_w: torch.Tensor,
        blens: Optional[List[int]], wlens: Optional[List[int]],
    ) -> torch.Tensor:
        """
        Returns:
            (batch, dim) fused representation
        """
        batch_size = len(blens) if blens else 1

        # Pool Black stream
        S_b = self.seed_b.expand(batch_size, -1).reshape(-1, self.seed_b.shape[-1])
        z_b = self.pool(S_b, h_b, None, blens)  # (batch, dim)

        # Pool White stream
        S_w = self.seed_w.expand(batch_size, -1).reshape(-1, self.seed_w.shape[-1])
        z_w = self.pool(S_w, h_w, None, wlens)  # (batch, dim)

        # Fuse
        z = torch.cat([z_b, z_w], dim=-1)
        z = self.fusion(z)

        return z


class SimpleStreamPooling(nn.Module):
    """
    Simpler pooling: mean pool each stream, then concat + project.
    Less expressive but more stable for small datasets.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h_b: torch.Tensor, h_w: torch.Tensor,
        blens: Optional[List[int]], wlens: Optional[List[int]],
    ) -> torch.Tensor:
        def mean_pool(h, lens):
            if lens is None:
                return h.mean(dim=0, keepdim=True)
            result = []
            offset = 0
            for l in lens:
                if l > 0:
                    result.append(h[offset:offset + l].mean(dim=0, keepdim=True))
                else:
                    result.append(torch.zeros(1, h.shape[-1], device=h.device))
                offset += l
            return torch.cat(result, dim=0)

        z_b = mean_pool(h_b, blens)
        z_w = mean_pool(h_w, wlens)
        z = torch.cat([z_b, z_w], dim=-1)
        return self.fusion(z)


# ─── Dual-View Set Transformer ────────────────────────────────────────────────

class DualViewSetTransformer(nn.Module):
    """
    Dual-View Set Transformer — the core KATA-RANK architecture.

    Processes Black and White moves in separate permutation-invariant
    streams, then cross-attends between them to model mutual influence.

    This is the combined innovation from:
      - Set Transformer (go-strength-model): permutation-invariant encoding
      - Black/White separation + cross-attention (go-analyzer v2)

    Architecture:
        Input (N, C) ─→ Split B/W
            ├── Black (N_b, C) ─→ Encoder_b ─→ CrossMAB ──┬── Pool ──→ Fuse ──→ Heads
            └── White (N_w, C) ─→ Encoder_w ─→ CrossMAB ──┘
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_inducing: int = 16,
        encoder_depth: int = 2,
        cross_depth: int = 1,
        dropout: float = 0.1,
        pooling: str = 'attention',  # 'attention' or 'mean'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Shared input projection (black and white share weights)
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.input_dropout = nn.Dropout(dropout)

        # Black stream encoder (input_dim = hidden_dim because we project above)
        self.encoder_b = SetEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_inducing=num_inducing,
            depth=encoder_depth,
            dropout=dropout,
        )

        # White stream encoder (separate instance for stream-specific features)
        self.encoder_w = SetEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_inducing=num_inducing,
            depth=encoder_depth,
            dropout=dropout,
        )

        # Cross-attention between streams
        self.cross_attn = BidirectionalCrossMAB(
            dim=hidden_dim,
            num_heads=num_heads,
            depth=cross_depth,
            dropout=dropout,
        )

        # Pooling
        if pooling == 'attention':
            self.pooling = StreamPooling(hidden_dim, num_heads, dropout)
        else:
            self.pooling = SimpleStreamPooling(hidden_dim, dropout)

        # Output dimension (fused representation)
        self.output_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor, xlens: Optional[List[int]] = None,
        player_dim: int = -1,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (total_seq, input_dim) packed batch of move features
            xlens: batch structure (sequence lengths)
            player_dim: which column in input_dim identifies player color.
                        -1 = auto-detect from interleaved sequence (B,W,B,W,...)
                        0..(input_dim-1) = use that feature column

        Returns:
            (batch, hidden_dim) fused representation
        """
        if xlens is None:
            xlens = [x.shape[0]]

        if player_dim is None or (isinstance(player_dim, int) and player_dim < 0):
            # Split by interleaving (assume B,W,B,W,... order)
            h = self.input_proj(x)
            x_b, x_w, blens, wlens = self._split_interleaved(h, xlens)
        elif isinstance(player_dim, int) and player_dim >= 0:
            # Split by a specific feature column (integer index)
            h = self.input_proj(x)
            x_b, x_w, blens, wlens = self._split_by_dim(h, x, player_dim, xlens)
        else:
            # player_dim is a tensor - use auto-detection
            if torch.any(player_dim >= 0):
                h = self.input_proj(x)
                x_b, x_w, blens, wlens = self._split_by_dim(h, x, 11, xlens)
            else:
                h = self.input_proj(x)
                x_b, x_w, blens, wlens = self._split_interleaved(h, xlens)

        # Encode each stream
        h_b = self.encoder_b(x_b, blens)
        h_w = self.encoder_w(x_w, wlens)

        # Cross-attend
        h_b, h_w = self.cross_attn(h_b, h_w, blens, wlens)

        # Pool and fuse
        z = self.pooling(h_b, h_w, blens, wlens)

        return z

    def _split_interleaved(
        self, h: torch.Tensor, xlens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
        """
        Split an interleaved B,W,B,W,... sequence into black and white streams.

        Assumes the first move in each game is Black.
        """
        blens = []
        wlens = []
        b_parts = []
        w_parts = []

        offset = 0
        for length in xlens:
            seq = h[offset:offset + length]
            n_b = (length + 1) // 2  # ceil: Black gets the extra move
            n_w = length // 2
            b_parts.append(seq[0::2])  # even indices: Black
            w_parts.append(seq[1::2])  # odd indices: White
            blens.append(n_b)
            wlens.append(n_w)
            offset += length

        x_b = torch.cat(b_parts, dim=0) if b_parts else torch.empty(0, h.shape[-1], device=h.device)
        x_w = torch.cat(w_parts, dim=0) if w_parts else torch.empty(0, h.shape[-1], device=h.device)

        return x_b, x_w, blens, wlens

    def _split_by_dim(
        self, h: torch.Tensor, x: torch.Tensor,
        player_dim: int, xlens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
        """
        Split by a feature dimension that encodes player color.
        The player_dim column should be 0 for Black, 1 for White.
        """
        blens = []
        wlens = []
        b_parts = []
        w_parts = []

        offset = 0
        for length in xlens:
            seq_h = h[offset:offset + length]
            seq_x = x[offset:offset + length]  # original features to read player_dim
            is_black = seq_x[:, player_dim] < 0.5

            b_parts.append(seq_h[is_black])
            w_parts.append(seq_h[~is_black])
            blens.append(is_black.sum().item())
            wlens.append((~is_black).sum().item())
            offset += length

        x_b = torch.cat(b_parts, dim=0) if b_parts else torch.empty(0, h.shape[-1], device=h.device)
        x_w = torch.cat(w_parts, dim=0) if w_parts else torch.empty(0, h.shape[-1], device=h.device)

        return x_b, x_w, blens, wlens

    def encode_black(self, x: torch.Tensor, xlens: Optional[List[int]] = None) -> torch.Tensor:
        """Encode only the Black stream (for analysis/debugging)."""
        if xlens is None:
            xlens = [x.shape[0]]
        h = self.input_proj(x)
        x_b, x_w, blens, wlens = self._split_interleaved(h, xlens)
        return self.encoder_b(x_b, blens)

    def encode_white(self, x: torch.Tensor, xlens: Optional[List[int]] = None) -> torch.Tensor:
        """Encode only the White stream (for analysis/debugging)."""
        if xlens is None:
            xlens = [x.shape[0]]
        h = self.input_proj(x)
        x_b, x_w, blens, wlens = self._split_interleaved(h, xlens)
        return self.encoder_w(x_w, wlens)


# ─── Neural Interface: Shared encoder weights variant ────────────────────────

class SharedDualViewSetTransformer(DualViewSetTransformer):
    """
    Variant where Black and White share encoder weights.

    This encodes the belief that "good moves are good moves regardless
    of color" — the encoder learns a universal move evaluator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make encoder_w share weights with encoder_b
        self.encoder_w = self.encoder_b
