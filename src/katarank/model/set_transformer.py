"""
KataRank — Set Transformer Core Module
========================================
Enhanced Set Transformer with proper multi-head attention,
based on "Set Transformer: A Framework for Attention-based
Permutation-Invariant Neural Networks" (Lee et al., 2019).

Compatible with the existing go-strength-model feature data format.
Supports both single-output (rating-only) and multi-task heads.

Architecture:
    Input: n x C feature matrix (per-player recent moves)
    → Linear projection → ISAB blocks → PMA → Output heads
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Callable


def unbatch(x: torch.Tensor, xlens: Optional[List[int]]) -> List[slice]:
    """
    Decompose a packed batch tensor into slices.

    Args:
        x: Packed tensor of shape (total_seq, dim)
        xlens: List of sequence lengths. If None, treats as single batch.

    Returns:
        List of slice objects for each sequence in the batch.
    """
    if xlens is None or len(xlens) == 0:
        return [slice(0, x.shape[0])]
    cum = np.cumsum([0] + xlens)
    return [slice(s, e) for s, e in zip(cum[:-1], cum[1:])]


def pack_batch(sequences: List[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
    """
    Pack a list of variable-length sequences into a single tensor.

    Args:
        sequences: List of tensors, each (seq_len_i, dim)

    Returns:
        packed: (total_seq, dim) tensor
        lens: List of sequence lengths
    """
    lens = [s.shape[0] for s in sequences]
    packed = torch.cat(sequences, dim=0)
    return packed, lens


# ─── Attention Components ─────────────────────────────────────────────────────

def _block_diagonal_mask(X_lens: List[int], Y_lens: List[int],
                         device: torch.device) -> torch.Tensor:
    """
    Additive (L, S) attention mask restricting each query to keys of the
    same game in a packed batch: 0 = same game, -inf = different game.

    Without this, packed games attend to each other and batch elements
    are no longer independent (training/inference mismatch).
    """
    L, S = sum(X_lens), sum(Y_lens)
    mask = torch.full((L, S), float('-inf'), device=device)
    xo = yo = 0
    for lx, ly in zip(X_lens, Y_lens):
        mask[xo: xo + lx, yo: yo + ly] = 0.0
        xo += lx
        yo += ly
    return mask


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head attention block with LayerNorm and residual connections.

    Implements: MAB(X, Y) = LayerNorm(H + rFF(H))
        where H = LayerNorm(X + MultiHead(X, Y, Y))
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.dim = dim
        self.num_heads = num_heads

        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,  # we use (seq, batch, dim) convention internally
        )
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, X: torch.Tensor, Y: torch.Tensor,
                X_lens: Optional[List[int]] = None,
                Y_lens: Optional[List[int]] = None) -> torch.Tensor:
        """
        Args:
            X: (seq_X, dim) query tensor (packed batch)
            Y: (seq_Y, dim) key/value tensor (packed batch)
            X_lens: per-game lengths for X; with Y_lens, attention is
                    restricted to within-game pairs (block-diagonal mask)
            Y_lens: per-game lengths for Y

        Returns:
            (seq_X, dim) output tensor
        """
        # Add batch dimension for MultiheadAttention: (seq, 1, dim)
        X_3d = X.unsqueeze(1)   # (seq_X, 1, dim)
        Y_3d = Y.unsqueeze(1)   # (seq_Y, 1, dim)

        attn_mask = None
        if X_lens is not None and Y_lens is not None and len(X_lens) > 1:
            attn_mask = _block_diagonal_mask(X_lens, Y_lens, X.device)

        if attn_mask is not None:
            # Queries of a game whose key segment is empty have all -inf
            # rows → softmax would produce NaN. Exclude them from mha and
            # let the residual pass the original representation through.
            all_masked = torch.isinf(attn_mask).all(dim=-1)
            if all_masked.any():
                valid = ~all_masked
                attn_out = torch.zeros_like(X)
                if valid.any():
                    out_v, _ = self.mha(
                        X[valid].unsqueeze(1), Y_3d, Y_3d,
                        attn_mask=attn_mask[valid],
                    )
                    attn_out[valid] = out_v.squeeze(1)
            else:
                attn_out, _ = self.mha(X_3d, Y_3d, Y_3d, attn_mask=attn_mask)
                attn_out = attn_out.squeeze(1)
        else:
            attn_out, _ = self.mha(X_3d, Y_3d, Y_3d)
            attn_out = attn_out.squeeze(1)

        # Residual + LayerNorm
        H = self.norm0(X + attn_out)

        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(H)
        out = self.norm1(H + ffn_out)

        return out


class SAB(nn.Module):
    """Set Attention Block: MAB(X, X) — self-attention within a set."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mab = MultiHeadAttentionBlock(dim, num_heads, dropout)

    def forward(self, X: torch.Tensor, xlens: Optional[List[int]] = None) -> torch.Tensor:
        return self.mab(X, X, xlens, xlens)


class ISAB(nn.Module):
    """
    Induced Set Attention Block.

    Uses M inducing points to project the input set into a smaller
    representation, reducing O(n^2) to O(nm) where m << n.
    """

    def __init__(self, dim: int, num_heads: int = 4,
                 num_inducing: int = 16, dropout: float = 0.1):
        super().__init__()
        self.num_inducing = num_inducing
        self.I = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.02)
        self.mab0 = MultiHeadAttentionBlock(dim, num_heads, dropout)
        self.mab1 = MultiHeadAttentionBlock(dim, num_heads, dropout)

    def forward(self, X: torch.Tensor, xlens: Optional[List[int]] = None) -> torch.Tensor:
        batch_size = len(xlens) if xlens else 1

        # Expand inducing points for batch: (batch, M, dim)
        I_batch = self.I.expand(batch_size, -1, -1)

        # Flatten for MAB processing: (batch * M, dim)
        I_flat = I_batch.reshape(-1, I_batch.shape[-1])

        # Per-game lengths for the inducing points (M each) — required so
        # the block-diagonal mask keeps games independent within the batch.
        I_lens = [self.num_inducing] * batch_size

        # MAB0: inducing points attend to input set
        # I acts as queries, X as keys/values
        H = self.mab0(I_flat, X, I_lens, xlens)  # (batch*M, dim)

        # Prepare H lens (each has same length = num_inducing)
        H_lens = [self.num_inducing] * batch_size

        # MAB1: input attends to inducing point outputs
        Y = self.mab1(X, H, xlens, H_lens)

        return Y


class PMA(nn.Module):
    """
    Pooling by Multi-head Attention.

    Uses a learnable seed vector to attend over the set and produce
    a fixed-size representation, regardless of input set size.
    """

    def __init__(self, dim: int, num_heads: int = 4,
                 num_seeds: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_seeds = num_seeds
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MultiHeadAttentionBlock(dim, num_heads, dropout)

    def forward(self, X: torch.Tensor, xlens: Optional[List[int]] = None) -> torch.Tensor:
        batch_size = len(xlens) if xlens else 1

        # Expand seed for batch: (batch, num_seeds, dim)
        S_batch = self.S.expand(batch_size, -1, -1)

        # Flatten: (batch * num_seeds, dim)
        S_flat = S_batch.reshape(-1, S_batch.shape[-1])

        # Seeds attend to the set, block-diagonal per game
        S_lens = [self.num_seeds] * batch_size
        Z = self.mab(S_flat, X, S_lens, xlens)  # (batch*num_seeds, dim)

        return Z


# ─── Encoder / Decoder ─────────────────────────────────────────────────────────

class SetEncoder(nn.Module):
    """
    Encoder that processes a set of move features.

    Architecture:
        Input → Linear → [ISAB × depth] → Output encoding
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, num_inducing: int = 16,
                 depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(ISAB(hidden_dim, num_heads, num_inducing, dropout))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, xlens: Optional[List[int]] = None) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.dropout(h)
        for block in self.blocks:
            h = block(h, xlens)
        return h


class SetDecoder(nn.Module):
    """
    Decoder that pools the encoded set representation.

    Architecture:
        Encoded set → PMA → Output projection
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4,
                 num_seeds: int = 1, dropout: float = 0.1):
        super().__init__()
        self.pma = PMA(hidden_dim, num_heads, num_seeds, dropout)

    def forward(self, h: torch.Tensor, xlens: Optional[List[int]] = None) -> torch.Tensor:
        # Pool to fixed size: (batch * num_seeds, hidden_dim)
        return self.pma(h, xlens)


# ─── Full Set Transformer Model ────────────────────────────────────────────────

class SetTransformer(nn.Module):
    """
    Full Set Transformer for permutation-invariant set processing.

    Can be used standalone (for rating-only) or as the encoder
    for multi-task models.

    Architecture:
        Input (n, C) → Encoder (ISABs) → Decoder (PMA) → Output
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, num_inducing: int = 16,
                 num_seeds: int = 1, depth: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = SetEncoder(input_dim, hidden_dim, num_heads,
                                  num_inducing, depth, dropout)
        self.decoder = SetDecoder(hidden_dim, num_heads, num_seeds, dropout)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, xlens: Optional[List[int]] = None) -> torch.Tensor:
        h = self.encoder(x, xlens)
        z = self.decoder(h, xlens)
        return z

    def encode(self, x: torch.Tensor, xlens: Optional[List[int]] = None) -> torch.Tensor:
        """Return encoder output (before pooling) for introspection."""
        return self.encoder(x, xlens)

    @classmethod
    def from_strengthnet_weights(cls, old_model: nn.Module,
                                 input_dim: int, hidden_dim: int,
                                 depth: int) -> 'SetTransformer':
        """
        Create a SetTransformer from a legacy StrengthNet's weights.

        This is a migration helper — copies what can be copied and
        initializes the rest randomly.
        """
        model = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_inducing=hidden_dim // 4,
            num_seeds=1,
            depth=depth,
        )
        # Copy input projection if dimensions match
        old_proj = old_model.enc.layers[0]
        if old_proj.weight.shape == model.encoder.input_proj.weight.shape:
            model.encoder.input_proj.weight.data = old_proj.weight.data

        return model


# ─── Single-output variant (legacy compatible) ─────────────────────────────────

class KataRankRatingModel(nn.Module):
    """
    Rating-only model — predicts a single scalar per player.

    This is the direct replacement for the original StrengthNet,
    producing the same output shape but using proper multi-head attention.

    Architecture:
        Move features → SetTransformer → Linear(1) → rating
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, num_inducing: int = 16,
                 depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = SetTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_inducing=num_inducing,
            num_seeds=1,
            depth=depth,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, xlens: Optional[List[int]] = None) -> torch.Tensor:
        z = self.encoder(x, xlens)
        return self.head(z).squeeze(-1)

    def save(self, path: str):
        torch.save({
            'model_state': self.state_dict(),
            'config': {
                'input_dim': self.encoder.encoder.input_proj.in_features,
                'hidden_dim': self.encoder.encoder.input_proj.out_features,
                'num_heads': self.encoder.decoder.pma.mab.num_heads,
                'num_inducing': self.encoder.encoder.blocks[0].num_inducing if self.encoder.encoder.blocks else 16,
                'depth': len(self.encoder.encoder.blocks),
            },
            'type': 'KataRankRatingModel',
        }, path)

    @staticmethod
    def load(path: str, device: str = 'cpu') -> 'KataRankRatingModel':
        data = torch.load(path, map_location=device, weights_only=True)
        cfg = data['config']
        model = KataRankRatingModel(
            input_dim=cfg['input_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_heads=cfg['num_heads'],
            num_inducing=cfg.get('num_inducing', cfg['hidden_dim'] // 4),
            depth=cfg['depth'],
        )
        model.load_state_dict(data['model_state'])
        return model


# ─── Bradley-Terry scoring helper ──────────────────────────────────────────────

def bradley_terry_score(black_rating: torch.Tensor,
                        white_rating: torch.Tensor,
                        scale: float = 315.8088) -> torch.Tensor:
    """
    Estimate win probability from two ratings.

    Args:
        black_rating: (batch,) tensor of Black's ratings in normalized scale
        white_rating: (batch,) tensor of White's ratings in normalized scale
        scale: Standard deviation of label distribution for unscaling

    Returns:
        (batch,) win probability for Black (0..1)
    """
    return 1.0 / (1.0 + (10 ** ((white_rating - black_rating) * scale / 400.0)))


def scale_rating(rating: torch.Tensor,
                 mean: float = 1623.1913,
                 std: float = 315.8088) -> torch.Tensor:
    """
    Convert normalized rating back to Glicko-2 scale.

    Args:
        rating: Normalized rating (N(0,1) scale)
        mean: Glicko-2 mean of training data
        std: Glicko-2 standard deviation

    Returns:
        Rating in Glicko-2 scale
    """
    return rating * std + mean
