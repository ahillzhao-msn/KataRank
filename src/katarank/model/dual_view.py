"""
KataRank — Dual-View Set Transformer
======================================
Architecture:
    Input (N, C) → split by player_dim=7 (isWhite)
        Black (N_b, C) → SetEncoder_b (ISAB) ─→ CausalCrossMAB ─→ SegPool ─→ Fuse → z
        White (N_w, C) → SetEncoder_w (ISAB) ─→ CausalCrossMAB ─┘

Key design choices:
  - Causal cross-attention: masks prevent future-move information leakage
  - Segmented attention pooling: opening / midgame / endgame weighted separately
  - turn_number (scalar[8]) propagated from split through cross-attention
  - No double-residual: CausalMAB owns the full residual path
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from katarank.model.set_transformer import SetEncoder, unbatch


# ─── Causal Mask Builders ─────────────────────────────────────────────────────

def build_causal_mask_bw(turn_b: torch.Tensor,
                          turn_w: torch.Tensor) -> torch.Tensor:
    """
    B→W causal mask: Black move i may only attend to White moves j
    where turn_w[j] < turn_b[i]  (White played strictly before Black).

    Returns (N_b, N_w) additive mask: 0=visible, -inf=masked.
    """
    # (1, N_w) < (N_b, 1)  →  (N_b, N_w) bool
    visible = turn_w.unsqueeze(0) < turn_b.unsqueeze(1)
    mask = torch.full((len(turn_b), len(turn_w)), float('-inf'),
                      dtype=torch.float32, device=turn_b.device)
    mask[visible] = 0.0
    return mask


def build_causal_mask_wb(turn_w: torch.Tensor,
                          turn_b: torch.Tensor) -> torch.Tensor:
    """
    W→B causal mask: White move j may only attend to Black moves i
    where turn_b[i] <= turn_w[j]  (Black played before or at same round — Black is first).

    Returns (N_w, N_b) additive mask: 0=visible, -inf=masked.
    """
    # (1, N_b) <= (N_w, 1)  →  (N_w, N_b) bool
    visible = turn_b.unsqueeze(0) <= turn_w.unsqueeze(1)
    mask = torch.full((len(turn_w), len(turn_b)), float('-inf'),
                      dtype=torch.float32, device=turn_w.device)
    mask[visible] = 0.0
    return mask


# ─── Causal MAB ───────────────────────────────────────────────────────────────

class CausalMAB(nn.Module):
    """
    Multi-head Attention Block that accepts an additive attn_mask.

    Implements standard post-norm cross-attention:
        H   = LayerNorm(X + CrossAttn(X, Y, mask))
        out = LayerNorm(H + FFN(H))

    Unlike the v2 CrossMAB (which called MultiHeadAttentionBlock and added
    an outer residual, producing a double-residual), this block owns the
    full residual path and returns the updated representation directly.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=False,
        )
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, X: torch.Tensor, Y: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            X: (N_q, dim)  query
            Y: (N_k, dim)  key/value
            attn_mask: (N_q, N_k) additive float mask, -inf = block, 0 = allow

        Returns:
            (N_q, dim) updated X

        Note on all-masked rows:
            A query whose entire row is -inf (e.g. Black move 0 with no prior
            White context) would produce NaN after softmax.  We detect such
            rows and replace their attention output with zeros, so the residual
            connection passes the original representation through unchanged.
        """
        Y_3d = Y.unsqueeze(1)   # (N_k, 1, dim)

        # Identify all-masked rows BEFORE calling mha.
        # softmax(all -inf) = NaN in both forward AND backward;
        # we must exclude those rows entirely from the mha call.
        if attn_mask is not None:
            all_masked = (attn_mask == float('-inf')).all(dim=-1)  # (N_q,)
        else:
            all_masked = None

        if all_masked is not None and all_masked.any():
            valid = ~all_masked                              # (N_q,) bool
            attn_out = torch.zeros_like(X)
            if valid.any():
                X_v = X[valid].unsqueeze(1)                  # (n_v, 1, dim)
                m_v = attn_mask[valid]                       # (n_v, N_k)
                out_v, _ = self.mha(X_v, Y_3d, Y_3d, attn_mask=m_v)
                attn_out[valid] = out_v.squeeze(1)
            # all_masked positions stay 0 → residual passes X through unchanged
        else:
            X_3d = X.unsqueeze(1)
            out, _ = self.mha(X_3d, Y_3d, Y_3d, attn_mask=attn_mask)
            attn_out = out.squeeze(1)

        H = self.norm0(X + attn_out)
        return self.norm1(H + self.ffn(H))


# ─── Cross-Attention with Causal Masking ─────────────────────────────────────

class CrossMAB(nn.Module):
    """
    Bidirectional cross-attention between Black and White streams
    with per-game causal masks built from turn numbers.

    Processes each game in the batch separately so that per-game
    masks of different shapes can be applied correctly.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mab_bw = CausalMAB(dim, num_heads, dropout)  # Black attends to White
        self.mab_wb = CausalMAB(dim, num_heads, dropout)  # White attends to Black

    def forward(
        self,
        h_b: torch.Tensor, h_w: torch.Tensor,
        blens: List[int], wlens: List[int],
        turn_b: torch.Tensor, turn_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_parts, w_parts = [], []
        b_off = w_off = 0

        for nb, nw in zip(blens, wlens):
            hb_i = h_b[b_off: b_off + nb]         # (nb, dim)
            hw_i = h_w[w_off: w_off + nw]         # (nw, dim)
            tb_i = turn_b[b_off: b_off + nb]
            tw_i = turn_w[w_off: w_off + nw]

            mask_bw = build_causal_mask_bw(tb_i, tw_i)   # (nb, nw)
            mask_wb = build_causal_mask_wb(tw_i, tb_i)   # (nw, nb)

            b_parts.append(self.mab_bw(hb_i, hw_i, attn_mask=mask_bw))
            w_parts.append(self.mab_wb(hw_i, hb_i, attn_mask=mask_wb))

            b_off += nb
            w_off += nw

        h_b_out = torch.cat(b_parts, dim=0) if b_parts else h_b
        h_w_out = torch.cat(w_parts, dim=0) if w_parts else h_w
        return h_b_out, h_w_out


class BidirectionalCrossMAB(nn.Module):
    """Stacked CrossMAB blocks. Each block owns its residual connection."""

    def __init__(self, dim: int, num_heads: int = 4,
                 depth: int = 1, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossMAB(dim, num_heads, dropout) for _ in range(depth)
        ])

    def forward(
        self,
        h_b: torch.Tensor, h_w: torch.Tensor,
        blens: List[int], wlens: List[int],
        turn_b: torch.Tensor, turn_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            # Each CrossMAB returns the fully updated h_b, h_w (residual inside CausalMAB)
            h_b, h_w = block(h_b, h_w, blens, wlens, turn_b, turn_w)
        return h_b, h_w


# ─── Segmented Attention Pooling ─────────────────────────────────────────────

class SegmentedAttentionPool(nn.Module):
    """
    Pool a sequence in three temporal segments: opening / midgame / endgame.

    Within each segment, learned attention weights emphasise high-complexity
    moves (high score_stdev, surprising policy choices) over routine ones.
    Avoids the problem of key moves being averaged away in long games.

    Output: (3 * d_model,) per sequence — concatenation of three pooled vectors.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def _pool_segment(self, seg: torch.Tensor) -> torch.Tensor:
        if seg.size(0) == 0:
            return torch.zeros(seg.size(-1), device=seg.device)
        w = torch.softmax(self.score(seg), dim=0)   # (k, 1)
        return (seg * w).sum(dim=0)                  # (d_model,)

    def forward_single(self, seq: torch.Tensor) -> torch.Tensor:
        """Pool one sequence. seq: (N, d_model) → (3 * d_model,)."""
        n = seq.size(0)
        s1 = max(n // 3, 1)
        s2 = max(2 * n // 3, s1 + 1)
        segs = [seq[:s1], seq[s1:s2], seq[s2:]]
        return torch.cat([self._pool_segment(s) for s in segs])

    def forward(self, h: torch.Tensor, lens: List[int]) -> torch.Tensor:
        """Pool packed batch. Returns (batch, 3 * d_model)."""
        out = []
        offset = 0
        for l in lens:
            out.append(self.forward_single(h[offset: offset + l]))
            offset += l
        return torch.stack(out)   # (batch, 3 * d_model)


class StreamPooling(nn.Module):
    """
    Pool Black and White streams independently via SegmentedAttentionPool,
    then fuse both into a single game-level representation.

    Output: (batch, hidden_dim)
    Fusion input: (batch, 6 * hidden_dim) = B_open + B_mid + B_end + W_open + W_mid + W_end
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.seg_pool = SegmentedAttentionPool(dim)
        self.fusion = nn.Sequential(
            nn.Linear(6 * dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h_b: torch.Tensor, h_w: torch.Tensor,
        blens: List[int], wlens: List[int],
    ) -> torch.Tensor:
        z_b = self.seg_pool(h_b, blens)          # (batch, 3*dim)
        z_w = self.seg_pool(h_w, wlens)          # (batch, 3*dim)
        z   = torch.cat([z_b, z_w], dim=-1)      # (batch, 6*dim)
        return self.fusion(z)                     # (batch, dim)


# ─── Dual-View Set Transformer v3 ────────────────────────────────────────────

class DualViewSetTransformer(nn.Module):
    """
    Dual-View Set Transformer v3.

    Key improvements over v2:
      - Causal cross-attention: each player only sees past opponent moves
      - Segmented attention pooling: opening/midgame/endgame weighted separately
      - turn_number extracted at split time and propagated to CrossMAB
      - No double-residual (bug fix vs v2 BidirectionalCrossMAB)

    Forward input contract:
        x:         (N_total, input_dim)  packed moves for one game (or batch)
        xlens:     [N_total] or list of per-game lengths
        player_dim: column index where isWhite flag is stored (default: 7)
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
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # Shared projection (Black and White share weights here)
        self.input_proj    = nn.Linear(input_dim, hidden_dim, bias=False)
        self.input_dropout = nn.Dropout(dropout)

        # Separate stream encoders (independent weights → stream-specific features)
        self.encoder_b = SetEncoder(hidden_dim, hidden_dim, num_heads,
                                    num_inducing, encoder_depth, dropout)
        self.encoder_w = SetEncoder(hidden_dim, hidden_dim, num_heads,
                                    num_inducing, encoder_depth, dropout)

        # Causal bidirectional cross-attention
        self.cross_attn = BidirectionalCrossMAB(
            hidden_dim, num_heads, cross_depth, dropout
        )

        # Segmented pooling + fusion
        self.pooling = StreamPooling(hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        xlens: Optional[List[int]] = None,
        player_dim: int = 7,
    ) -> torch.Tensor:
        """
        Returns:
            (batch, hidden_dim) game-level fused representation
        """
        if xlens is None:
            xlens = [x.shape[0]]

        h = self.input_proj(x)
        h = self.input_dropout(h)

        h_b, h_w, blens, wlens, turn_b, turn_w = self._split(h, x, player_dim, xlens)

        h_b = self.encoder_b(h_b, blens)
        h_w = self.encoder_w(h_w, wlens)

        h_b, h_w = self.cross_attn(h_b, h_w, blens, wlens, turn_b, turn_w)

        return self.pooling(h_b, h_w, blens, wlens)

    def _split(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        player_dim: int,
        xlens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int],
               torch.Tensor, torch.Tensor]:
        """
        Split packed batch by player color (scalar[player_dim]).
        Also extracts turn numbers from scalar[8] for causal masking.

        Returns:
            h_b, h_w: projected packed tensors per stream
            blens, wlens: per-game sequence lengths
            turn_b, turn_w: packed turn-number tensors
        """
        b_parts, w_parts = [], []
        tb_parts, tw_parts = [], []
        blens, wlens = [], []
        offset = 0

        for length in xlens:
            seq_h = h[offset: offset + length]
            seq_x = x[offset: offset + length]

            is_black = seq_x[:, player_dim] < 0.5   # isWhite=0 → Black

            b_parts.append(seq_h[is_black])
            w_parts.append(seq_h[~is_black])
            nb = int(is_black.sum())
            nw = int((~is_black).sum())
            blens.append(nb)
            wlens.append(nw)
            # Go strictly alternates: Black at even turns (0,2,4…), White at odd (1,3,5…)
            # Positional index within each stream gives the correct causal ordering.
            dev_x = seq_h.device
            tb_parts.append(torch.arange(nb, dtype=torch.float32, device=dev_x) * 2)
            tw_parts.append(torch.arange(nw, dtype=torch.float32, device=dev_x) * 2 + 1)
            offset += length

        dev = h.device
        dim = h.shape[-1]

        def _cat(parts, fallback_shape):
            return (torch.cat(parts, dim=0) if any(len(p) > 0 for p in parts)
                    else torch.empty(0, fallback_shape, device=dev))

        h_b    = _cat(b_parts, dim)
        h_w    = _cat(w_parts, dim)
        turn_b = torch.cat(tb_parts) if tb_parts else torch.empty(0, device=dev)
        turn_w = torch.cat(tw_parts) if tw_parts else torch.empty(0, device=dev)

        return h_b, h_w, blens, wlens, turn_b, turn_w
