"""
KataRank — Interpretability Capture
====================================
Groundwork for SAE-based analysis of the model's attention dimensions.

ActivationCapture records, during a forward pass:
  - residual-stream activations: the token-level output of every attention
    block (MultiHeadAttentionBlock, CausalMAB) — the natural SAE training
    substrate
  - attention maps: the (queries, keys) weight matrices of every
    nn.MultiheadAttention call (head-averaged)

Usage::

    model = KataRankModel.load('best.pt').eval()
    with ActivationCapture(model) as cap:
        out = model(x, xlens)

    cap.activations            # {module_name: (tokens, dim) tensor}
    cap.attention_maps         # {module_name: (queries, keys) tensor}
    cap.stack('encoder_b')     # concatenated activations matching a prefix

Collected tensors are detached and moved to CPU by default, so capture is
safe inside large-batch loops feeding an SAE training corpus.
"""

from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from katarank.model.set_transformer import MultiHeadAttentionBlock
from katarank.model.dual_view import CausalMAB


class ActivationCapture:
    """Context manager recording activations via forward hooks.

    Args:
        model:          Any module tree containing attention blocks.
        capture_blocks: Record residual-stream outputs of attention blocks.
        capture_attn:   Record attention weight matrices (head-averaged;
                        only present where the forward pass requests
                        need_weights, which is the default).
        detach:         Detach + move to CPU (recommended for SAE corpus
                        collection; set False to inspect gradients).
        accumulate:     Concatenate repeated fires of the same hook along the
                        token dimension instead of overwriting. Needed for
                        corpus collection: CrossMAB calls each CausalMAB once
                        per game in a batch, so overwrite mode keeps only the
                        last game. Applies to block activations only —
                        attention maps have per-game shapes and stay
                        last-write-wins.
        block_types:    Override which module classes count as blocks.
    """

    def __init__(
        self,
        model: nn.Module,
        capture_blocks: bool = True,
        capture_attn: bool = True,
        detach: bool = True,
        accumulate: bool = False,
        block_types: Optional[Tuple[Type[nn.Module], ...]] = None,
    ):
        self.model = model
        self.capture_blocks = capture_blocks
        self.capture_attn = capture_attn
        self.detach = detach
        self.accumulate = accumulate
        self.block_types = block_types or (MultiHeadAttentionBlock, CausalMAB)

        self.activations: Dict[str, torch.Tensor] = {}
        self.attention_maps: Dict[str, torch.Tensor] = {}
        self._handles: List = []

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> 'ActivationCapture':
        for name, mod in self.model.named_modules():
            if self.capture_blocks and isinstance(mod, self.block_types):
                self._handles.append(
                    mod.register_forward_hook(self._block_hook(name))
                )
            if self.capture_attn and isinstance(mod, nn.MultiheadAttention):
                self._handles.append(
                    mod.register_forward_hook(self._attn_hook(name))
                )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ── hooks ─────────────────────────────────────────────────────────────────

    def _prep(self, t: torch.Tensor) -> torch.Tensor:
        return t.detach().cpu() if self.detach else t

    def _block_hook(self, name: str):
        def fn(_mod, _inputs, output):
            t = self._prep(output[0] if isinstance(output, tuple) else output)
            if self.accumulate and name in self.activations:
                t = torch.cat([self.activations[name], t], dim=0)
            self.activations[name] = t
        return fn

    def _attn_hook(self, name: str):
        def fn(_mod, _inputs, output):
            # nn.MultiheadAttention returns (attn_output, attn_weights);
            # weights are None when called with need_weights=False.
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                # (batch, queries, keys) with batch=1 internally → squeeze
                self.attention_maps[name] = self._prep(output[1]).squeeze(0)
        return fn

    # ── convenience ───────────────────────────────────────────────────────────

    def stack(self, prefix: str = '') -> torch.Tensor:
        """Concatenate captured block activations whose name starts with
        `prefix`, along the token dimension — an SAE training matrix.

        Example: stack('encoder.encoder_b') → all Black-stream tokens.
        """
        parts = [
            t for name, t in sorted(self.activations.items())
            if name.startswith(prefix) and t.dim() == 2
        ]
        if not parts:
            raise KeyError(
                f"No 2-D activations match prefix {prefix!r}; "
                f"captured: {sorted(self.activations)}"
            )
        return torch.cat(parts, dim=0)

    def clear(self):
        """Drop captured tensors (keep hooks active) between batches."""
        self.activations.clear()
        self.attention_maps.clear()
