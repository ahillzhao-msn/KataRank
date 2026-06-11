"""
KataRank — Sparse Autoencoder Interface
========================================
SAE scaffolding for labeling model behaviour during game review.
Design rationale and data flow: docs/SAE_DESIGN.md.

Pipeline:

    KataRankModel.forward
      └─ ActivationCapture           per-move residual-stream activations
           └─ SparseAutoencoder      sparse feature decomposition
                └─ FeatureExtractor  per-move top-k features, move-aligned
                     └─ FeatureRegistry  human labels for feature ids

Training the SAE is deliberately out of scope here: `collect_sae_corpus`
prepares the training matrices; the training loop will live in the future
ReviewWorkflow.

Usage::

    sae = SparseAutoencoder.load('sae_xattn_b_lite.pt')
    fx  = FeatureExtractor(model, sae_b=sae,
                           registry=FeatureRegistry('features.json'))
    moves = fx.extract(x, top_k=8)
    # [{'move_no': 1, 'color': 'B', 'feature_ids': [...], ...}, ...]
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from katarank.model.dual_view import CausalMAB
from katarank.model.interpret import ActivationCapture


# ─── Sparse Autoencoder ───────────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """
    Overcomplete sparse autoencoder over residual-stream activations.

        f  = ReLU(W_enc · (x − b_dec) + b_enc)     (tokens, d_features)
        x̂ = W_dec · f + b_dec                      (tokens, d_model)

    Sparsity modes:
        k=None  — L1 penalty (weight `l1_coeff`) in loss()
        k=int   — keep only the k largest activations per token (hard L0);
                  loss() then skips the L1 term

    W_dec rows are the dictionary atoms (feature directions). Call
    `normalize_decoder_()` after each optimizer step during training to keep
    them unit-norm, otherwise L1 can be gamed by shrinking f and growing W_dec.
    """

    VERSION = '0.1'

    def __init__(
        self,
        d_model: int,
        expansion: int = 8,
        k: Optional[int] = None,
        l1_coeff: float = 1e-3,
    ):
        super().__init__()
        self.d_model    = d_model
        self.d_features = d_model * expansion
        self.k          = k
        self.l1_coeff   = l1_coeff
        self._cfg = dict(d_model=d_model, expansion=expansion,
                         k=k, l1_coeff=l1_coeff)

        # Encoder initialised as decoder transpose (standard tied init).
        w = torch.randn(self.d_features, d_model) / d_model ** 0.5
        w = w / w.norm(dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(w)                  # (d_features, d_model)
        self.W_enc = nn.Parameter(w.t().clone())      # (d_model, d_features)
        self.b_enc = nn.Parameter(torch.zeros(self.d_features))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(tokens, d_model) → (tokens, d_features) sparse feature activations."""
        f = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        if self.k is not None and self.k < self.d_features:
            vals, idx = f.topk(self.k, dim=-1)
            f = torch.zeros_like(f).scatter_(-1, idx, vals)
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """(tokens, d_features) → (tokens, d_model) reconstruction."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.encode(x)
        return {'recon': self.decode(f), 'features': f}

    def loss(self, x: torch.Tensor,
             out: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Reconstruction + sparsity losses. 'l0' is diagnostic only."""
        if out is None:
            out = self.forward(x)
        f   = out['features']
        mse = F.mse_loss(out['recon'], x)
        l1  = f.abs().sum(dim=-1).mean()
        l0  = (f > 0).float().sum(dim=-1).mean()
        total = mse if self.k is not None else mse + self.l1_coeff * l1
        return {'total': total, 'mse': mse, 'l1': l1, 'l0': l0}

    @torch.no_grad()
    def normalize_decoder_(self):
        """Renormalise dictionary atoms to unit norm (call after each step)."""
        self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    # ── persistence (same envelope as KataRankModel) ──────────────────────────

    def save(self, path: str):
        torch.save({
            'version':     self.VERSION,
            'type':        'SparseAutoencoder',
            'config':      self._cfg,
            'model_state': self.state_dict(),
        }, path)

    @staticmethod
    def load(path: str, device: str = 'cpu') -> 'SparseAutoencoder':
        data = torch.load(path, map_location=device, weights_only=True)
        assert data.get('type') == 'SparseAutoencoder', \
            f"Expected SparseAutoencoder, got {data.get('type')}"
        sae = SparseAutoencoder(**data['config'])
        sae.load_state_dict(data['model_state'])
        return sae.to(device)

    def get_config(self) -> dict:
        return dict(self._cfg)


# ─── Feature label registry ──────────────────────────────────────────────────

class FeatureRegistry:
    """
    JSON-backed store mapping SAE feature ids to human labels.

    Single-writer by design: labeling is a low-frequency human activity,
    so atomic replace (tmp + os.replace) suffices — no locking.
    """

    def __init__(self, path: str):
        self.path = path
        self._entries: Dict[int, dict] = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                self._entries = {int(k): v for k, v in json.load(fh).items()}

    def label(self, feature_id: int, label: str,
              author: str = '', notes: str = '') -> dict:
        entry = {
            'label':   label,
            'author':  author,
            'notes':   notes,
            'updated': date.today().isoformat(),
        }
        self._entries[int(feature_id)] = entry
        self._save()
        return entry

    def get(self, feature_id: int) -> Optional[dict]:
        return self._entries.get(int(feature_id))

    def all(self) -> Dict[int, dict]:
        return dict(self._entries)

    def _save(self):
        d = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=d, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump({str(k): v for k, v in self._entries.items()},
                          fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


# ─── Review-time feature extraction ──────────────────────────────────────────

class MoveFeature(TypedDict):
    """Per-move SAE feature readout — JSON-serializable review unit."""
    move_no:     int             # 1-based global move number
    color:       str             # 'B' | 'W'
    stream_pos:  int             # position within the player's stream
    feature_ids: List[int]       # top-k active features, activation-descending
    activations: List[float]
    labels:      List[Optional[str]]  # registry labels, None if unlabeled


def default_cross_sites(model: nn.Module) -> Tuple[str, str]:
    """Resolve the last cross-attention block's (mab_bw, mab_wb) module names.

    These tokens have seen opponent context — the recommended extraction site
    (see docs/SAE_DESIGN.md §3.2).
    """
    bw = [n for n, m in model.named_modules()
          if isinstance(m, CausalMAB) and n.endswith('mab_bw')]
    wb = [n for n, m in model.named_modules()
          if isinstance(m, CausalMAB) and n.endswith('mab_wb')]
    if not bw or not wb:
        raise ValueError("Model has no CrossMAB blocks (mab_bw/mab_wb)")
    return bw[-1], wb[-1]


class FeatureExtractor:
    """
    Run one game through the model, read activations at the chosen sites,
    and decompose each move into top-k SAE features.

    Single-game only: CrossMAB fires its hooks once per game in a batch, so
    overwrite-mode capture is only well-defined for batch size 1 — which is
    the natural granularity of game review anyway.

    Args:
        model:    KataRankModel (or any module with CrossMAB blocks).
        sae_b:    SAE for the Black-stream site.
        sae_w:    SAE for the White-stream site; defaults to sae_b (fine for
                  interface testing; train per-stream SAEs for real semantics).
        site_b:   Black capture site; default = last cross block's mab_bw.
        site_w:   White capture site; default = last cross block's mab_wb.
        registry: Optional FeatureRegistry to attach human labels.
    """

    def __init__(
        self,
        model: nn.Module,
        sae_b: SparseAutoencoder,
        sae_w: Optional[SparseAutoencoder] = None,
        site_b: Optional[str] = None,
        site_w: Optional[str] = None,
        registry: Optional[FeatureRegistry] = None,
    ):
        self.model = model
        self.sae_b = sae_b
        self.sae_w = sae_w if sae_w is not None else sae_b
        if site_b is None or site_w is None:
            auto_b, auto_w = default_cross_sites(model)
            site_b = site_b or auto_b
            site_w = site_w or auto_w
        self.site_b = site_b
        self.site_w = site_w
        self.registry = registry

    @torch.no_grad()
    def extract(self, x: torch.Tensor, top_k: int = 8) -> List[MoveFeature]:
        """
        Args:
            x:     (N, input_dim) packed move features for ONE game
            top_k: features reported per move (≤ this when fewer are active)

        Returns:
            One MoveFeature per move, ascending move_no. Move numbers assume
            strict B/W alternation with Black first — the same assumption
            DualViewSetTransformer uses for its causal masks.
        """
        assert x.dim() == 2, "extract() takes a single game: x must be 2-D"
        self.model.eval()
        with ActivationCapture(self.model, capture_attn=False) as cap:
            self.model(x, xlens=[x.size(0)])

        results: List[MoveFeature] = []
        plan = (
            (self.site_b, self.sae_b, 'B'),
            (self.site_w, self.sae_w, 'W'),
        )
        for site, sae, color in plan:
            acts = cap.activations.get(site)
            if acts is None or acts.numel() == 0:
                continue   # empty stream (e.g. fragment with one color only)
            device = next(sae.parameters()).device
            f = sae.encode(acts.to(device))
            k = min(top_k, f.size(-1))
            vals, idx = f.topk(k, dim=-1)
            for pos in range(f.size(0)):
                keep = vals[pos] > 0
                ids  = idx[pos][keep].tolist()
                results.append(MoveFeature(
                    move_no     = 2 * pos + 1 if color == 'B' else 2 * pos + 2,
                    color       = color,
                    stream_pos  = pos,
                    feature_ids = ids,
                    activations = vals[pos][keep].tolist(),
                    labels      = [self._label(i) for i in ids],
                ))
        results.sort(key=lambda r: r['move_no'])
        return results

    def _label(self, feature_id: int) -> Optional[str]:
        if self.registry is None:
            return None
        entry = self.registry.get(feature_id)
        return entry['label'] if entry else None


# ─── Corpus collection (feeds the future ReviewWorkflow SAE training) ────────

def collect_sae_corpus(
    model: nn.Module,
    batches: Iterable[Tuple[torch.Tensor, List[int]]],
    sites: List[str],
    max_tokens: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Run batches through the model and accumulate per-site activations into
    SAE training matrices.

    Uses accumulate-mode capture so cross-attention sites (which fire once
    per game within a batch) collect every game, not just the last one.

    Args:
        model:      Model to capture from (set to eval internally).
        batches:    Iterable of (x, xlens) — e.g. built from KAB2Samples.
        sites:      Module names to collect (see default_cross_sites()).
        max_tokens: Stop once every site has at least this many tokens
                    (0 = consume the whole iterable).

    Returns:
        {site: (n_tokens, d_model) cpu tensor} — sites that never fired
        are omitted.
    """
    model.eval()
    store:  Dict[str, List[torch.Tensor]] = {s: [] for s in sites}
    counts: Dict[str, int] = {s: 0 for s in sites}

    with torch.no_grad():
        for x, xlens in batches:
            with ActivationCapture(model, capture_attn=False,
                                   accumulate=True) as cap:
                model(x, xlens)
            for s in sites:
                t = cap.activations.get(s)
                if t is not None and t.numel() > 0:
                    store[s].append(t)
                    counts[s] += t.size(0)
            if max_tokens and all(c >= max_tokens for c in counts.values()):
                break

    return {s: torch.cat(parts, dim=0) for s, parts in store.items() if parts}
