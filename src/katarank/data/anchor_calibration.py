"""
KataRank — HumanSL anchor calibration
=====================================
Maps rank labels written by an older HumanSL anchor onto the current anchor's scale, so a
KAB2 cache labelled by several anchor versions trains on one consistent target without
re-running HumanSL over it.

Why: anchor v2 (5 fixed profiles + spline, katago-fork batch_analysis) is biased against
a full 29-profile scan of the same moves (8k–5k players ~1.25 ranks low; mean |err| 0.65).
Anchor v3 (7 coarse + ±2 refine) matches the full scan to 0.02. The bias is a property of
the v2 procedure, so it is learnable from the cached v2 output: pair v2 labels with v3 /
full-scan labels of the same players, fit the correction once, apply it at load time.

The cache itself is never rewritten: ``_meta.csv`` keeps each row's raw label and
``humanAnchorVer``; ``effective_rank`` decides per row which label training sees.

Model: Nadaraya–Watson regression on the v2 posterior mean, evaluated on a fixed grid
(``grid``): ``mean_shift`` (target − v2 mean) and ``log_sd_ratio`` (log target sd / v2
sd). ``prior_weight`` pseudo-observations of "no correction" keep the sparse tails from
extrapolating. Stored as JSON so the artifact is readable and diffable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

NUM_RANKS = 29                       # 20k(0) … 9d(28)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CALIBRATION = REPO_ROOT / 'nets' / 'anchor_calibration' / 'v2_to_v3.json'


@dataclass
class AnchorCalibration:
    source_anchor: int               # humanAnchorVer this maps FROM
    target_anchor: int               # humanAnchorVer whose scale it maps TO
    grid: np.ndarray                 # v2 posterior-mean grid points (ranks)
    mean_shift: np.ndarray           # target mean − source mean at each grid point
    log_sd_ratio: np.ndarray         # log(target sd / source sd) at each grid point
    meta: Dict = field(default_factory=dict)   # fit provenance + validation report

    # ── fit / apply ───────────────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        src_exp: np.ndarray, src_sd: np.ndarray,
        tgt_exp: np.ndarray, tgt_sd: np.ndarray,
        *,
        source_anchor: int = 2, target_anchor: int = 3,
        bandwidth: float = 0.6, prior_weight: float = 3.0, grid_step: float = 0.1,
        meta: Optional[Dict] = None,
    ) -> 'AnchorCalibration':
        grid = np.round(np.arange(0.0, NUM_RANKS - 1 + 1e-9, grid_step), 6)
        shift = _kernel_mean(src_exp, tgt_exp - src_exp, grid, bandwidth, prior_weight)
        ratio = _kernel_mean(src_exp, np.log(tgt_sd / src_sd), grid, bandwidth, prior_weight)
        m = dict(meta or {})
        m.update(bandwidth=bandwidth, prior_weight=prior_weight, n_train=int(len(src_exp)))
        return cls(source_anchor, target_anchor, grid, shift, ratio, m)

    def apply(self, src_exp, src_sd=None):
        """Calibrated (mean, sd) for source-anchor posterior mean/sd (scalars or arrays).
        sd is None when ``src_sd`` is None."""
        x = np.asarray(src_exp, dtype=float)
        mean = np.clip(x + np.interp(x, self.grid, self.mean_shift), 0.0, NUM_RANKS - 1)
        if src_sd is None:
            return mean, None
        sd = np.asarray(src_sd, dtype=float) * np.exp(np.interp(x, self.grid, self.log_sd_ratio))
        return mean, sd

    # ── persistence ───────────────────────────────────────────────────────────

    def to_json(self) -> str:
        """JSON with each array on one line, so a refit diffs as a few changed lines."""
        arrays = {
            'grid':         [round(float(v), 6) for v in self.grid],
            'mean_shift':   [round(float(v), 5) for v in self.mean_shift],
            'log_sd_ratio': [round(float(v), 5) for v in self.log_sd_ratio],
        }
        lines = [f' "source_anchor": {self.source_anchor},',
                 f' "target_anchor": {self.target_anchor},']
        lines += [f' "{k}": {json.dumps(v)},' for k, v in arrays.items()]
        meta = json.dumps(self.meta, ensure_ascii=False, indent=1).replace('\n', '\n ')
        lines.append(f' "meta": {meta}')
        return '{\n' + '\n'.join(lines) + '\n}'

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json() + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path) -> 'AnchorCalibration':
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(int(d['source_anchor']), int(d['target_anchor']), np.array(d['grid']),
                   np.array(d['mean_shift']), np.array(d['log_sd_ratio']), d.get('meta', {}))

    @staticmethod
    def file_sha256(path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _kernel_mean(x, y, grid, bandwidth, prior_weight):
    """Gaussian-kernel weighted mean of y at each grid point, shrunk toward 0 by
    ``prior_weight`` pseudo-observations."""
    w = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bandwidth) ** 2)
    den = w.sum(1) + prior_weight
    return np.divide(w @ y, den, out=np.zeros(len(grid)), where=den > 0)


def load_default_calibration(path=DEFAULT_CALIBRATION) -> Optional[AnchorCalibration]:
    """The repo's calibration artifact, or None when it does not exist."""
    p = Path(path)
    return AnchorCalibration.load(p) if p.exists() else None


# ─── Per-row label resolution ─────────────────────────────────────────────────

def _float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def effective_rank_idx(row: Dict, side: str,
                       calibration: Optional[AnchorCalibration]) -> Optional[int]:
    """Training label for one side of a ``_meta.csv`` row, or None to keep the row's own
    ``{side}_humanRank``.

    A row written by ``calibration.source_anchor`` is mapped onto the target scale and
    rounded. Every other row (current anchor, legacy window, no calibration) keeps its
    own label: the anchor already rounds its posterior mean into ``humanRank``.
    """
    if calibration is None:
        return None
    ver = _float(row.get('humanAnchorVer'))
    exp = _float(row.get(f'{side}_humanRankExp'))
    if ver is None or exp is None or int(round(ver)) != calibration.source_anchor:
        return None
    mean, _ = calibration.apply(exp)
    return int(np.floor(float(mean) + 0.5))
