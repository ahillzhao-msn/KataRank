"""
KataRank — KAB2 File Dataset

Reads _B.npz / _W.npz file pairs produced by `katago batch_analysis`.
Supports random-access indexing (standard Map-style Dataset).

This is the canonical dataset for offline training: generate KAB2 files
once, iterate many epochs.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from katarank.data.base import KAB2Base, KAB2Sample
from katarank.data.preprocess import read_kab2, probe_kab2_dim


# ─── Rank string ↔ index ─────────────────────────────────────────────────────

_RANK_NAMES = [
    '20k','19k','18k','17k','16k','15k','14k','13k','12k','11k',
    '10k','9k','8k','7k','6k','5k','4k','3k','2k','1k',
    '1d','2d','3d','4d','5d','6d','7d','8d','9d',
]
_RANK_TO_IDX: Dict[str, int] = {f'rank_{n}': i for i, n in enumerate(_RANK_NAMES)}
NUM_RANK_CLASSES = len(_RANK_NAMES)   # 29


def rank_str_to_idx(s: str) -> int:
    """'rank_8d' → 28,  'rank_20k' → 0,  unknown → -1."""
    return _RANK_TO_IDX.get(str(s).strip(), -1)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class KAB2FileDataset(KAB2Base):
    """
    Map-style Dataset over KAB2 file pairs.

    Layout on disk::

        data_dir/
          <stem>_B.npz    ← Black player's moves
          <stem>_W.npz    ← White player's moves
          _meta.csv       ← per-game labels (logPrior, humanRank, …)

    One sample = one game (both players concatenated).
    """

    def __init__(
        self,
        data_dir: str,
        meta_csv: Optional[str] = None,
        split: str = 'T',
        max_moves_per_player: int = 400,
        min_moves_per_player: int = 5,
        cache: bool = True,
    ):
        self.data_dir  = Path(data_dir)
        self.split     = split
        self.max_moves = max_moves_per_player
        self.min_moves = min_moves_per_player
        self.cache     = cache
        self._move_cache: Dict[str, np.ndarray] = {}

        meta_path   = meta_csv or str(self.data_dir / '_meta.csv')
        self._games = self._load_meta(meta_path, split)
        self._input_dim  = self._detect_input_dim()
        self._scalar_dim = 10
        self._trunk_dim  = (self._input_dim - self._scalar_dim) // 2

        print(
            f"KAB2FileDataset: {len(self._games)} games  "
            f"split='{split}'  input_dim={self._input_dim}  "
            f"{'lite' if self.is_lite else 'full'}"
        )

    # ── KAB2Base properties ───────────────────────────────────────────────────

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def scalar_dim(self) -> int:
        return self._scalar_dim

    @property
    def trunk_dim(self) -> int:
        return self._trunk_dim

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._games)

    def __getitem__(self, idx: int) -> KAB2Sample:
        g = self._games[idx]
        gid = g['id']

        b_moves = self._load_moves(gid, 'B')
        w_moves = self._load_moves(gid, 'W')

        if len(b_moves) < self.min_moves or len(w_moves) < self.min_moves:
            return KAB2Base._empty_sample(gid, self._input_dim, g['target_b'], g['target_w'])

        return KAB2Base._make_sample(
            moves_b    = b_moves,
            moves_w    = w_moves,
            game_id    = gid,
            target_b   = g['target_b'],
            target_w   = g['target_w'],
            rank_b     = g['rank_b'],
            rank_w     = g['rank_w'],
            human_lp_b = g['human_lp_b'],
            human_lp_w = g['human_lp_w'],
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_meta(self, csv_path: str, split: str) -> List[Dict]:
        games = []
        with open(csv_path, 'r', newline='', encoding='utf-8', errors='replace') as f:
            for row in csv.DictReader(f):
                if split and row.get('set', '') != split:
                    continue
                rank_b = rank_str_to_idx(row.get('B_humanRank', ''))
                rank_w = rank_str_to_idx(row.get('W_humanRank', ''))
                b_hlp  = float(row.get('B_humanLogPrior', 0.0))
                w_hlp  = float(row.get('W_humanLogPrior', 0.0))
                if b_hlp == 0.0: rank_b = -1
                if w_hlp == 0.0: rank_w = -1
                games.append({
                    'id':        row['file'].strip(),
                    'sgf_path':  row.get('sgf_path', '').strip(),
                    'target_b':  float(row.get('B_logPrior', 0.0)),
                    'target_w':  float(row.get('W_logPrior', 0.0)),
                    'rank_b':    rank_b,
                    'rank_w':    rank_w,
                    'human_lp_b': b_hlp,
                    'human_lp_w': w_hlp,
                })
        return games

    def _detect_input_dim(self) -> int:
        for g in self._games:
            p = self.data_dir / f"{g['id']}_B.npz"
            if p.exists():
                return probe_kab2_dim(str(p))
        raise FileNotFoundError(f"No _B.npz files found in {self.data_dir}")

    def _load_moves(self, game_id: str, side: str) -> np.ndarray:
        key = f"{game_id}_{side}"
        if self.cache and key in self._move_cache:
            return self._move_cache[key]
        moves, _ = read_kab2(str(self.data_dir / f"{game_id}_{side}.npz"))
        if len(moves) > self.max_moves:
            moves = moves[-self.max_moves:]
        moves = np.ascontiguousarray(moves, dtype=np.float32)
        if self.cache:
            self._move_cache[key] = moves
        return moves


# ─── DataLoader factory ───────────────────────────────────────────────────────

def make_file_loader(
    data_dir: str,
    split: str = 'T',
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle: bool = True,
    **dataset_kwargs,
) -> tuple:
    """Returns (DataLoader, KAB2FileDataset)."""
    ds = KAB2FileDataset(data_dir, split=split, **dataset_kwargs)
    loader = DataLoader(
        ds,
        batch_size   = batch_size,
        shuffle      = shuffle,
        num_workers  = num_workers,
        collate_fn   = KAB2Base.collate,
        pin_memory   = False,
    )
    return loader, ds
