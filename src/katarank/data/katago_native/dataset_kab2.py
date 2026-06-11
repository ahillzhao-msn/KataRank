"""
KataRank — KAB2 File Dataset
=============================
PyTorch Dataset for training KataRankModel on KAB2-format NPZ files.

File naming convention:
    game_{id:016X}_B.npz   ← Black player's moves
    game_{id:016X}_W.npz   ← White player's moves

Inherits from BaseKAB2Dataset — each item is a KAB2Sample.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from katarank.data.katago_native.dataset_base import (
    BaseKAB2Dataset,
    KAB2Sample,
    kab2_make_sample,
    kab2_collate,
)
from katarank.data.preprocess import read_kab2, probe_kab2_dim

# ─── Rank string → integer index ──────────────────────────────────────────────

_RANK_NAMES = [
    '20k','19k','18k','17k','16k','15k','14k','13k','12k','11k',
    '10k','9k','8k','7k','6k','5k','4k','3k','2k','1k',
    '1d','2d','3d','4d','5d','6d','7d','8d','9d',
]
_RANK_TO_IDX: Dict[str, int] = {f'rank_{name}': i for i, name in enumerate(_RANK_NAMES)}
NUM_RANK_CLASSES = len(_RANK_NAMES)   # 29


def rank_str_to_idx(s: str) -> int:
    """Convert 'rank_8d' → 28, 'rank_20k' → 0; unknown → -1."""
    return _RANK_TO_IDX.get(str(s).strip(), -1)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class KAB2Dataset(BaseKAB2Dataset):
    """
    Game-level dataset reading KAB2 binary NPZ files from disk.

    Each __getitem__ returns one game: both players' move features packed
    together, plus scalar training targets derived from the file headers.

    Implements BaseKAB2Dataset.map-style access.
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
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_moves = max_moves_per_player
        self.min_moves = min_moves_per_player
        self.cache = cache
        self._move_cache: Dict[str, np.ndarray] = {}

        meta_path = meta_csv or str(self.data_dir / '_meta.csv')
        self.games = self._load_meta(meta_path, split)

        self._input_dim = self._detect_input_dim()

        print(
            f"KAB2Dataset: {len(self.games)} games "
            f"split='{split}'  input_dim={self._input_dim}"
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def __len__(self) -> int:
        return len(self.games)

    def _load_meta(self, csv_path: str, split: str) -> List[Dict]:
        games = []
        with open(csv_path, 'r', newline='', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split and row.get('set', '') != split:
                    continue

                rank_b_idx = rank_str_to_idx(row.get('B_humanRank', ''))
                rank_w_idx = rank_str_to_idx(row.get('W_humanRank', ''))

                b_hlp = float(row.get('B_humanLogPrior', 0.0))
                w_hlp = float(row.get('W_humanLogPrior', 0.0))
                if b_hlp == 0.0:
                    rank_b_idx = -1
                if w_hlp == 0.0:
                    rank_w_idx = -1

                games.append({
                    'id':          row['file'].strip(),
                    'sgf_path':    row.get('sgf_path', '').strip(),
                    'target_b':    float(row.get('B_logPrior', 0.0)),
                    'target_w':    float(row.get('W_logPrior', 0.0)),
                    'rank_b':      rank_b_idx,
                    'rank_w':      rank_w_idx,
                    'human_lp_b':  b_hlp,
                    'human_lp_w':  w_hlp,
                })
        return games

    def _detect_input_dim(self) -> int:
        for g in self.games:
            b_path = self.data_dir / f"{g['id']}_B.npz"
            if b_path.exists():
                return probe_kab2_dim(str(b_path))
        raise FileNotFoundError(
            f"No _B.npz files found in {self.data_dir}"
        )

    def _load_moves(self, game_id: str, side: str) -> np.ndarray:
        key = f"{game_id}_{side}"
        if self.cache and key in self._move_cache:
            return self._move_cache[key]

        path = str(self.data_dir / f"{game_id}_{side}.npz")
        moves, _ = read_kab2(path)

        if len(moves) > self.max_moves:
            moves = moves[-self.max_moves:]

        moves = np.ascontiguousarray(moves, dtype=np.float32)
        if self.cache:
            self._move_cache[key] = moves
        return moves

    def __getitem__(self, idx: int) -> KAB2Sample:
        g = self.games[idx]
        gid = g['id']

        b_moves = self._load_moves(gid, 'B')
        w_moves = self._load_moves(gid, 'W')

        if len(b_moves) < self.min_moves or len(w_moves) < self.min_moves:
            return self._empty_sample(g)

        return kab2_make_sample(
            moves_b=b_moves, moves_w=w_moves, game_id=gid,
            target_b=g['target_b'], target_w=g['target_w'],
            rank_b=g['rank_b'], rank_w=g['rank_w'],
            human_lp_b=g['human_lp_b'], human_lp_w=g['human_lp_w'],
        )

    def _empty_sample(self, g: Dict) -> KAB2Sample:
        return KAB2Sample(
            x=torch.zeros(0, self._input_dim),
            seq_len=0,
            target_b=torch.tensor(g['target_b'], dtype=torch.float32),
            target_w=torch.tensor(g['target_w'], dtype=torch.float32),
            rank_b=torch.tensor(-1, dtype=torch.long),
            rank_w=torch.tensor(-1, dtype=torch.long),
            human_lp_b=torch.tensor(0.0, dtype=torch.float32),
            human_lp_w=torch.tensor(0.0, dtype=torch.float32),
            game_id=g['id'],
        )


# ─── DataLoader factory ───────────────────────────────────────────────────────

def make_kab2_loader(
    data_dir: str,
    split: str = 'T',
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """Convenience factory that wires KAB2Dataset + kab2_collate."""
    ds = KAB2Dataset(data_dir, split=split, **dataset_kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=kab2_collate,
        pin_memory=False,
    ), ds
