"""
KataRank — KAB2 Dataset
========================
PyTorch Dataset for training KataRankModel on KAB2-format game files.

File naming convention:
    game_{id:016X}_B.npz   ← Black player's moves
    game_{id:016X}_W.npz   ← White player's moves

One Dataset sample = ONE GAME (both players concatenated).
The model receives the full game and predicts both players' ratings.

Layout of the packed move tensor x (N_total, input_dim):
    First  N_b rows: Black moves  (scalar[7] == 0)
    Last   N_w rows: White moves  (scalar[7] == 1)

Targets returned per sample:
    target_b      float   meanLogPrior for Black (primary rating signal)
    target_w      float   meanLogPrior for White
    rank_b        int     humanRankIdx (0=20k…28=9d, -1 if not computed)
    rank_w        int     humanRankIdx for White
    human_lp_b    float   HumanSL log-prior confidence for Black
    human_lp_w    float   HumanSL log-prior confidence for White
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from katarank.data.preprocess import read_kab2, probe_kab2_dim


# ─── Canonical sample type ────────────────────────────────────────────────────

class KAB2Sample(TypedDict):
    """One paired game sample — both players' moves concatenated."""
    x:          torch.Tensor   # (N_b + N_w, input_dim) float32
    seq_len:    int
    target_b:   torch.Tensor   # meanLogPrior for Black
    target_w:   torch.Tensor   # meanLogPrior for White
    rank_b:     torch.Tensor   # humanRankIdx (0-28 or -1)
    rank_w:     torch.Tensor
    human_lp_b: torch.Tensor   # HumanSL confidence weight
    human_lp_w: torch.Tensor
    game_id:    str


def kab2_make_sample(
    moves_b: np.ndarray,
    moves_w: np.ndarray,
    game_id: str,
    target_b: float = 0.0,
    target_w: float = 0.0,
    rank_b: int = -1,
    rank_w: int = -1,
    human_lp_b: float = 0.0,
    human_lp_w: float = 0.0,
) -> KAB2Sample:
    """Build a KAB2Sample from numpy move arrays (Black first, then White)."""
    x = np.concatenate([moves_b, moves_w], axis=0)
    return KAB2Sample(
        x          = torch.from_numpy(x),
        seq_len    = len(x),
        target_b   = torch.tensor(target_b,   dtype=torch.float32),
        target_w   = torch.tensor(target_w,   dtype=torch.float32),
        rank_b     = torch.tensor(rank_b,     dtype=torch.long),
        rank_w     = torch.tensor(rank_w,     dtype=torch.long),
        human_lp_b = torch.tensor(human_lp_b, dtype=torch.float32),
        human_lp_w = torch.tensor(human_lp_w, dtype=torch.float32),
        game_id    = game_id,
    )


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

class KAB2Dataset(Dataset):
    """
    Game-level dataset reading KAB2 binary files.

    Each __getitem__ returns one game:  both players' move features packed
    together, plus scalar training targets derived from the file headers.
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
        """
        Args:
            data_dir: Directory containing game_*_B.npz / game_*_W.npz files.
            meta_csv: Path to _meta.csv; auto-detected as data_dir/_meta.csv if None.
            split: 'T' (train), 'V' (val), 'E' (test), or '' (all).
            max_moves_per_player: Truncate long sequences.
            min_moves_per_player: Skip games shorter than this.
            cache: Cache loaded move arrays in memory.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_moves = max_moves_per_player
        self.min_moves = min_moves_per_player
        self.cache = cache
        self._move_cache: Dict[str, np.ndarray] = {}

        meta_path = meta_csv or str(self.data_dir / '_meta.csv')
        self.games = self._load_meta(meta_path, split)

        # Detect input_dim from the first available file
        self.input_dim = self._detect_input_dim()

        print(
            f"KAB2Dataset: {len(self.games)} games "
            f"split='{split}'  input_dim={self.input_dim}"
        )

    # ── Meta loading ──────────────────────────────────────────────────────────

    def _load_meta(self, csv_path: str, split: str) -> List[Dict]:
        games = []
        with open(csv_path, 'r', newline='', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split and row.get('set', '') != split:
                    continue

                rank_b_idx = rank_str_to_idx(row.get('B_humanRank', ''))
                rank_w_idx = rank_str_to_idx(row.get('W_humanRank', ''))

                # humanLogPrior=0.0 with a valid rank string means HumanSL wasn't
                # actually run — treat as no-label.
                b_hlp = float(row.get('B_humanLogPrior', 0.0))
                w_hlp = float(row.get('W_humanLogPrior', 0.0))
                if b_hlp == 0.0:
                    rank_b_idx = -1
                if w_hlp == 0.0:
                    rank_w_idx = -1

                games.append({
                    'id':          row['file'].strip(),
                    'sgf_path':    row.get('sgf_path', '').strip(),  # added in newer meta
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

    # ── Move loading ──────────────────────────────────────────────────────────

    def _load_moves(self, game_id: str, side: str) -> np.ndarray:
        """Load and cache move records for one player side ('B' or 'W')."""
        key = f"{game_id}_{side}"
        if self.cache and key in self._move_cache:
            return self._move_cache[key]

        path = str(self.data_dir / f"{game_id}_{side}.npz")
        moves, _ = read_kab2(path)

        # Truncate long sequences (keep the last max_moves — endgame has higher info)
        if len(moves) > self.max_moves:
            moves = moves[-self.max_moves:]

        moves = np.ascontiguousarray(moves, dtype=np.float32)
        if self.cache:
            self._move_cache[key] = moves
        return moves

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, idx: int) -> Dict:
        g = self.games[idx]
        gid = g['id']

        b_moves = self._load_moves(gid, 'B')  # (N_b, input_dim)
        w_moves = self._load_moves(gid, 'W')  # (N_w, input_dim)

        if len(b_moves) < self.min_moves or len(w_moves) < self.min_moves:
            # Return empty placeholder — collate will drop or handle these
            return self._empty_sample(g)

        # Concatenate: Black first, then White.
        # _split() in DualViewSetTransformer uses scalar[7] (isWhite) to re-separate them.
        x = np.concatenate([b_moves, w_moves], axis=0)   # (N_b + N_w, input_dim)

        return {
            'x':          torch.from_numpy(x),
            'seq_len':    len(x),
            'target_b':   torch.tensor(g['target_b'],   dtype=torch.float32),
            'target_w':   torch.tensor(g['target_w'],   dtype=torch.float32),
            'rank_b':     torch.tensor(g['rank_b'],     dtype=torch.long),
            'rank_w':     torch.tensor(g['rank_w'],     dtype=torch.long),
            'human_lp_b': torch.tensor(g['human_lp_b'], dtype=torch.float32),
            'human_lp_w': torch.tensor(g['human_lp_w'], dtype=torch.float32),
            'game_id':    gid,
        }

    def _empty_sample(self, g: Dict) -> Dict:
        return {
            'x':          torch.zeros(0, self.input_dim),
            'seq_len':    0,
            'target_b':   torch.tensor(g['target_b'],   dtype=torch.float32),
            'target_w':   torch.tensor(g['target_w'],   dtype=torch.float32),
            'rank_b':     torch.tensor(-1, dtype=torch.long),
            'rank_w':     torch.tensor(-1, dtype=torch.long),
            'human_lp_b': torch.tensor(0.0, dtype=torch.float32),
            'human_lp_w': torch.tensor(0.0, dtype=torch.float32),
            'game_id':    g['id'],
        }


# ─── Collate ──────────────────────────────────────────────────────────────────

def kab2_collate(batch: List[Dict]) -> Dict:
    """
    Pack variable-length game sequences into a flat tensor + xlens list.

    Filters out empty samples (games with too few moves).
    """
    batch = [item for item in batch if item['seq_len'] > 0]
    if not batch:
        return {}

    x_list   = [item['x'] for item in batch]
    xlens    = [item['seq_len'] for item in batch]

    return {
        'x':          torch.cat(x_list, dim=0),      # (sum_N, input_dim)
        'xlens':      xlens,
        'target_b':   torch.stack([item['target_b']   for item in batch]),
        'target_w':   torch.stack([item['target_w']   for item in batch]),
        'rank_b':     torch.stack([item['rank_b']     for item in batch]),
        'rank_w':     torch.stack([item['rank_w']     for item in batch]),
        'human_lp_b': torch.stack([item['human_lp_b'] for item in batch]),
        'human_lp_w': torch.stack([item['human_lp_w'] for item in batch]),
        'game_ids':   [item['game_id'] for item in batch],
    }


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
