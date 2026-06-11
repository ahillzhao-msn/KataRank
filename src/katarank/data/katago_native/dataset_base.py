"""
KataRank — Base KAB2 Dataset (Abstract)
========================================
Defines the canonical contract for all KAB2 data sources.

Two concrete subclasses:
  - KAB2Dataset         (file-based: reads _B.npz/_W.npz from disk)
  - KAB2StreamDataset   (stream-based: pipes from katago.exe via KataGoEngine)

Every dataset yields KAB2Sample — one game with both players' moves concatenated.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TypedDict, Union

import torch


# ─── Canonical sample type ────────────────────────────────────────────────────

class KAB2Sample(TypedDict):
    """One paired game sample — both players' moves concatenated.

    Tensor layout: x is (N_b + N_w, input_dim) float32.
    First N_b rows: Black moves (scalar[7] == 0).
    Last  N_w rows: White moves (scalar[7] == 1).
    """
    x:          torch.Tensor   # (N_b + N_w, input_dim) float32
    seq_len:    int
    target_b:   torch.Tensor   # meanLogPrior for Black (rating signal)
    target_w:   torch.Tensor   # meanLogPrior for White
    rank_b:     torch.Tensor   # humanRankIdx (0-28 or -1)
    rank_w:     torch.Tensor
    human_lp_b: torch.Tensor   # HumanSL confidence weight
    human_lp_w: torch.Tensor
    game_id:    str


class KAB2Batch(TypedDict):
    """Collated batch for model.forward().

    Produced by kab2_collate(). Variable-length games are packed into
    a flat tensor; xlens tracks per-game sequence lengths.
    """
    x:          torch.Tensor   # (sum_N, input_dim)
    xlens:      List[int]      # per-game lengths, len = batch_size
    target_b:   torch.Tensor   # (batch_size,)
    target_w:   torch.Tensor
    rank_b:     torch.Tensor
    rank_w:     torch.Tensor
    human_lp_b: torch.Tensor
    human_lp_w: torch.Tensor
    game_ids:   List[str]


# ─── Abstract base ───────────────────────────────────────────────────────────

class BaseKAB2Dataset(ABC):
    """
    Abstract base for all KAB2 data sources.

    Properties:
        input_dim: Number of features per move (10 for lite mode,
                   10 + 2*trunkCh for full mode).

    Yields (via __getitem__ or __iter__):
        KAB2Sample — one game with both players' moves concatenated.

    Subclasses must implement:
        input_dim (property)
        __len__ (int, number of games)
    """

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Feature dimensionality per move."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of games in this dataset."""
        ...

    # Optional: allow __getitem__ for map-style access
    @abstractmethod
    def __getitem__(self, idx: int) -> KAB2Sample:
        ...


def kab2_make_sample(
    moves_b: 'np.ndarray',
    moves_w: 'np.ndarray',
    game_id: str,
    target_b: float = 0.0,
    target_w: float = 0.0,
    rank_b: int = -1,
    rank_w: int = -1,
    human_lp_b: float = 0.0,
    human_lp_w: float = 0.0,
) -> KAB2Sample:
    """Build a KAB2Sample from numpy move arrays (Black first, then White)."""
    import numpy as np
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


# ─── Collate ──────────────────────────────────────────────────────────────────

def kab2_collate(batch: List[Union[KAB2Sample, Dict]]) -> KAB2Batch:
    """
    Pack variable-length game sequences into a flat tensor + xlens list.

    Filters out empty samples (games with too few moves).
    """
    batch = [item for item in batch if item['seq_len'] > 0]
    if not batch:
        return KAB2Batch(
            x=torch.empty(0), xlens=[], target_b=torch.empty(0),
            target_w=torch.empty(0), rank_b=torch.empty(0),
            rank_w=torch.empty(0), human_lp_b=torch.empty(0),
            human_lp_w=torch.empty(0), game_ids=[],
        )

    x_list  = [item['x'] for item in batch]
    xlens   = [item['seq_len'] for item in batch]

    return KAB2Batch(
        x          = torch.cat(x_list, dim=0),
        xlens      = xlens,
        target_b   = torch.stack([item['target_b']   for item in batch]),
        target_w   = torch.stack([item['target_w']   for item in batch]),
        rank_b     = torch.stack([item['rank_b']     for item in batch]),
        rank_w     = torch.stack([item['rank_w']     for item in batch]),
        human_lp_b = torch.stack([item['human_lp_b'] for item in batch]),
        human_lp_w = torch.stack([item['human_lp_w'] for item in batch]),
        game_ids   = [item['game_id'] for item in batch],
    )
