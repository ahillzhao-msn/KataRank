"""
KataRank — KAB2 Dataset Abstractions

KAB2Base is the common contract for all KAB2-backed datasets.
Subclasses differ in data source (files vs live stream) and feature mode
(full trunk+scalar vs lite scalar-only).

Hierarchy
---------
KAB2Base  (ABC, torch Dataset)
├── KAB2FileDataset    — reads _B.npz/_W.npz file pairs (indexed, random-access)
└── KAB2StreamDataset  — backed by a StreamQueue (iterable, live stream)

A KAB2Sample is the canonical dict returned by every dataset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset


# ─── Sample schema ────────────────────────────────────────────────────────────

class KAB2Sample(TypedDict):
    """One paired game sample, both players concatenated."""
    x:          torch.Tensor   # (N_b + N_w, input_dim) float32
    seq_len:    int
    target_b:   torch.Tensor   # scalar float32 — meanLogPrior for Black
    target_w:   torch.Tensor   # scalar float32 — meanLogPrior for White
    rank_b:     torch.Tensor   # scalar long    — humanRankIdx (0-28 or -1)
    rank_w:     torch.Tensor   # scalar long
    human_lp_b: torch.Tensor   # scalar float32 — HumanSL confidence weight
    human_lp_w: torch.Tensor   # scalar float32
    game_id:    str


# ─── Abstract base ────────────────────────────────────────────────────────────

class KAB2Base(Dataset, ABC):
    """
    Abstract base class for all KAB2 datasets.

    Subclasses must expose dimensionality properties and implement __getitem__.
    All datasets use the same KAB2Sample schema so collate functions are
    shared regardless of source (files, stream, in-memory).
    """

    # ── Required properties ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Total feature dimension per move: scalar_dim + 2 * trunk_dim."""

    @property
    @abstractmethod
    def scalar_dim(self) -> int:
        """Number of scalar fields per move (always 10)."""

    @property
    @abstractmethod
    def trunk_dim(self) -> int:
        """Trunk channels per move (0 in lite mode)."""

    # ── Convenience predicates ────────────────────────────────────────────────

    @property
    def is_lite(self) -> bool:
        """True when trunk_dim == 0 (scalars only, -no-trunk mode)."""
        return self.trunk_dim == 0

    @property
    def is_full(self) -> bool:
        """True when trunk_dim > 0 (full trunk + scalar features)."""
        return self.trunk_dim > 0

    # ── Shared collate ────────────────────────────────────────────────────────

    @staticmethod
    def collate(batch) -> Dict:
        """
        Pack a list of KAB2Sample dicts into a batched dict.

        Compatible with torch.utils.data.DataLoader(collate_fn=KAB2Base.collate).
        Filters out empty samples (seq_len == 0).
        """
        batch = [b for b in batch if b and b.get('seq_len', 0) > 0]
        if not batch:
            return {}
        return {
            'x':          torch.cat([b['x'] for b in batch], dim=0),
            'xlens':      [b['seq_len'] for b in batch],
            'target_b':   torch.stack([b['target_b']   for b in batch]),
            'target_w':   torch.stack([b['target_w']   for b in batch]),
            'rank_b':     torch.stack([b['rank_b']     for b in batch]),
            'rank_w':     torch.stack([b['rank_w']     for b in batch]),
            'human_lp_b': torch.stack([b['human_lp_b'] for b in batch]),
            'human_lp_w': torch.stack([b['human_lp_w'] for b in batch]),
            'game_ids':   [b['game_id'] for b in batch],
        }

    # ── Factory helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _make_sample(
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
        """Build a KAB2Sample from numpy move arrays."""
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

    @staticmethod
    def _empty_sample(
        game_id: str,
        input_dim: int,
        target_b: float = 0.0,
        target_w: float = 0.0,
    ) -> KAB2Sample:
        return KAB2Sample(
            x          = torch.zeros(0, input_dim),
            seq_len    = 0,
            target_b   = torch.tensor(target_b, dtype=torch.float32),
            target_w   = torch.tensor(target_w, dtype=torch.float32),
            rank_b     = torch.tensor(-1, dtype=torch.long),
            rank_w     = torch.tensor(-1, dtype=torch.long),
            human_lp_b = torch.tensor(0.0, dtype=torch.float32),
            human_lp_w = torch.tensor(0.0, dtype=torch.float32),
            game_id    = game_id,
        )
