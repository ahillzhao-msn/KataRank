"""
KataRank — KAB2 Stream Dataset

IterableDataset backed by a live StreamQueue (KataGo stdout pipe).
Suitable for online training without writing KAB2 files to disk.

Usage::

    engine = KataGoEngine(model='kata1-b18c384nbt.bin.gz')
    sq     = engine.stream_queue(sgfs, mode='full', buffer_size=64)
    ds     = KAB2StreamDataset(sq)

    loader = DataLoader(ds, batch_size=16, collate_fn=KAB2Base.collate)
    for batch in loader:
        train_step(batch)

Note: IterableDataset does not support random access or len().
      Use KAB2FileDataset for multi-epoch indexed training.
"""

from typing import Dict, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from katarank.data.base import KAB2Base, KAB2Sample
from katarank.engine import StreamQueue


class KAB2StreamDataset(KAB2Base, IterableDataset):
    """
    Wraps a StreamQueue as a PyTorch IterableDataset.

    Dimensionality (input_dim, scalar_dim, trunk_dim) is inferred from
    the first sample received; before the first sample, properties
    raise RuntimeError unless `hint_input_dim` is provided.

    Args:
        queue:          A StreamQueue returned by KataGoEngine.stream_queue().
        hint_input_dim: Optional pre-declared input_dim (avoids waiting for
                        first sample to resolve dimensionality).
        min_moves:      Skip samples where either player has fewer moves.
        max_moves:      Truncate each player to last N moves.
    """

    def __init__(
        self,
        queue: StreamQueue,
        hint_input_dim: Optional[int] = None,
        min_moves: int = 5,
        max_moves: int = 400,
    ):
        self._queue     = queue
        self._min_moves = min_moves
        self._max_moves = max_moves

        # Resolved lazily from first sample unless hinted
        self._input_dim: Optional[int] = hint_input_dim
        self._scalar_dim = 10
        self._trunk_dim: Optional[int] = (
            (hint_input_dim - 10) // 2 if hint_input_dim else None
        )

    # ── KAB2Base properties ───────────────────────────────────────────────────

    @property
    def input_dim(self) -> int:
        if self._input_dim is None:
            raise RuntimeError(
                "input_dim not yet known. Pass hint_input_dim= or "
                "iterate at least one sample first."
            )
        return self._input_dim

    @property
    def scalar_dim(self) -> int:
        return self._scalar_dim

    @property
    def trunk_dim(self) -> int:
        if self._trunk_dim is None:
            raise RuntimeError("trunk_dim not yet resolved.")
        return self._trunk_dim

    def __len__(self):
        raise TypeError(
            "KAB2StreamDataset has no fixed length (live stream). "
            "Use KAB2FileDataset for indexed/multi-epoch training."
        )

    # ── Iterable interface ────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[KAB2Sample]:
        for moves_b, moves_w, info_b, info_w in self._queue:
            moves_b, moves_w = self._clip(moves_b), self._clip(moves_w)
            if (len(moves_b) < self._min_moves or
                    len(moves_w) < self._min_moves):
                continue

            # Resolve dim on first sample
            if self._input_dim is None:
                self._input_dim = moves_b.shape[1]
                self._trunk_dim = (self._input_dim - self._scalar_dim) // 2

            game_id = info_b.get('game_id', '')

            yield KAB2Base._make_sample(
                moves_b    = moves_b,
                moves_w    = moves_w,
                game_id    = game_id,
                target_b   = info_b.get('mean_log_prior', 0.0),
                target_w   = info_w.get('mean_log_prior', 0.0),
                rank_b     = info_b.get('human_rank_idx', -1),
                rank_w     = info_w.get('human_rank_idx', -1),
                human_lp_b = info_b.get('human_log_prior', 0.0),
                human_lp_w = info_w.get('human_log_prior', 0.0),
            )

    def _clip(self, moves: np.ndarray) -> np.ndarray:
        if len(moves) > self._max_moves:
            return moves[-self._max_moves:]
        return moves

    def close(self) -> None:
        """Terminate the underlying stream."""
        self._queue.close()
