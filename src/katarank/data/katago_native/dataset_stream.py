"""
KataRank — KAB2 Stream Dataset

IterableDataset backed by a live StreamQueue (KataGo stdout pipe).
Suitable for online training without writing KAB2 files to disk.

Usage::

    engine = KataGoEngine(model='kata1-b18c384nbt.bin.gz')
    sq     = engine.stream_queue(sgfs, mode='full', buffer_size=64)
    ds     = KAB2StreamDataset(sq)

    loader = DataLoader(ds, batch_size=16, collate_fn=kab2_collate)
    for batch in loader:
        train_step(batch)
"""

from typing import Iterator, Optional

import numpy as np
from torch.utils.data import IterableDataset

from katarank.data.katago_native.dataset_kab2 import KAB2Sample, kab2_make_sample
from katarank.engine import StreamQueue


class KAB2StreamDataset(IterableDataset):
    """
    Wraps a StreamQueue as a PyTorch IterableDataset.

    input_dim is inferred from the first sample unless hint_input_dim is given.
    Does not support len() or random access — use KAB2Dataset for indexed training.
    """

    def __init__(
        self,
        queue: StreamQueue,
        hint_input_dim: Optional[int] = None,
        min_moves: int = 5,
        max_moves: int = 400,
    ):
        self._queue      = queue
        self._min_moves  = min_moves
        self._max_moves  = max_moves
        self._input_dim: Optional[int] = hint_input_dim

    @property
    def input_dim(self) -> int:
        if self._input_dim is None:
            raise RuntimeError(
                "input_dim not yet known — pass hint_input_dim= or "
                "iterate at least one sample first."
            )
        return self._input_dim

    def __iter__(self) -> Iterator[KAB2Sample]:
        for moves_b, moves_w, info_b, info_w in self._queue:
            moves_b = self._clip(moves_b)
            moves_w = self._clip(moves_w)
            if len(moves_b) < self._min_moves or len(moves_w) < self._min_moves:
                continue
            if self._input_dim is None:
                self._input_dim = moves_b.shape[1]
            yield kab2_make_sample(
                moves_b    = moves_b,
                moves_w    = moves_w,
                game_id    = info_b.get('game_id', ''),
                target_b   = info_b.get('mean_log_prior', 0.0),
                target_w   = info_w.get('mean_log_prior', 0.0),
                rank_b     = info_b.get('human_rank_idx', -1),
                rank_w     = info_w.get('human_rank_idx', -1),
                human_lp_b = info_b.get('human_log_prior', 0.0),
                human_lp_w = info_w.get('human_log_prior', 0.0),
            )

    def _clip(self, moves: np.ndarray) -> np.ndarray:
        return moves[-self._max_moves:] if len(moves) > self._max_moves else moves

    def close(self) -> None:
        self._queue.close()
