"""
KataRank — KAB2 Stream Dataset
===============================
PyTorch IterableDataset that wraps a KataGoEngine.stream_to_tensors() pipeline.

Reads games on-the-fly from katago.exe via stdin pipe — zero disk I/O.
Yields KAB2Sample items compatible with kab2_collate.

Supports all SgfInput modes: file paths, SGF strings, SGF directories.

Usage::

    engine = KataGoEngine(model='kata1-b18c384nbt.bin.gz')
    ds     = KAB2StreamDataset(engine, sgf_paths=['g1.sgf', 'g2.sgf'], mode='lite')

    loader = DataLoader(ds, batch_size=16, collate_fn=kab2_collate)
    for batch in loader:
        train_step(batch)
"""

from typing import Iterator, List, Optional, Union

import numpy as np
from torch.utils.data import IterableDataset

from katarank.engine import KataGoEngine
from katarank.schema import (
    BaseKAB2Dataset,
    KAB2Sample,
    kab2_make_sample,
)


class KAB2StreamDataset(IterableDataset, BaseKAB2Dataset):
    """
    Wraps a KataGoEngine stream as a PyTorch IterableDataset.

    input_dim is inferred from the first sample; not available at construction
    time. Call .input_dim only after iterating at least one sample.
    Supports iter-style only (no random access, no __getitem__).
    """

    def __init__(
        self,
        engine: KataGoEngine,
        sgf_paths: Optional[Union[List[str], str]] = None,
        *,
        sgf_strings: Optional[List[str]] = None,
        sgf_dir: Optional[str] = None,
        mode: str = 'full',
        min_moves: int = 5,
        max_moves: int = 400,
        **engine_kwargs,
    ):
        self._engine      = engine
        self._mode        = mode
        self._min_moves   = min_moves
        self._max_moves   = max_moves
        self._engine_kwargs = engine_kwargs
        self._input_dim: Optional[int] = None

        # Store all input variants for __iter__ resolution
        self._sgf_paths   = [sgf_paths] if isinstance(sgf_paths, str) else sgf_paths
        self._sgf_strings = sgf_strings
        self._sgf_dir     = sgf_dir

        if not any([self._sgf_paths, self._sgf_strings, self._sgf_dir]):
            raise ValueError(
                "Provide at least one of: sgf_paths, sgf_strings, sgf_dir"
            )

    @property
    def input_dim(self) -> int:
        if self._input_dim is None:
            raise RuntimeError(
                "input_dim not yet known — iterate at least one sample first, "
                "or pass hint_input_dim= to constructor."
            )
        return self._input_dim

    def __len__(self) -> int:
        raise TypeError("KAB2StreamDataset has no __len__ — it's an IterableDataset")

    def __getitem__(self, idx: int) -> KAB2Sample:
        raise TypeError("KAB2StreamDataset has no __getitem__ — use iteration instead")

    def __iter__(self) -> Iterator[KAB2Sample]:
        for x_b, x_w, info_b, info_w in self._engine.stream_to_tensors(
            sgf_paths=self._sgf_paths,
            sgf_strings=self._sgf_strings,
            sgf_dir=self._sgf_dir,
            mode=self._mode,
            **self._engine_kwargs
        ):
            x_b_clip = x_b[-self._max_moves:] if len(x_b) > self._max_moves else x_b
            x_w_clip = x_w[-self._max_moves:] if len(x_w) > self._max_moves else x_w
            if len(x_b_clip) < self._min_moves or len(x_w_clip) < self._min_moves:
                continue
            if self._input_dim is None:
                self._input_dim = x_b_clip.shape[1]
            yield kab2_make_sample(
                moves_b    = x_b_clip.numpy(),
                moves_w    = x_w_clip.numpy(),
                game_id    = info_b.get('game_id', ''),
                target_b   = info_b.get('mean_log_prior', 0.0),
                target_w   = info_w.get('mean_log_prior', 0.0),
                rank_b     = info_b.get('human_rank_idx', -1),
                rank_w     = info_w.get('human_rank_idx', -1),
                human_lp_b = info_b.get('human_log_prior', 0.0),
                human_lp_w = info_w.get('human_log_prior', 0.0),
            )
