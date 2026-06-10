"""KataRank — KAB2 data pipeline"""

# File-based dataset (map-style, indexed)
from katarank.data.katago_native.dataset_file import (
    KAB2FileDataset,
    make_file_loader,
    rank_str_to_idx,
    NUM_RANK_CLASSES,
)

# Stream dataset (iterable, live pipe)
from katarank.data.katago_native.dataset_stream import KAB2StreamDataset

# Legacy alias so existing code using KAB2Dataset still works
from katarank.data.katago_native.dataset_kab2 import (
    KAB2Dataset,
    kab2_collate,
    make_kab2_loader,
)

__all__ = [
    # New canonical names
    'KAB2FileDataset', 'make_file_loader',
    'KAB2StreamDataset',
    'rank_str_to_idx', 'NUM_RANK_CLASSES',
    # Legacy aliases
    'KAB2Dataset', 'kab2_collate', 'make_kab2_loader',
]
