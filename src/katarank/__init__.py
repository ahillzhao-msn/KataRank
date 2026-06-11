"""
KataRank — Python library
==========================
High-level interface exposing KataGo analysis as a Python-native pipeline.

Quick start:

    from katarank import KataGoEngine
    from model import KataRankModel
    import torch

    engine = KataGoEngine(model='kata1-b18c384nbt.bin.gz')
    rank_model = KataRankModel.load('katarank.pt')

    # Stream mode — no disk I/O
    for x_b, x_w, info_b, info_w in engine.stream_to_tensors(sgf_paths=['game.sgf']):
        x = torch.cat([x_b, x_w], dim=0)
        out = rank_model(x, xlens=[len(x_b) + len(x_w)])
        print('b_rating:', out['b_rating'].item())

    # Lite mode — scalars only (10 dims, 100× faster)
    for side, moves, info in engine.stream_games(sgf_paths=['game.sgf'], mode='lite'):
        print(side, moves.shape, info['mean_log_prior'])
"""

from katarank.engine import KataGoEngine, parse_kab2_buffer
from katarank.workflow import TrainingWorkflow, InferenceWorkflow, RankResult
from katarank.schema import KAB2Sample, KAB2Batch, BaseKAB2Dataset, KAB2Output
from katarank.schema import output_to_json, output_from_json, save_output, load_output
from katarank.schema import save_outputs_batch, load_outputs_batch, rank_idx_to_str
from katarank.data.datasets import KAB2Dataset, KAB2StreamDataset

__all__ = [
    # Engine
    'KataGoEngine', 'parse_kab2_buffer',
    # Workflows
    'TrainingWorkflow', 'InferenceWorkflow',
    # Data classes
    'KAB2Dataset', 'KAB2StreamDataset', 'KAB2Sample', 'KAB2Batch',
    # Schema
    'BaseKAB2Dataset', 'KAB2Output',
    'output_to_json', 'output_from_json',
    'save_output', 'load_output',
    'save_outputs_batch', 'load_outputs_batch',
    'rank_idx_to_str',
    # Result type
    'RankResult',
]
