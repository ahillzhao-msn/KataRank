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
    for x_b, x_w, info_b, info_w in engine.stream_to_tensors(['game.sgf']):
        x = torch.cat([x_b, x_w], dim=0)
        out = rank_model(x, xlens=[len(x_b) + len(x_w)])
        print('b_rating:', out['b_rating'].item())

    # Lite mode — scalars only (10 dims, 100× faster)
    for side, moves, info in engine.stream_games(['game.sgf'], mode='lite'):
        print(side, moves.shape, info['mean_log_prior'])

    # Queue mode — producer/consumer for training loops (no blocking on I/O)
    sq = engine.stream_queue(sgfs, mode='full', buffer_size=64)
    for moves_b, moves_w, info_b, info_w in sq:
        x = torch.cat([torch.from_numpy(moves_b), torch.from_numpy(moves_w)])
        out = rank_model(x, xlens=[len(moves_b) + len(moves_w)])
"""

from katarank.engine import KataGoEngine, StreamQueue, parse_kab2_buffer
from katarank.pool import KataGoPool
from katarank.workflow import TrainingWorkflow, InferenceWorkflow, MixedWorkflow

__all__ = [
    'KataGoEngine', 'StreamQueue', 'parse_kab2_buffer',
    'KataGoPool',
    'TrainingWorkflow', 'InferenceWorkflow', 'MixedWorkflow',
]
