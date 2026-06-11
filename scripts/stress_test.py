#!/usr/bin/env python
"""
KataRank — 50-game stress test: optimal vs slowest pipeline.

Two groups of 25 games (20 analysis+training, 5 inference), sampled randomly
across rank-level subdirectories of the SGF corpus.

Group A (optimal):  persistent daemon engine, lite stream (10 dims),
                    in-memory training samples, inference via the same daemon.
Group B (slowest):  one-shot processes, full trunk features (10+2C dims),
                    compressed disk round-trip, one cold engine per
                    inference game.

Both groups run HumanSL annotation (-human-model), train the same model
architecture for the same number of epochs, and report per-stage timings.

Usage:
    uv run python scripts/stress_test.py [--seed 42] [--epochs 10]
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from katarank import KataGoEngine, PersistentKataGoEngine, KAB2Dataset
from katarank.schema import kab2_make_sample, kab2_collate
from katarank.workflow import TrainingWorkflow, InferenceWorkflow, run_rank_files
from katarank.model import KataRankModel, KataRankLoss

KATAGO_BIN  = os.environ.get('KATAGO_BIN', 'katago')
MODEL       = os.environ.get('KATAGO_MODEL', os.path.expanduser('~/.katago/default_model.bin.gz'))
HUMAN_MODEL = os.environ.get('KATAGO_HUMAN_MODEL', os.path.expanduser('~/.katago/b18c384nbt-humanv0.bin.gz'))
SGF_ROOT    = Path(os.environ.get('KATARANK_SGF_CORPUS', os.path.expanduser('~/sgf-corpus')))

MAX_MOVES = 400   # clip per player, matches KAB2Dataset default


# ─── Sampling ─────────────────────────────────────────────────────────────────

def sample_games(n: int, seed: int):
    """Round-robin across rank-level dirs for a scattered rank distribution."""
    rng = random.Random(seed)
    groups = sorted(d for d in SGF_ROOT.iterdir() if d.is_dir())
    pools = {d.name: rng.sample(sorted(d.glob('*.sgf')),
                                min(n, len(list(d.glob('*.sgf')))))
             for d in groups}
    picked, names = [], sorted(pools)
    i = 0
    while len(picked) < n:
        name = names[i % len(names)]
        if pools[name]:
            picked.append((str(pools[name].pop()), name))
        i += 1
    rng.shuffle(picked)
    return picked


# ─── Shared training/inference harness ────────────────────────────────────────

class MemDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]


def train_model(samples_or_ds, input_dim: int, epochs: int, batch_size: int = 8):
    ds = MemDataset(samples_or_ds) if isinstance(samples_or_ds, list) else samples_or_ds
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=kab2_collate)
    model = KataRankModel(input_dim=input_dim, hidden_dim=64, num_heads=2,
                          num_inducing=8, encoder_depth=1, cross_depth=1)
    loss_fn = KataRankLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    wf = TrainingWorkflow(model, loss_fn, opt, log_every=0)
    wf.run_loader(loader, epochs=epochs)
    return model


def collect_stream_samples(engine, paths, mode):
    samples = []
    for x_b, x_w, ib, iw in engine.stream_to_tensors(sgf_paths=paths, mode=mode):
        samples.append(kab2_make_sample(
            moves_b=x_b[-MAX_MOVES:].numpy(), moves_w=x_w[-MAX_MOVES:].numpy(),
            game_id=ib.get('game_id', ''),
            target_b=ib['mean_log_prior'], target_w=iw['mean_log_prior'],
            rank_b=ib['human_rank_idx'], rank_w=iw['human_rank_idx'],
            human_lp_b=ib['human_log_prior'], human_lp_w=iw['human_log_prior'],
        ))
    return samples


# ─── Group A: optimal ─────────────────────────────────────────────────────────

def run_group_a(train_games, infer_games, epochs):
    t = {}
    t0 = time.time()
    engine = PersistentKataGoEngine(
        model=MODEL, human_model=HUMAN_MODEL, katago_bin=KATAGO_BIN, mode='lite')
    engine.start()
    t['startup'] = time.time() - t0

    t0 = time.time()
    samples = collect_stream_samples(engine, train_games, mode='lite')
    t['analysis_20'] = time.time() - t0

    t0 = time.time()
    model = train_model(samples, input_dim=10, epochs=epochs)
    t['train'] = time.time() - t0

    t0 = time.time()
    wf = InferenceWorkflow(model, engine)
    outputs = run_rank_files(engine, wf, infer_games, mode='lite')
    t['inference_5'] = time.time() - t0

    engine.close()
    t['total'] = sum(t.values())
    return t, len(samples), len(outputs), 10


# ─── Group B: slowest ─────────────────────────────────────────────────────────

def run_group_b(train_games, infer_games, epochs):
    t = {}
    out_dir = Path(tempfile.mkdtemp(prefix='katarank_stress_'))
    try:
        # Analysis: one-shot process, full trunk, compressed disk write
        t0 = time.time()
        engine = KataGoEngine(model=MODEL, human_model=HUMAN_MODEL,
                              katago_bin=KATAGO_BIN)
        rc = engine.batch_to_files(str(out_dir), sgf_paths=train_games, mode='full')
        t['analysis_20'] = time.time() - t0
        if rc != 0:
            raise RuntimeError(f'batch_to_files exit code {rc}')

        # Training: disk round-trip through KAB2Dataset
        t0 = time.time()
        ds = KAB2Dataset(str(out_dir), split='T', cache=False)
        input_dim = ds.input_dim
        model = train_model(ds, input_dim=input_dim, epochs=epochs)
        t['train'] = time.time() - t0

        # Inference: one cold engine per game (model reload every time)
        t0 = time.time()
        outputs = []
        for g in infer_games:
            cold = KataGoEngine(model=MODEL, human_model=HUMAN_MODEL,
                                katago_bin=KATAGO_BIN)
            wf = InferenceWorkflow(model, cold)
            outputs += run_rank_files(cold, wf, [g], mode='full')
        t['inference_5'] = time.time() - t0

        t['total'] = sum(t.values())
        return t, len(ds), len(outputs), input_dim
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--report', default='scripts/stress_report.json')
    args = ap.parse_args()

    games = sample_games(50, args.seed)
    dist = {}
    for _, grp in games:
        dist[grp] = dist.get(grp, 0) + 1
    print(f'sampled 50 games, rank distribution: {dist}', flush=True)

    a_paths = [p for p, _ in games[:25]]
    b_paths = [p for p, _ in games[25:]]
    a_train, a_infer = a_paths[:20], a_paths[20:]
    b_train, b_infer = b_paths[:20], b_paths[20:]

    print('\n=== Group A: optimal (daemon + lite stream + in-memory) ===', flush=True)
    ta, a_n, a_out, a_dim = run_group_a(a_train, a_infer, args.epochs)
    for k, v in ta.items():
        print(f'  {k:14s} {v:8.1f}s', flush=True)

    print('\n=== Group B: slowest (one-shot + full trunk + disk) ===', flush=True)
    tb, b_n, b_out, b_dim = run_group_b(b_train, b_infer, args.epochs)
    for k, v in tb.items():
        print(f'  {k:14s} {v:8.1f}s', flush=True)

    print('\n=== Comparison (B / A) ===')
    print(f'  {"stage":14s} {"A (s)":>8s} {"B (s)":>8s} {"ratio":>7s}')
    for k in ('analysis_20', 'train', 'inference_5', 'total'):
        a, b = ta.get(k, 0), tb.get(k, 0)
        ratio = b / a if a > 0 else float('inf')
        print(f'  {k:14s} {a:8.1f} {b:8.1f} {ratio:6.1f}x')
    print(f'  (A startup {ta.get("startup", 0):.1f}s is included in A total; '
          f'B pays model loads inside its stages)')
    print(f'  feature dims: A={a_dim}  B={b_dim}')
    print(f'  trained on:   A={a_n} games  B={b_n} games; '
          f'inferred: A={a_out}  B={b_out}')

    report = {
        'seed': args.seed, 'epochs': args.epochs,
        'rank_distribution': dist,
        'group_a': {'timings': ta, 'train_games': a_n,
                    'inferred': a_out, 'input_dim': a_dim},
        'group_b': {'timings': tb, 'train_games': b_n,
                    'inferred': b_out, 'input_dim': b_dim},
    }
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'\nreport -> {args.report}')


if __name__ == '__main__':
    main()
