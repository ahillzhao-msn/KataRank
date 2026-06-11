#!/usr/bin/env python
"""
KataRank — Pressure Test: Optimal vs Heaviest
===============================================
Two groups of 25 games (20 train + 5 inference each)
  - Group A: stream lite + quick training + lite inference
  - Group B: file full + heavier training + full inference

Outputs timing + convergence metrics for comparison.
"""
import os, sys, random, time, shutil, json, subprocess
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

KATAGO = r'C:\Users\bzhao\katago-fork\cpp\katago.exe'
MODEL  = r'C:\Users\bzhao\.katago\default_model.bin.gz'
HUMAN  = r'C:\Users\bzhao\.katago\b18c384nbt-humanv0.bin.gz'
SGF_DIR = os.path.expanduser('~/go-analyzer/training')
WORK   = os.path.expanduser('~/katarank/_stress_test')

ADVANCE_INFO = "pressure test output"

def pick_sgfs(n_per_group: int = 25) -> list:
    """Random stratified sampling across rank levels."""
    level_dirs = sorted(Path(SGF_DIR).iterdir())
    level_dirs = [d for d in level_dirs if d.is_dir()]
    random.shuffle(level_dirs)
    
    all_sgfs = []
    for ld in level_dirs:
        sgfs = sorted(ld.glob('*.sgf'))
        random.shuffle(sgfs)
        all_sgfs.extend(sgfs)
    
    random.shuffle(all_sgfs)
    selected = all_sgfs[:n_per_group * 2]
    print(f"  Sampled {len(selected)} SGFs from {len(level_dirs)} rank levels")
    return selected

def run_analysis(sgfs, label: str, mode: str, out_dir: str) -> float:
    """Run batch_analysis and return elapsed seconds."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Write CSV list with train/val split
    csv_path = os.path.join(out_dir, '_games.csv')
    with open(csv_path, 'w') as f:
        f.write('File,Player Black,Player White,Score,BlackRating,WhiteRating,Set\n')
        for i, s in enumerate(sgfs):
            splt = 'T' if i < len(sgfs) - 4 else 'V'
            f.write(f'{s},unknown,unknown,0.5,1500,1500,{splt}\n')
    
    cmd = [KATAGO, 'batch_analysis', '-model', MODEL,
           '-human-model', HUMAN,
           '-list', csv_path,
           '-min-moves', '10']
    
    if mode == 'lite':
        cmd += ['-no-trunk']
    else:
        cmd += ['-output-dir', out_dir]

    t0 = time.time()
    if mode == 'lite':
        cmd += ['-output-dir', out_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
    
    npz_count = len(list(Path(out_dir).glob('*.npz')))
    print(f"  ├─ files: {npz_count} .npz")
    print(f"  └─ analysis: {elapsed:.1f}s")
    return elapsed

def run_training(data_dir: str, label: str, hidden_dim: int, epochs: int) -> float:
    """Run KAB2 training and return elapsed seconds."""
    import torch
    from katarank.model import KataRankModel, KataRankLoss
    from katarank.data.datasets import KAB2Dataset, make_kab2_loader
    from katarank.train.training import Trainer, evaluate_metrics
    
    meta_csv = os.path.join(data_dir, '_meta.csv')
    
    train_loader, train_ds = make_kab2_loader(
        data_dir=data_dir, meta_csv=meta_csv, split='T',
        batch_size=8, shuffle=True, num_workers=0,
    )
    val_loader, val_ds = make_kab2_loader(
        data_dir=data_dir, meta_csv=meta_csv, split='V',
        batch_size=8, shuffle=False, num_workers=0,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KataRankModel(
        input_dim=train_ds.input_dim,
        hidden_dim=hidden_dim, num_heads=4, num_inducing=8,
        encoder_depth=2, cross_depth=1, dropout=0.1,
    ).to(device)
    
    loss_fn = KataRankLoss(w_rating=1.0, w_bt=0.5, w_rank=0.3)
    
    t0 = time.time()
    trainer = Trainer(model, loss_fn, train_loader, val_loader, {
        'device': str(device), 'learning_rate': 0.001, 'weight_decay': 1e-5,
        'lr_min': 1e-5, 'gradient_clip': 1.0, 'patience': 50,
    })
    os.makedirs(os.path.join(data_dir, 'nets'), exist_ok=True)
    result = trainer.train(epochs, os.path.join(data_dir, 'nets'))
    elapsed = time.time() - t0
    
    metrics = evaluate_metrics(model, val_loader, device)
    print(f"  ├─ best val: {result['best_val_loss']:.4f} @ epoch {trainer.best_epoch}")
    print(f"  ├─ metrics: rank_mae={metrics.get('rank_mae','N/A')}  rating_corr={metrics.get('rating_corr','N/A')}")
    print(f"  └─ training: {elapsed:.1f}s")
    return elapsed

def run_inference(sgfs, label: str, mode: str, checkpoint: str, out_dir: str) -> float:
    """Run inference on held-out SGFs and return elapsed seconds."""
    import torch
    from katarank.engine import KataGoEngine
    from katarank.model import KataRankModel
    from katarank.workflow import run_rank_files, InferenceWorkflow
    
    engine = KataGoEngine(model=MODEL, human_model=HUMAN, katago_bin=KATAGO)
    rank_model = KataRankModel.load(checkpoint)
    inf_workflow = InferenceWorkflow(rank_model, engine)
    
    t0 = time.time()
    results = run_rank_files(
        engine=engine, inf_workflow=inf_workflow,
        paths=[str(s) for s in sgfs],
        mode='lite' if mode == 'lite' else 'full',
        min_moves=10,
    )
    elapsed = time.time() - t0
    for r in results:
        bname = katarank.schema.rank_idx_to_str(r['b_rank'])
        wname = katarank.schema.rank_idx_to_str(r['w_rank'])
        print(f"  ├─ {r['game_id']}: B={bname} ({r['b_rating']:.3f})  "
              f"W={wname} ({r['w_rating']:.3f})")
    print(f"  └─ inference: {elapsed:.1f}s ({len(results)} outputs)")
    return elapsed

import katarank.schema

def main():
    print("=" * 60)
    print("KataRank — Pressure Test: Optimal vs Heaviest")
    print("=" * 60)
    
    # ── 1. Sample SGFs ───────────────────────────────────────────────────
    print("\n[1] Sampling SGFs...")
    os.makedirs(WORK, exist_ok=True)
    all_sgfs = pick_sgfs(25)
    group_a_sgfs = [str(s) for s in all_sgfs[:25]]
    group_b_sgfs = [str(s) for s in all_sgfs[25:50]]
    
    train_a = group_a_sgfs[:20]
    infer_a = group_a_sgfs[20:25]
    train_b = group_b_sgfs[:20]
    infer_b = group_b_sgfs[20:25]
    
    print(f"\n  Group A: {len(train_a)} train + {len(infer_a)} inference")
    print(f"  Group B: {len(train_b)} train + {len(infer_b)} inference")
    
    # ── Group A: OPTIMAL ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Group A — OPTIMAL (stream lite + quick train)")
    print("=" * 60)
    
    dir_a = os.path.join(WORK, 'group_a')
    t_a = {}
    
    print("\n  Phase 1+2: Analysis (file lite) + Training...")
    t_a['data_gen'] = run_analysis(train_a, 'A', 'lite', os.path.join(dir_a, 'data'))
    t_a['training'] = run_training(os.path.join(dir_a, 'data'), 'A', hidden_dim=64, epochs=15)

    print("\n  Phase 3: Inference (stream lite)...")
    # Use the trained model for inference
    ckpt = os.path.join(dir_a, 'data', 'nets', 'best.pt')
    if os.path.exists(ckpt):
        t_a['inference'] = run_inference(infer_a, 'A', 'lite', ckpt, dir_a)
    
    # ── Group B: HEAVIEST ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Group B — HEAVIEST (file full + heavy train)")
    print("=" * 60)
    
    dir_b = os.path.join(WORK, 'group_b')
    t_b = {}
    
    print("\n  Phase 1: Analysis (file full)...")
    t_b['analysis'] = run_analysis(train_b, 'B', 'full', os.path.join(dir_b, 'analysis'))
    
    print("\n  Phase 2: Training (full features, 15 epochs)...")
    t_b['training'] = run_training(os.path.join(dir_b, 'analysis'), 'B',
                                    hidden_dim=128, epochs=15)
    
    print("\n  Phase 3: Inference (file full)...")
    ckpt = os.path.join(dir_b, 'analysis', 'nets', 'best.pt')
    if os.path.exists(ckpt):
        t_b['inference'] = run_inference(infer_b, 'B', 'full', ckpt, dir_b)
    
    # ── Report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON REPORT")
    print("=" * 60)
    
    def fmt(sec):
        m, s = divmod(int(sec), 60)
        return f"{m}m{s:02d}s"
    
    print(f"\n  {'Phase':<25} {'Group A (Optimal)':<20} {'Group B (Heaviest)':<20}")
    print(f"  {'-'*20:<25} {'-'*18:<20} {'-'*18:<20}")
    phases = [('Analysis (20 games)', 'analysis'),
              ('Training', 'training'),
              ('Inference (5 games)', 'inference'),
              ('TOTAL', None)]
    
    total_a = sum(t_a.get(k, 0) for k in ['analysis', 'training', 'inference'])
    total_b = sum(t_b.get(k, 0) for k in ['analysis', 'training', 'inference'])
    
    for name, key in phases:
        if key:
            va = fmt(t_a.get(key, 0))
            vb = fmt(t_b.get(key, 0))
        else:
            va = fmt(total_a)
            vb = fmt(total_b)
        print(f"  {name:<25} {va:<20} {vb:<20}")
    
    ratio = total_b / max(total_a, 1)
    print(f"\n  ⚡ Total speedup: {ratio:.1f}x")
    
    # Save report
    report = {
        'group_a': {k: round(v, 1) for k, v in t_a.items() if v},
        'group_b': {k: round(v, 1) for k, v in t_b.items() if v},
        'total_a': round(total_a, 1),
        'total_b': round(total_b, 1),
        'speedup_ratio': round(ratio, 2),
    }
    report_path = os.path.join(WORK, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
