#!/usr/bin/env python
"""
KataRank — Quality Comparison (File Gen + Multi-epoch Train)
===============================================================
Full vs Lite: same architecture, 15 epochs each, compare quality.

Pipeline:
  1. File-mode analysis → .npz files (full & lite)
  2. Multi-epoch KAB2Dataset training → convergence metrics
  3. File-mode inference on held-out → rank agreement
"""
import os, sys, random, time, json, subprocess
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

KATAGO = r'C:\Users\bzhao\katago-fork\cpp\katago.exe'
CONFIG = r'C:\Users\bzhao\katago-fork\cpp\analysis_config_opt.cfg'
MODEL  = r'C:\Users\bzhao\.katago\default_model.bin.gz'
HUMAN  = r'C:\Users\bzhao\.katago\b18c384nbt-humanv0.bin.gz'
SGF_DIR = os.path.expanduser('~/go-analyzer/training')
WORK   = os.path.expanduser('~/katarank/_stress_test')

import torch
import torch.nn as nn
import torch.optim as optim
from katarank.data.datasets import make_kab2_loader
from katarank.model import KataRankModel, KataRankLoss
from katarank.workflow import InferenceWorkflow, run_rank_files
from katarank.schema import rank_idx_to_str

def pick_sgfs(n: int = 50) -> list:
    level_dirs = [d for d in sorted(Path(SGF_DIR).iterdir()) if d.is_dir()]
    all_sgfs = []
    for ld in level_dirs:
        sgfs = sorted(ld.glob('*.sgf'))
        random.shuffle(sgfs)
        all_sgfs.extend(sgfs)
    random.shuffle(all_sgfs)
    return [str(s) for s in all_sgfs[:n]]

def gen_kab2(sgfs: list, mode: str, out_dir: str) -> float:
    """Generate KAB2 .npz files via one-shot batch_analysis."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, '_games.csv')
    with open(csv_path, 'w') as f:
        f.write('File,Player Black,Player White,Score,BlackRating,WhiteRating,Set\n')
        for i, s in enumerate(sgfs):
            splt = 'T' if i < len(sgfs) - 5 else 'V'
            f.write(f'{s},unknown,unknown,0.5,1500,1500,{splt}\n')
    cmd = [KATAGO, 'batch_analysis', '-model', MODEL,
           '-config', CONFIG, '-human-model', HUMAN,
           '-list', csv_path, '-output-dir', out_dir, '-min-moves', '10']
    if mode == 'lite':
        cmd.append('-no-trunk')
    t0 = time.time()
    subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    n_npz = len(list(Path(out_dir).glob('*.npz')))
    print(f'  {dt:.0f}s  {n_npz}/{len(sgfs)} .npz')
    return dt

def train_model(data_dir: str, hidden_dim: int, epochs: int):
    """Multi-epoch training from KAB2 files. Splits last 5 as val."""
    tc = {'batch_size': 8, 'num_workers': 0,
          'max_moves_per_player': 400, 'min_moves_per_player': 5}
    # Fix meta CSV: label last 5 games as V
    meta_path = os.path.join(data_dir, '_meta.csv')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            lines = f.readlines()
        # Header: ...,white_moves,set,... — set is column 8 (0-indexed)
        if len(lines) > 1:
            for i in range(max(1, len(lines)-5), len(lines)):
                cols = lines[i].split(',')
                if len(cols) > 8:
                    cols[8] = 'V'
                    lines[i] = ','.join(cols)
            with open(meta_path, 'w') as f:
                f.writelines(lines)
    train_loader, train_ds = make_kab2_loader(data_dir, split='T', **tc)
    val_loader,   val_ds   = make_kab2_loader(data_dir, split='V', **tc)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  device={device}  train={len(train_ds.games)} val={len(val_ds.games)} input_dim={train_ds.input_dim}')

    model = KataRankModel(train_ds.input_dim, hidden_dim=hidden_dim,
        num_heads=4, num_inducing=8).to(device)
    loss_fn = KataRankLoss(w_rating=1.0, w_bt=0.5, w_rank=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    from katarank.train.training import Trainer, evaluate_metrics
    trainer = Trainer(model, loss_fn, train_loader, val_loader, {
        'device': str(device), 'learning_rate': 0.001, 'weight_decay': 1e-5,
        'lr_min': 1e-5, 'gradient_clip': 1.0, 'patience': 20,
    })
    t0 = time.time()
    result = trainer.train(epochs, os.path.join(data_dir, 'nets'))
    dt = time.time() - t0
    metrics = evaluate_metrics(model, val_loader, device)
    print(f'  └─ best_val={trainer.best_val_loss:.4f} @ epoch {trainer.best_epoch}')
    print(f'     rank_mae={metrics.get("rank_mae","N/A")}  rank_acc={metrics.get("rank_acc","N/A")}  '
          f'rating_corr={metrics.get("rating_corr","N/A")}')
    return model, trainer, dt, metrics

def do_inference(model, infer_sgfs, mode):
    """Run inference on held-out games."""
    from katarank.engine import KataGoEngine
    device = next(model.parameters()).device
    model.to(device).eval()
    engine = KataGoEngine(model=MODEL, config=CONFIG, human_model=HUMAN, katago_bin=KATAGO)
    inf = InferenceWorkflow(model, engine, device=str(device))
    t0 = time.time()
    results = run_rank_files(engine, inf, infer_sgfs, mode=mode)
    dt = time.time() - t0
    for r in results:
        print(f'  ├─ {Path(r["game_id"]).stem}: '
              f'B={rank_idx_to_str(r["b_rank"])} ({r["b_rating"]:.3f})  '
              f'W={rank_idx_to_str(r["w_rank"])} ({r["w_rating"]:.3f})  '
              f'conf=({r["b_confidence"]:.2f},{r["w_confidence"]:.2f})')
    return results, dt

def main():
    print('='*60)
    print('Quality Comparison: Full vs Lite (Multi-epoch Training)')
    print('='*60)

    os.makedirs(WORK, exist_ok=True)
    sgfs = pick_sgfs(50)
    train_sgfs, infer_sgfs = sgfs[:45], sgfs[45:]
    print(f'\n[1] Sampled: {len(train_sgfs)} train + {len(infer_sgfs)} inference')
    for s in infer_sgfs:
        print(f'  inference: {Path(s).parent.name}/{Path(s).name}')

    results = {}
    for label, mode, hdim in [('FULL','full',128), ('LITE','lite',64)]:
        print(f'\n{"="*60}\n{label} (mode={mode}, hidden={hdim})\n{"="*60}')
        d = os.path.join(WORK, f'qual_{mode}')

        print('\n  Phase 1: KAB2 generation...')
        ta = gen_kab2(train_sgfs, mode, d)

        print('\n  Phase 2: Training (15 epochs)...')
        model, trainer, tt, metrics = train_model(d, hdim, 15)
        model.save(os.path.join(d, 'best.pt'))

        print('\n  Phase 3: Inference (5 held-out)...')
        infer_results, ti = do_inference(model, infer_sgfs, mode)

        results[mode] = {
            'analysis_s': round(ta,1), 'train_s': round(tt,1), 'infer_s': round(ti,1),
            'input_dim': 10+2*512 if mode=='full' else 10,
            'hidden_dim': hdim, 'val_loss': trainer.best_val_loss,
            'best_epoch': trainer.best_epoch,
            'train_loss': [round(v,4) for v in trainer.train_hist],
            'val_loss_curve': [round(v,4) for v in trainer.val_hist],
            'metrics': {k: round(float(v),4) for k,v in metrics.items()},
            'inference': [{'game': r['game_id'],
                'b_rank': r['b_rank'], 'w_rank': r['w_rank'],
                'b_rating': round(r['b_rating'],3), 'w_rating': round(r['w_rating'],3),
                'b_conf': round(r['b_confidence'],2), 'w_conf': round(r['w_confidence'],2)}
                for r in infer_results],
        }

    # ── Report ──
    print('\n' + '='*60)
    print('QUALITY COMPARISON REPORT')
    print('='*60)
    f,l = results['full'], results['lite']
    print(f'\n  {"Metric":<25} {"FULL":<22} {"LITE":<22}')
    print(f'  {"-":-<25} {"-":-<22} {"-":-<22}')
    print(f'  {"Input/hidden":<25} {f["input_dim"]}/{f["hidden_dim"]:<18} {l["input_dim"]}/{l["hidden_dim"]}')
    print(f'  {"Epochs":<25} {len(f["train_loss"]):<22} {len(l["train_loss"])}')
    print(f'  {"Best epoch":<25} {f["best_epoch"]:<22} {l["best_epoch"]}')
    print(f'  {"Best val loss":<25} {f["val_loss"]:<22.4f} {l["val_loss"]:<.4f}')
    print(f'  {"Rank MAE":<25} {f["metrics"].get("rank_mae","N/A"):<22} {l["metrics"].get("rank_mae","N/A")}')
    print(f'  {"Rank Acc":<25} {f["metrics"].get("rank_acc","N/A"):<22} {l["metrics"].get("rank_acc","N/A")}')
    print(f'  {"Rating Corr":<25} {f["metrics"].get("rating_corr","N/A"):<22} {l["metrics"].get("rating_corr","N/A")}')
    print(f'\n  {"Game":<36} {"FULL rank/rating":<24} {"LITE rank/rating":<24} {"Match?":<8}')
    for fi, li in zip(f['inference'], l['inference']):
        match = fi['b_rank']==li['b_rank'] and fi['w_rank']==li['w_rank']
        print(f'  {Path(fi["game"]).stem:<36}'
              f' B={fi["b_rating"]:+.1f}/W={fi["w_rating"]:+.1f}  '
              f' B={li["b_rating"]:+.1f}/W={li["w_rating"]:+.1f}  '
              f'{"✅" if match else "❌"}')

    jpath = os.path.join(WORK, 'qual_report.json')
    with open(jpath,'w') as f:
        json.dump(results,f,indent=2,default=str)
    print(f'\n  Report: {jpath}')

if __name__ == '__main__':
    main()
