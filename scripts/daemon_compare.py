#!/usr/bin/env python
"""
KataRank — Daemon Mode Quality Comparison
============================================
Full vs Lite: same data, same epochs, compare convergence + inference quality.

Pipeline:
  1. Sample 50 SGFs (45 train + 5 inference)
  2. Daemon full → train model_full → infer on 5 held-out
  3. Daemon lite → train model_lite → infer on 5 held-out
  4. Compare loss curves + rank accuracy + rating correlation
"""
import os, sys, random, time, json
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

KATAGO = r'C:\Users\bzhao\katago-fork\cpp\katago.exe'
MODEL  = r'C:\Users\bzhao\.katago\default_model.bin.gz'
HUMAN  = r'C:\Users\bzhao\.katago\b18c384nbt-humanv0.bin.gz'
SGF_DIR = os.path.expanduser('~/go-analyzer/training')
WORK   = os.path.expanduser('~/katarank/_stress_test')

import torch
import torch.nn as nn
import torch.optim as optim
from katarank.engine import PersistentKataGoEngine
from katarank.model import KataRankModel, KataRankLoss
from katarank.workflow import TrainingWorkflow, InferenceWorkflow, run_rank_files, result_to_output
from katarank.schema import KAB2Output, rank_idx_to_str

def pick_sgfs(n: int = 50) -> list:
    level_dirs = sorted(Path(SGF_DIR).iterdir())
    level_dirs = [d for d in level_dirs if d.is_dir()]
    all_sgfs = []
    for ld in level_dirs:
        sgfs = sorted(ld.glob('*.sgf'))
        random.shuffle(sgfs)
        all_sgfs.extend(sgfs)
    random.shuffle(all_sgfs)
    return [str(s) for s in all_sgfs[:n]]

def train_and_eval(mode: str, train_sgfs, infer_sgfs,
                   hidden_dim: int, epochs: int) -> dict:
    """Train a model via streaming daemon, then inference on held-out.
    
    Returns dict with timing, loss curve, validation metrics, inference results.
    """
    label = mode.upper()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ── Step 1: Daemon streaming training ──────────────────────────────
    print(f"\n[{label}] Starting daemon (mode={mode}, hidden={hidden_dim})...")
    eng = PersistentKataGoEngine(
        model=MODEL, config=os.path.join(os.path.dirname(KATAGO), 'analysis_config_opt.cfg'),
        human_model=HUMAN,
        katago_bin=KATAGO, mode=mode,
    )
    eng.start()
    
    # Determine input_dim from config — full=1034 (b28c512: 10+2*512), lite=10
    input_dim = 10 + 2 * 512 if mode == 'full' else 10
    print(f"  input_dim={input_dim}, device={device}")
    
    # Build model
    model = KataRankModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim, num_heads=4, num_inducing=8,
        encoder_depth=2, cross_depth=1, dropout=0.1,
    ).to(device)
    loss_fn = KataRankLoss(w_rating=1.0, w_bt=0.5, w_rank=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training workflow with streaming
    t0 = time.time()
    wf = TrainingWorkflow(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        device=str(device), gradient_clip=1.0, log_every=10,
    )
    
    # Stream all training games
    print(f"  Streaming {len(train_sgfs)} training games...")
    train_stream = eng.stream_to_tensors(sgf_paths=train_sgfs, mode=mode)
    train_result = wf.run_stream(train_stream, batch_size=8, max_steps=epochs * 5)
    train_elapsed = time.time() - t0
    
    # ── Step 2: Val loss via separate stream ───────────────────────────
    # Use a subset of training games for validation
    val_sgfs = train_sgfs[-5:]  # last 5 for val
    val_stream = eng.stream_to_tensors(sgf_paths=val_sgfs, mode=mode)
    val_losses = []
    model.eval()
    with torch.no_grad():
        for x_b, x_w, info_b, info_w in val_stream:
            sample = {
                'x': torch.cat([x_b, x_w]).to(device),
                'seq_len': len(x_b) + len(x_w),
                'target_b': torch.tensor(info_b['mean_log_prior']).to(device),
                'target_w': torch.tensor(info_w['mean_log_prior']).to(device),
            }
            if x_b.shape[0] >= 5 and x_w.shape[0] >= 5:
                sample_x = torch.cat([x_b, x_w]).to(device)
                out = model(sample_x, [len(sample_x)])
                targets = {
                    'target_b': sample['target_b'].unsqueeze(0),
                    'target_w': sample['target_w'].unsqueeze(0),
                    'rank_b': torch.tensor([info_b.get('human_rank_idx', -1)]).to(device),
                    'rank_w': torch.tensor([info_w.get('human_rank_idx', -1)]).to(device),
                    'human_lp_b': torch.tensor([info_b.get('human_log_prior', 0.0)]).to(device),
                    'human_lp_w': torch.tensor([info_w.get('human_log_prior', 0.0)]).to(device),
                }
                val_losses.append(loss_fn(out, targets)['total'].item())
    
    val_loss = sum(val_losses) / max(len(val_losses), 1) if val_losses else float('nan')
    
    # ── Step 3: Inference on held-out ──────────────────────────────────
    print(f"  Inferring {len(infer_sgfs)} held-out games...")
    t1 = time.time()
    inf_workflow = InferenceWorkflow(model, eng, device=str(device))
    results = run_rank_files(eng, inf_workflow, infer_sgfs, mode=mode)
    infer_elapsed = time.time() - t1
    
    # ── Close engine ───────────────────────────────────────────────────
    eng.close()
    
    # ── Quality metrics ────────────────────────────────────────────────
    qual = {
        'mode': mode,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'train_steps': train_result['steps'],
        'train_elapsed': round(train_elapsed, 1),
        'val_loss': round(val_loss, 4),
        'infer_elapsed': round(infer_elapsed, 1),
        'results': results,
    }
    
    # Show inference
    for r in results:
        bname = rank_idx_to_str(r['b_rank'])
        wname = rank_idx_to_str(r['w_rank'])
        print(f"  ├─ {r['game_id']}: B={bname} ({r['b_rating']:.3f})  "
              f"W={wname} ({r['w_rating']:.3f})  "
              f"conf=({r['b_confidence']:.2f},{r['w_confidence']:.2f})")
    
    return qual

def main():
    print("=" * 60)
    print("Daemon Mode — Quality Comparison: Full vs Lite")
    print("=" * 60)
    
    os.makedirs(WORK, exist_ok=True)
    
    # ── Sample ─────────────────────────────────────────────────────────
    print("\n[1] Sampling 50 SGFs (45 train + 5 inference)...")
    all_sgfs = pick_sgfs(50)
    train_sgfs = all_sgfs[:45]
    infer_sgfs = all_sgfs[45:50]
    print(f"  Train: {len(train_sgfs)}  Infer: {len(infer_sgfs)}")
    
    # Determine rank level distribution
    for sg in [infer_sgfs[0]]:
        print(f"  e.g. inference game: {Path(sg).stem} from {Path(sg).parent.name}")
    
    # ── Full mode ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Group A — FULL (trunk+pick, hidden=128)")
    print("=" * 60)
    qual_full = train_and_eval('full', train_sgfs, infer_sgfs,
                                hidden_dim=128, epochs=15)
    
    # ── Lite mode ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Group B — LITE (10-dim scalars, hidden=64)")
    print("=" * 60)
    qual_lite = train_and_eval('lite', train_sgfs, infer_sgfs,
                                hidden_dim=64, epochs=15)
    
    # ── Report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("QUALITY COMPARISON REPORT")
    print("=" * 60)
    
    f, l = qual_full, qual_lite
    print(f"\n  {'Metric':<25} {'FULL':<20} {'LITE':<20}")
    print(f"  {'-'*20:<25} {'-'*18:<20} {'-'*18:<20}")
    print(f"  {'Input dim':<25} {f['input_dim']:<20} {l['input_dim']:<20}")
    print(f"  {'Hidden dim':<25} {f['hidden_dim']:<20} {l['hidden_dim']:<20}")
    print(f"  {'Training steps':<25} {f['train_steps']:<20} {l['train_steps']:<20}")
    print(f"  {'Training time':<25} {f['train_elapsed']}s{'':<16} {l['train_elapsed']}s")
    print(f"  {'Val loss':<25} {f['val_loss']:<20} {l['val_loss']:<20}")
    print(f"  {'Inference time':<25} {f['infer_elapsed']}s{'':<16} {l['infer_elapsed']}s")
    
    # Compare per-game ratings
    print(f"\n  Per-game rating comparison:")
    print(f"  {'Game':<30} {'FULL B/W rating':<24} {'LITE B/W rating':<24} {'Agree?':<10}")
    for rf, rl in zip(f['results'], l['results']):
        agree = rf['b_rank'] == rl['b_rank'] and rf['w_rank'] == rl['w_rank']
        print(f"  {rf['game_id']:<30}"
              f"  B={rf['b_rating']:+.3f}/W={rf['w_rating']:+.3f}  "
              f"  B={rl['b_rating']:+.3f}/W={rl['w_rating']:+.3f}"
              f"  {'✅' if agree else '❌':<10}")
    
    # Save report
    report = {
        'full': qual_full,
        'lite': qual_lite,
        'inference_comparison': [
            {'game_id': rf['game_id'],
             'full_b_rank': rf['b_rank'], 'full_w_rank': rf['w_rank'],
             'full_b_rating': rf['b_rating'], 'full_w_rating': rf['w_rating'],
             'lite_b_rank': rl['b_rank'], 'lite_w_rank': rl['w_rank'],
             'lite_b_rating': rl['b_rating'], 'lite_w_rating': rl['w_rating'],
             'rank_agree': rf['b_rank'] == rl['b_rank'] and rf['w_rank'] == rl['w_rank']}
            for rf, rl in zip(qual_full['results'], qual_lite['results'])
        ],
    }
    report_path = os.path.join(WORK, 'daemon_quality_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report: {report_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
