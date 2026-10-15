"""
KataRank — Training Entry Point

Usage:
    uv run katarank-train --data-dir data/kab2 --meta data/kab2/_meta.csv
    uv run katarank-train --config src/katarank/train/config_kata_native.yaml --resume nets/best.pt
"""

import json
import os
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.optim as optim

from katarank.model import KataRankModel, KataRankLoss
from katarank.data.datasets import make_kab2_loader
from katarank.schema import TrainingReport, save_training_report

# Same path server.py's /train/status and train_session.py's JOB_FILE read/
# write — this process runs as a separate `-m katarank.train.training`
# subprocess (spawned by train_session.py), so it has no access to
# train_session.py's in-memory job state and merges into the file directly.
JOB_FILE = Path.home() / ".katarank" / "train_job.json"


HISTORY_LEN = 40  # enough points to see the convergence curve without the file growing unbounded


def _write_epoch_progress(progress: dict) -> None:
    """Merge current epoch + a rolling loss-curve history into the shared
    training job file, so /train/status can show live progress and the
    convergence trend (not just the latest point) without polling logs."""
    try:
        job = json.loads(JOB_FILE.read_text(encoding='utf-8')) if JOB_FILE.exists() else {}
    except (json.JSONDecodeError, OSError):
        job = {}
    job['epoch_progress'] = progress
    history = job.get('epoch_history') or []
    history.append({'epoch': progress['epoch'], 'train_loss': round(progress['train_loss'], 4),
                     'val_loss': round(progress['val_loss'], 4)})
    job['epoch_history'] = history[-HISTORY_LEN:]
    try:
        JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOB_FILE.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
    except OSError:
        pass


def _device(cfg_val: str) -> torch.device:
    if cfg_val == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(cfg_val)


class Trainer:
    def __init__(self, model, loss_fn, train_loader, val_loader, config,
                resume_optimizer_path: str | None = None):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config

        self.device = _device(config.get('device', 'auto'))
        self.model = self.model.to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5),
        )

        epochs = config.get('epochs', 100)
        warmup = config.get('warmup_epochs', 0)
        self.warmup_epochs = warmup

        if warmup > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, total_iters=warmup,
            )
            cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(epochs - warmup, 1),
                eta_min=config.get('lr_min', 1e-5),
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup],
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs,
                eta_min=config.get('lr_min', 1e-5),
            )

        self.gradient_clip = config.get('gradient_clip', 1.0)
        self.patience = config.get('patience', 20)
        self.best_val_loss = float('inf')
        self.best_state = None
        self.best_epoch = 0
        self.bad_epochs = 0
        self.early_stopped = False
        self.train_hist: list = []
        self.val_hist: list = []

        # Continual/incremental training: --resume only restores model
        # WEIGHTS by itself (see main()) — a fresh AdamW starts with zero
        # moment estimates, and Adam's bias-correction term amplifies the
        # very first steps when those estimates are still zero, so even a
        # warmed-up LR schedule can knock an already-converged model off
        # its optimum before it settles back down (confirmed: a real run
        # degraded for ~30 epochs after resuming and never recovered).
        # Restoring the optimizer + scheduler state alongside the model
        # weights makes this a true continuation of the SAME optimization
        # trajectory instead of a fresh cold start on pre-trained weights.
        # Training from scratch (no resume_optimizer_path) is completely
        # unaffected — self.optimizer/self.scheduler are freshly built
        # above either way.
        if resume_optimizer_path and os.path.exists(resume_optimizer_path):
            state = torch.load(resume_optimizer_path, map_location=self.device,
                               weights_only=True)
            self.optimizer.load_state_dict(state['optimizer_state'])
            self.scheduler.load_state_dict(state['scheduler_state'])
            print(f"Resumed optimizer/scheduler state from {resume_optimizer_path} "
                  f"(was at epoch {state.get('epoch', '?')})")

    def save_optimizer_state(self, path: str, epoch: int):
        torch.save({
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'epoch': epoch,
        }, path)

    def _batch_loss(self, batch) -> torch.Tensor:
        x     = batch['x'].to(self.device)
        xlens = batch['xlens']
        out   = self.model(x, xlens)
        targets = {
            k: batch[k].to(self.device)
            for k in ('target_b', 'target_w', 'rank_b', 'rank_w',
                      'human_lp_b', 'human_lp_w')
        }
        return self.loss_fn(out, targets)['total']

    def train_epoch(self) -> float:
        self.model.train()
        total, n = 0.0, 0
        for batch in self.train_loader:
            if not batch['xlens']:
                continue
            loss = self._batch_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            total += loss.item()
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            if not batch['xlens']:
                continue
            total += self._batch_loss(batch).item()
            n += 1
        return {'val_loss': total / max(n, 1)}

    def train(self, epochs: int, ckpt_dir: str, on_epoch=None) -> dict:
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Training on {self.device}  |  train={len(self.train_loader.dataset)}  "
              f"val={len(self.val_loader.dataset)}")
        print("-" * 60)

        epoch = 0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val = self.validate()
            dt = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            self.train_hist.append(round(train_loss, 6))
            self.val_hist.append(round(val['val_loss'], 6))

            print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  "
                  f"val={val['val_loss']:.4f}  lr={lr:.2e}  ({dt:.1f}s)",
                  flush=True)

            self.scheduler.step()

            if val['val_loss'] < self.best_val_loss:
                self.best_val_loss = val['val_loss']
                self.best_epoch = epoch
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.bad_epochs = 0
                self.model.save(os.path.join(ckpt_dir, 'best.pt'))
                self.save_optimizer_state(os.path.join(ckpt_dir, 'optimizer_state.pt'), epoch)
            else:
                self.bad_epochs += 1

            if on_epoch is not None:
                on_epoch(epoch=epoch, epochs=epochs, train_loss=train_loss,
                          val_loss=val['val_loss'], lr=lr,
                          best_val_loss=self.best_val_loss, best_epoch=self.best_epoch)

            if self.bad_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                self.early_stopped = True
                break

        if self.best_state:
            self.model.load_state_dict(self.best_state)

        return {'best_val_loss': self.best_val_loss, 'epochs_trained': epoch}


# ─── Validation metrics (for TrainingReport) ─────────────────────────────────

@torch.no_grad()
def evaluate_metrics(model, loader, device, input_dim: int | None = None) -> dict:
    """Rank/rating quality on a validation loader. See TrainingReport docs.

    input_dim: slice batch['x']'s feature dim down to this width before
    feeding the model — for evaluating a lite (10-dim) checkpoint against
    a loader built from full (1034-dim) KAB2 data, same convention as
    distill.py's own end-of-run evaluation (SlicedDataset). batch['x'] is
    a flat (total_moves_in_batch, feat_dim) tensor — kab2_collate
    torch.cat's per-game sequences along dim 0 rather than padding to
    (B, T, D) — so this is a 2D slice on the last axis, not 3D.
    """
    model.eval()
    ratings_pred, ratings_tgt = [], []
    rank_pred, rank_lab = [], []

    for batch in loader:
        if not batch['xlens']:
            continue
        x = batch['x'][:, :input_dim] if input_dim else batch['x']
        out = model(x.to(device), batch['xlens'])
        for side, key in (('b', 'rank_b'), ('w', 'rank_w')):
            pred = out[f'rank_probs_{side}'].argmax(-1).cpu()
            lab  = batch[key]
            mask = lab >= 0
            rank_pred += pred[mask].tolist()
            rank_lab  += lab[mask].tolist()
        ratings_pred += out['b_rating'].cpu().tolist() + out['w_rating'].cpu().tolist()
        ratings_tgt  += batch['target_b'].tolist() + batch['target_w'].tolist()

    metrics = {'n_rank_labeled': len(rank_lab)}
    if rank_lab:
        diff = np.abs(np.asarray(rank_pred) - np.asarray(rank_lab))
        metrics['rank_mae']     = float(diff.mean())
        metrics['rank_acc']     = float((diff == 0).mean())
        metrics['rank_acc_pm1'] = float((diff <= 1).mean())
    if len(ratings_pred) > 1:
        corr = np.corrcoef(ratings_pred, ratings_tgt)[0, 1]
        metrics['rating_corr'] = float(corr) if np.isfinite(corr) else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description='KataRank Training')
    parser.add_argument('--config', default='src/katarank/train/config_kata_native.yaml')
    parser.add_argument('--data-dir', help='Override data directory')
    parser.add_argument('--meta', help='Override _meta.csv path')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--resume', help='Resume from checkpoint .pt')
    parser.add_argument('--ckpt-dir', help='Override output.checkpoint_dir '
                        '(e.g. a staging dir, so this run does not overwrite '
                        'the live production checkpoint before it is evaluated)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tc = config.get('training', {})
    mc = config.get('model', {})
    oc = config.get('output', {})

    if args.data_dir: tc['data_dir']  = args.data_dir
    if args.meta:     tc['meta_csv']  = args.meta
    if args.epochs:   tc['epochs']    = args.epochs
    if args.patience: tc['patience']  = args.patience
    if args.lr:       tc['learning_rate'] = args.lr
    if args.batch_size: tc['batch_size'] = args.batch_size

    print("=" * 60)
    print("KataRank — KAB2 Training")
    print("=" * 60)

    print("\nLoading datasets...")
    train_loader, train_ds = make_kab2_loader(
        data_dir    = tc['data_dir'],
        meta_csv    = tc.get('meta_csv'),
        split       = tc.get('train_split', 'T'),
        batch_size  = tc.get('batch_size', 16),
        shuffle     = True,
        num_workers = tc.get('num_workers', 0),
        max_moves_per_player = tc.get('max_moves', 400),
        min_moves_per_player = tc.get('min_moves', 5),
        stratified  = tc.get('stratified', False),
        n_bands     = tc.get('n_bands', 5),
    )
    val_loader, val_ds = make_kab2_loader(
        data_dir    = tc['data_dir'],
        meta_csv    = tc.get('meta_csv'),
        split       = tc.get('val_split', 'V'),
        batch_size  = tc.get('batch_size', 16),
        shuffle     = False,
        num_workers = tc.get('num_workers', 0),
        max_moves_per_player = tc.get('max_moves', 400),
        min_moves_per_player = tc.get('min_moves', 5),
    )

    # Training requires HumanSL rank anchors: the data must have been
    # generated with -human-model. Inference does not need them.
    labeled = sum(1 for g in train_ds.games if g['rank_b'] >= 0 or g['rank_w'] >= 0)
    if labeled == 0:
        raise SystemExit(
            "ERROR: no HumanSL rank labels found in the training split.\n"
            "Training data must be generated with:  katago batch_analysis "
            "-human-model <human.bin.gz> ...\n"
            "(humanRankIdx is -1 for every game in _meta.csv)"
        )
    print(f"  HumanSL labels: {labeled}/{len(train_ds.games)} games")

    print("\nBuilding model...")
    model = KataRankModel(
        input_dim    = train_ds.input_dim,
        hidden_dim   = mc.get('hidden_dim', 128),
        num_heads    = mc.get('num_heads', 4),
        num_inducing = mc.get('num_inducing', 16),
        encoder_depth= mc.get('encoder_depth', 2),
        cross_depth  = mc.get('cross_depth', 1),
        dropout      = mc.get('dropout', 0.1),
        n_rank_classes = mc.get('num_rank_classes', 29),
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Input dim:  {train_ds.input_dim}")

    lw = tc.get('loss_weights', {})
    loss_fn = KataRankLoss(
        w_rating = lw.get('rating_mse', 1.0),
        w_bt     = lw.get('bradley_terry', 0.5),
        w_rank   = lw.get('rank_anchor', 0.3),
    )

    resume_optimizer_path = None
    if args.resume:
        ckpt = KataRankModel.load(args.resume)
        model.load_state_dict(ckpt.state_dict())
        print(f"Resumed from {args.resume}")
        # Companion file saved next to any checkpoint by Trainer.
        # save_optimizer_state — carries this same continuation forward
        # whether --resume points at last run's own candidate or all the
        # way back at the live production checkpoint.
        candidate = os.path.join(os.path.dirname(args.resume), 'optimizer_state.pt')
        if os.path.exists(candidate):
            resume_optimizer_path = candidate
        else:
            print(f"No optimizer_state.pt next to {args.resume} — "
                  f"continuing with a fresh optimizer/scheduler (weights only)")

    trainer = Trainer(
        model        = model,
        loss_fn      = loss_fn,
        train_loader = train_loader,
        val_loader   = val_loader,
        config       = tc,
        resume_optimizer_path = resume_optimizer_path,
    )

    ckpt_dir = args.ckpt_dir or oc.get('checkpoint_dir', 'nets/katarank')
    t_start = time.time()

    def on_epoch(**progress):
        _write_epoch_progress(progress)

    print("\nStarting training...")
    result = trainer.train(
        epochs   = tc.get('epochs', 100),
        ckpt_dir = ckpt_dir,
        on_epoch = on_epoch,
    )

    model.save(os.path.join(ckpt_dir, 'final.pt'))

    # ── TrainingReport: the canonical training output ─────────────────────────
    print("\nComputing validation metrics...")
    final_metrics = evaluate_metrics(model, val_loader, trainer.device)

    val_labeled = sum(1 for g in val_ds.games if g['rank_b'] >= 0 or g['rank_w'] >= 0)
    report = TrainingReport(
        version          = '1.0',
        created_at       = datetime.now(timezone.utc).isoformat(),
        model_config     = model.get_config(),
        training_config  = dict(tc),
        data             = {
            'train_games':        len(train_ds.games),
            'val_games':          len(val_ds.games),
            'train_rank_labeled': labeled,
            'val_rank_labeled':   val_labeled,
            'input_dim':          train_ds.input_dim,
        },
        epochs_trained   = result['epochs_trained'],
        early_stopped    = trainer.early_stopped,
        best_epoch       = trainer.best_epoch,
        best_val_loss    = trainer.best_val_loss,
        train_loss       = trainer.train_hist,
        val_loss         = trainer.val_hist,
        final_metrics    = final_metrics,
        ordinal_thresholds_b = model.rank_head_b.thresholds.detach().cpu().tolist(),
        ordinal_thresholds_w = model.rank_head_w.thresholds.detach().cpu().tolist(),
        elapsed_seconds  = round(time.time() - t_start, 1),
    )
    report_path = os.path.join(ckpt_dir, 'training_report.json')
    save_training_report(report, report_path)

    print(f"\nDone — best val loss: {result['best_val_loss']:.4f}  "
          f"epochs: {result['epochs_trained']}")
    print(f"  metrics: {final_metrics}")
    print(f"  report:  {report_path}")


if __name__ == '__main__':
    main()
