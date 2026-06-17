"""
KataRank — Training Entry Point

Usage:
    uv run katarank-train --data-dir data/kab2 --meta data/kab2/_meta.csv
    uv run katarank-train --config src/katarank/train/config_kata_native.yaml --resume nets/best.pt
"""

import os
import time
import argparse
from datetime import datetime, timezone

import numpy as np
import yaml
import torch
import torch.optim as optim

from katarank.model import KataRankModel, KataRankLoss
from katarank.data.datasets import KAB2Dataset, make_kab2_loader
from katarank.schema import TrainingReport, save_training_report


def _device(cfg_val: str) -> torch.device:
    if cfg_val == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(cfg_val)


class Trainer:
    def __init__(self, model, loss_fn, train_loader, val_loader, config):
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

    def train(self, epochs: int, ckpt_dir: str) -> dict:
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
                  f"val={val['val_loss']:.4f}  lr={lr:.2e}  ({dt:.1f}s)")

            self.scheduler.step()

            if val['val_loss'] < self.best_val_loss:
                self.best_val_loss = val['val_loss']
                self.best_epoch = epoch
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.bad_epochs = 0
                self.model.save(os.path.join(ckpt_dir, 'best.pt'))
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.early_stopped = True
                    break

        if self.best_state:
            self.model.load_state_dict(self.best_state)

        return {'best_val_loss': self.best_val_loss, 'epochs_trained': epoch}


# ─── Validation metrics (for TrainingReport) ─────────────────────────────────

@torch.no_grad()
def evaluate_metrics(model, loader, device) -> dict:
    """Rank/rating quality on a validation loader. See TrainingReport docs."""
    model.eval()
    ratings_pred, ratings_tgt = [], []
    rank_pred, rank_lab = [], []

    for batch in loader:
        if not batch['xlens']:
            continue
        out = model(batch['x'].to(device), batch['xlens'])
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
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--resume', help='Resume from checkpoint .pt')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tc = config.get('training', {})
    mc = config.get('model', {})
    oc = config.get('output', {})

    if args.data_dir: tc['data_dir']  = args.data_dir
    if args.meta:     tc['meta_csv']  = args.meta
    if args.epochs:   tc['epochs']    = args.epochs
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

    if args.resume:
        ckpt = KataRankModel.load(args.resume)
        model.load_state_dict(ckpt.state_dict())
        print(f"Resumed from {args.resume}")

    trainer = Trainer(
        model        = model,
        loss_fn      = loss_fn,
        train_loader = train_loader,
        val_loader   = val_loader,
        config       = tc,
    )

    ckpt_dir = oc.get('checkpoint_dir', 'nets/katarank')
    t_start = time.time()

    print("\nStarting training...")
    result = trainer.train(
        epochs   = tc.get('epochs', 100),
        ckpt_dir = ckpt_dir,
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
