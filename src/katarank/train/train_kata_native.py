"""
KataRank — Training Entry Point

Usage:
    uv run katarank-train --data-dir data/kab2 --meta data/kab2/_meta.csv
    uv run katarank-train --config src/katarank/train/config_kata_native.yaml --resume nets/best.pt
"""

import os
import time
import argparse
import yaml
import torch
import torch.optim as optim

from katarank.model import KataRankModel, KataRankLoss
from katarank.data.katago_native import KAB2Dataset, make_kab2_loader


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
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100),
            eta_min=config.get('lr_min', 1e-5),
        )

        self.gradient_clip = config.get('gradient_clip', 1.0)
        self.patience = config.get('patience', 20)
        self.best_val_loss = float('inf')
        self.best_state = None
        self.bad_epochs = 0

    def _forward(self, batch):
        x     = batch['x'].to(self.device)
        xlens = batch['xlens']
        out   = self.model(x, xlens)
        return out, batch

    def train_epoch(self) -> float:
        self.model.train()
        total, n = 0.0, 0
        for batch in self.train_loader:
            if not batch:
                continue
            out, b = self._forward(batch)
            loss, _ = self.loss_fn(
                out,
                target_b    = b['target_b'].to(self.device),
                target_w    = b['target_w'].to(self.device),
                rank_b      = b['rank_b'].to(self.device),
                rank_w      = b['rank_w'].to(self.device),
                human_lp_b  = b['human_lp_b'].to(self.device),
                human_lp_w  = b['human_lp_w'].to(self.device),
            )
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
            if not batch:
                continue
            out, b = self._forward(batch)
            loss, _ = self.loss_fn(
                out,
                target_b    = b['target_b'].to(self.device),
                target_w    = b['target_w'].to(self.device),
                rank_b      = b['rank_b'].to(self.device),
                rank_w      = b['rank_w'].to(self.device),
                human_lp_b  = b['human_lp_b'].to(self.device),
                human_lp_w  = b['human_lp_w'].to(self.device),
            )
            total += loss.item()
            n += 1
        return {'val_loss': total / max(n, 1)}

    def train(self, epochs: int, ckpt_dir: str) -> dict:
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Training on {self.device}  |  train={len(self.train_loader.dataset)}  "
              f"val={len(self.val_loader.dataset)}")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val = self.validate()
            dt = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  "
                  f"val={val['val_loss']:.4f}  lr={lr:.2e}  ({dt:.1f}s)")

            self.scheduler.step()

            if val['val_loss'] < self.best_val_loss:
                self.best_val_loss = val['val_loss']
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.bad_epochs = 0
                self.model.save(os.path.join(ckpt_dir, 'best.pt'))
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if self.best_state:
            self.model.load_state_dict(self.best_state)

        return {'best_val_loss': self.best_val_loss, 'epochs_trained': epoch}


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
    )
    val_loader, _ = make_kab2_loader(
        data_dir    = tc['data_dir'],
        meta_csv    = tc.get('meta_csv'),
        split       = tc.get('val_split', 'V'),
        batch_size  = tc.get('batch_size', 16),
        shuffle     = False,
        num_workers = tc.get('num_workers', 0),
        max_moves_per_player = tc.get('max_moves', 400),
        min_moves_per_player = tc.get('min_moves', 5),
    )

    print("\nBuilding model...")
    model = KataRankModel(
        input_dim    = train_ds.input_dim,
        hidden_dim   = mc.get('hidden_dim', 128),
        num_heads    = mc.get('num_heads', 4),
        num_inducing = mc.get('num_inducing', 16),
        encoder_depth= mc.get('encoder_depth', 2),
        cross_depth  = mc.get('cross_depth', 1),
        dropout      = mc.get('dropout', 0.1),
        num_rank_classes = mc.get('num_rank_classes', 29),
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Input dim:  {train_ds.input_dim}")

    lw = tc.get('loss_weights', {})
    loss_fn = KataRankLoss(
        rating_w   = lw.get('rating_mse', 1.0),
        bt_w       = lw.get('bradley_terry', 0.5),
        rank_w     = lw.get('rank_anchor', 0.3),
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

    print("\nStarting training...")
    result = trainer.train(
        epochs   = tc.get('epochs', 100),
        ckpt_dir = oc.get('checkpoint_dir', 'nets/katarank'),
    )

    model.save(os.path.join(oc.get('checkpoint_dir', 'nets/katarank'), 'final.pt'))

    print(f"\nDone — best val loss: {result['best_val_loss']:.4f}  "
          f"epochs: {result['epochs_trained']}")


if __name__ == '__main__':
    main()
