"""
KataRank — Knowledge Distillation (v2 Lite Model)
===================================================
Train a lite student model (input_dim=10, scalar-only) to mimic the full
teacher model (input_dim=1034, trunk vectors).

The student receives only the 10-dim scalar features per move but learns
to reproduce the teacher's rank probability distributions and ratings.
This gives the lite model quality approaching the full model without
requiring expensive trunk vector computation at inference time.

Usage:
    uv run python -m katarank.train.distill --teacher nets/katarank/best.pt --data-dir ~/.katarank/kab2_cache
    uv run python -m katarank.train.distill --config src/katarank/train/config_distill.yaml
"""

from __future__ import annotations

import os
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from katarank.model import KataRankModel
from katarank.data.preprocess import read_kab2_combined
from katarank.data.datasets.dataset_kab2 import (
    KAB2Dataset, StratifiedRankSampler, rank_str_to_idx, NUM_RANK_CLASSES,
)
from katarank.schema import kab2_collate, save_training_report, TrainingReport


# ─── Distillation Loss ──────────────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation.

    Components:
        - KL divergence: student rank_probs vs teacher rank_probs (both sides)
        - Rating MSE: student ratings vs teacher ratings (both sides)
        - Hard label anchor: original RankAnchorLoss for calibration (optional)

    Temperature controls softness of teacher distributions. Higher T
    spreads probability mass more evenly, giving richer gradient signal
    for the student.
    """

    def __init__(
        self,
        w_kl: float = 1.0,
        w_rating: float = 1.0,
        w_hard: float = 0.1,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.w_kl = w_kl
        self.w_rating = w_rating
        self.w_hard = w_hard
        self.temperature = temperature

    def forward(
        self,
        student: Dict[str, torch.Tensor],
        teacher: Dict[str, torch.Tensor],
        hard_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        T = self.temperature

        # KL divergence on rank probabilities (both sides)
        # Teacher probs are sharpened/softened by temperature
        kl_b = self._kl_with_temp(student['rank_probs_b'], teacher['rank_probs_b'], T)
        kl_w = self._kl_with_temp(student['rank_probs_w'], teacher['rank_probs_w'], T)
        l_kl = kl_b + kl_w

        # Rating MSE vs teacher predictions
        l_rating = (
            F.mse_loss(student['b_rating'], teacher['b_rating'])
            + F.mse_loss(student['w_rating'], teacher['w_rating'])
        )

        total = self.w_kl * l_kl + self.w_rating * l_rating

        result = {
            'total': total,
            'kl': l_kl,
            'rating': l_rating,
        }

        # Optional hard label anchor for rank calibration
        if hard_targets is not None and self.w_hard > 0:
            from katarank.model.losses import RankAnchorLoss
            anchor = RankAnchorLoss()
            l_hard_b = anchor(
                student['rank_probs_b'],
                hard_targets['rank_b'],
                hard_targets['human_lp_b'],
            )
            l_hard_w = anchor(
                student['rank_probs_w'],
                hard_targets['rank_w'],
                hard_targets['human_lp_w'],
            )
            l_hard = l_hard_b + l_hard_w
            total = total + self.w_hard * l_hard
            result['hard'] = l_hard
            result['total'] = total

        return result

    @staticmethod
    def _kl_with_temp(
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        T: float,
    ) -> torch.Tensor:
        """KL(teacher_soft || student_soft) with temperature scaling."""
        teacher_logits = torch.log(teacher_probs + 1e-8) / T
        student_logits = torch.log(student_probs + 1e-8) / T
        teacher_soft = F.softmax(teacher_logits, dim=-1)
        student_log_soft = F.log_softmax(student_logits, dim=-1)
        # Scale by T^2 so gradient magnitude is independent of temperature
        return T * T * F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')


# ─── Teacher target generation ───────────────────────────────────────────────

@torch.no_grad()
def generate_teacher_targets(
    teacher: KataRankModel,
    dataset: KAB2Dataset,
    device: str = 'cuda',
    batch_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """Run teacher inference on all games, return soft targets.

    Returns dict of tensors indexed by dataset position:
        teacher_b_rating:     (N,)
        teacher_w_rating:     (N,)
        teacher_rank_probs_b: (N, 29)
        teacher_rank_probs_w: (N, 29)
    """
    teacher = teacher.to(device).eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=kab2_collate, num_workers=0,
    )

    all_br, all_wr, all_rpb, all_rpw = [], [], [], []

    for batch in loader:
        if not batch['xlens']:
            continue
        x = batch['x'].to(device)
        out = teacher(x, batch['xlens'])
        all_br.append(out['b_rating'].cpu())
        all_wr.append(out['w_rating'].cpu())
        all_rpb.append(out['rank_probs_b'].cpu())
        all_rpw.append(out['rank_probs_w'].cpu())

    return {
        'teacher_b_rating': torch.cat(all_br),
        'teacher_w_rating': torch.cat(all_wr),
        'teacher_rank_probs_b': torch.cat(all_rpb),
        'teacher_rank_probs_w': torch.cat(all_rpw),
    }


# ─── Distillation Dataset ───────────────────────────────────────────────────

class DistillationDataset(Dataset):
    """Wraps a KAB2Dataset with teacher soft targets.

    Each item returns the same KAB2Sample but with:
    - x sliced to first ``student_dim`` columns (10-dim scalar features)
    - teacher soft targets attached
    """

    def __init__(
        self,
        base_dataset: KAB2Dataset,
        teacher_targets: Dict[str, torch.Tensor],
        student_dim: int = 10,
    ):
        self.base = base_dataset
        self.targets = teacher_targets
        self.student_dim = student_dim

    def __len__(self) -> int:
        return len(self.base)

    @property
    def input_dim(self) -> int:
        return self.student_dim

    @property
    def games(self):
        return self.base.games

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        # Slice move features to student_dim (first 10 scalar cols)
        if sample['seq_len'] > 0:
            sample['x'] = sample['x'][:, :self.student_dim]
        else:
            sample['x'] = torch.zeros(0, self.student_dim)

        # Attach teacher targets
        sample['teacher_b_rating'] = self.targets['teacher_b_rating'][idx]
        sample['teacher_w_rating'] = self.targets['teacher_w_rating'][idx]
        sample['teacher_rank_probs_b'] = self.targets['teacher_rank_probs_b'][idx]
        sample['teacher_rank_probs_w'] = self.targets['teacher_rank_probs_w'][idx]
        return sample


def distill_collate(batch):
    """Collate for DistillationDataset — extends kab2_collate with teacher targets."""
    batch = [item for item in batch if item['seq_len'] > 0]
    if not batch:
        return {
            'x': torch.empty(0), 'xlens': [],
            'target_b': torch.empty(0), 'target_w': torch.empty(0),
            'rank_b': torch.empty(0), 'rank_w': torch.empty(0),
            'human_lp_b': torch.empty(0), 'human_lp_w': torch.empty(0),
            'game_ids': [],
            'teacher_b_rating': torch.empty(0),
            'teacher_w_rating': torch.empty(0),
            'teacher_rank_probs_b': torch.empty(0, 29),
            'teacher_rank_probs_w': torch.empty(0, 29),
        }

    base = kab2_collate(batch)
    base['teacher_b_rating'] = torch.stack([b['teacher_b_rating'] for b in batch])
    base['teacher_w_rating'] = torch.stack([b['teacher_w_rating'] for b in batch])
    base['teacher_rank_probs_b'] = torch.stack([b['teacher_rank_probs_b'] for b in batch])
    base['teacher_rank_probs_w'] = torch.stack([b['teacher_rank_probs_w'] for b in batch])
    return base


# ─── Distillation Trainer ───────────────────────────────────────────────────

class DistillTrainer:

    def __init__(self, student, loss_fn, train_loader, val_loader, config):
        self.student = student
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config

        device_str = config.get('device', 'auto')
        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)
        self.student = self.student.to(self.device)

        self.optimizer = optim.AdamW(
            student.parameters(),
            lr=config.get('learning_rate', 5e-4),
            weight_decay=config.get('weight_decay', 1e-4),
        )

        epochs = config.get('epochs', 100)
        warmup = config.get('warmup_epochs', 5)
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
        self.patience = config.get('patience', 30)
        self.best_val_loss = float('inf')
        self.best_state = None
        self.best_epoch = 0
        self.bad_epochs = 0
        self.early_stopped = False
        self.train_hist: list = []
        self.val_hist: list = []

    def _batch_loss(self, batch) -> torch.Tensor:
        x = batch['x'].to(self.device)
        xlens = batch['xlens']
        student_out = self.student(x, xlens)

        teacher_out = {
            'b_rating': batch['teacher_b_rating'].to(self.device),
            'w_rating': batch['teacher_w_rating'].to(self.device),
            'rank_probs_b': batch['teacher_rank_probs_b'].to(self.device),
            'rank_probs_w': batch['teacher_rank_probs_w'].to(self.device),
        }

        hard_targets = {
            'rank_b': batch['rank_b'].to(self.device),
            'rank_w': batch['rank_w'].to(self.device),
            'human_lp_b': batch['human_lp_b'].to(self.device),
            'human_lp_w': batch['human_lp_w'].to(self.device),
        }

        return self.loss_fn(student_out, teacher_out, hard_targets)['total']

    def train_epoch(self) -> float:
        self.student.train()
        total, n = 0.0, 0
        for batch in self.train_loader:
            if not batch['xlens']:
                continue
            loss = self._batch_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.gradient_clip)
            self.optimizer.step()
            total += loss.item()
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self) -> float:
        self.student.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            if not batch['xlens']:
                continue
            total += self._batch_loss(batch).item()
            n += 1
        return total / max(n, 1)

    def train(self, epochs: int, ckpt_dir: str) -> dict:
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Distillation on {self.device}  |  "
              f"train={len(self.train_loader.dataset)}  "
              f"val={len(self.val_loader.dataset)}")
        print("-" * 60)

        epoch = 0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate()
            dt = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            self.train_hist.append(round(train_loss, 6))
            self.val_hist.append(round(val_loss, 6))

            print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  lr={lr:.2e}  ({dt:.1f}s)")

            self.scheduler.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_state = {k: v.cpu().clone()
                                   for k, v in self.student.state_dict().items()}
                self.bad_epochs = 0
                self.student.save(os.path.join(ckpt_dir, 'best_lite.pt'))
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.early_stopped = True
                    break

        if self.best_state:
            self.student.load_state_dict(self.best_state)

        return {'best_val_loss': self.best_val_loss, 'epochs_trained': epoch}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='KataRank — Knowledge Distillation')
    parser.add_argument('--teacher', default='nets/katarank/best.pt',
                        help='Path to teacher checkpoint')
    parser.add_argument('--data-dir', required=True,
                        help='KAB2 cache directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Student model hidden dim (smaller than teacher)')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    print("=" * 60)
    print("KataRank — Knowledge Distillation (v2 Lite)")
    print("=" * 60)

    # ── Load teacher ─────────────────────────────────────────────────────
    print(f"\nLoading teacher from {args.teacher}...")
    teacher = KataRankModel.load(args.teacher)
    teacher_cfg = teacher.get_config()
    print(f"  Teacher: input_dim={teacher_cfg['input_dim']}, "
          f"hidden_dim={teacher_cfg['hidden_dim']}, "
          f"params={teacher.count_parameters():,}")

    # ── Load datasets ────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = KAB2Dataset(args.data_dir, split='T', max_moves_per_player=400,
                           min_moves_per_player=5)
    val_ds = KAB2Dataset(args.data_dir, split='V', max_moves_per_player=400,
                         min_moves_per_player=5)

    # ── Generate teacher soft targets ────────────────────────────────────
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nGenerating teacher targets on {device}...")
    t0 = time.time()
    train_targets = generate_teacher_targets(teacher, train_ds, device, args.batch_size)
    val_targets = generate_teacher_targets(teacher, val_ds, device, args.batch_size)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Wrap in distillation datasets ────────────────────────────────────
    student_dim = 10
    dist_train = DistillationDataset(train_ds, train_targets, student_dim)
    dist_val = DistillationDataset(val_ds, val_targets, student_dim)

    train_sampler = StratifiedRankSampler(dist_train, batch_size=args.batch_size)
    train_loader = DataLoader(dist_train, batch_size=args.batch_size,
                              sampler=train_sampler, collate_fn=distill_collate)
    val_loader = DataLoader(dist_val, batch_size=args.batch_size,
                            shuffle=False, collate_fn=distill_collate)

    # ── Build student model ──────────────────────────────────────────────
    print(f"\nBuilding student model (input_dim={student_dim}, "
          f"hidden_dim={args.hidden_dim})...")
    student = KataRankModel(
        input_dim=student_dim,
        hidden_dim=args.hidden_dim,
        num_heads=4,
        num_inducing=16,
        encoder_depth=2,
        cross_depth=1,
        dropout=0.2,
    )
    print(f"  Student params: {student.count_parameters():,}")
    print(f"  Teacher params: {teacher.count_parameters():,}")
    print(f"  Compression: {teacher.count_parameters() / student.count_parameters():.1f}x")

    # ── Train ────────────────────────────────────────────────────────────
    loss_fn = DistillationLoss(
        w_kl=1.0,
        w_rating=1.0,
        w_hard=0.1,
        temperature=args.temperature,
    )

    config = {
        'learning_rate': args.lr,
        'lr_min': 1e-5,
        'warmup_epochs': 5,
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'patience': 30,
        'epochs': args.epochs,
        'device': args.device,
    }

    ckpt_dir = 'nets/katarank_lite'
    trainer = DistillTrainer(student, loss_fn, train_loader, val_loader, config)

    print("\nStarting distillation...")
    t_start = time.time()
    result = trainer.train(args.epochs, ckpt_dir)

    student.save(os.path.join(ckpt_dir, 'final_lite.pt'))

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\nComputing validation metrics...")
    from katarank.train.training import evaluate_metrics
    # Need a standard loader (not distillation) for evaluation
    eval_ds = KAB2Dataset(args.data_dir, split='V', max_moves_per_player=400,
                          min_moves_per_player=5)
    # Slice eval data to student_dim
    class SlicedDataset(Dataset):
        def __init__(self, base, dim):
            self.base = base
            self.dim = dim
        def __len__(self):
            return len(self.base)
        @property
        def input_dim(self):
            return self.dim
        def __getitem__(self, idx):
            s = self.base[idx]
            if s['seq_len'] > 0:
                s['x'] = s['x'][:, :self.dim]
            else:
                s['x'] = torch.zeros(0, self.dim)
            return s

    eval_sliced = SlicedDataset(eval_ds, student_dim)
    eval_loader = DataLoader(eval_sliced, batch_size=args.batch_size,
                             shuffle=False, collate_fn=kab2_collate)
    metrics = evaluate_metrics(student, eval_loader, trainer.device)

    elapsed = round(time.time() - t_start, 1)
    report = TrainingReport(
        version='1.0-distill',
        created_at=datetime.now(timezone.utc).isoformat(),
        model_config=student.get_config(),
        training_config=config,
        data={
            'train_games': len(train_ds.games),
            'val_games': len(val_ds.games),
            'input_dim': student_dim,
            'teacher': args.teacher,
            'teacher_input_dim': teacher_cfg['input_dim'],
            'temperature': args.temperature,
        },
        epochs_trained=result['epochs_trained'],
        early_stopped=trainer.early_stopped,
        best_epoch=trainer.best_epoch,
        best_val_loss=trainer.best_val_loss,
        train_loss=trainer.train_hist,
        val_loss=trainer.val_hist,
        final_metrics=metrics,
        ordinal_thresholds_b=student.rank_head_b.thresholds.detach().cpu().tolist(),
        ordinal_thresholds_w=student.rank_head_w.thresholds.detach().cpu().tolist(),
        elapsed_seconds=elapsed,
    )
    report_path = os.path.join(ckpt_dir, 'training_report.json')
    save_training_report(report, report_path)

    print(f"\nDone — best val loss: {result['best_val_loss']:.4f}  "
          f"epochs: {result['epochs_trained']}")
    print(f"  metrics: {metrics}")
    print(f"  report:  {report_path}")


if __name__ == '__main__':
    main()
