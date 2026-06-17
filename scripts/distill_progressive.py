"""
Progressive distillation experiment — validate pipeline with tiny samples.
Runs 10 → 20 → 50 → 100 → 200 games, each warm-starting from prior.

Usage:
    uv run python scripts/distill_progressive.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from katarank.model import KataRankModel
from katarank.data.datasets.dataset_kab2 import KAB2Dataset
from katarank.schema import kab2_collate
from katarank.train.distill import (
    DistillationLoss, DistillationDataset, DistillTrainer,
    distill_collate, generate_teacher_targets,
)
from katarank.train.training import evaluate_metrics


CACHE_DIR = str(Path.home() / ".katarank" / "kab2_cache")
TEACHER_PATH = "nets/katarank/best.pt"
STEPS = [10, 20, 50, 100, 200]
STUDENT_DIM = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_step(
    n_games: int,
    teacher: KataRankModel,
    full_train_ds: KAB2Dataset,
    full_val_ds: KAB2Dataset,
    full_train_targets: dict,
    full_val_targets: dict,
    resume_from: str | None = None,
) -> str:
    """Train student on n_games, return checkpoint path."""
    print(f"\n{'='*60}")
    print(f"  Step: {n_games} games")
    print(f"{'='*60}")

    # Subset selection — stratified: pick from evenly across the dataset
    n_train = min(n_games, len(full_train_ds))
    n_val = max(n_train // 10, 5)

    stride = max(len(full_train_ds) // n_train, 1)
    train_indices = list(range(0, len(full_train_ds), stride))[:n_train]
    val_stride = max(len(full_val_ds) // n_val, 1)
    val_indices = list(range(0, len(full_val_ds), val_stride))[:n_val]

    # Slice targets for subset
    def slice_targets(targets, indices):
        return {k: v[indices] for k, v in targets.items()}

    sub_train_targets = slice_targets(full_train_targets, train_indices)
    sub_val_targets = slice_targets(full_val_targets, val_indices)

    # Create distillation datasets
    sub_train_ds = Subset(full_train_ds, train_indices)
    sub_val_ds = Subset(full_val_ds, val_indices)

    # Wrap Subset to look like KAB2Dataset for DistillationDataset
    class SubsetWrapper:
        def __init__(self, subset, targets, student_dim):
            self.subset = subset
            self.targets = targets
            self.student_dim = student_dim
            self.games = [subset.dataset.games[i] for i in subset.indices]
        def __len__(self):
            return len(self.subset)
        @property
        def input_dim(self):
            return self.student_dim
        def __getitem__(self, idx):
            sample = self.subset[idx]
            if sample['seq_len'] > 0:
                sample['x'] = sample['x'][:, :self.student_dim]
            else:
                sample['x'] = torch.zeros(0, self.student_dim)
            sample['teacher_b_rating'] = self.targets['teacher_b_rating'][idx]
            sample['teacher_w_rating'] = self.targets['teacher_w_rating'][idx]
            sample['teacher_rank_probs_b'] = self.targets['teacher_rank_probs_b'][idx]
            sample['teacher_rank_probs_w'] = self.targets['teacher_rank_probs_w'][idx]
            return sample

    dist_train = SubsetWrapper(sub_train_ds, sub_train_targets, STUDENT_DIM)
    dist_val = SubsetWrapper(sub_val_ds, sub_val_targets, STUDENT_DIM)

    batch_size = min(16, n_train)
    train_loader = DataLoader(dist_train, batch_size=batch_size,
                              shuffle=True, collate_fn=distill_collate)
    val_loader = DataLoader(dist_val, batch_size=batch_size,
                            shuffle=False, collate_fn=distill_collate)

    # Build or resume student
    if resume_from and Path(resume_from).exists():
        student = KataRankModel.load(resume_from)
        lr = 2e-4  # lower LR for warm-start
        print(f"  Warm-start from {resume_from}, lr={lr}")
    else:
        student = KataRankModel(
            input_dim=STUDENT_DIM, hidden_dim=64, num_heads=4,
            num_inducing=16, encoder_depth=2, cross_depth=1, dropout=0.2,
        )
        lr = 5e-4
        print(f"  Fresh student, params={student.count_parameters():,}, lr={lr}")

    loss_fn = DistillationLoss(w_kl=1.0, w_rating=1.0, w_hard=0.1, temperature=2.0)

    epochs = min(30 + n_games, 100)  # more epochs for tiny datasets
    config = {
        'learning_rate': lr,
        'lr_min': 1e-5,
        'warmup_epochs': 3,
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'patience': 20,
        'epochs': epochs,
        'device': DEVICE,
    }

    ckpt_dir = f"nets/katarank_lite/step_{n_games}"
    trainer = DistillTrainer(student, loss_fn, train_loader, val_loader, config)

    t0 = time.time()
    result = trainer.train(epochs, ckpt_dir)
    elapsed = time.time() - t0

    ckpt_path = os.path.join(ckpt_dir, 'best_lite.pt')
    student.save(os.path.join(ckpt_dir, 'final_lite.pt'))

    # Quick eval on the val subset (10-dim sliced)
    class SlicedSubset:
        def __init__(self, subset, dim):
            self.subset = subset
            self.dim = dim
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            s = self.subset[idx]
            if s['seq_len'] > 0:
                s['x'] = s['x'][:, :self.dim]
            else:
                s['x'] = torch.zeros(0, self.dim)
            return s

    eval_loader = DataLoader(SlicedSubset(sub_val_ds, STUDENT_DIM),
                             batch_size=batch_size, shuffle=False,
                             collate_fn=kab2_collate)
    metrics = evaluate_metrics(student, eval_loader, trainer.device)

    print(f"\n  Result: val_loss={result['best_val_loss']:.4f}  "
          f"epochs={result['epochs_trained']}  time={elapsed:.0f}s")
    print(f"  Metrics: {metrics}")

    return ckpt_path


def main():
    print("=" * 60)
    print("KataRank — Progressive Distillation Experiment")
    print(f"Steps: {STEPS}  Device: {DEVICE}")
    print("=" * 60)

    # Load teacher
    print(f"\nLoading teacher from {TEACHER_PATH}...")
    teacher = KataRankModel.load(TEACHER_PATH, device=DEVICE)
    teacher.eval()
    tcfg = teacher.get_config()
    print(f"  Teacher: input_dim={tcfg['input_dim']}, params={teacher.count_parameters():,}")

    # Load full datasets
    print("\nLoading datasets...")
    full_train = KAB2Dataset(CACHE_DIR, split='T', max_moves_per_player=400,
                             min_moves_per_player=5)
    full_val = KAB2Dataset(CACHE_DIR, split='V', max_moves_per_player=400,
                           min_moves_per_player=5)

    # Generate all teacher targets once
    print(f"\nGenerating teacher targets ({DEVICE})...")
    t0 = time.time()
    train_targets = generate_teacher_targets(teacher, full_train, DEVICE, batch_size=32)
    val_targets = generate_teacher_targets(teacher, full_val, DEVICE, batch_size=32)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Progressive training
    ckpt = None
    results = []

    for n in STEPS:
        ckpt = run_step(
            n_games=n,
            teacher=teacher,
            full_train_ds=full_train,
            full_val_ds=full_val,
            full_train_targets=train_targets,
            full_val_targets=val_targets,
            resume_from=ckpt,
        )
        results.append((n, ckpt))

    print(f"\n{'='*60}")
    print("Progressive Distillation Complete!")
    print(f"{'='*60}")
    for n, path in results:
        print(f"  {n:>4d} games → {path}")


if __name__ == "__main__":
    main()
