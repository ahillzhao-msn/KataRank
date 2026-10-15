"""
KataRank — Checkpoint evaluation (no training).

Loads a saved KataRankModel checkpoint and computes the same holdout
metrics training.py reports (val_loss, rank_mae, rank_acc, rank_acc_pm1,
rating_corr) against a given KAB2 data directory/split.

Used by scripts/train_session.py's evaluation gate: a freshly-trained
candidate checkpoint and the currently-live production checkpoint are
both evaluated on the *same* up-to-date holdout split, so "did training
actually help" is an apples-to-apples comparison rather than trusting
each checkpoint's own historical training_report.json (which may have
been computed against a smaller/older holdout set).
"""
from __future__ import annotations

import torch

from katarank.model import KataRankModel, KataRankLoss
from katarank.data.datasets import make_kab2_loader
from katarank.train.training import evaluate_metrics


def _device(cfg_val: str = 'auto') -> torch.device:
    if cfg_val == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(cfg_val)


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: str,
    data_dir: str,
    meta_csv: str | None = None,
    val_split: str = 'V',
    batch_size: int = 32,
    device: str = 'auto',
    loss_weights: dict | None = None,
    input_dim: int | None = None,
) -> dict:
    """Return {'val_loss', 'n_rank_labeled', 'rank_mae', 'rank_acc',
    'rank_acc_pm1', 'rating_corr'} for ``ckpt_path`` on ``data_dir``'s
    ``val_split``. Raises if the checkpoint's input_dim doesn't match the
    data (e.g. comparing a full model against lite-only features).

    ``loss_weights`` MUST match whatever ``KataRankLoss`` weights the
    checkpoint being compared against was actually trained/selected with
    (see training.py's ``main()`` — currently {rating_mse: 1.0,
    bradley_terry: 0.5, rank_anchor: 0.3} in config_kata_native.yaml).
    ``KataRankLoss()``'s own constructor defaults (w_bt=2.0, w_rank=0.1)
    are NOT the training config's values — passing none here silently
    computes val_loss under a different weighting than the number this
    is being compared against, making the two incomparable even though
    they'd look like the same metric.

    ``input_dim``: pass 10 to evaluate a lite checkpoint against this
    (full, 1034-dim) ``data_dir`` — slices batch['x'] down before feeding
    the model, same convention distill.py's own end-of-run eval uses.
    batch['x'] is a flat (total_moves_in_batch, feat_dim) tensor (see
    kab2_collate — sequences are torch.cat'd along dim 0, not padded to
    (B, T, D)), so this is a 2D slice on the last axis.
    """
    dev = _device(device)
    model = KataRankModel.load(ckpt_path)
    model.to(dev)
    model.eval()

    val_loader, _ = make_kab2_loader(
        data_dir=data_dir,
        meta_csv=meta_csv,
        split=val_split,
        batch_size=batch_size,
        shuffle=False,
    )

    lw = loss_weights or {}
    loss_fn = KataRankLoss(
        w_rating=lw.get('rating_mse', 1.0),
        w_bt=lw.get('bradley_terry', 0.5),
        w_rank=lw.get('rank_anchor', 0.3),
    )
    total, n = 0.0, 0
    for batch in val_loader:
        if not batch['xlens']:
            continue
        x = batch['x'][:, :input_dim] if input_dim else batch['x']
        out = model(x.to(dev), batch['xlens'])
        targets = {
            k: batch[k].to(dev)
            for k in ('target_b', 'target_w', 'rank_b', 'rank_w',
                      'human_lp_b', 'human_lp_w')
        }
        total += loss_fn(out, targets)['total'].item()
        n += 1

    metrics = {'val_loss': total / max(n, 1)}
    metrics.update(evaluate_metrics(model, val_loader, dev, input_dim=input_dim))
    return metrics


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Evaluate a KataRank checkpoint')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--meta', default=None)
    parser.add_argument('--split', default='V')
    parser.add_argument('--input-dim', type=int, default=None,
                        help='Pass 10 to evaluate a lite checkpoint against full-dim data_dir')
    args = parser.parse_args()

    metrics = evaluate_checkpoint(args.checkpoint, args.data_dir,
                                  meta_csv=args.meta, val_split=args.split,
                                  input_dim=args.input_dim)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
