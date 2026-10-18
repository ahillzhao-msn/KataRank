"""
KataRank — fit the HumanSL anchor calibration (katarank.data.anchor_calibration).

Pairs each player's cached source-anchor label (``--cache-meta``, e.g. kab2_cache/_meta.csv
written by anchor v2) with the same player's target-anchor label from one or more
batch_analysis runs of the current anchor (``--truth``: _meta.csv files, any games of the
cache). Chooses the kernel bandwidth by grouped (per-game) K-fold CV, reports the
held-out error per rank band, and writes the artifact that KAB2Dataset applies at load
time.

Usage (from the KataRank root):
    .venv/Scripts/python.exe scripts/fit_anchor_calibration.py \
        --cache-meta ~/.katarank/kab2_cache/_meta.csv \
        --truth runA/_meta.csv runB/_meta.csv \
        [--out nets/anchor_calibration/v2_to_v3.json]
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from katarank.data.anchor_calibration import DEFAULT_CALIBRATION, AnchorCalibration  # noqa: E402

BANDS = [(0, 9), (9, 12), (12, 16), (16, 20), (20, 24), (24, 29)]   # 20k-12k … 5d-9d


def _rows(path):
    with open(path, encoding='utf-8', errors='replace', newline='') as f:
        return list(csv.DictReader(f))


def _ver(row):
    try:
        return int(round(float(row.get('humanAnchorVer') or 0)))
    except ValueError:
        return 0


def load_pairs(cache_meta, truth_metas, source_anchor, target_anchor):
    cache = {r['file']: r for r in _rows(cache_meta) if _ver(r) == source_anchor}
    pairs = {}   # (game, side) → (src_exp, src_sd, tgt_exp, tgt_sd)
    for path in truth_metas:
        for r in _rows(path):
            c = cache.get(r['file'])
            if c is None or _ver(r) != target_anchor:
                continue
            for s in 'BW':
                vals = [c[f'{s}_humanRankExp'], c[f'{s}_humanRankSd'],
                        r[f'{s}_humanRankExp'], r[f'{s}_humanRankSd']]
                if all(vals):
                    pairs[(r['file'], s)] = tuple(float(v) for v in vals)
    keys = sorted(pairs)
    return np.array([k[0] for k in keys]), np.array([pairs[k] for k in keys])


def grouped_folds(groups, k, seed=0):
    uniq = np.unique(groups)
    np.random.default_rng(seed).shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def cv_predict(X, groups, k, **fit_kw):
    mean, sd = np.zeros(len(X)), np.zeros(len(X))
    for tr, te in grouped_folds(groups, k):
        cal = AnchorCalibration.fit(X[tr, 0], X[tr, 1], X[tr, 2], X[tr, 3], **fit_kw)
        mean[te], sd[te] = cal.apply(X[te, 0], X[te, 1])
    return mean, sd


def error_report(est, truth):
    e = est - truth
    rep = {'mean_abs_err': round(float(np.abs(e).mean()), 4),
           'bias': round(float(e.mean()), 4),
           'max_abs_err': round(float(np.abs(e).max()), 3),
           'rounded_equal': round(float(np.mean(np.floor(est + .5) == np.floor(truth + .5))), 4),
           'bands': {}}
    for a, b in BANDS:
        sel = (truth >= a) & (truth < b)
        if sel.any():
            rep['bands'][f'{a}-{b}'] = {'n': int(sel.sum()), 'bias': round(float(e[sel].mean()), 3),
                                        'mean_abs_err': round(float(np.abs(e[sel]).mean()), 3)}
    return rep


def _fmt(name, rep):
    bands = ' '.join(f"[{k}] {v['bias']:+.2f}" for k, v in rep['bands'].items())
    return (f"  {name:12s} |err| {rep['mean_abs_err']:.3f}  bias {rep['bias']:+.3f}  "
            f"max {rep['max_abs_err']:.2f}  round= {rep['rounded_equal']:.1%}  | {bands}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--cache-meta', required=True, type=Path)
    ap.add_argument('--truth', required=True, nargs='+', type=Path)
    ap.add_argument('--source-anchor', type=int, default=2)
    ap.add_argument('--target-anchor', type=int, default=3)
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--bandwidths', type=float, nargs='+', default=[0.3, 0.45, 0.6, 0.8, 1.0, 1.3])
    ap.add_argument('--prior-weight', type=float, default=3.0)
    ap.add_argument('--cv-tolerance', type=float, default=0.02,
                    help='largest bandwidth whose CV mean |err| is within this fraction of the best')
    ap.add_argument('--out', type=Path, default=DEFAULT_CALIBRATION)
    args = ap.parse_args()

    groups, X = load_pairs(args.cache_meta.expanduser(), [p.expanduser() for p in args.truth],
                           args.source_anchor, args.target_anchor)
    if len(X) == 0:
        raise SystemExit('no players carry both a source- and a target-anchor label')
    truth = X[:, 2]
    print(f"{len(X)} paired players from {len(np.unique(groups))} games "
          f"(v{args.source_anchor} cache ↔ v{args.target_anchor} truth)")

    raw = error_report(X[:, 0], truth)
    print(_fmt('raw', raw))
    reps = {}
    for h in args.bandwidths:
        est, _ = cv_predict(X, groups, args.folds, source_anchor=args.source_anchor,
                            target_anchor=args.target_anchor, bandwidth=h,
                            prior_weight=args.prior_weight)
        reps[h] = error_report(est, truth)
        print(_fmt(f'h={h}', reps[h]))
    # The smoothest map whose CV error is within tolerance of the best: a narrow kernel
    # fits the sparse tails' sampling noise, and a wiggly map stretches some label
    # ranges apart and leaves rank classes empty in the remapped cache.
    best_err = min(r['mean_abs_err'] for r in reps.values())
    h = max(b for b, r in reps.items() if r['mean_abs_err'] <= best_err * (1 + args.cv_tolerance))
    cv_rep = reps[h]
    est, est_sd = cv_predict(X, groups, args.folds, source_anchor=args.source_anchor,
                             target_anchor=args.target_anchor, bandwidth=h,
                             prior_weight=args.prior_weight)
    sd_raw = float(np.abs(np.log(X[:, 1] / X[:, 3])).mean())
    sd_cal = float(np.abs(np.log(est_sd / X[:, 3])).mean())
    print(f"chosen bandwidth {h}; sd mean |log ratio| raw {sd_raw:.3f} → calibrated {sd_cal:.3f}")

    cal = AnchorCalibration.fit(
        X[:, 0], X[:, 1], X[:, 2], X[:, 3],
        source_anchor=args.source_anchor, target_anchor=args.target_anchor,
        bandwidth=h, prior_weight=args.prior_weight,
        meta={
            'fitted_at': datetime.now(timezone.utc).isoformat(),
            'truth_sources': [{'path': str(p), 'rows': len(_rows(p.expanduser()))}
                              for p in args.truth],
            'n_games': int(len(np.unique(groups))),
            'cv': {'folds': args.folds, 'grouped_by': 'game', 'raw': raw, 'calibrated': cv_rep,
                   'tolerance': args.cv_tolerance,
                   'by_bandwidth': {str(b): r['mean_abs_err'] for b, r in reps.items()},
                   'sd_mean_abs_log_ratio': {'raw': round(sd_raw, 4), 'calibrated': round(sd_cal, 4)}},
        })
    slope = np.gradient(cal.grid + cal.mean_shift, cal.grid)
    cal.meta['map_slope'] = [round(float(slope.min()), 3), round(float(slope.max()), 3)]
    print(f"map slope d(calibrated)/d(v2) in [{slope.min():.2f}, {slope.max():.2f}]")
    if slope.min() <= 0:
        raise SystemExit('calibration map is not monotone — refusing to write it')
    cal.save(args.out)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
