# HumanSL anchor calibration

KataRank's rank head is trained against a HumanSL rank label per player, written into each
KAB2 file by `katago batch_analysis -human-model` (katago-fork). The procedure that produces
the label is versioned as `humanAnchorVer`. When it changes, the cache holds labels from more
than one procedure. Calibration maps the older labels onto the current scale at load time, so
the cache trains on one consistent target without re-running HumanSL over it.

## Anchor versions

| `humanAnchorVer` | Procedure | Against a full 29-profile scan of the same moves |
|---|---|---|
| 0 / 1 | legacy 3-profile window around a logPrior formula | label ≈ formula; not an anchor |
| 2 | 5 fixed profiles (20k/13k/6k/2d/9d) + natural spline | mean \|err\| 0.65, rounded label equal 46%, 8k–5k **−1.25** |
| 3 | 7 coarse profiles + ±2 refine + natural spline | mean \|err\| 0.02, rounded label equal 98% |

v2's split-half stability was close to the full scan's. It is biased, though, and stability
does not measure bias. Seven ranks between knots is too far for the spline to follow the
likelihood curve. Measured on 600 players (300 random V-set games), katago-fork PR #4.

## How it is applied

- Artifact: `nets/anchor_calibration/v2_to_v3.json`: `mean_shift` and `log_sd_ratio` on a
  0–28 grid of the v2 posterior mean, plus fit provenance and the held-out validation report
  (`meta`).
- `KAB2Dataset(anchor_calibration='default')` loads it when present. A row whose
  `humanAnchorVer` equals the artifact's `source_anchor` trains on
  `round(v2 mean + mean_shift(v2 mean))`. Every other row keeps its own `humanRank`.
  Training, the train-session gate (`evaluate_checkpoint`) and distillation all build their
  data through `KAB2Dataset`, so they see the same labels.
- `_meta.csv` and the KAB2 files are never rewritten. Each row keeps its raw label and its
  anchor version, so the mapping can be refit or dropped.
- `training_report.json` → `data.anchor_calibration` records the artifact path, its sha256
  and how many labels it remapped (`null` when none was applied).
- Override per run with `anchor_calibration:` in the training config (a path, or `null` to
  train on the raw labels).

## Refitting

The truth labels come from batch_analysis runs of the current anchor over cached games:

```bash
.venv/Scripts/python.exe scripts/fit_anchor_calibration.py \
    --cache-meta ~/.katarank/kab2_cache/_meta.csv \
    --truth nets/anchor_calibration/truth/*.csv
```

The script chooses the kernel bandwidth by grouped (per-game) 10-fold CV: the largest
bandwidth within `--cv-tolerance` (2%) of the best error, so the map stays smooth. It prints
raw vs calibrated held-out error per rank band and writes the artifact with the map's slope
range. It refuses to write a map that is not monotone. Refit whenever more
target-anchor labels exist, for example once new games are labelled by v3 and later relabelled
from the cache. The truth CSVs it was fitted on are kept under `nets/anchor_calibration/truth/`.

## Limits

- The mapping uses only the v2 posterior mean, so it corrects systematic bias, not each
  player's individual error. Held-out mean |err| is about 0.3 ranks, against a per-label
  noise floor of about 1.15 ranks (posterior sd from about 90 voting moves). Adding v2 sd,
  humanLogPrior or main-model per-player statistics did not improve it (gradient boosting,
  same CV).
- The sparse tails (20k–12k, 7d–9d) shrink toward no correction (`prior_weight`) instead of
  extrapolating.
- Binned by the **true** label, the calibrated labels read about +0.3 in 20k–9k. This is
  regression to the mean, which any conditional-mean estimate shows; it is not a residual
  bias of the map. Binned by v2 label or by calibrated label, every band is within ±0.04, and
  the population spread is preserved: sd 4.00 ranks for the truth, 3.98 calibrated, 4.39 raw v2.

## First fit (2026-10-18)

2600 players / 1300 games: `v3_random300_20261017.csv` (random V-set games) and
`v3_strat1000_20261018.csv` (500 random, plus 500 games with a v2 label below 16 or above 24).
Grouped 10-fold CV. Bandwidth 0.6: the largest within 2% of the best CV error (0.3 scored
0.318 but its map's slope ranged 0.15–2.73, which stretched v2 ≈ 9.5 apart and emptied 10k in
the remapped cache). The chosen map's slope is 0.52–1.59.

| | mean \|err\| | bias | rounded label equal | 8k–5k bias | sd \|log ratio\| |
|---|---|---|---|---|---|
| raw v2 | 0.701 | −0.344 | 44.0% | −1.31 | 0.191 |
| calibrated | 0.321 | −0.004 | 69.5% | −0.02 | 0.099 |

On the 1600 players of the random-sample games, the label distribution moves onto the v3 one:
mean / sd 18.22 / 3.66 (v3), 18.02 / 4.04 (raw v2), 18.22 / 3.64 (calibrated). v3 peaks at
3k–2k; v2 had spread 8k–5k players downward.
