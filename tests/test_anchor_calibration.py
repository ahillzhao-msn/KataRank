"""HumanSL anchor calibration: fit/apply/persistence and how KAB2Dataset resolves labels."""
import csv

import numpy as np

from katarank.data.anchor_calibration import AnchorCalibration, effective_rank_idx
from katarank.data.datasets.dataset_kab2 import KAB2Dataset


def _bump(x):
    return 1.2 * np.exp(-0.5 * ((x - 13.0) / 1.0) ** 2)


def _synthetic(n=4000, seed=0):
    """v2-like labels: around 13 they read up to 1.2 ranks low, sd 20% too wide."""
    rng = np.random.default_rng(seed)
    src = rng.uniform(4, 26, n)
    tgt = src + _bump(src) + rng.normal(0, 0.3, n)
    tgt_sd = rng.uniform(0.8, 1.5, n)
    return src, tgt_sd * 1.2, tgt, tgt_sd


def test_fit_recovers_band_bias():
    src, src_sd, tgt, tgt_sd = _synthetic()
    cal = AnchorCalibration.fit(src, src_sd, tgt, tgt_sd, bandwidth=0.4)
    mean, sd = cal.apply(src, src_sd)
    inside = (src >= 12) & (src < 14)
    assert (tgt - src)[inside].mean() > 1.0                  # the bias being corrected
    assert abs((mean - tgt)[inside].mean()) < 0.1
    assert abs((mean - tgt)[(src > 18) & (src < 24)].mean()) < 0.05
    assert abs(np.log(sd / tgt_sd).mean()) < 0.05            # raw: log 1.2 ≈ 0.18


def test_sparse_tails_shrink_to_no_correction():
    src = np.full(50, 15.0)
    cal = AnchorCalibration.fit(src, np.ones(50), src - 1.0, np.ones(50), bandwidth=0.5)
    assert abs(cal.apply(2.0)[0] - 2.0) < 1e-6               # no data near 2: unchanged
    assert abs(cal.apply(15.0)[0] - 14.0) < 0.1


def test_apply_clips_to_rank_range():
    cal = AnchorCalibration.fit(np.full(100, 27.8), np.ones(100), np.full(100, 30.0),
                                np.ones(100), bandwidth=0.5, prior_weight=0.0)
    assert cal.apply(27.8)[0] == 28.0


def test_save_load_roundtrip(tmp_path):
    src, src_sd, tgt, tgt_sd = _synthetic(500)
    cal = AnchorCalibration.fit(src, src_sd, tgt, tgt_sd, meta={'note': '测试'})
    p = tmp_path / 'cal.json'
    cal.save(p)
    back = AnchorCalibration.load(p)
    assert back.source_anchor == 2 and back.target_anchor == 3 and back.meta['note'] == '测试'
    x = np.linspace(0, 28, 57)
    assert np.allclose(back.apply(x)[0], cal.apply(x)[0], atol=1e-4)


def _shift_cal(shift=-1.0):
    grid = np.arange(0.0, 28.01, 0.1)
    return AnchorCalibration(2, 3, grid, np.full(len(grid), shift), np.zeros(len(grid)))


def test_effective_rank_only_for_source_anchor():
    cal = _shift_cal()
    row = {'humanAnchorVer': '2', 'B_humanRankExp': '14.4'}
    assert effective_rank_idx(row, 'B', cal) == 13
    assert effective_rank_idx({**row, 'humanAnchorVer': '3'}, 'B', cal) is None
    assert effective_rank_idx({**row, 'humanAnchorVer': '0'}, 'B', cal) is None
    assert effective_rank_idx({**row, 'B_humanRankExp': ''}, 'B', cal) is None
    assert effective_rank_idx(row, 'B', None) is None


def _write_meta(path, rows):
    cols = ['file', 'set', 'humanAnchorVer', 'B_humanRank', 'W_humanRank', 'B_humanLogPrior',
            'W_humanLogPrior', 'B_humanRankExp', 'W_humanRankExp', 'B_logPrior', 'W_logPrior']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})


def test_dataset_applies_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(KAB2Dataset, '_detect_input_dim', lambda self: 10)
    _write_meta(tmp_path / '_meta.csv', [
        # v2 row: calibrated from its posterior mean (14.4 − 1 → 13.4 → 13 = 7k)
        dict(file='g2', set='T', humanAnchorVer='2', B_humanRank='rank_6k', W_humanRank='rank_1d',
             B_humanLogPrior='-1.5', W_humanLogPrior='-1.5', B_humanRankExp='14.4',
             W_humanRankExp='20.2'),
        # v3 row: kept as written
        dict(file='g3', set='T', humanAnchorVer='3', B_humanRank='rank_6k', W_humanRank='rank_1d',
             B_humanLogPrior='-1.5', W_humanLogPrior='-1.5', B_humanRankExp='14.4',
             W_humanRankExp='20.2'),
        # v2 row whose HumanSL pass did not run for White: stays unlabelled
        dict(file='g4', set='T', humanAnchorVer='2', B_humanRank='rank_6k', W_humanRank='',
             B_humanLogPrior='-1.5', W_humanLogPrior='0', B_humanRankExp='14.4'),
    ])
    p = tmp_path / 'cal.json'
    _shift_cal().save(p)

    ds = KAB2Dataset(str(tmp_path), split='T', anchor_calibration=p)
    by_id = {g['id']: g for g in ds.games}
    assert (by_id['g2']['rank_b'], by_id['g2']['rank_w']) == (13, 19)
    assert (by_id['g3']['rank_b'], by_id['g3']['rank_w']) == (14, 20)
    assert (by_id['g4']['rank_b'], by_id['g4']['rank_w']) == (13, -1)
    assert ds.calibrated_labels == 3
    info = ds.anchor_calibration_info()
    assert info['source_anchor'] == 2 and info['calibrated_labels'] == 3 and len(info['sha256']) == 64

    raw = KAB2Dataset(str(tmp_path), split='T', anchor_calibration=None)
    assert {g['id']: g['rank_b'] for g in raw.games} == {'g2': 14, 'g3': 14, 'g4': 14}
    assert raw.anchor_calibration_info() is None
