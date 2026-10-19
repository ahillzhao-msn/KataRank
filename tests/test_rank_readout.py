"""The inference readout: calibrated posterior mean, not the head's argmax."""

from __future__ import annotations

import numpy as np
import torch

from katarank.data.anchor_calibration import load_default_calibration
from katarank.workflow import rank_readout, result_to_output


def _one_hot(center: int, n: int = 29) -> np.ndarray:
    p = np.zeros(n)
    p[center] = 1.0
    return p


def test_readout_is_the_calibrated_posterior_mean():
    cal = load_default_calibration()
    assert cal is not None
    exp_rank, idx, top = rank_readout(_one_hot(14), cal)
    assert top == 14
    assert idx == 15
    assert abs(exp_rank - 15.3408) < 1e-3


def test_readout_without_calibration_is_the_plain_mean():
    exp_rank, idx, top = rank_readout(_one_hot(14), None)
    assert (round(exp_rank, 6), idx, top) == (14.0, 14, 14)


def test_readout_handles_missing_probs():
    assert rank_readout(None) == (None, -1, -1)
    assert rank_readout(np.zeros(29)) == (None, 0, 0)


def test_result_to_output_prefers_the_calibrated_rank():
    result = {
        'game_id': 'g', 'b_rating': 0.0, 'w_rating': 0.0,
        'b_rank_probs': torch.tensor(_one_hot(14)),
        'w_rank_probs': None,
        'b_rank_top': 14, 'w_rank_top': -1,
        'b_rank': 15, 'w_rank': -1,
        'b_rank_exp': 15.3408, 'w_rank_exp': None,
    }
    out = result_to_output(result)
    assert out['b_rank'] == 15
    assert abs(out['b_rank_exp'] - 15.3408) < 1e-4
    assert out['w_rank'] == -1


def test_output_calibration_depends_on_the_models_label_scale(tmp_path):
    from katarank.model import KataRankModel
    from katarank.workflow import output_calibration

    cal = load_default_calibration()
    model = KataRankModel(input_dim=10, hidden_dim=16, num_heads=2, num_inducing=4)
    assert model.rank_label_anchor is None
    assert output_calibration(model, cal) is cal            # pre-field checkpoint: v2 labels
    model.rank_label_anchor = cal.source_anchor
    assert output_calibration(model, cal) is cal
    model.rank_label_anchor = cal.target_anchor
    assert output_calibration(model, cal) is None           # already on the v3 scale
    assert output_calibration(model, None) is None

    p = tmp_path / 'm.pt'
    model.save(str(p))
    assert KataRankModel.load(str(p)).rank_label_anchor == cal.target_anchor
