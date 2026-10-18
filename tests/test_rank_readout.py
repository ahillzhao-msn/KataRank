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
