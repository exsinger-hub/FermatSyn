import os
import sys

import pytest
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.frequency_loss import (  # noqa: E402
    FrequencyLoss,
    focal_frequency_loss,
    frequency_mode_weights,
    rapsd_profile_loss,
)
from train import _calibration_summary_from_sums  # noqa: E402


def _image_pair():
    target = torch.zeros((2, 1, 16, 16))
    pred = target.clone()
    pred[:, :, 4:12, 4:12] = 0.25
    return pred, target


def test_default_frequency_mode_leaves_generator_loss_unchanged():
    base_generator_loss = torch.tensor(3.25)
    lambda_ffl, lambda_rapsd = frequency_mode_weights("none", gamma=9.0)
    extra_frequency_loss = lambda_ffl * torch.tensor(11.0) + lambda_rapsd * torch.tensor(13.0)

    assert (base_generator_loss + extra_frequency_loss).item() == pytest.approx(base_generator_loss.item())


@pytest.mark.parametrize(
    "mode,gamma,expected_weights",
    [
        ("none", 2.5, (0.0, 0.0)),
        ("ffl", 2.5, (1.0, 0.0)),
        ("rapsd", 2.5, (0.0, 1.0)),
        ("full", 2.5, (1.0, 2.5)),
    ],
)
def test_frequency_modes_select_expected_ffl_and_rapsd_terms(mode, gamma, expected_weights):
    pred, target = _image_pair()
    lambda_ffl, lambda_rapsd = frequency_mode_weights(mode, gamma=gamma)
    criterion = FrequencyLoss(
        lambda_ffl=lambda_ffl,
        lambda_rapsd=lambda_rapsd,
        use_mask=False,
        use_window=False,
        num_rapsd_bins=8,
    )

    loss, metrics = criterion(pred, target, return_metrics=True)
    expected = (
        expected_weights[0] * focal_frequency_loss(pred, target, use_mask=False, use_window=False)
        + expected_weights[1] * rapsd_profile_loss(pred, target, use_mask=False, use_window=False, num_bins=8)
    )

    assert (lambda_ffl, lambda_rapsd) == expected_weights
    assert loss.item() == pytest.approx(expected.item())
    assert metrics["loss_frequency"].item() == pytest.approx(expected.item())


def test_frequency_calibration_summary_formula():
    summary = _calibration_summary_from_sums(
        l1_sum=40.0,
        ssim_sum=8.0,
        ffl_sum=4.0,
        rapsd_sum=2.0,
        sample_count=2,
    )

    assert summary["mean_reconstruction_contribution"] == pytest.approx(24.0)
    assert summary["mean_ffl"] == pytest.approx(2.0)
    assert summary["mean_rapsd"] == pytest.approx(1.0)
    assert summary["gamma"] == pytest.approx(2.0)
    assert summary["mean_frequency_ffl"] == pytest.approx(2.0)
    assert summary["mean_frequency_rapsd"] == pytest.approx(1.0)
    assert summary["mean_frequency_full"] == pytest.approx(4.0)
    assert summary["lambda_freq_suggestions_by_mode"]["ffl"]["kappa_0.1"] == pytest.approx(1.2)
    assert summary["lambda_freq_suggestions_by_mode"]["rapsd"]["kappa_0.1"] == pytest.approx(2.4)
    assert summary["lambda_freq_suggestions_by_mode"]["full"]["kappa_0.1"] == pytest.approx(0.6)
    assert summary["lambda_freq_suggestions"]["kappa_0.1"] == pytest.approx(0.6)
