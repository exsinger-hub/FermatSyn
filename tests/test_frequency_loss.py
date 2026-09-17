import os
import sys

import pytest
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.frequency_loss import (  # noqa: E402
    FrequencyLoss,
    anatomical_mask,
    focal_frequency_loss,
    hann_window2d,
    radial_average_power_spectral_density,
    rapsd_errors,
)


def _rand_synthrad(shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=generator) * 2.0 - 1.0


def test_same_image_has_near_zero_loss_and_metrics():
    target = _rand_synthrad((2, 3, 32, 40), seed=1)
    pred = target.clone().requires_grad_(True)
    criterion = FrequencyLoss(num_rapsd_bins=64)

    loss, metrics = criterion(pred, target, return_metrics=True)

    assert loss.item() == pytest.approx(0.0, abs=1.0e-7)
    assert metrics["loss_ffl"].item() == pytest.approx(0.0, abs=1.0e-7)
    assert metrics["loss_rapsd"].item() == pytest.approx(0.0, abs=1.0e-7)
    assert metrics["rapsd_full_error"].item() == pytest.approx(0.0, abs=1.0e-7)
    assert metrics["rapsd_high_error"].item() == pytest.approx(0.0, abs=1.0e-7)


def test_perturbation_has_positive_loss_and_finite_gradients():
    target = _rand_synthrad((2, 2, 32, 32), seed=2)
    pred = (target + 0.05 * _rand_synthrad(target.shape, seed=3)).clone()
    pred.requires_grad_(True)
    criterion = FrequencyLoss(num_rapsd_bins=64)

    loss = criterion(pred, target)
    loss.backward()

    assert loss.item() > 0.0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0


def test_anatomical_mask_closing_fills_small_holes():
    image = torch.full((1, 1, 24, 24), -1.0)
    image[..., 6:18, 6:18] = 0.0
    image[..., 11:13, 11:13] = -1.0

    mask = anatomical_mask(image, threshold=-0.95, closing_kernel_size=5)

    assert mask.shape == image.shape
    assert mask[..., 12, 12].item() == pytest.approx(1.0)
    assert mask[..., 0, 0].item() == pytest.approx(0.0)


def test_mask_suppresses_outside_anatomy_perturbation():
    target = torch.full((1, 1, 32, 32), -1.0)
    target[..., 10:22, 10:22] = 0.0
    pred = target.clone()
    pred[..., 0:8, 0:8] = 1.0

    masked = focal_frequency_loss(
        pred,
        target,
        use_mask=True,
        use_window=False,
        anatomy_threshold=-0.95,
        closing_kernel_size=3,
    )
    unmasked = focal_frequency_loss(
        pred,
        target,
        use_mask=False,
        use_window=False,
    )

    assert masked.item() == pytest.approx(0.0, abs=1.0e-7)
    assert unmasked.item() > 0.0


def test_hann_window_and_rapsd_shapes_are_stable():
    image = _rand_synthrad((2, 3, 19, 23), seed=4)
    window = hann_window2d(19, 23, device=image.device, dtype=image.dtype)
    profile = radial_average_power_spectral_density(image * window, num_bins=64)

    assert window.shape == (1, 1, 19, 23)
    assert profile.shape == (2, 3, 64)
    assert torch.isfinite(profile).all()


def test_window_and_mask_ablation_combinations_autograd():
    target = _rand_synthrad((2, 2, 24, 20), seed=5)
    base_pred = target + 0.03 * _rand_synthrad(target.shape, seed=6)

    for use_mask in (False, True):
        for use_window in (False, True):
            pred = base_pred.clone().requires_grad_(True)
            criterion = FrequencyLoss(
                use_mask=use_mask,
                use_window=use_window,
                num_rapsd_bins=64,
            )
            loss = criterion(pred, target)
            loss.backward()

            assert torch.isfinite(loss)
            assert pred.grad is not None
            assert torch.isfinite(pred.grad).all()


def test_high_frequency_rapsd_metric_detects_checkerboard_noise():
    target = torch.zeros((1, 1, 32, 32))
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    checker = ((xx + yy) % 2).float() * 2.0 - 1.0
    pred = target + 0.1 * checker.view(1, 1, 32, 32)

    metrics = rapsd_errors(
        pred,
        target,
        use_mask=False,
        use_window=True,
        num_bins=64,
        high_freq_fraction=0.5,
    )

    assert metrics["rapsd_full_error"].item() > 0.0
    assert metrics["rapsd_high_error"].item() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_frequency_loss_cuda_smoke():
    target = _rand_synthrad((1, 1, 16, 16), seed=7).cuda()
    pred = (target + 0.01).clone().detach().requires_grad_(True)
    criterion = FrequencyLoss(num_rapsd_bins=64).cuda()

    loss = criterion(pred, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
