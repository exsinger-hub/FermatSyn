import pytest
import torch
import torch.nn as nn

from evaluate_frozen_3tap import (
    ALPHAS,
    axial_guard,
    frozen_center_triplet,
    register_unique_center,
    summarize_controls,
    three_tap_output,
)


def test_alpha_grid_is_fixed_and_alpha_zero_is_exact_identity():
    assert ALPHAS == (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
    left = torch.randn(2, 1, 3, 3)
    center = torch.randn(2, 1, 3, 3)
    right = torch.randn(2, 1, 3, 3)
    assert torch.equal(three_tap_output(left, center, right, 0.0), center)


def test_three_tap_matches_hand_calculation_and_preserves_linear_sequence():
    left = torch.tensor([[[[0.0]]]])
    center = torch.tensor([[[[2.0]]]])
    right = torch.tensor([[[[0.0]]]])
    assert torch.equal(
        three_tap_output(left, center, right, 0.20),
        torch.tensor([[[[1.6]]]]),
    )
    right_linear = torch.tensor([[[[4.0]]]])
    assert torch.equal(three_tap_output(left, center, right_linear, 0.40), center)


class _RecordingGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image):
        self.calls.append(image.detach().clone())
        return image + 10.0


def test_frozen_triplet_calls_generator_once_per_time_without_batch_folding():
    generator = _RecordingGenerator()
    source = torch.arange(5.0).reshape(1, 5, 1, 1, 1)
    outputs = frozen_center_triplet(generator, source)
    assert len(generator.calls) == 3
    assert all(call.shape == (1, 1, 1, 1) for call in generator.calls)
    assert [float(call.item()) for call in generator.calls] == [1.0, 2.0, 3.0]
    assert [float(output.item()) for output in outputs] == [11.0, 12.0, 13.0]


def test_unique_center_rejects_duplicate_patient_slice_pair():
    seen = set()
    assert register_unique_center(seen, "P001", 7) == ("P001", 7)
    register_unique_center(seen, "P002", 7)
    with pytest.raises(RuntimeError, match="duplicate validation centre"):
        register_unique_center(seen, "P001", 7)


def test_axial_guard_applies_psnr_ssim_and_l1_thresholds():
    baseline = {"a2_psnr": 26.0, "a2_ssim": 0.900, "a2_l1": 0.040}
    passing = {"a2_psnr": 25.90, "a2_ssim": 0.899, "a2_l1": 0.0404}
    assert axial_guard(passing, baseline)["pass"] is True

    failing = dict(passing)
    failing["a2_psnr"] = 25.89
    result = axial_guard(failing, baseline)
    assert result["pass"] is False
    assert result["checks"]["psnr"]["pass"] is False


class _MetricBank:
    def __init__(self, primary):
        self.primary = primary

    def summarize(self):
        return {
            "a2_delta_l1_anatomy": self.primary,
            "a2_psnr": 26.0,
            "a2_ssim": 0.900,
            "a2_l1": 0.040,
        }


def test_summary_selects_best_guarded_alpha_from_fixed_grid():
    primary = {
        0.0: 0.0320,
        0.05: 0.0315,
        0.10: 0.0310,
        0.20: 0.0312,
        0.30: 0.0317,
        0.40: 0.0325,
    }
    summary = summarize_controls(
        {alpha: _MetricBank(primary[alpha]) for alpha in ALPHAS}
    )
    assert summary["selection"]["best_guard_pass_alpha"] == 0.10
    assert summary["alpha_results"]["0.10"][
        "primary_change_vs_alpha_zero"
    ]["relative_reduction_percent"] == pytest.approx(3.125)
