import torch
import torch.nn as nn

from models.a2_spatial import SpatialAxialDifferenceAdapter
from models.a2_v2 import anatomy_delta_loss, forward_window


class _DummyGenerator(nn.Module):
    def extract_a2_cin_features(self, image):
        # Keep a batch-sensitive term so folding time into batch would change
        # the result, matching the identity regression covered by the real path.
        feature = image + image.mean(dim=0, keepdim=True)
        return {
            "descriptor": feature.mean(dim=(-2, -1)),
            "skip_fusion2": feature,
        }

    def build_a2_center_features(self, cin_features, indices):
        return {
            "skip_fusion2": cin_features["skip_fusion2"].index_select(0, indices)
        }

    def decode_a2_features(self, features):
        return features["skip_fusion2"]

    def forward(self, image):
        cin = self.extract_a2_cin_features(image)
        indices = torch.arange(image.shape[0], device=image.device)
        return self.decode_a2_features(self.build_a2_center_features(cin, indices))


def test_perfect_prediction_has_zero_anatomy_delta_loss():
    target = torch.linspace(-0.5, 0.9, 3 * 4 * 4).reshape(1, 3, 1, 4, 4)
    loss = anatomy_delta_loss(target.clone(), target)
    assert loss.item() == 0.0


def test_wrong_first_difference_has_positive_loss():
    target = torch.zeros(1, 3, 1, 4, 4)
    prediction = target.clone()
    prediction[:, 2] = 0.25
    assert anatomy_delta_loss(prediction, target).item() > 0.0


def test_empty_anatomy_mask_fallback_stays_finite():
    target = torch.full((1, 3, 1, 4, 4), -1.0)
    prediction = target.clone()
    prediction[:, 1] = -0.8
    loss = anatomy_delta_loss(prediction, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_joint_helper_and_zero_adapter_match_three_direct_slices():
    generator = _DummyGenerator().eval()
    adapter = SpatialAxialDifferenceAdapter(feature_dim=1, rank=1).eval()
    source = torch.randn(2, 5, 1, 4, 4)
    prediction, baseline, residual = forward_window(
        generator,
        adapter,
        source,
        input_mode="neighbor_difference",
        amp_enabled=False,
        need_baseline=True,
    )
    direct = torch.stack([generator(source[:, index]) for index in (1, 2, 3)], dim=1)
    assert prediction.shape == (2, 3, 1, 4, 4)
    assert torch.count_nonzero(residual) == 0
    assert torch.equal(baseline, direct)
    assert torch.equal(prediction, direct)
