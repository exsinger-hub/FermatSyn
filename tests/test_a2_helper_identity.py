import torch
import torch.nn as nn

from models.a2_context import extract_ordered_frozen_features


class _BatchSensitiveGenerator(nn.Module):
    """Small stand-in whose features change when time is folded into batch."""

    def extract_a2_cin_features(self, image):
        batch_context = image.mean(dim=0, keepdim=True)
        feature = image + batch_context
        return {
            "descriptor": feature.mean(dim=(-2, -1)),
            "skip": feature,
        }

    def build_a2_center_features(self, cin_features, indices):
        return {"skip": cin_features["skip"].index_select(0, indices)}

    def decode_a2_features(self, features):
        return features["skip"]

    def forward(self, image):
        cin = self.extract_a2_cin_features(image)
        indices = torch.arange(image.shape[0], device=image.device)
        features = self.build_a2_center_features(cin, indices)
        return self.decode_a2_features(features)


def test_frozen_helper_preserves_center_generator_path_without_time_batch_folding():
    generator = _BatchSensitiveGenerator().eval()
    source = torch.tensor([[[[[1.0]]], [[[3.0]]], [[[8.0]]]]])

    descriptors, features = extract_ordered_frozen_features(
        generator, source, center=1
    )
    helper = generator.decode_a2_features(features)
    direct = generator(source[:, 1])

    assert descriptors.shape == (1, 3, 1)
    assert torch.equal(descriptors[0, :, 0], torch.tensor([2.0, 6.0, 16.0]))
    assert torch.equal(helper, direct)
