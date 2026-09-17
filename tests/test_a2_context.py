import torch

from models.a2_context import (
    InterSliceContextModulator,
    apply_channel_affine,
    intervene_descriptors,
)


def test_zero_initialized_modulator_is_identity():
    module = InterSliceContextModulator(
        descriptor_dim=8, feature_dim=6, hidden_dim=5, mode="bigru"
    )
    descriptors = torch.randn(2, 4, 8)
    features = torch.randn(2, 4, 6, 3, 3)
    gamma, beta = module(descriptors)
    assert torch.count_nonzero(gamma) == 0
    assert torch.count_nonzero(beta) == 0
    refined = apply_channel_affine(features, gamma, beta)
    assert torch.equal(refined, features)


def test_affine_head_receives_gradient_at_identity_initialization():
    module = InterSliceContextModulator(
        descriptor_dim=8, feature_dim=6, hidden_dim=5, mode="gru"
    )
    descriptors = torch.randn(2, 4, 8)
    gamma, beta = module(descriptors)
    loss = gamma.square().mean() + (beta - 0.1).square().mean()
    loss.backward()
    assert module.affine[-1].bias.grad.abs().sum().item() > 0


def test_independent_mode_shapes():
    module = InterSliceContextModulator(
        descriptor_dim=7, feature_dim=9, hidden_dim=4, mode="independent"
    )
    gamma, beta = module(torch.randn(3, 5, 7))
    assert gamma.shape == (3, 5, 9)
    assert beta.shape == (3, 5, 9)


def test_descriptor_interventions_preserve_or_replace_the_center():
    descriptors = torch.arange(5.0).reshape(1, 5, 1)
    assert torch.equal(intervene_descriptors(descriptors, "ordered"), descriptors)
    repeated = intervene_descriptors(descriptors, "center_repeat")
    assert torch.equal(repeated, torch.full_like(descriptors, 2.0))
    reversed_descriptors = intervene_descriptors(descriptors, "reverse")
    assert torch.equal(reversed_descriptors, torch.flip(descriptors, dims=(1,)))
    assert reversed_descriptors[0, 2, 0].item() == descriptors[0, 2, 0].item()
