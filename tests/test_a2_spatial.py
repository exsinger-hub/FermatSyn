import torch

from models.a2_spatial import SpatialAxialDifferenceAdapter, channel_layer_norm


def test_channel_layer_norm_is_per_spatial_position():
    features = torch.randn(2, 7, 4, 5)
    normalized = channel_layer_norm(features)
    assert normalized.shape == features.shape
    assert torch.allclose(normalized.mean(dim=1), torch.zeros(2, 4, 5), atol=1e-5)


def test_spatial_adapter_is_exact_identity_at_initialization():
    module = SpatialAxialDifferenceAdapter(feature_dim=8, rank=4)
    ordered = torch.randn(2, 5, 8, 6, 6)
    refined, residual = module(
        ordered, output_indices=(1, 2, 3), input_mode="neighbor_difference"
    )
    assert torch.count_nonzero(residual) == 0
    assert torch.equal(refined, ordered[:, 1:4])


def test_center_only_is_independent_of_neighbour_values():
    module = SpatialAxialDifferenceAdapter(feature_dim=8, rank=4)
    torch.nn.init.normal_(module.output_projection.weight)
    first = torch.randn(2, 5, 8, 6, 6)
    second = first.clone()
    second[:, (0, 1, 3, 4)] = torch.randn_like(second[:, (0, 1, 3, 4)])
    refined_first, _ = module(first, output_indices=(2,), input_mode="center_only")
    refined_second, _ = module(second, output_indices=(2,), input_mode="center_only")
    assert torch.equal(refined_first, refined_second)


def test_ordered_context_changes_when_neighbours_change():
    module = SpatialAxialDifferenceAdapter(feature_dim=8, rank=4)
    torch.nn.init.normal_(module.output_projection.weight)
    first = torch.randn(2, 5, 8, 6, 6)
    second = first.clone()
    second[:, 1] = second[:, 1] + 0.5 * torch.randn_like(second[:, 1])
    refined_first, _ = module(
        first, output_indices=(2,), input_mode="neighbor_difference"
    )
    refined_second, _ = module(
        second, output_indices=(2,), input_mode="neighbor_difference"
    )
    assert not torch.equal(refined_first, refined_second)


def test_center_replacement_has_exactly_zero_residual_without_biases():
    module = SpatialAxialDifferenceAdapter(feature_dim=8, rank=4)
    torch.nn.init.normal_(module.output_projection.weight)
    ordered = torch.randn(2, 5, 8, 6, 6)
    refined, residual = module(
        ordered, output_indices=(2,), input_mode="center_replace"
    )
    assert torch.count_nonzero(residual) == 0
    assert torch.equal(refined[:, 0], ordered[:, 2])


def test_center_and_context_arms_have_identical_parameter_count():
    center = SpatialAxialDifferenceAdapter(feature_dim=8, rank=4)
    context = SpatialAxialDifferenceAdapter(feature_dim=8, rank=4)
    center_count = sum(parameter.numel() for parameter in center.parameters())
    context_count = sum(parameter.numel() for parameter in context.parameters())
    assert center_count == context_count


def test_zero_initialized_output_projection_receives_gradient():
    module = SpatialAxialDifferenceAdapter(feature_dim=8, rank=4)
    ordered = torch.randn(2, 5, 8, 6, 6)
    refined, _ = module(
        ordered, output_indices=(2,), input_mode="neighbor_difference"
    )
    loss = (refined - 0.1).square().mean()
    loss.backward()
    gradient = module.output_projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0
