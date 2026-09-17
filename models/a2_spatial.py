"""Spatial axial-difference refinement for the A2-v2 experiment.

The original A2 pilot globally pooled every slice and predicted one affine
value per decoder channel.  This module keeps the CIN spatial map, represents
ordered context as signed neighbouring-slice differences, and predicts a
zero-initialized residual in the same feature space.  The exact same module is
used for neighbour-difference and centre-only controls; only its two input maps
differ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_INPUT_MODES = {
    "center_only",
    "neighbor_difference",
    "center_replace",
    "swap",
    "outer_neighbors",
}


def channel_layer_norm(features, eps=1e-5):
    """Layer-normalize channels independently at every spatial position."""

    if features.ndim != 4:
        raise ValueError("features must have shape B x C x H x W")
    normalized = F.layer_norm(
        features.permute(0, 2, 3, 1),
        (features.shape[1],),
        eps=float(eps),
    )
    return normalized.permute(0, 3, 1, 2).contiguous()


class SpatialAxialDifferenceAdapter(nn.Module):
    """Predict a spatial CIN residual from a parameter-matched two-map input."""

    def __init__(self, feature_dim=288, rank=96):
        super().__init__()
        if feature_dim < 1 or rank < 1:
            raise ValueError("feature_dim and rank must be positive")
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)

        # Both the centre-only and neighbour-difference arms use this exact
        # parameterization.  Biases are disabled so a centre-replacement
        # intervention (zero signed differences) collapses exactly to Frozen G.
        self.input_projection = nn.Conv2d(
            2 * self.feature_dim, self.rank, kernel_size=1, bias=False
        )
        self.mixer = nn.Sequential(
            nn.Conv2d(
                self.rank,
                self.rank,
                kernel_size=3,
                padding=1,
                groups=self.rank,
                bias=False,
            ),
            nn.SiLU(),
        )
        self.output_projection = nn.Conv2d(
            self.rank, self.feature_dim, kernel_size=1, bias=False
        )
        self.reset_identity()

    def reset_identity(self):
        """Make adapter insertion an exact identity transformation."""

        nn.init.zeros_(self.output_projection.weight)

    def _adapter_input(self, ordered, time_index, input_mode):
        if input_mode not in SUPPORTED_INPUT_MODES:
            raise ValueError(
                "input mode must be one of %s" % sorted(SUPPORTED_INPUT_MODES)
            )
        center = channel_layer_norm(ordered[:, time_index])
        if input_mode == "center_only":
            return torch.cat([center, center], dim=1)
        if input_mode == "center_replace":
            return torch.zeros(
                center.shape[0],
                2 * center.shape[1],
                center.shape[2],
                center.shape[3],
                device=center.device,
                dtype=center.dtype,
            )

        distance = 2 if input_mode == "outer_neighbors" else 1
        if time_index - distance < 0 or time_index + distance >= ordered.shape[1]:
            raise ValueError("requested neighbours fall outside the ordered window")
        left = channel_layer_norm(ordered[:, time_index - distance]) - center
        right = channel_layer_norm(ordered[:, time_index + distance]) - center
        if input_mode == "swap":
            left, right = right, left
        return torch.cat([left, right], dim=1)

    def refine_one(self, ordered, time_index, input_mode="neighbor_difference"):
        """Refine one non-boundary slice from ``B x T x C x H x W`` maps."""

        if ordered.ndim != 5:
            raise ValueError("ordered features must have shape B x T x C x H x W")
        if ordered.shape[2] != self.feature_dim:
            raise ValueError(
                "feature channel mismatch: expected %d, got %d"
                % (self.feature_dim, ordered.shape[2])
            )
        if time_index <= 0 or time_index >= ordered.shape[1] - 1:
            raise ValueError("time_index must have both an immediate left and right neighbour")

        center = ordered[:, time_index]
        adapter_input = self._adapter_input(ordered, time_index, input_mode)
        mixed = self.mixer(self.input_projection(adapter_input))
        residual = self.output_projection(mixed)
        return center + residual, residual

    def forward(self, ordered, output_indices=None, input_mode="neighbor_difference"):
        """Refine requested interior slices and stack them on a time dimension."""

        if ordered.ndim != 5:
            raise ValueError("ordered features must have shape B x T x C x H x W")
        if output_indices is None:
            output_indices = range(1, ordered.shape[1] - 1)
        refined, residuals = [], []
        for time_index in output_indices:
            value, residual = self.refine_one(
                ordered, int(time_index), input_mode=input_mode
            )
            refined.append(value)
            residuals.append(residual)
        if not refined:
            raise ValueError("at least one interior output index is required")
        return torch.stack(refined, dim=1), torch.stack(residuals, dim=1)


def extract_ordered_frozen_cin(generator, source):
    """Extract all CIN maps without folding the time axis into the batch axis."""

    if source.ndim != 5:
        raise ValueError("source must have shape B x T x C x H x W")
    ordered = {}
    with torch.no_grad():
        for time_index in range(source.shape[1]):
            cin = generator.extract_a2_cin_features(source[:, time_index])
            for key, value in cin.items():
                ordered.setdefault(key, []).append(value)
    return {key: torch.stack(values, dim=1) for key, values in ordered.items()}
