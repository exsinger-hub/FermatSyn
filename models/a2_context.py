"""Inter-slice context propagation for the A2 refinement pilot.

The module consumes one descriptor per ordered slice and predicts channel-wise
affine parameters for the frozen generator bottleneck.  Its final projection is
zero initialized, so inserting it is exactly an identity transformation before
training.
"""

import torch
import torch.nn as nn


SUPPORTED_DESCRIPTOR_INTERVENTIONS = {"ordered", "center_repeat", "reverse"}


def intervene_descriptors(descriptors, policy):
    """Apply a deterministic context intervention without changing the centre."""

    if descriptors.ndim != 3:
        raise ValueError("descriptors must have shape B x T x C")
    if policy not in SUPPORTED_DESCRIPTOR_INTERVENTIONS:
        raise ValueError(
            "descriptor intervention must be one of %s"
            % sorted(SUPPORTED_DESCRIPTOR_INTERVENTIONS)
        )
    if policy == "ordered":
        return descriptors
    if policy == "center_repeat":
        center = descriptors.shape[1] // 2
        return descriptors[:, center : center + 1].expand_as(descriptors)
    return torch.flip(descriptors, dims=(1,))


class InterSliceContextModulator(nn.Module):
    """Predict per-slice channel affine parameters from ordered descriptors."""

    def __init__(
        self,
        descriptor_dim=288,
        feature_dim=288,
        hidden_dim=128,
        mode="bigru",
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        if mode not in {"independent", "gru", "bigru"}:
            raise ValueError("mode must be independent, gru, or bigru")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.descriptor_dim = int(descriptor_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.mode = mode
        self.num_layers = int(num_layers)
        self.input_norm = nn.LayerNorm(self.descriptor_dim)

        recurrent_dropout = float(dropout) if self.num_layers > 1 else 0.0
        if mode == "independent":
            self.context = nn.Sequential(
                nn.Linear(self.descriptor_dim, self.hidden_dim),
                nn.SiLU(),
            )
            context_dim = self.hidden_dim
        else:
            bidirectional = mode == "bigru"
            self.context = nn.GRU(
                input_size=self.descriptor_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=recurrent_dropout,
                bidirectional=bidirectional,
                batch_first=True,
            )
            context_dim = self.hidden_dim * (2 if bidirectional else 1)

        self.affine = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 2 * self.feature_dim),
        )
        self.reset_identity()

    def reset_identity(self):
        """Make the predicted affine transform exactly identity at start."""

        last = self.affine[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, descriptors):
        if descriptors.ndim != 3:
            raise ValueError("descriptors must have shape B x T x C")
        if descriptors.shape[-1] != self.descriptor_dim:
            raise ValueError(
                "descriptor channel mismatch: expected %d, got %d"
                % (self.descriptor_dim, descriptors.shape[-1])
            )

        normalized = self.input_norm(descriptors)
        if self.mode == "independent":
            context = self.context(normalized)
        else:
            context, _ = self.context(normalized)
        gamma, beta = self.affine(context).chunk(2, dim=-1)
        return gamma, beta


def apply_channel_affine(features, gamma, beta):
    """Apply ``(1 + gamma) * features + beta`` to B x T feature maps."""

    if features.ndim != 5:
        raise ValueError("features must have shape B x T x C x H x W")
    if gamma.ndim != 3 or beta.ndim != 3:
        raise ValueError("gamma and beta must have shape B x T x C")
    if gamma.shape != beta.shape or gamma.shape != features.shape[:3]:
        raise ValueError("affine parameters must match feature B/T/C dimensions")
    gamma = gamma.unsqueeze(-1).unsqueeze(-1)
    beta = beta.unsqueeze(-1).unsqueeze(-1)
    return (1.0 + gamma) * features + beta


def extract_ordered_frozen_features(generator, source, center):
    """Extract per-slice descriptors without folding time into the batch."""

    if source.ndim != 5:
        raise ValueError("source must have shape B x T x C x H x W")
    batch, window, _, _, _ = source.shape
    if center < 0 or center >= window:
        raise ValueError("center index is outside the source window")

    descriptors = []
    features = None
    with torch.no_grad():
        # The ordinary frozen-generator path receives a batch B of 2-D slices.
        # Folding time into that batch changed B to B*T and changed the real
        # SAM2/CIN centre feature enough to violate the no-A2 identity gate.
        # Preserve the original effective batch at every ordered time point.
        selected = torch.arange(batch, device=source.device)
        for time_index in range(window):
            cin = generator.extract_a2_cin_features(source[:, time_index])
            descriptors.append(cin["descriptor"])
            if time_index == center:
                features = generator.build_a2_center_features(cin, selected)
    return torch.stack(descriptors, dim=1), features
