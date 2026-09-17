"""Joint decoding and through-plane objectives for the A2-v2 pilot."""

import torch

from models.a2_spatial import SUPPORTED_INPUT_MODES, extract_ordered_frozen_cin
from models.frequency_loss import anatomical_mask


def output_indices(window_size):
    if window_size != 5:
        raise ValueError("A2-v2 pilot is preregistered for window-size 5")
    return (1, 2, 3)


def _features_at_time(generator, ordered, time_index):
    batch = ordered["skip_fusion2"].shape[0]
    cin = {key: value[:, time_index] for key, value in ordered.items()}
    selected = torch.arange(batch, device=cin["skip_fusion2"].device)
    return generator.build_a2_center_features(cin, selected)


def forward_window(
    generator,
    adapter,
    source,
    input_mode,
    amp_enabled,
    need_baseline=False,
):
    """Jointly refine the middle three slices without folding time into batch."""

    if input_mode not in SUPPORTED_INPUT_MODES:
        raise ValueError("unsupported input mode: %s" % input_mode)
    indices = output_indices(source.shape[1])
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        ordered = extract_ordered_frozen_cin(generator, source)
        refined, residuals = adapter(
            ordered["skip_fusion2"],
            output_indices=indices,
            input_mode=input_mode,
        )
        predictions, baselines = [], []
        for output_position, time_index in enumerate(indices):
            features = _features_at_time(generator, ordered, time_index)
            if need_baseline:
                baselines.append(generator.decode_a2_features(features))
            refined_features = dict(features)
            # Inject only through the deepest skip.  Frozen bottleneck
            # activations remain identical across all paired arms.
            refined_features["skip_fusion2"] = refined[:, output_position]
            predictions.append(generator.decode_a2_features(refined_features))
    prediction = torch.stack(predictions, dim=1)
    baseline = torch.stack(baselines, dim=1) if need_baseline else None
    return prediction, baseline, residuals


def anatomy_delta_loss(prediction, target):
    """Match predicted and true first differences inside union anatomy masks."""

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must match as B x K x C x H x W")
    if prediction.shape[1] < 2:
        raise ValueError("at least two ordered outputs are required")
    masks = anatomical_mask(
        target.flatten(0, 1), threshold=-0.95, closing_kernel_size=7
    ).reshape_as(target)
    union = torch.maximum(masks[:, 1:], masks[:, :-1])
    error = (
        (prediction[:, 1:] - prediction[:, :-1])
        - (target[:, 1:] - target[:, :-1])
    ).abs()
    denominator = union.sum()
    if float(denominator.detach().item()) <= 0.0:
        return error.mean()
    return (error * union).sum() / denominator

