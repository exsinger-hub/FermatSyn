"""Frequency-domain losses and RAPSD metrics for SynthRAD images.

The helpers in this module assume 2-D image tensors shaped ``B x C x H x W``
and normalized to ``[-1, 1]``.  They are intentionally independent from the
training loop so ablations can import the same implementation without touching
the model code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_image_pair(pred, target):
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if pred.dim() != 4:
        raise ValueError("expected tensors shaped B x C x H x W")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("pred and target must be floating point tensors")


def _expand_mask(mask, image):
    if mask.dim() != 4:
        raise ValueError("mask must be shaped B x 1 x H x W or B x C x H x W")
    if mask.shape[0] != image.shape[0] or mask.shape[-2:] != image.shape[-2:]:
        raise ValueError("mask batch/spatial shape must match the image")
    if mask.shape[1] == 1:
        mask = mask.expand(-1, image.shape[1], -1, -1)
    elif mask.shape[1] != image.shape[1]:
        raise ValueError("mask channel count must be 1 or match the image")
    return mask.to(device=image.device, dtype=image.dtype).detach()


def anatomical_mask(image, threshold=-0.95, closing_kernel_size=7):
    """Build a body/anatomy mask for ``[-1, 1]`` SynthRAD-like slices.

    Air/background in SynthRAD-style normalized volumes is usually near ``-1``.
    The threshold therefore keeps pixels above ``threshold`` and then applies a
    differentiability-free 2-D morphological closing to fill small holes.  For
    multi-channel tensors the channel-wise maximum defines a shared anatomy mask
    that is expanded back to all channels.
    """

    if image.dim() != 4:
        raise ValueError("expected image shaped B x C x H x W")
    if not image.is_floating_point():
        raise TypeError("image must be a floating point tensor")

    with torch.no_grad():
        ref = image.detach()
        if ref.shape[1] > 1:
            ref = ref.amax(dim=1, keepdim=True)
        mask = (ref > threshold).to(dtype=image.dtype)

        k = int(closing_kernel_size or 0)
        if k > 1:
            if k % 2 == 0:
                k += 1
            pad = k // 2
            dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
            mask = -F.max_pool2d(-dilated, kernel_size=k, stride=1, padding=pad)

    return mask.expand(-1, image.shape[1], -1, -1).contiguous()


def hann_window2d(height, width, device=None, dtype=None):
    """Return a separable 2-D Hann window shaped ``1 x 1 x H x W``."""

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    dtype = dtype or torch.float32
    wy = torch.hann_window(height, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(width, periodic=False, device=device, dtype=dtype)
    return (wy[:, None] * wx[None, :]).view(1, 1, height, width)


def _prepare_frequency_inputs(
    pred,
    target,
    mask=None,
    use_mask=True,
    use_window=True,
    anatomy_threshold=-0.95,
    closing_kernel_size=7,
):
    _check_image_pair(pred, target)

    if use_mask:
        if mask is None:
            mask = anatomical_mask(target, anatomy_threshold, closing_kernel_size)
        else:
            mask = _expand_mask(mask, target)
        pred = pred * mask
        target = target * mask

    if use_window:
        window = hann_window2d(
            pred.shape[-2],
            pred.shape[-1],
            device=pred.device,
            dtype=pred.dtype,
        )
        pred = pred * window
        target = target * window

    return pred, target


def _fft_power(image, eps=1.0e-12):
    spectrum = torch.fft.fft2(image, dim=(-2, -1), norm="ortho")
    return spectrum.real.square() + spectrum.imag.square() + eps


def focal_frequency_loss(
    pred,
    target,
    mask=None,
    alpha=1.0,
    use_mask=True,
    use_window=True,
    anatomy_threshold=-0.95,
    closing_kernel_size=7,
    eps=1.0e-12,
):
    """Focal Frequency Loss with detached dynamic focus weights.

    The focus matrix is derived from the current spectral residual but detached
    before it weights the squared complex FFT error.  This keeps gradients on the
    residual itself while avoiding second-order feedback through the weights.
    """

    pred, target = _prepare_frequency_inputs(
        pred,
        target,
        mask=mask,
        use_mask=use_mask,
        use_window=use_window,
        anatomy_threshold=anatomy_threshold,
        closing_kernel_size=closing_kernel_size,
    )

    diff = torch.fft.fft2(pred - target, dim=(-2, -1), norm="ortho")
    diff_power = diff.real.square() + diff.imag.square()
    focus = (diff_power.detach() + eps).pow(0.5 * alpha)
    focus = focus / focus.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (focus * diff_power).mean()


def _radial_bin_index(height, width, num_bins, device):
    fy = torch.fft.fftfreq(height, d=1.0, device=device)
    fx = torch.fft.fftfreq(width, d=1.0, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    radius = radius / radius.max().clamp_min(1.0e-12)
    bins = torch.clamp((radius * num_bins).long(), max=num_bins - 1)
    return bins.reshape(-1)


def radial_average_power_spectral_density(
    image,
    num_bins=64,
    eps=1.0e-12,
):
    """Compute RAPSD profiles shaped ``B x C x num_bins``.

    The implementation is tensor-only and works on CPU or CUDA.  Empty radial
    bins are clamped with ``eps`` in the denominator for numerical stability.
    """

    if image.dim() != 4:
        raise ValueError("expected image shaped B x C x H x W")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    b, c, h, w = image.shape
    power = _fft_power(image, eps=0.0).reshape(b * c, h * w)
    bins = _radial_bin_index(h, w, num_bins, image.device)
    expanded_bins = bins.unsqueeze(0).expand(b * c, -1)

    profile = image.new_zeros((b * c, num_bins))
    profile.scatter_add_(1, expanded_bins, power)

    counts = image.new_zeros((num_bins,))
    counts.scatter_add_(0, bins, torch.ones_like(bins, dtype=image.dtype))
    profile = profile / counts.clamp_min(eps).view(1, num_bins)
    return profile.view(b, c, num_bins)


def rapsd_profile(
    image,
    mask=None,
    use_mask=True,
    use_window=True,
    anatomy_threshold=-0.95,
    closing_kernel_size=7,
    num_bins=64,
    eps=1.0e-12,
):
    """Compute RAPSD after applying the same mask/window policy as the loss."""

    if image.dim() != 4:
        raise ValueError("expected image shaped B x C x H x W")
    target = image
    if use_mask:
        if mask is None:
            mask = anatomical_mask(target, anatomy_threshold, closing_kernel_size)
        else:
            mask = _expand_mask(mask, target)
        image = image * mask
    if use_window:
        image = image * hann_window2d(
            image.shape[-2], image.shape[-1], device=image.device, dtype=image.dtype
        )
    return radial_average_power_spectral_density(image, num_bins=num_bins, eps=eps)


def rapsd_profile_loss(
    pred,
    target,
    mask=None,
    use_mask=True,
    use_window=True,
    anatomy_threshold=-0.95,
    closing_kernel_size=7,
    num_bins=64,
    eps=1.0e-12,
):
    """Differentiable log-RAPSD L1 loss."""

    pred, target = _prepare_frequency_inputs(
        pred,
        target,
        mask=mask,
        use_mask=use_mask,
        use_window=use_window,
        anatomy_threshold=anatomy_threshold,
        closing_kernel_size=closing_kernel_size,
    )
    pred_profile = radial_average_power_spectral_density(pred, num_bins, eps)
    target_profile = radial_average_power_spectral_density(target, num_bins, eps)
    return F.l1_loss(torch.log(pred_profile + eps), torch.log(target_profile + eps))


def rapsd_errors(
    pred,
    target,
    mask=None,
    use_mask=True,
    use_window=True,
    anatomy_threshold=-0.95,
    closing_kernel_size=7,
    num_bins=64,
    high_freq_fraction=0.5,
    eps=1.0e-12,
):
    """Return full-band and high-frequency relative RAPSD errors."""

    pred, target = _prepare_frequency_inputs(
        pred,
        target,
        mask=mask,
        use_mask=use_mask,
        use_window=use_window,
        anatomy_threshold=anatomy_threshold,
        closing_kernel_size=closing_kernel_size,
    )
    pred_profile = radial_average_power_spectral_density(pred, num_bins, eps)
    target_profile = radial_average_power_spectral_density(target, num_bins, eps)

    def relative_l1(a, b):
        return (a - b).abs().sum(dim=-1) / b.abs().sum(dim=-1).clamp_min(eps)

    start = int(round(num_bins * float(high_freq_fraction)))
    start = min(max(start, 0), num_bins - 1)
    full = relative_l1(pred_profile, target_profile).mean()
    high = relative_l1(pred_profile[..., start:], target_profile[..., start:]).mean()
    return {
        "rapsd_full_error": full,
        "rapsd_high_error": high,
    }


def composite_frequency_loss(
    pred,
    target,
    mask=None,
    lambda_ffl=1.0,
    lambda_rapsd=1.0,
    alpha=1.0,
    use_mask=True,
    use_window=True,
    anatomy_threshold=-0.95,
    closing_kernel_size=7,
    num_rapsd_bins=64,
    eps=1.0e-12,
):
    ffl = focal_frequency_loss(
        pred,
        target,
        mask=mask,
        alpha=alpha,
        use_mask=use_mask,
        use_window=use_window,
        anatomy_threshold=anatomy_threshold,
        closing_kernel_size=closing_kernel_size,
        eps=eps,
    )
    rapsd = rapsd_profile_loss(
        pred,
        target,
        mask=mask,
        use_mask=use_mask,
        use_window=use_window,
        anatomy_threshold=anatomy_threshold,
        closing_kernel_size=closing_kernel_size,
        num_bins=num_rapsd_bins,
        eps=eps,
    )
    return lambda_ffl * ffl + lambda_rapsd * rapsd


class FrequencyLoss(nn.Module):
    """Combined FFL + log-RAPSD loss with RAPSD error reporting."""

    def __init__(
        self,
        lambda_ffl=1.0,
        lambda_rapsd=1.0,
        alpha=1.0,
        use_mask=True,
        use_window=True,
        anatomy_threshold=-0.95,
        closing_kernel_size=7,
        num_rapsd_bins=64,
        high_freq_fraction=0.5,
        eps=1.0e-12,
    ):
        super(FrequencyLoss, self).__init__()
        self.lambda_ffl = lambda_ffl
        self.lambda_rapsd = lambda_rapsd
        self.alpha = alpha
        self.use_mask = use_mask
        self.use_window = use_window
        self.anatomy_threshold = anatomy_threshold
        self.closing_kernel_size = closing_kernel_size
        self.num_rapsd_bins = num_rapsd_bins
        self.high_freq_fraction = high_freq_fraction
        self.eps = eps

    def forward(self, pred, target, mask=None, return_metrics=False):
        ffl = focal_frequency_loss(
            pred,
            target,
            mask=mask,
            alpha=self.alpha,
            use_mask=self.use_mask,
            use_window=self.use_window,
            anatomy_threshold=self.anatomy_threshold,
            closing_kernel_size=self.closing_kernel_size,
            eps=self.eps,
        )
        rapsd = rapsd_profile_loss(
            pred,
            target,
            mask=mask,
            use_mask=self.use_mask,
            use_window=self.use_window,
            anatomy_threshold=self.anatomy_threshold,
            closing_kernel_size=self.closing_kernel_size,
            num_bins=self.num_rapsd_bins,
            eps=self.eps,
        )
        loss = self.lambda_ffl * ffl + self.lambda_rapsd * rapsd
        if not return_metrics:
            return loss

        errors = rapsd_errors(
            pred,
            target,
            mask=mask,
            use_mask=self.use_mask,
            use_window=self.use_window,
            anatomy_threshold=self.anatomy_threshold,
            closing_kernel_size=self.closing_kernel_size,
            num_bins=self.num_rapsd_bins,
            high_freq_fraction=self.high_freq_fraction,
            eps=self.eps,
        )
        metrics = {
            "loss_frequency": loss.detach(),
            "loss_ffl": ffl.detach(),
            "loss_rapsd": rapsd.detach(),
        }
        metrics.update({key: value.detach() for key, value in errors.items()})
        return loss, metrics


def frequency_mode_weights(freq_mode, gamma=1.0):
    """Map the training frequency mode to raw FFL/RAPSD weights."""

    mode = str(freq_mode or "none").lower()
    if mode == "none":
        return 0.0, 0.0
    if mode == "ffl":
        return 1.0, 0.0
    if mode == "rapsd":
        return 0.0, 1.0
    if mode == "full":
        return 1.0, float(gamma)
    raise ValueError("freq_mode must be one of: none, ffl, rapsd, full")
