"""Evaluate a fixed Frozen-G axial 3-tap smoothing control on SynthRAD.

The control has no trainable parameters.  For every unique centre slice, the
frozen 2-D generator is called separately on the immediate left, centre, and
right source slices.  Only the smoothed centre output enters the same
patient-balanced ``SequenceMetrics`` used by A2-v2, so overlapping five-slice
windows never duplicate metric observations.
"""

import argparse
import json
import math
import os
import time
from collections import Counter

import torch


ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
WINDOW_SIZE = 5
TRIPLET_INDICES = (1, 2, 3)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--source-opt", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--load-size", type=int, default=256)
    parser.add_argument("--fine-size", type=int, default=256)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument(
        "--scan-use-multipath", dest="scan_use_multipath", action="store_true"
    )
    parser.add_argument(
        "--scan-use-legacy", dest="scan_use_multipath", action="store_false"
    )
    parser.set_defaults(scan_use_multipath=True)
    parser.add_argument("--scan-name", default="fermat")
    parser.add_argument("--scan-k", type=int, default=1)
    parser.add_argument("--scan-lambda-c", type=float, default=0.7)
    parser.add_argument("--scan-mu", type=float, default=0.03)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    return parser.parse_args()


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def alpha_key(alpha):
    return "%.2f" % float(alpha)


def alpha_tag(alpha):
    return "%03d" % int(round(float(alpha) * 100.0))


def three_tap_output(left, center, right, alpha):
    """Apply ``[alpha/2, 1-alpha, alpha/2]`` to matching image tensors."""

    if left.shape != center.shape or right.shape != center.shape:
        raise ValueError("left, center, and right outputs must have matching shapes")
    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and lie in [0, 1]")
    return (1.0 - alpha) * center + 0.5 * alpha * (left + right)


@torch.no_grad()
def frozen_center_triplet(generator, source):
    """Call Frozen G once per time point; never fold time into the batch."""

    if source.ndim != 5 or source.shape[1] != WINDOW_SIZE:
        raise ValueError("source must have shape B x 5 x C x H x W")
    outputs = []
    for time_index in TRIPLET_INDICES:
        outputs.append(generator(source[:, time_index]))
    return tuple(outputs)


def register_unique_center(seen, patient, slice_index):
    """Reject duplicate metric observations for the same physical slice."""

    key = (str(patient), int(slice_index))
    if key in seen:
        raise RuntimeError(
            "duplicate validation centre: patient=%s slice=%d" % key
        )
    seen.add(key)
    return key


def axial_guard(candidate, baseline):
    """Apply the A2 axial-quality guard against the alpha-zero baseline."""

    required = ("a2_psnr", "a2_ssim", "a2_l1")
    missing = [
        name for name in required if name not in candidate or name not in baseline
    ]
    if missing:
        raise ValueError("missing axial guard metrics: %s" % sorted(set(missing)))
    checks = {
        "psnr": {
            "value": float(candidate["a2_psnr"]),
            "minimum": float(baseline["a2_psnr"] - 0.10),
        },
        "ssim": {
            "value": float(candidate["a2_ssim"]),
            "minimum": float(baseline["a2_ssim"] - 0.001),
        },
        "l1": {
            "value": float(candidate["a2_l1"]),
            "maximum": float(baseline["a2_l1"] * 1.01),
        },
    }
    checks["psnr"]["pass"] = checks["psnr"]["value"] >= checks["psnr"]["minimum"]
    checks["ssim"]["pass"] = checks["ssim"]["value"] >= checks["ssim"]["minimum"]
    checks["l1"]["pass"] = checks["l1"]["value"] <= checks["l1"]["maximum"]
    return {
        "pass": all(record["pass"] for record in checks.values()),
        "checks": checks,
    }


def summarize_controls(metric_banks):
    """Return patient-equal summaries and the guarded validation upper bound."""

    if tuple(metric_banks) != ALPHAS:
        raise ValueError("metric banks must use the fixed alpha grid in order")
    aggregates = {
        alpha: metric_banks[alpha].summarize() for alpha in ALPHAS
    }
    baseline = aggregates[0.0]
    primary_name = "a2_delta_l1_anatomy"
    results = {}
    eligible = []
    for alpha in ALPHAS:
        aggregate = aggregates[alpha]
        guard = axial_guard(aggregate, baseline)
        primary = aggregate.get(primary_name)
        base_primary = baseline.get(primary_name)
        change = None
        if primary is not None and base_primary is not None:
            absolute = float(base_primary - primary)
            relative = (
                100.0 * absolute / base_primary if base_primary > 0.0 else None
            )
            change = {
                "absolute_reduction": absolute,
                "relative_reduction_percent": relative,
            }
            if guard["pass"]:
                eligible.append((float(primary), alpha))
        results[alpha_key(alpha)] = {
            "alpha": float(alpha),
            "metrics": aggregate,
            "axial_guard": guard,
            "primary_change_vs_alpha_zero": change,
        }
    best_alpha = min(eligible)[1] if eligible else None
    return {
        "primary_metric": primary_name,
        "aggregation": "mean within patient, then unweighted mean across patients",
        "alpha_results": results,
        "selection": {
            "rule": "minimum patient-equal primary among axial-guard-pass alphas",
            "best_guard_pass_alpha": (
                float(best_alpha) if best_alpha is not None else None
            ),
            "interpretation": "validation-optimized smoothing upper bound",
        },
    }


@torch.no_grad()
def main():
    # Keep project-heavy imports out of the pure helper surface so the fixed
    # smoothing algebra and accounting rules remain unit-testable on CPU.
    from torch.utils.data import DataLoader

    from data.volume_window_dataset import SynthRADVolumeWindowDataset
    from models.frequency_loss import anatomical_mask
    from models.mamba_one import ssim_loss
    from train_a2 import (
        SequenceMetrics,
        build_generator,
        read_and_validate_source_options,
        set_seed,
    )

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    if os.path.exists(summary_path):
        raise RuntimeError("summary.json already exists; use a new output directory")
    if args.fine_size != args.load_size:
        raise ValueError(
            "the fixed validation control requires fine-size == load-size"
        )

    started = time.time()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source_options = read_and_validate_source_options(args)
    generator = build_generator(args, device)
    generator.eval()

    dataset = SynthRADVolumeWindowDataset(
        dataroot=args.dataroot,
        phase="val",
        window_size=WINDOW_SIZE,
        stride=1,
        load_size=args.load_size,
        fine_size=args.fine_size,
        direction="AtoB",
        augment=False,
        max_windows=args.max_val_windows,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    metric_banks = {alpha: SequenceMetrics() for alpha in ALPHAS}
    seen = set()
    centers_per_patient = Counter()
    alpha_zero_max_abs = 0.0
    for batch in loader:
        source = batch["A"].to(device, non_blocking=True)
        target = batch["B_center"].to(device, non_blocking=True)
        patient = batch["patient_id"][0]
        slice_index = int(batch["center_slice_index"].item())
        register_unique_center(seen, patient, slice_index)
        centers_per_patient[patient] += 1

        left, center, right = frozen_center_triplet(generator, source)
        left = left.float().clamp(-1.0, 1.0)
        center = center.float().clamp(-1.0, 1.0)
        right = right.float().clamp(-1.0, 1.0)
        anatomy = anatomical_mask(target, threshold=-0.95, closing_kernel_size=7)
        for alpha in ALPHAS:
            prediction = three_tap_output(left, center, right, alpha).clamp(-1.0, 1.0)
            if alpha == 0.0:
                alpha_zero_max_abs = max(
                    alpha_zero_max_abs,
                    float((prediction - center).abs().max().item()),
                )
            ssim_value = float((1.0 - ssim_loss(prediction, target)).item())
            metric_banks[alpha].update(
                "a2",
                patient,
                slice_index,
                prediction[0],
                target[0],
                anatomy[0],
                ssim_value,
            )

    if alpha_zero_max_abs != 0.0:
        raise RuntimeError(
            "alpha-zero control does not reproduce Frozen G: max_abs=%g"
            % alpha_zero_max_abs
        )

    for alpha in ALPHAS:
        write_json(
            os.path.join(
                args.output_dir,
                "patient_metrics_alpha_%s.json" % alpha_tag(alpha),
            ),
            metric_banks[alpha].patient_summaries(),
        )
    summary = summarize_controls(metric_banks)
    write_json(summary_path, summary)
    write_json(
        os.path.join(args.output_dir, "manifest.json"),
        {
            "arguments": vars(args),
            "source_generator_options": source_options,
            "alpha_grid": list(ALPHAS),
            "formula": "(1-alpha)*G_t + alpha/2*(G_t-1 + G_t+1)",
            "window_size": WINDOW_SIZE,
            "generator_time_indices": list(TRIPLET_INDICES),
            "metric_output_index": 2,
            "boundary_policy": (
                "use only Window-5 centres; no padding; never cross patient or slice gap"
            ),
            "temporal_forward_policy": "three separate Frozen-G calls; time never folded into batch",
            "aggregation": "mean within patient, then unweighted mean across patients",
            "validation_centers": len(seen),
            "validation_patients": len(centers_per_patient),
            "centers_per_patient": dict(sorted(centers_per_patient.items())),
            "alpha_zero_max_abs": alpha_zero_max_abs,
            "generator_forward_calls": 3 * len(seen),
            "elapsed_seconds": time.time() - started,
        },
    )


if __name__ == "__main__":
    main()
