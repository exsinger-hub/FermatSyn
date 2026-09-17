"""Evaluate whether a trained A2-v1 BiGRU uses its neighbouring descriptors."""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from data.volume_window_dataset import SynthRADVolumeWindowDataset
from models.a2_context import InterSliceContextModulator, intervene_descriptors
from models.frequency_loss import anatomical_mask
from models.mamba_one import ssim_loss
from train_a2 import (
    SequenceMetrics,
    build_generator,
    frozen_features,
    read_and_validate_source_options,
    set_seed,
)


POLICIES = ("ordered", "center_repeat", "reverse")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--source-opt", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--a2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--load-size", type=int, default=256)
    parser.add_argument("--fine-size", type=int, default=256)
    parser.add_argument("--scan-use-multipath", action="store_true", default=True)
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


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    if os.path.exists(summary_path):
        raise RuntimeError("summary.json already exists; use a new output directory")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    read_and_validate_source_options(args)
    generator = build_generator(args, device)

    raw = torch.load(args.a2_checkpoint, map_location="cpu")
    saved_args = raw.get("args", {})
    modulator = InterSliceContextModulator(
        descriptor_dim=288,
        feature_dim=144,
        hidden_dim=int(saved_args.get("hidden_dim", 128)),
        mode=saved_args.get("mode", "bigru"),
    ).to(device)
    modulator.load_state_dict(raw["a2_state_dict"], strict=True)
    modulator.eval()

    dataset = SynthRADVolumeWindowDataset(
        dataroot=args.dataroot,
        phase="val",
        window_size=int(saved_args.get("window_size", 5)),
        stride=1,
        load_size=args.load_size,
        fine_size=args.fine_size,
        direction="AtoB",
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    metrics = {policy: SequenceMetrics() for policy in POLICIES}
    helper_checked = False
    for batch in loader:
        source = batch["A"].to(device, non_blocking=True)
        target = batch["B_center"].to(device, non_blocking=True)
        center = source.shape[1] // 2
        descriptors, features = frozen_features(generator, source, center)
        baseline = generator.decode_a2_features(features).float().clamp(-1.0, 1.0)
        if not helper_checked:
            direct = generator(source[:, center]).float().clamp(-1.0, 1.0)
            helper_max_abs = float((direct - baseline).abs().max().item())
            if helper_max_abs > 1e-6:
                raise RuntimeError("helper identity failed: max_abs=%g" % helper_max_abs)
            helper_checked = True

        patient = batch["patient_id"][0]
        slice_index = int(batch["center_slice_index"].item())
        anatomy = anatomical_mask(target, threshold=-0.95, closing_kernel_size=7)
        base_ssim = float((1.0 - ssim_loss(baseline, target)).item())
        for policy in POLICIES:
            gamma, beta = modulator(intervene_descriptors(descriptors, policy))
            prediction = generator.decode_a2_features(
                features, gamma[:, center], beta[:, center]
            ).float().clamp(-1.0, 1.0)
            a2_ssim = float((1.0 - ssim_loss(prediction, target)).item())
            metrics[policy].update(
                "base",
                patient,
                slice_index,
                baseline[0],
                target[0],
                anatomy[0],
                base_ssim,
            )
            metrics[policy].update(
                "a2",
                patient,
                slice_index,
                prediction[0],
                target[0],
                anatomy[0],
                a2_ssim,
            )

    summary = {}
    for policy in POLICIES:
        patients = metrics[policy].patient_summaries()
        write_json(
            os.path.join(args.output_dir, "patient_metrics_%s.json" % policy),
            patients,
        )
        summary[policy] = metrics[policy].summarize()
    write_json(
        os.path.join(args.output_dir, "manifest.json"),
        {
            "arguments": vars(args),
            "saved_a2_arguments": saved_args,
            "policies": list(POLICIES),
            "validation_windows": len(dataset),
            "helper_max_abs": helper_max_abs,
        },
    )
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
