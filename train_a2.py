"""Train and evaluate the SynthRAD A2 inter-slice refinement module.

The pretrained 2-D generator is kept frozen.  Ordered input slices are encoded
through CIN, a small context module predicts FiLM parameters, and only the
centre slice of each window is decoded.  This makes the pilot independent of
the existing training jobs and keeps the original generator checkpoint intact.
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.volume_window_dataset import SynthRADVolumeWindowDataset
from models.a2_context import (
    InterSliceContextModulator,
    extract_ordered_frozen_features,
)
from models.frequency_loss import anatomical_mask
from models.mamba_one import ssim_loss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--source-opt", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("independent", "gru", "bigru"), default="bigru")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--val-stride", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--lambda-ssim", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--load-size", type=int, default=256)
    parser.add_argument("--fine-size", type=int, default=256)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
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
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_and_validate_source_options(args):
    source = {}
    with open(args.source_opt, "r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            source[key.strip()] = value.strip()
    expected = {
        "which_model_netG": "mamba",
        "vit_name": "Mamba-B_16",
        "input_nc": "1",
        "output_nc": "1",
        "fineSize": str(args.fine_size),
        "norm": "batch",
        "scan_use_multipath": str(args.scan_use_multipath),
        "scan_name": str(args.scan_name),
        "scan_K": str(args.scan_k),
        "scan_lambda_c": str(args.scan_lambda_c),
        "scan_mu": str(args.scan_mu),
        "no_lora": str(args.no_lora),
        "lora_rank": str(args.lora_rank),
        "lora_alpha": str(args.lora_alpha),
        "lora_dropout": str(args.lora_dropout),
    }
    mismatches = {
        key: {"source": source.get(key), "requested": value}
        for key, value in expected.items()
        if source.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "A2 configuration does not match source generator opt.txt: %s"
            % json.dumps(mismatches, sort_keys=True)
        )
    return source


def load_state_dict(path):
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("state_dict", "netG", "generator"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    if not isinstance(raw, dict):
        raise RuntimeError("generator checkpoint does not contain a state dict")
    if raw and all(key.startswith("module.") for key in raw):
        raw = {key[len("module.") :]: value for key, value in raw.items()}
    return raw


def build_generator(args, device):
    os.environ["AFS_SAM2_CHECKPOINT"] = os.path.abspath(args.sam2_checkpoint)
    from models import modules, networks

    modules.SCAN_CONFIG.update(
        {
            "use_multipath": args.scan_use_multipath,
            "scan": args.scan_name,
            "K": args.scan_k,
            "lambda_c": args.scan_lambda_c,
            "mu": args.scan_mu,
        }
    )
    generator = networks.define_G(
        input_nc=1,
        output_nc=1,
        ngf=64,
        which_model_netG="mamba",
        vit_name="Mamba-B_16",
        img_size=args.fine_size,
        pre_trained_path="",
        norm="batch",
        use_dropout=False,
        init_type="normal",
        gpu_ids=[],
        pre_trained_trans=False,
        pre_trained_resnet=False,
    )
    if not args.no_lora:
        generator.sam2encoder.enable_lora(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    state = load_state_dict(args.generator_checkpoint)
    missing, unexpected = generator.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "generator checkpoint mismatch; missing=%s unexpected=%s"
            % (missing[:12], unexpected[:12])
        )
    generator.to(device)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    return generator


def make_loaders(args):
    common = dict(
        dataroot=args.dataroot,
        window_size=args.window_size,
        load_size=args.load_size,
        fine_size=args.fine_size,
        direction="AtoB",
    )
    train_set = SynthRADVolumeWindowDataset(
        phase="train",
        stride=args.train_stride,
        augment=True,
        max_windows=args.max_train_windows,
        **common,
    )
    val_set = SynthRADVolumeWindowDataset(
        phase="val",
        stride=args.val_stride,
        augment=False,
        max_windows=args.max_val_windows,
        **common,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def frozen_features(generator, source, center):
    return extract_ordered_frozen_features(generator, source, center)


def forward_pair(generator, modulator, source, amp_enabled, need_baseline=False):
    center = source.shape[1] // 2
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        descriptors, features = frozen_features(generator, source, center)
        gamma, beta = modulator(descriptors)
        prediction = generator.decode_a2_features(
            features, gamma[:, center], beta[:, center]
        )
        baseline = None
        if need_baseline:
            baseline = generator.decode_a2_features(features)
    return prediction, baseline


def identity_gate(generator, modulator, loader, device, amp_enabled):
    batch = next(iter(loader))
    source = batch["A"].to(device, non_blocking=True)
    modulator.eval()
    with torch.no_grad():
        prediction, baseline = forward_pair(
            generator, modulator, source, amp_enabled, need_baseline=True
        )
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            direct = generator(source[:, source.shape[1] // 2])
    helper_max_abs = float((direct.float() - baseline.float()).abs().max().item())
    if helper_max_abs > 1e-6:
        raise RuntimeError(
            "A2 helper does not reproduce generator.forward: max_abs=%g"
            % helper_max_abs
        )
    max_abs = float((prediction.float() - baseline.float()).abs().max().item())
    if max_abs > 1e-6:
        raise RuntimeError("zero-init identity gate failed: max_abs=%g" % max_abs)
    return {"helper_max_abs": helper_max_abs, "identity_max_abs": max_abs}


class SequenceMetrics:
    """Patient-balanced axial and through-plane metrics."""

    def __init__(self):
        self.stats = defaultdict(
            lambda: defaultdict(lambda: {"sum": 0.0, "count": 0})
        )
        self.previous = {}

    def update(
        self,
        method,
        patient,
        slice_index,
        prediction,
        target,
        anatomy,
        ssim_value,
    ):
        pred = prediction.detach().float().cpu()
        truth = target.detach().float().cpu()
        anatomy = anatomy.detach().bool().cpu()
        l1 = float(F.l1_loss(pred, truth).item())
        mse = float(F.mse_loss(pred, truth).item())
        self._add(patient, "%s_l1" % method, l1)
        self._add(patient, "%s_mse" % method, mse)
        self._add(patient, "%s_ssim" % method, ssim_value)

        key = (method, patient)
        prior = self.previous.get(key)
        delta = None
        if prior is not None and slice_index == prior["slice_index"] + 1:
            delta_pred = pred - prior["prediction"]
            delta_truth = truth - prior["target"]
            delta = delta_pred - delta_truth
            delta_mask = anatomy | prior["anatomy"]
            self._add(
                patient,
                "%s_delta_l1_full" % method,
                float(delta.abs().mean().item()),
            )
            if delta_mask.any():
                self._add(
                    patient,
                    "%s_delta_l1_anatomy" % method,
                    float(delta.abs()[delta_mask].mean().item()),
                )
            if prior["delta"] is not None:
                second = delta - prior["delta"]
                self._add(
                    patient,
                    "%s_delta2_l1_full" % method,
                    float(second.abs().mean().item()),
                )
                second_mask = delta_mask | prior["delta_mask"]
                if second_mask.any():
                    self._add(
                        patient,
                        "%s_delta2_l1_anatomy" % method,
                        float(second.abs()[second_mask].mean().item()),
                    )
        else:
            delta_mask = None
        self.previous[key] = {
            "slice_index": int(slice_index),
            "prediction": pred,
            "target": truth,
            "anatomy": anatomy,
            "delta": delta,
            "delta_mask": delta_mask,
        }

    def _add(self, patient, metric, value):
        record = self.stats[patient][metric]
        record["sum"] += float(value)
        record["count"] += 1

    def patient_summaries(self):
        summaries = {}
        for patient_id, patient in self.stats.items():
            record = {
                name: values["sum"] / values["count"]
                for name, values in patient.items()
                if values["count"]
            }
            for method in ("base", "a2"):
                mse = record.get("%s_mse" % method)
                if mse is not None:
                    record["%s_psnr" % method] = 10.0 * math.log10(
                        4.0 / max(mse, 1e-12)
                    )
            summaries[patient_id] = record
        return summaries

    def summarize(self):
        patients = self.patient_summaries()
        metric_names = sorted({name for patient in patients.values() for name in patient})
        summary = {}
        for name in metric_names:
            values = [patient[name] for patient in patients.values() if name in patient]
            summary[name] = float(np.mean(values))
            summary[name + "_patients"] = len(values)
        return summary


def save_preview(output_dir, epoch, index, batch, baseline, prediction):
    from PIL import Image, ImageDraw

    os.makedirs(os.path.join(output_dir, "previews"), exist_ok=True)
    source = batch["A_center"][0, 0].detach().float().cpu().numpy()
    target = batch["B_center"][0, 0].detach().float().cpu().numpy()
    base = baseline[0, 0].detach().float().cpu().numpy()
    a2 = prediction[0, 0].detach().float().cpu().numpy()
    arrays = [source, target, base, a2, np.abs(base - target), np.abs(a2 - target)]
    labels = ["Input", "Target", "Frozen G", "G + A2", "G error", "A2 error"]
    tiles = []
    for array, label in zip(arrays, labels):
        if "error" in label:
            scale = max(float(np.percentile(array, 99)), 1e-6)
            image = np.clip(array / scale, 0.0, 1.0)
        else:
            image = np.clip((array + 1.0) * 0.5, 0.0, 1.0)
        tile = Image.fromarray((image * 255).astype(np.uint8), mode="L").convert("RGB")
        canvas = Image.new("RGB", (tile.width, tile.height + 24), "white")
        canvas.paste(tile, (0, 24))
        ImageDraw.Draw(canvas).text((6, 5), label, fill="black")
        tiles.append(canvas)
    montage = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white")
    offset = 0
    for tile in tiles:
        montage.paste(tile, (offset, 0))
        offset += tile.width
    patient = batch["patient_id"][0]
    slice_index = int(batch["center_slice_index"].item())
    stem = "epoch%03d_%03d_%s_slice%04d" % (epoch, index, patient, slice_index)
    montage.save(os.path.join(output_dir, "previews", stem + ".png"))
    np.savez_compressed(
        os.path.join(output_dir, "previews", stem + ".npz"),
        source=source,
        target=target,
        baseline=base,
        a2=a2,
    )


@torch.no_grad()
def evaluate(generator, modulator, loader, device, amp_enabled, output_dir, epoch, preview_count):
    generator.eval()
    modulator.eval()
    metrics = SequenceMetrics()
    for index, batch in enumerate(loader):
        source = batch["A"].to(device, non_blocking=True)
        target = batch["B_center"].to(device, non_blocking=True)
        prediction, baseline = forward_pair(
            generator, modulator, source, amp_enabled, need_baseline=True
        )
        prediction = prediction.float().clamp(-1.0, 1.0)
        baseline = baseline.float().clamp(-1.0, 1.0)
        patient = batch["patient_id"][0]
        slice_index = int(batch["center_slice_index"].item())
        base_ssim = float((1.0 - ssim_loss(baseline, target)).item())
        a2_ssim = float((1.0 - ssim_loss(prediction, target)).item())
        anatomy = anatomical_mask(target, threshold=-0.95, closing_kernel_size=7)
        metrics.update(
            "base", patient, slice_index, baseline[0], target[0], anatomy[0], base_ssim
        )
        metrics.update(
            "a2", patient, slice_index, prediction[0], target[0], anatomy[0], a2_ssim
        )
        if index < preview_count:
            save_preview(output_dir, epoch, index, batch, baseline, prediction)
    with open(
        os.path.join(output_dir, "patient_metrics_epoch%03d.json" % epoch),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics.patient_summaries(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics.summarize()


def train_epoch(generator, modulator, loader, optimizer, scaler, device, args, amp_enabled):
    generator.eval()
    modulator.train()
    totals = defaultdict(float)
    samples = 0
    gradient_gate = None
    for batch in loader:
        source = batch["A"].to(device, non_blocking=True)
        target = batch["B_center"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            prediction, _ = forward_pair(
                generator, modulator, source, amp_enabled, need_baseline=False
            )
            l1 = F.l1_loss(prediction, target)
            ssim = ssim_loss(prediction, target)
            loss = args.lambda_l1 * l1 + args.lambda_ssim * ssim
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if gradient_gate is None:
            gradient = modulator.affine[-1].weight.grad
            gradient_gate = float(gradient.abs().sum().item()) if gradient is not None else 0.0
            if not math.isfinite(gradient_gate) or gradient_gate <= 0.0:
                raise RuntimeError("A2 gradient gate failed: %s" % gradient_gate)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(target.shape[0])
        totals["train_loss"] += float(loss.item()) * batch_size
        totals["train_l1"] += float(l1.item()) * batch_size
        totals["train_ssim_loss"] += float(ssim.item()) * batch_size
        samples += batch_size
    return {key: value / samples for key, value in totals.items()}, gradient_gate


def save_checkpoint(path, modulator, optimizer, epoch, args, metrics):
    torch.save(
        {
            "epoch": int(epoch),
            "a2_state_dict": modulator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def main():
    args = parse_args()
    if args.window_size % 2 != 1 or args.window_size < 3:
        raise ValueError("window-size must be odd and at least 3")
    if args.smoke_only:
        args.epochs = 1
        if args.max_train_windows <= 0:
            args.max_train_windows = 8
        if args.max_val_windows <= 0:
            args.max_val_windows = 4
        args.preview_count = min(args.preview_count, 2)
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        raise RuntimeError(
            "output directory already contains metrics.jsonl; use a new run directory"
        )
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = device.type == "cuda" and not args.no_amp

    source_options = read_and_validate_source_options(args)
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"arguments": vars(args), "source_generator_options": source_options},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    train_loader, val_loader = make_loaders(args)
    generator = build_generator(args, device)
    modulator = InterSliceContextModulator(
        descriptor_dim=288,
        feature_dim=144,
        hidden_dim=args.hidden_dim,
        mode=args.mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        modulator.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    gate_metrics = identity_gate(
        generator, modulator, train_loader, device, amp_enabled
    )
    parameter_count = sum(parameter.numel() for parameter in modulator.parameters())
    print(
        json.dumps(
            {
                "event": "gates",
                **gate_metrics,
                "a2_parameters": parameter_count,
                "train_windows": len(train_loader.dataset),
                "val_windows": len(val_loader.dataset),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    best_primary = float("inf")
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loader.dataset.set_epoch(epoch - 1)
        train_metrics, gradient_gate = train_epoch(
            generator,
            modulator,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            amp_enabled,
        )
        val_metrics = evaluate(
            generator,
            modulator,
            val_loader,
            device,
            amp_enabled,
            args.output_dir,
            epoch,
            args.preview_count,
        )
        guard_pass = (
            val_metrics["a2_psnr"] >= val_metrics["base_psnr"] - 0.10
            and val_metrics["a2_ssim"] >= val_metrics["base_ssim"] - 0.001
            and val_metrics["a2_l1"] <= val_metrics["base_l1"] * 1.01
        )
        record = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - started,
            "gradient_gate": gradient_gate,
            "axial_guard_pass": guard_pass,
            **train_metrics,
            **val_metrics,
        }
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        save_checkpoint(
            os.path.join(args.output_dir, "latest_a2.pt"),
            modulator,
            optimizer,
            epoch,
            args,
            record,
        )
        primary = val_metrics.get("a2_delta_l1_anatomy", val_metrics["a2_l1"])
        if guard_pass and primary < best_primary:
            best_primary = primary
            save_checkpoint(
                os.path.join(args.output_dir, "best_a2.pt"),
                modulator,
                optimizer,
                epoch,
                args,
                record,
            )


if __name__ == "__main__":
    main()
