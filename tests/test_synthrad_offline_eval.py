import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_synthrad_offline import (
    PairSpec,
    discover_pairs,
    evaluate_pairs,
    evaluate_tensor_pair,
    main,
)


def _body_sample(delta=0.0):
    target = torch.full((1, 1, 32, 32), -1.0)
    target[..., 8:24, 8:24] = 0.0
    pred = target.clone()
    pred[..., 10:22, 10:22] += delta
    return pred, target


def test_identical_pair_has_zero_errors_and_capped_psnr():
    pred, target = _body_sample(delta=0.0)
    metrics = evaluate_tensor_pair(pred, target, num_rapsd_bins=16)
    assert metrics["full_mae"] == pytest.approx(0.0)
    assert metrics["anatomy_mae"] == pytest.approx(0.0)
    assert metrics["full_psnr"] == pytest.approx(100.0)
    assert metrics["anatomy_psnr"] == pytest.approx(100.0)
    assert metrics["full_rapsd_relative_error"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["anatomy_rapsd_high_relative_error"] == pytest.approx(0.0, abs=1e-7)


def test_masked_metrics_ignore_background_only_error():
    pred, target = _body_sample(delta=0.0)
    pred[..., :4, :4] = 1.0
    metrics = evaluate_tensor_pair(pred, target, num_rapsd_bins=16)
    assert metrics["full_mae"] > 0.0
    assert metrics["anatomy_mae"] == pytest.approx(0.0)
    assert metrics["full_psnr"] < metrics["anatomy_psnr"]

def test_anatomy_ssim_averages_ssim_map_only_inside_mask():
    target = torch.full((1, 1, 64, 64), -1.0)
    target[..., 24:40, 24:40] = 0.0
    pred = target.clone()
    pred[..., 28:36, 28:36] = 0.75

    metrics = evaluate_tensor_pair(pred, target, num_rapsd_bins=16)

    mask = (target.numpy()[0, 0] > -0.95)
    _, ssim_map = structural_similarity(
        target.numpy()[0, 0],
        pred.numpy()[0, 0],
        data_range=2.0,
        full=True,
    )
    old_zeroed_full_ssim = structural_similarity(
        target.numpy()[0, 0] * mask,
        pred.numpy()[0, 0] * mask,
        data_range=2.0,
    )
    assert metrics["anatomy_ssim"] == pytest.approx(float(np.mean(ssim_map[mask])))
    assert metrics["anatomy_ssim"] < old_zeroed_full_ssim

def test_evaluate_pairs_splits_batched_arrays_and_records_normalized_space(tmp_path):
    pred = np.zeros((2, 1, 16, 16), dtype=np.float32)
    target = np.zeros((2, 1, 16, 16), dtype=np.float32)
    pred[1] = 0.1
    pred_path = tmp_path / "pred.npy"
    target_path = tmp_path / "target.npy"
    np.save(pred_path, pred)
    np.save(target_path, target)

    rows, summary = evaluate_pairs([PairSpec("case", pred_path, target_path)], num_rapsd_bins=8)

    assert len(rows) == 2
    assert summary["num_pairs"] == 1
    assert summary["num_samples"] == 2
    assert summary["aggregation"] == "unweighted arithmetic mean over evaluated 2-D samples"
    assert summary["intensity_space"] == "normalized [-1, 1]; no HU conversion applied"
    assert summary["metrics_mean"]["full_mae"] == pytest.approx(0.05, abs=1e-6)


def test_discover_pairs_matches_relative_stems(tmp_path):
    pred_dir = tmp_path / "pred"
    target_dir = tmp_path / "target"
    (pred_dir / "sub").mkdir(parents=True)
    (target_dir / "sub").mkdir(parents=True)
    np.save(pred_dir / "sub" / "case001.npy", np.zeros((8, 8), dtype=np.float32))
    np.save(target_dir / "sub" / "case001.npy", np.zeros((8, 8), dtype=np.float32))
    pairs = discover_pairs(pred_dir, target_dir)
    assert pairs == [PairSpec("sub/case001", pred_dir / "sub" / "case001.npy", target_dir / "sub" / "case001.npy")]

def test_discover_pairs_errors_on_mismatched_relative_stems(tmp_path):
    pred_dir = tmp_path / "pred"
    target_dir = tmp_path / "target"
    pred_dir.mkdir()
    target_dir.mkdir()
    np.save(pred_dir / "case001.npy", np.zeros((8, 8), dtype=np.float32))
    np.save(target_dir / "case002.npy", np.zeros((8, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="mismatched paired files"):
        discover_pairs(pred_dir, target_dir)

def test_main_writes_json_and_csv_from_pairs_csv(tmp_path):
    pred = tmp_path / "pred.npy"
    target = tmp_path / "target.npy"
    np.save(pred, np.zeros((16, 16), dtype=np.float32))
    np.save(target, np.zeros((16, 16), dtype=np.float32))
    pairs_csv = tmp_path / "pairs.csv"
    with pairs_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "pred", "target"])
        writer.writeheader()
        writer.writerow({"id": "case", "pred": pred, "target": target})
    outdir = tmp_path / "out"

    assert main(["--pairs-csv", str(pairs_csv), "--outdir", str(outdir), "--num-rapsd-bins", "8"]) == 0

    assert (outdir / "per_sample_metrics.csv").exists()
    assert (outdir / "summary_metrics.csv").exists()
    summary = json.loads((outdir / "summary.json").read_text())
    assert summary["num_samples"] == 1
    assert summary["outputs"]["summary_json"] == str(outdir / "summary.json")