import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from compare_a2_v2_epoch010 import (
    ARM_CENTER,
    ARM_DELTA,
    ARM_SPATIAL,
    FIRST_EPOCH_AT_UTC,
    PROTOCOL_FROZEN_AT_UTC,
    _bootstrap_indices,
    compare_a2_v2_epoch010,
    exact_wilcoxon_signed_rank,
    main,
)


def _manifest(arm, output_dir):
    input_mode = "center_only" if arm == ARM_CENTER else "neighbor_difference"
    delta_weight = 7.25 if arm == ARM_DELTA else 0.0
    return {
        "arguments": {
            "dataroot": "/data/synthrad_patient_disjoint_m11_v1",
            "generator_checkpoint": "/results/full_g_seed2028/best_net_G.pth",
            "source_opt": "/results/full_g_seed2028/opt.txt",
            "sam2_checkpoint": "/models/sam2_hiera_large.pt",
            "output_dir": str(output_dir),
            "input_mode": input_mode,
            "window_size": 5,
            "train_stride": 5,
            "val_stride": 1,
            "adapter_rank": 96,
            "epochs": 10,
            "batch_size": 1,
            "workers": 2,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lambda_l1": 100.0,
            "lambda_ssim": 10.0,
            "delta_kappa": 0.0,
            "delta_weight": delta_weight,
            "calibration_points_per_patient": 3,
            "seed": 2026,
            "load_size": 256,
            "fine_size": 256,
            "max_train_windows": 0,
            "max_val_windows": 0,
            "preview_count": 4,
            "device": "cuda:0",
            "no_amp": True,
            "calibrate_only": False,
            "smoke_only": False,
            "scan_use_multipath": True,
            "scan_name": "fermat",
            "scan_k": 1,
            "scan_lambda_c": 0.7,
            "scan_mu": 0.03,
            "no_lora": False,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
        },
        "source_generator_options": {
            "which_model_netG": "mamba",
            "scan_name": "fermat",
        },
        "repo": {
            "branch": "codex/a2v2-spatial-difference",
            "commit": "0123456789abcdef",
            "tracked_dirty": False,
        },
        "environment": {
            "hostname": "server26",
            "torch": "2.5.0",
            "cuda_runtime": "12.1",
            "cudnn": 90100,
            "gpu": "NVIDIA GeForce RTX 4090",
            "precision": "FP32",
        },
        "adapter_parameters": 100_000,
        "train_windows_epoch0": 2_679,
        "train_windows_all_offsets": 13_393,
        "val_windows": 1_689,
        "output_indices": [1, 2, 3],
        "validation_uses_unique_center_only": True,
        "calibration": {
            "delta_kappa": 0.0,
            "delta_weight": delta_weight,
            "source": "explicit" if delta_weight else "disabled",
        },
        "gates": {"helper_max_abs": 0.0, "identity_max_abs": 0.0},
    }


def _patient_record(primary, delta2, l1=0.036, psnr=26.0, ssim=0.895):
    return {
        "a2_delta_l1_anatomy": primary,
        "a2_delta2_l1_anatomy": delta2,
        "a2_l1": l1,
        "a2_psnr": psnr,
        "a2_ssim": ssim,
        "base_delta_l1_anatomy": 0.032,
        "base_delta2_l1_anatomy": 0.045,
        "base_l1": 0.037,
        "base_psnr": 25.8,
        "base_ssim": 0.890,
    }


def _write_arm(root, arm, primary, delta2):
    run_dir = root / arm
    run_dir.mkdir()
    manifest = _manifest(arm, run_dir)
    patients = {
        "P%03d" % index: _patient_record(primary, delta2)
        for index in range(1, 12)
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (run_dir / "patient_metrics_epoch010.json").write_text(
        json.dumps(patients), encoding="utf-8"
    )
    return run_dir


def _write_valid_triad(root, delta_primary=0.0303):
    center = _write_arm(root, ARM_CENTER, primary=0.0310, delta2=0.0430)
    spatial = _write_arm(root, ARM_SPATIAL, primary=0.0306, delta2=0.0420)
    delta = _write_arm(root, ARM_DELTA, primary=delta_primary, delta2=0.0415)
    return center, spatial, delta


class CompareA2V2Epoch010Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_wilcoxon_is_explicitly_enumerated_with_ties(self):
        result = exact_wilcoxon_signed_rank(np.ones(6, dtype=np.float64))
        self.assertEqual(result["method"], "enumerated_signed_rank_two_sided")
        self.assertEqual(result["n_nonzero"], 6)
        self.assertEqual(result["T_plus"], 21.0)
        self.assertEqual(result["T_minus"], 0.0)
        self.assertEqual(result["p_exact"], 0.03125)

    def test_bootstrap_indices_are_common_and_reproducible(self):
        first = _bootstrap_indices(11)
        second = _bootstrap_indices(11)
        self.assertEqual(first.shape, (10_000, 11))
        self.assertTrue(np.array_equal(first, second))

    def test_delta_is_kept_only_when_frozen_delta_gate_passes(self):
        center, spatial, delta = _write_valid_triad(self.root)
        decision = compare_a2_v2_epoch010(center, spatial, delta)

        self.assertEqual(decision["input_status"], "VALID")
        self.assertEqual(
            decision["protocol_frozen_at_utc"], PROTOCOL_FROZEN_AT_UTC
        )
        self.assertEqual(decision["first_epoch_at_utc"], FIRST_EPOCH_AT_UTC)
        self.assertLess(
            decision["protocol_frozen_at_utc"], decision["first_epoch_at_utc"]
        )
        self.assertIs(
            decision["constants"]["thresholds_cli_overridable"], False
        )
        self.assertEqual(
            decision["constants"]["delta_keep_min_relative_improvement_pct"],
            0.5,
        )
        self.assertIs(decision["contrasts"]["delta_vs_spatial"]["pass"], True)
        self.assertEqual(
            decision["contrasts"]["delta_vs_spatial"]["n_better"], 11
        )
        self.assertEqual(
            decision["contrasts"]["delta_vs_spatial"]["wilcoxon"]["p_exact"],
            0.0009765625,
        )
        guard = decision["contrasts"]["delta_vs_spatial"]["guards"]
        self.assertIs(guard["all_pass"], True)
        self.assertEqual(
            set(guard["candidate_vs_control"]["atoms"]),
            {"l1", "psnr", "ssim", "delta2"},
        )
        self.assertEqual(
            decision["decisions"],
            {
                "spatial_context_pass": True,
                "full_context_pass": True,
                "delta_keep_pass": True,
                "selected_candidate": ARM_DELTA,
                "delta_status": "KEEP",
                "training_gate": "PASS",
                "intervention_gate": "PENDING",
                "promotion_status": "PENDING_INTERVENTION",
            },
        )

    def test_delta_is_dropped_without_post_hoc_threshold_relaxation(self):
        center, spatial, delta = _write_valid_triad(
            self.root, delta_primary=0.03058
        )
        decision = compare_a2_v2_epoch010(center, spatial, delta)

        delta_contrast = decision["contrasts"]["delta_vs_spatial"]
        self.assertGreater(delta_contrast["relative_improvement_pct"], 0.0)
        self.assertLess(delta_contrast["relative_improvement_pct"], 0.5)
        self.assertIs(
            delta_contrast["gate_conditions"][
                "minimum_relative_improvement"
            ]["pass"],
            False,
        )
        self.assertIs(decision["decisions"]["delta_keep_pass"], False)
        self.assertEqual(decision["decisions"]["selected_candidate"], ARM_SPATIAL)
        self.assertEqual(decision["decisions"]["delta_status"], "DROP")
        self.assertEqual(decision["decisions"]["training_gate"], "PASS")

    def test_mismatched_frozen_base_invalidates_the_triad(self):
        center, spatial, delta = _write_valid_triad(self.root)
        path = delta / "patient_metrics_epoch010.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["P001"]["base_l1"] = 0.047
        path.write_text(json.dumps(metrics), encoding="utf-8")

        decision = compare_a2_v2_epoch010(center, spatial, delta)
        self.assertEqual(decision["input_status"], "INVALID")
        self.assertEqual(decision["decisions"]["training_gate"], "INVALID")
        self.assertTrue(
            any(
                "base metric base_l1 differs" in item
                for item in decision["invalid_reasons"]
            )
        )

    def test_cli_writes_machine_readable_decision_without_threshold_arguments(self):
        center, spatial, delta = _write_valid_triad(self.root)
        output = self.root / "decision" / "a2v2_epoch010_decision.json"
        code = main(
            [
                "--center-dir",
                str(center),
                "--spatial-dir",
                str(spatial),
                "--delta-dir",
                str(delta),
                "--output",
                str(output),
            ]
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(payload["schema_version"], "a2v2-stat-decision-1.0")
        self.assertEqual(
            payload["decisions"]["promotion_status"], "PENDING_INTERVENTION"
        )


if __name__ == "__main__":
    unittest.main()
