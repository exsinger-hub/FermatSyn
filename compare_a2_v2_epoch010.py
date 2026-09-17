"""Apply the frozen A2V2-STATS-v1 decision rule to Epoch-10 results.

The program is intentionally read-only with respect to experiment directories.
It reads one manifest and one patient-level metrics file from each preregistered
arm, then writes a single machine-readable decision JSON.  All scientific
thresholds are constants: the command line can select inputs and output only.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "a2v2-stat-decision-1.0"
PROTOCOL_ID = "A2V2-STATS-v1"
PROTOCOL_FROZEN_AT_UTC = "2026-08-29T06:55:22Z"
FREEZE_EVIDENCE = "remote_confirmation_no_epoch1_metrics"
FIRST_EPOCH_AT_UTC = "2026-08-29T06:59:43.478356778Z"

EPOCH = 10
REQUIRED_PATIENTS = 11
PRIMARY_METRIC = "a2_delta_l1_anatomy"
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 2026
BOOTSTRAP_CI = 0.95
CONTEXT_MIN_RELATIVE_PERCENT = 1.0
DELTA_KEEP_MIN_RELATIVE_PERCENT = 0.5
WILCOXON_ALPHA = 0.05
MIN_BETTER_PATIENTS = 8
BASE_RTOL = 1e-6
BASE_ATOL = 1e-8

ARM_CENTER = "center_pm"
ARM_SPATIAL = "axial_spatial"
ARM_DELTA = "axial_delta_k005"
ARM_ORDER = (ARM_CENTER, ARM_SPATIAL, ARM_DELTA)

REQUIRED_METRICS = (
    "a2_delta_l1_anatomy",
    "a2_delta2_l1_anatomy",
    "a2_l1",
    "a2_psnr",
    "a2_ssim",
    "base_delta_l1_anatomy",
    "base_delta2_l1_anatomy",
    "base_l1",
    "base_psnr",
    "base_ssim",
)
BASE_METRICS = tuple(name for name in REQUIRED_METRICS if name.startswith("base_"))

BASE_GUARD_LIMITS = {
    "l1_max_relative_increase": 0.01,
    "psnr_max_absolute_drop_db": 0.10,
    "ssim_max_absolute_drop": 0.001,
}
CONTROL_GUARD_LIMITS = {
    "l1_max_relative_increase": 0.002,
    "psnr_max_absolute_drop_db": 0.05,
    "ssim_max_absolute_drop": 0.0005,
    "delta2_max_relative_increase": 0.005,
}

ARGUMENT_DIFFERENCES_ALLOWED = {
    "output_dir",
    "input_mode",
    "delta_kappa",
    "delta_weight",
}


def _read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _arm_paths(run_dir):
    root = Path(run_dir).resolve()
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "patient_metrics": root / "patient_metrics_epoch010.json",
    }


def _load_arm(arm, run_dir, reasons):
    paths = _arm_paths(run_dir)
    result = {"arm": arm, "paths": {key: str(value) for key, value in paths.items()}}
    for name in ("manifest", "patient_metrics"):
        path = paths[name]
        if not path.is_file():
            reasons.append("%s missing %s: %s" % (arm, name, path))
            continue
        try:
            result[name] = _read_json(path)
        except (OSError, ValueError, TypeError) as error:
            reasons.append("%s invalid %s: %s" % (arm, name, error))
    return result


def _canonical_common_arguments(arguments):
    return {
        key: value
        for key, value in arguments.items()
        if key not in ARGUMENT_DIFFERENCES_ALLOWED
    }


def _validate_manifest(arms, reasons):
    manifests = {}
    for arm in ARM_ORDER:
        manifest = arms[arm].get("manifest")
        if not isinstance(manifest, dict):
            reasons.append("%s manifest must be an object" % arm)
            continue
        manifests[arm] = manifest

    if len(manifests) != len(ARM_ORDER):
        return

    expected_modes = {
        ARM_CENTER: "center_only",
        ARM_SPATIAL: "neighbor_difference",
        ARM_DELTA: "neighbor_difference",
    }
    common_arguments = None
    common_options = None
    commits = []
    parameter_counts = []
    val_windows = []

    for arm in ARM_ORDER:
        manifest = manifests[arm]
        arguments = manifest.get("arguments")
        if not isinstance(arguments, dict):
            reasons.append("%s manifest.arguments must be an object" % arm)
            continue

        required_arguments = {
            "seed": 2026,
            "epochs": 10,
            "window_size": 5,
            "val_stride": 1,
            "no_amp": True,
        }
        for name, expected in required_arguments.items():
            if arguments.get(name) != expected:
                reasons.append(
                    "%s arguments.%s must equal %r (got %r)"
                    % (arm, name, expected, arguments.get(name))
                )
        if arguments.get("input_mode") != expected_modes[arm]:
            reasons.append(
                "%s input_mode must equal %s" % (arm, expected_modes[arm])
            )

        calibration = manifest.get("calibration")
        delta_weight = None
        if isinstance(calibration, dict):
            delta_weight = calibration.get("delta_weight")
        if not _finite_number(delta_weight):
            reasons.append("%s calibration.delta_weight must be finite" % arm)
        elif arm == ARM_DELTA and float(delta_weight) <= 0.0:
            reasons.append("%s must have a positive delta weight" % arm)
        elif arm != ARM_DELTA and float(delta_weight) != 0.0:
            reasons.append("%s must have delta weight exactly zero" % arm)

        environment = manifest.get("environment")
        if not isinstance(environment, dict) or environment.get("precision") != "FP32":
            reasons.append("%s must record FP32 precision" % arm)

        repo = manifest.get("repo")
        if not isinstance(repo, dict):
            reasons.append("%s manifest.repo must be an object" % arm)
        else:
            commit = repo.get("commit")
            if not isinstance(commit, str) or not commit:
                reasons.append("%s must record a git commit" % arm)
            else:
                commits.append(commit)
            if repo.get("tracked_dirty") is not False:
                reasons.append("%s must have repo.tracked_dirty=false" % arm)

        parameter_count = manifest.get("adapter_parameters")
        if not isinstance(parameter_count, int) or isinstance(parameter_count, bool):
            reasons.append("%s adapter_parameters must be an integer" % arm)
        elif parameter_count <= 0:
            reasons.append("%s adapter_parameters must be positive" % arm)
        else:
            parameter_counts.append(parameter_count)

        current_val_windows = manifest.get("val_windows")
        if not isinstance(current_val_windows, int) or current_val_windows <= 0:
            reasons.append("%s val_windows must be a positive integer" % arm)
        else:
            val_windows.append(current_val_windows)
        if manifest.get("validation_uses_unique_center_only") is not True:
            reasons.append("%s must validate unique centers only" % arm)
        if manifest.get("output_indices") != [1, 2, 3]:
            reasons.append("%s output_indices must equal [1, 2, 3]" % arm)

        normalized_arguments = _canonical_common_arguments(arguments)
        if common_arguments is None:
            common_arguments = normalized_arguments
        elif normalized_arguments != common_arguments:
            reasons.append("%s has non-preregistered argument differences" % arm)

        options = manifest.get("source_generator_options")
        if not isinstance(options, dict):
            reasons.append("%s source_generator_options must be an object" % arm)
        elif common_options is None:
            common_options = options
        elif options != common_options:
            reasons.append("%s source_generator_options differ across arms" % arm)

    if commits and (len(commits) != len(ARM_ORDER) or len(set(commits)) != 1):
        reasons.append("all arms must use the same git commit")
    if parameter_counts and (
        len(parameter_counts) != len(ARM_ORDER) or len(set(parameter_counts)) != 1
    ):
        reasons.append("all arms must have exactly equal adapter parameter counts")
    if val_windows and (
        len(val_windows) != len(ARM_ORDER) or len(set(val_windows)) != 1
    ):
        reasons.append("all arms must have the same validation window count")


def _validate_patient_metrics(arms, reasons):
    patient_sets = {}
    for arm in ARM_ORDER:
        metrics = arms[arm].get("patient_metrics")
        if not isinstance(metrics, dict):
            reasons.append("%s patient metrics must be an object" % arm)
            continue
        patient_sets[arm] = set(metrics)
        if len(metrics) != REQUIRED_PATIENTS:
            reasons.append(
                "%s must contain exactly %d patients (got %d)"
                % (arm, REQUIRED_PATIENTS, len(metrics))
            )
        for patient, record in metrics.items():
            if not isinstance(record, dict):
                reasons.append("%s/%s metrics must be an object" % (arm, patient))
                continue
            for metric in REQUIRED_METRICS:
                value = record.get(metric)
                if not _finite_number(value):
                    reasons.append(
                        "%s/%s missing finite metric %s" % (arm, patient, metric)
                    )

    if len(patient_sets) != len(ARM_ORDER):
        return
    first = patient_sets[ARM_CENTER]
    for arm in (ARM_SPATIAL, ARM_DELTA):
        if patient_sets[arm] != first:
            reasons.append("patient sets differ between %s and %s" % (ARM_CENTER, arm))
    if reasons:
        return

    patients = sorted(first)
    for metric in BASE_METRICS:
        reference = np.asarray(
            [arms[ARM_CENTER]["patient_metrics"][patient][metric] for patient in patients],
            dtype=np.float64,
        )
        for arm in (ARM_SPATIAL, ARM_DELTA):
            candidate = np.asarray(
                [arms[arm]["patient_metrics"][patient][metric] for patient in patients],
                dtype=np.float64,
            )
            if not np.allclose(candidate, reference, rtol=BASE_RTOL, atol=BASE_ATOL):
                reasons.append("base metric %s differs for arm %s" % (metric, arm))


def _bootstrap_indices(patient_count):
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return generator.integers(
        0,
        patient_count,
        size=(BOOTSTRAP_REPS, patient_count),
        endpoint=False,
    )


def exact_wilcoxon_signed_rank(differences):
    """Return an enumerated, two-sided Wilcoxon signed-rank result."""

    values = np.asarray(differences, dtype=np.float64)
    values = values[values != 0.0]
    count = int(values.size)
    if count == 0:
        return {
            "method": "enumerated_signed_rank_two_sided",
            "n_nonzero": 0,
            "T_plus": 0.0,
            "T_minus": 0.0,
            "W_min": 0.0,
            "p_exact": 1.0,
        }

    absolute = np.abs(values)
    order = np.argsort(absolute, kind="mergesort")
    ranks = np.empty(count, dtype=np.float64)
    start = 0
    while start < count:
        end = start + 1
        while end < count and absolute[order[end]] == absolute[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    t_plus = float(ranks[values > 0.0].sum())
    rank_sum = float(ranks.sum())
    t_minus = rank_sum - t_plus
    center = rank_sum / 2.0
    observed_deviation = abs(t_plus - center)
    extreme = 0
    assignments = 1 << count
    for assignment in range(assignments):
        permuted_t_plus = 0.0
        for index, rank in enumerate(ranks):
            if assignment & (1 << index):
                permuted_t_plus += float(rank)
        if abs(permuted_t_plus - center) >= observed_deviation - 1e-12:
            extreme += 1

    return {
        "method": "enumerated_signed_rank_two_sided",
        "n_nonzero": count,
        "T_plus": t_plus,
        "T_minus": float(t_minus),
        "W_min": float(min(t_plus, t_minus)),
        "p_exact": float(extreme / assignments),
    }


def _atomic_guard(name, candidate, reference, limit, operator, passed):
    return {
        "name": name,
        "candidate_value": float(candidate),
        "reference_value": float(reference),
        "limit": float(limit),
        "operator": operator,
        "pass": bool(passed),
    }


def _base_guard(summary):
    atoms = {
        "l1": _atomic_guard(
            "a2_l1_vs_frozen",
            summary["a2_l1"],
            summary["base_l1"],
            summary["base_l1"] * 1.01,
            "<=",
            summary["a2_l1"] <= summary["base_l1"] * 1.01,
        ),
        "psnr": _atomic_guard(
            "a2_psnr_vs_frozen",
            summary["a2_psnr"],
            summary["base_psnr"],
            summary["base_psnr"] - 0.10,
            ">=",
            summary["a2_psnr"] >= summary["base_psnr"] - 0.10,
        ),
        "ssim": _atomic_guard(
            "a2_ssim_vs_frozen",
            summary["a2_ssim"],
            summary["base_ssim"],
            summary["base_ssim"] - 0.001,
            ">=",
            summary["a2_ssim"] >= summary["base_ssim"] - 0.001,
        ),
    }
    return {"atoms": atoms, "all_pass": all(atom["pass"] for atom in atoms.values())}


def _control_guard(candidate, control):
    atoms = {
        "l1": _atomic_guard(
            "candidate_l1_vs_control",
            candidate["a2_l1"],
            control["a2_l1"],
            control["a2_l1"] * 1.002,
            "<=",
            candidate["a2_l1"] <= control["a2_l1"] * 1.002,
        ),
        "psnr": _atomic_guard(
            "candidate_psnr_vs_control",
            candidate["a2_psnr"],
            control["a2_psnr"],
            control["a2_psnr"] - 0.05,
            ">=",
            candidate["a2_psnr"] >= control["a2_psnr"] - 0.05,
        ),
        "ssim": _atomic_guard(
            "candidate_ssim_vs_control",
            candidate["a2_ssim"],
            control["a2_ssim"],
            control["a2_ssim"] - 0.0005,
            ">=",
            candidate["a2_ssim"] >= control["a2_ssim"] - 0.0005,
        ),
        "delta2": _atomic_guard(
            "candidate_delta2_vs_control",
            candidate["a2_delta2_l1_anatomy"],
            control["a2_delta2_l1_anatomy"],
            control["a2_delta2_l1_anatomy"] * 1.005,
            "<=",
            candidate["a2_delta2_l1_anatomy"]
            <= control["a2_delta2_l1_anatomy"] * 1.005,
        ),
    }
    return {"atoms": atoms, "all_pass": all(atom["pass"] for atom in atoms.values())}


def _arm_summary(patient_metrics, patients):
    return {
        metric: float(
            np.mean([float(patient_metrics[patient][metric]) for patient in patients])
        )
        for metric in REQUIRED_METRICS
    }


def _contrast(
    control_name,
    candidate_name,
    arms,
    patients,
    arm_summaries,
    bootstrap_indices,
    minimum_relative_percent,
):
    control_values = np.asarray(
        [arms[control_name]["patient_metrics"][p][PRIMARY_METRIC] for p in patients],
        dtype=np.float64,
    )
    candidate_values = np.asarray(
        [arms[candidate_name]["patient_metrics"][p][PRIMARY_METRIC] for p in patients],
        dtype=np.float64,
    )
    differences = control_values - candidate_values
    control_mean = float(control_values.mean())
    candidate_mean = float(candidate_values.mean())
    absolute_improvement = float(differences.mean())
    relative_improvement = float(100.0 * absolute_improvement / control_mean)

    sampled_control = control_values[bootstrap_indices].mean(axis=1)
    sampled_candidate = candidate_values[bootstrap_indices].mean(axis=1)
    if np.any(~np.isfinite(sampled_control)) or np.any(sampled_control <= 0.0):
        raise ValueError("bootstrap produced a non-positive control mean")
    sampled_absolute = sampled_control - sampled_candidate
    sampled_relative = 100.0 * sampled_absolute / sampled_control
    absolute_ci = np.quantile(sampled_absolute, [0.025, 0.975], method="linear")
    relative_ci = np.quantile(sampled_relative, [0.025, 0.975], method="linear")

    wilcoxon = exact_wilcoxon_signed_rank(differences)
    n_better = int(np.sum(differences > 0.0))
    n_equal = int(np.sum(differences == 0.0))
    n_worse = int(np.sum(differences < 0.0))

    candidate_base_guard = _base_guard(arm_summaries[candidate_name])
    control_base_guard = _base_guard(arm_summaries[control_name])
    candidate_control_guard = _control_guard(
        arm_summaries[candidate_name], arm_summaries[control_name]
    )
    guards_pass = (
        candidate_base_guard["all_pass"]
        and control_base_guard["all_pass"]
        and candidate_control_guard["all_pass"]
    )
    conditions = {
        "minimum_relative_improvement": {
            "value": relative_improvement,
            "threshold": float(minimum_relative_percent),
            "operator": ">=",
            "pass": relative_improvement >= minimum_relative_percent,
        },
        "bootstrap_absolute_ci_lower_positive": {
            "value": float(absolute_ci[0]),
            "threshold": 0.0,
            "operator": ">",
            "pass": float(absolute_ci[0]) > 0.0,
        },
        "exact_wilcoxon": {
            "value": float(wilcoxon["p_exact"]),
            "threshold": WILCOXON_ALPHA,
            "operator": "<",
            "pass": float(wilcoxon["p_exact"]) < WILCOXON_ALPHA,
        },
        "minimum_better_patients": {
            "value": n_better,
            "threshold": MIN_BETTER_PATIENTS,
            "operator": ">=",
            "pass": n_better >= MIN_BETTER_PATIENTS,
        },
        "all_guards": {
            "value": guards_pass,
            "threshold": True,
            "operator": "is",
            "pass": guards_pass,
        },
    }
    gate_pass = all(condition["pass"] for condition in conditions.values())

    return {
        "control": control_name,
        "candidate": candidate_name,
        "metric": PRIMARY_METRIC,
        "direction": "lower_is_better",
        "patient_differences": {
            patient: float(value) for patient, value in zip(patients, differences)
        },
        "control_mean": control_mean,
        "candidate_mean": candidate_mean,
        "absolute_improvement": absolute_improvement,
        "relative_improvement_pct": relative_improvement,
        "bootstrap": {
            "repetitions": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "generator": "PCG64",
            "common_resample_indices": True,
            "absolute_improvement_ci95": [float(value) for value in absolute_ci],
            "relative_improvement_pct_ci95": [float(value) for value in relative_ci],
        },
        "n_better": n_better,
        "n_equal": n_equal,
        "n_worse": n_worse,
        "wilcoxon": wilcoxon,
        "guards": {
            "candidate_vs_frozen": candidate_base_guard,
            "control_vs_frozen": control_base_guard,
            "candidate_vs_control": candidate_control_guard,
            "all_pass": guards_pass,
        },
        "gate_conditions": conditions,
        "pass": gate_pass,
    }


def _constants():
    return {
        "epoch": EPOCH,
        "primary_metric": PRIMARY_METRIC,
        "required_patients": REQUIRED_PATIENTS,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_generator": "PCG64",
        "bootstrap_ci": BOOTSTRAP_CI,
        "context_min_relative_improvement_pct": CONTEXT_MIN_RELATIVE_PERCENT,
        "delta_keep_min_relative_improvement_pct": DELTA_KEEP_MIN_RELATIVE_PERCENT,
        "wilcoxon_alpha": WILCOXON_ALPHA,
        "minimum_better_patients": MIN_BETTER_PATIENTS,
        "base_guard_limits": BASE_GUARD_LIMITS,
        "control_guard_limits": CONTROL_GUARD_LIMITS,
        "thresholds_cli_overridable": False,
    }


def _invalid_decision(arms, reasons):
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_frozen_at_utc": PROTOCOL_FROZEN_AT_UTC,
        "first_epoch_at_utc": FIRST_EPOCH_AT_UTC,
        "freeze_evidence": FREEZE_EVIDENCE,
        "preregistered": True,
        "input_status": "INVALID",
        "invalid_reasons": sorted(set(reasons)),
        "inputs": {arm: arms[arm]["paths"] for arm in ARM_ORDER},
        "constants": _constants(),
        "arm_summaries": {},
        "contrasts": {},
        "decisions": {
            "spatial_context_pass": False,
            "full_context_pass": False,
            "delta_keep_pass": False,
            "selected_candidate": None,
            "delta_status": "DROP",
            "training_gate": "INVALID",
            "intervention_gate": "PENDING",
            "promotion_status": "INVALID",
        },
    }


def compare_a2_v2_epoch010(center_dir, spatial_dir, delta_dir):
    reasons = []
    arms = {
        ARM_CENTER: _load_arm(ARM_CENTER, center_dir, reasons),
        ARM_SPATIAL: _load_arm(ARM_SPATIAL, spatial_dir, reasons),
        ARM_DELTA: _load_arm(ARM_DELTA, delta_dir, reasons),
    }
    _validate_manifest(arms, reasons)
    _validate_patient_metrics(arms, reasons)
    if reasons:
        return _invalid_decision(arms, reasons)

    patients = sorted(arms[ARM_CENTER]["patient_metrics"])
    arm_summaries = {
        arm: _arm_summary(arms[arm]["patient_metrics"], patients)
        for arm in ARM_ORDER
    }
    bootstrap_indices = _bootstrap_indices(len(patients))
    try:
        spatial_vs_center = _contrast(
            ARM_CENTER,
            ARM_SPATIAL,
            arms,
            patients,
            arm_summaries,
            bootstrap_indices,
            CONTEXT_MIN_RELATIVE_PERCENT,
        )
        delta_vs_center = _contrast(
            ARM_CENTER,
            ARM_DELTA,
            arms,
            patients,
            arm_summaries,
            bootstrap_indices,
            CONTEXT_MIN_RELATIVE_PERCENT,
        )
        delta_vs_spatial = _contrast(
            ARM_SPATIAL,
            ARM_DELTA,
            arms,
            patients,
            arm_summaries,
            bootstrap_indices,
            DELTA_KEEP_MIN_RELATIVE_PERCENT,
        )
    except ValueError as error:
        reasons.append(str(error))
        return _invalid_decision(arms, reasons)

    spatial_pass = bool(spatial_vs_center["pass"])
    full_pass = bool(delta_vs_center["pass"])
    delta_keep_pass = bool(delta_vs_spatial["pass"])
    if full_pass and delta_keep_pass:
        selected = ARM_DELTA
        delta_status = "KEEP"
        training_gate = "PASS"
    elif spatial_pass:
        selected = ARM_SPATIAL
        delta_status = "DROP"
        training_gate = "PASS"
    else:
        selected = None
        delta_status = "DROP"
        training_gate = "FAIL"

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_frozen_at_utc": PROTOCOL_FROZEN_AT_UTC,
        "first_epoch_at_utc": FIRST_EPOCH_AT_UTC,
        "freeze_evidence": FREEZE_EVIDENCE,
        "preregistered": True,
        "input_status": "VALID",
        "invalid_reasons": [],
        "inputs": {arm: arms[arm]["paths"] for arm in ARM_ORDER},
        "constants": _constants(),
        "patients": patients,
        "arm_summaries": arm_summaries,
        "contrasts": {
            "spatial_vs_center": spatial_vs_center,
            "delta_vs_center": delta_vs_center,
            "delta_vs_spatial": delta_vs_spatial,
        },
        "decisions": {
            "spatial_context_pass": spatial_pass,
            "full_context_pass": full_pass,
            "delta_keep_pass": delta_keep_pass,
            "selected_candidate": selected,
            "delta_status": delta_status,
            "training_gate": training_gate,
            "intervention_gate": "PENDING",
            "promotion_status": (
                "PENDING_INTERVENTION" if training_gate == "PASS" else "FAIL"
            ),
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center-dir", required=True)
    parser.add_argument("--spatial-dir", required=True)
    parser.add_argument("--delta-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    decision = compare_a2_v2_epoch010(
        args.center_dir,
        args.spatial_dir,
        args.delta_dir,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0 if decision["input_status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
