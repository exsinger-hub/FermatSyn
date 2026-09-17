import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")

from models.scan_paths import generate_fermat_indices
from scripts.analyze_scan_endpoints import (
    endpoint_metrics,
    generate_lambda0_endpoint,
    generate_lambda1_endpoint,
    run_experiment,
)


def test_lambda0_fast_endpoint_matches_reference_pointwise_on_small_grids():
    for H, W in [(3, 5), (8, 8), (17, 29)]:
        expected = generate_fermat_indices(H, W, lambda_c=0.0).detach().cpu().numpy()
        actual = generate_lambda0_endpoint(H, W)
        np.testing.assert_array_equal(actual, expected)


def test_lambda1_fast_endpoint_matches_reference_pointwise_on_small_grids():
    for H, W in [(3, 5), (8, 8), (13, 17)]:
        expected = generate_fermat_indices(H, W, lambda_c=1.0).detach().cpu().numpy()
        actual = generate_lambda1_endpoint(H, W)
        np.testing.assert_array_equal(actual, expected)


def test_lambda1_has_unit_steps_and_tight_triangle_bound():
    H, W = 64, 64
    path = generate_lambda1_endpoint(H, W)
    metrics = endpoint_metrics(path, H, W)
    assert metrics["all_steps_equal_one"] is True
    assert metrics["delta_le_k"] is True
    assert metrics["delta_equals_k_any_after_start"] is True
    assert metrics["delta_equals_k_count_after_start"] > 0


def test_endpoint_anisotropy_separates_direction_and_within_sector_variance():
    path = generate_lambda0_endpoint(32, 32)
    metrics = endpoint_metrics(path, 32, 32, sectors=16)
    anis = metrics["anisotropy"]
    assert anis["sectors"] == 16
    assert len(anis["sector_counts"]) == 16
    assert len(anis["sector_mean_steps"]) == 16
    assert anis["direction_mean_variance"] >= 0.0
    assert anis["normalized_direction_anisotropy"] >= 0.0
    assert anis["within_sector_variance_mean"] >= 0.0


def test_run_experiment_writes_csv_json_npz_and_flags_missing_jacobian(tmp_path):
    args = type(
        "Args",
        (),
        {
            "height": 16,
            "width": 16,
            "dataset": "SynthRAD",
            "outdir": tmp_path,
            "sectors": 8,
            "n_candidates": 64,
            "lambdas": [0.0, 0.5, 1.0],
        },
    )()
    summary = run_experiment(args)
    assert (tmp_path / "endpoint_metrics.csv").exists()
    assert (tmp_path / "lambda_grid_metrics.csv").exists()
    assert (tmp_path / "reference_validation.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "paths.npz").exists()
    assert summary["assertions"]["spearman_sigma_J"]["status"] == "needs_jacobian_data"
    assert len(summary["lambda_grid_rows"]) == 3


def test_large_endpoint_code_does_not_call_reference_quadratic_generator():
    source = (ROOT / "scripts" / "analyze_scan_endpoints.py").read_text()
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name == "generate_fermat_indices":
                calls.append(node.lineno)
    assert len(calls) == 1


def test_endpoint_generation_is_complete_without_reference_call():
    for fn in (generate_lambda0_endpoint, generate_lambda1_endpoint):
        path = fn(32, 32)
        assert path.shape == (1024,)
        assert len(np.unique(path)) == 1024
        assert int(path.min()) == 0
        assert int(path.max()) == 1023