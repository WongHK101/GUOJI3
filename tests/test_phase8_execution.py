from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in [PROJECT_ROOT, SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase8_protocol import file_sha256, load_json, load_run_matrix  # noqa: E402
from phase8_results import aggregate_pilot, aggregate_replication, validate_gpu_preflight  # noqa: E402
from experiments.phase8_gpu_runner import generate_var1_data  # noqa: E402


CONFIG = PROJECT_ROOT / "configs" / "phase8" / "phase8_execution_lock.json"
MATRIX = PROJECT_ROOT / "configs" / "phase8" / "phase8_run_matrix.csv"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric_payload(row: dict) -> dict:
    method = row["method"]
    partial_auroc = 0.90
    total_auroc = 0.90
    auprc = 0.60
    coefficient_partial = 0.90
    coefficient_total = 0.90
    mse = 0.10
    if row["block"] == "capacity_replication" and method != "baseline_jrngc":
        d_cond = int(method.rsplit("_", 1)[1])
        partial_auroc = 0.88 if d_cond < 4 else 0.65
        total_auroc = 0.89
        mse = 0.095 if d_cond < 4 else 0.08
    if row["block"] == "coefficient_replication" and method == "concat_x_only":
        partial_auroc = 0.55
        total_auroc = 0.85
        coefficient_partial = 0.30
        coefficient_total = 0.80
    if row["block"] == "repair_pilot":
        values = {
            "baseline_jrngc": (0.65, 0.60, 0.50, 0.100),
            "concat_x_only": (0.50, 0.50, 0.30, 0.100),
            "full_aux_equal_lambda": (0.62, 0.58, 0.48, 0.102),
            "full_aux_lc10": (0.61, 0.57, 0.47, 0.103),
            "coverage_aligned_raw_chain": (0.60, 0.55, 0.40, 0.105),
        }
        total_auroc, auprc, coefficient_total, mse = values[method]
        partial_auroc = total_auroc - (0.10 if method == "concat_x_only" else 0.0)
    payload = {
        "fixed_target_prediction_mse": mse,
        "jacobian_penalty": 0.01,
        "total_regularized_objective": mse + 0.01,
        "graph_metrics": {
            "partial_nominal": {"auroc": partial_auroc, "auprc": auprc},
            "total_nominal": {"auroc": total_auroc, "auprc": auprc},
        },
        "coefficient_r_partial_lag1": coefficient_partial,
        "coefficient_r_total_lag1": coefficient_total,
        "nominal_partial_total_pearson": 0.80 if method == "concat_x_only" else 1.0,
        "nominal_partial_total_topk_jaccard": 0.70 if method == "concat_x_only" else 1.0,
        "M_missing": 0.10 if method == "concat_x_only" else 0.0,
    }
    if row["block"] == "fixed_target_interventions":
        if method == "concat_fixed_target_interventions":
            interventions = {
                "fixed_target_prediction_mse_delta": {
                    "clean": 0.0,
                    "mask_x": 0.1,
                    "mask_c": 0.4,
                    "mask_both": 0.5,
                    "shuffle_x_only": 0.1,
                    "shuffle_c_only": 0.3,
                    "shuffle_both_routes": 0.5,
                },
                "legacy_objective_delta": {"clean": 0.0, "mask_x": 0.2},
            }
        else:
            interventions = {
                "fixed_target_prediction_mse_delta": {"clean": 0.0, "mask_x": 0.5, "shuffle_x": 0.4},
                "control_status": "new_matched_input_space_control_not_legacy_replication",
            }
        payload["interventions"] = interventions
    return payload


def _create_run(root: Path, row: dict, metrics: dict | None = None, extras: dict[str, object] | None = None) -> None:
    run_dir = root / "runs" / row["record_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        **row,
        "formal_result": row["formal_result"].lower() == "true",
        "data_seed": int(row["data_seed"]),
        "model_seed": int(row["model_seed"]),
        "perturbation_seed": None if not row["perturbation_seed"] else int(row["perturbation_seed"]),
        "gating_checkpoint": "final" if row["block"] == "repair_pilot" else "restored_legacy_best",
        "config_sha256": file_sha256(CONFIG),
        "run_matrix_sha256": file_sha256(MATRIX),
        "execution_device": "cuda",
        "deterministic_settings": {"torch_deterministic_algorithms": True},
        "source_release": {
            "approved_commit": "test-release",
            "actual_commit": None,
            "source_manifest_sha256": "test-manifest",
        },
    }
    _write_json(run_dir / "metrics.json", metrics or _metric_payload(row))
    _write_json(run_dir / "resolved_config.json", resolved)
    _write_json(run_dir / "runtime.json", {
        "training_seconds": 1.0,
        "evaluation_seconds": 1.0,
        "total_seconds": 2.0,
        "projected_2000_iteration_training_seconds": 100.0,
        "peak_reserved_fraction": 0.01,
    })
    for name, value in (extras or {}).items():
        path = run_dir / name
        if name.endswith(".json"):
            _write_json(path, value)
        else:
            torch.save(value, path)
    files = {}
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name not in {"artifact_manifest.json", "status.json"}:
            files[path.name] = file_sha256(path)
    _write_json(run_dir / "artifact_manifest.json", {"files": files})
    _write_json(run_dir / "status.json", {
        "status": "complete",
        "output_complete": True,
        "no_nan_inf": True,
        "formal_result": resolved["formal_result"],
    })


def test_replication_aggregator_uses_partial_tracks_and_fixed_target_mse(tmp_path):
    rows = [row for row in load_run_matrix(MATRIX) if row["block"] in {
        "capacity_replication", "fixed_target_interventions", "coefficient_replication"
    }]
    for row in rows:
        _create_run(tmp_path, row)
    report = aggregate_replication(tmp_path, config_path=CONFIG, matrix_path=MATRIX)
    assert report["record_count"] == 50
    assert report["all_three_replicated"]
    assert set(report["classifications"].values()) == {"REPLICATED"}
    assert report["capacity"]["pairs"][0]["baseline"]["partial_nominal_auroc"] == 0.90
    assert "total_nominal_auroc" in report["capacity"]["pairs"][0]["baseline"]
    assert report["fixed_target_interventions"]["pairs"][0]["new_no_aux_control"]["control_status"].startswith("new_")


def test_pilot_aggregator_applies_total_nominal_and_lag1_gates(tmp_path):
    rows = [row for row in load_run_matrix(MATRIX) if row["block"] == "repair_pilot"]
    for row in rows:
        _create_run(tmp_path, row)
    report = aggregate_pilot(
        tmp_path,
        config_path=CONFIG,
        matrix_path=MATRIX,
        preflight_report={"passed": True},
        replication_report={"all_three_replicated": True},
    )
    assert report["record_count"] == 30
    assert report["pilot_go_passed"]
    assert report["confirmation_eligible"]
    assert report["confirmation_executed"] is False
    assert np.isclose(report["repair_minus_concat"]["means"]["delta_auroc"], 0.10)


def test_gpu_preflight_validator_checks_duplicate_and_hard_stops(tmp_path):
    rows = [row for row in load_run_matrix(MATRIX) if row["phase"] == "preflight"]
    state = {"weight": torch.tensor([1.0, 2.0])}
    checkpoint = {
        "model_state_dict": state,
        "objective_trace_first20": [1.0 - index * 0.01 for index in range(20)],
    }
    schedule = [{"lag_ids": [1, 2], "target_indices": [2, 3], "output_targets": [0, 1]} for _ in range(100)]
    for row in rows:
        metrics = _metric_payload(row)
        extras = {}
        if row["record_id"] in {"P8-PRE-004", "P8-PRE-005"}:
            extras["checkpoint_iter20.pt"] = checkpoint
            count = 20 if row["record_id"] == "P8-PRE-004" else 100
            extras["jacobian_schedule.json"] = {"entries": schedule[:count], "seed": 32001, "sha256": "test"}
        if row["record_id"] == "P8-PRE-005":
            scale_snapshot = {
                "fixed_target_prediction_mse": 0.90,
                "output_target_variance_ratio": 0.5,
                "nominal_jacobian_penalty": 0.01,
                "nominal_regularizer_predictor_gradient_norm": 0.1,
                "nominal_regularizer_preprocessor_gradient_norm": 0.1,
                "predictor_gradient_reachable": True,
                "preprocessor_gradient_reachable": True,
                "gradients_finite": True,
            }
            metrics["regularizer_scale_preflight"] = {
                "initial": {"fixed_target_prediction_mse": 1.0},
                "final": scale_snapshot,
                "by_completed_iteration": {
                    str(point): scale_snapshot for point in [0, 20, 40, 60, 80, 100]
                },
                "pure_mse_relative_change": -0.10,
            }
            iteration_rows = [
                {
                    "completed_iteration": index + 1,
                    "historical_stratum": ["B1", "B2", "B3"][index % 3],
                    "historical_lag": 2 + index,
                }
                for index in range(100)
            ]
            metrics["stratified_benchmark_trace"] = {
                "iterations": iteration_rows,
                "strata": {
                    "B1": {"sample_count": 34, "nonzero_float32_contribution_count": 20},
                    "B2": {"sample_count": 33, "nonzero_float32_contribution_count": 10},
                    "B3": {"sample_count": 33, "nonzero_float32_contribution_count": 0},
                },
                "cumulative_historical_contribution": 0.1,
                "cumulative_historical_predictor_gradient_norm": 0.2,
                "cumulative_historical_preprocessor_gradient_norm": 0.3,
                "all_strata_sampled": True,
            }
        _create_run(tmp_path, row, metrics=metrics, extras=extras)
    report = validate_gpu_preflight(
        tmp_path,
        config_path=CONFIG,
        matrix_path=MATRIX,
        cpu_preflight_summary={"passed": True},
    )
    assert report["passed"]
    assert report["determinism"]["checkpoint_state_max_abs_difference"] == 0.0
    assert report["compute_projection"]["includes_confirmation_cost_projection_only"] is True
    assert report["compute_projection"]["confirmation_execution_authorized"] is False

    # Long-stratum zero draws are disclosed but allowed; a zero-only medium
    # stratum must fail the revised aggregate gate.
    metrics_path = tmp_path / "runs" / "P8-PRE-005" / "metrics.json"
    changed = load_json(metrics_path)
    changed["stratified_benchmark_trace"]["strata"]["B2"]["nonzero_float32_contribution_count"] = 0
    _write_json(metrics_path, changed)
    artifact_path = metrics_path.parent / "artifact_manifest.json"
    artifact = load_json(artifact_path)
    artifact["files"]["metrics.json"] = file_sha256(metrics_path)
    _write_json(artifact_path, artifact)
    rejected = validate_gpu_preflight(
        tmp_path,
        config_path=CONFIG,
        matrix_path=MATRIX,
        cpu_preflight_summary={"passed": True},
    )
    assert not rejected["passed"]
    assert any(
        failure["gate"] == "historical_stratum_nonzero_float32_draw" and failure["stratum"] == "B2"
        for failure in rejected["failures"]
    )


def test_confirmation_rows_are_never_selected_by_authorized_stage_sets():
    rows = load_run_matrix(MATRIX)
    selected_blocks = {
        "infrastructure_smoke",
        "repair_scale_benchmark",
        "capacity_replication",
        "fixed_target_interventions",
        "coefficient_replication",
        "repair_pilot",
    }
    selected = [row for row in rows if row["block"] in selected_blocks]
    assert len(selected) == 85
    assert all(row["phase"] != "gated_confirmation" for row in selected)


def _legacy_var1(seed: int, *, numpy_seed_offset: int):
    torch.manual_seed(seed)
    A = torch.randn(4, 4) * (torch.rand(4, 4) < 0.3).float()
    sr = torch.linalg.eigvals(A).abs().max()
    if numpy_seed_offset == 0:
        A = A * (0.8 / sr)
    else:
        A = A * (0.8 / max(float(sr), 0.01))
    A_np = A.numpy()
    np.random.seed(seed + numpy_seed_offset)
    x = np.zeros((4, 20))
    x[:, 0] = np.random.randn(4) * 0.1
    for t in range(1, 20):
        x[:, t] = A_np @ x[:, t - 1] + np.random.randn(4) * 0.1
    return x, (np.abs(A_np) > 0.01).astype(np.float64), A_np


def test_block_specific_var_generators_match_frozen_legacy_numerics():
    for offset, name in [(0, "legacy_shortcut"), (1, "legacy_full_aux")]:
        expected = _legacy_var1(711, numpy_seed_offset=offset)
        actual = generate_var1_data(
            4,
            20,
            711,
            numpy_seed_offset=offset,
            generator_name=name,
        )
        for expected_array, actual_array in zip(expected, actual[:3]):
            assert np.array_equal(expected_array, actual_array)
