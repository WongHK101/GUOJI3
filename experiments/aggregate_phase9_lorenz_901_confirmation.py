"""Aggregate the frozen Phase 9 Lorenz-96 901 confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from phase9_lorenz_901_confirmation import (
    atomic_json,
    file_sha256,
    load_protocol,
    load_run_matrix,
    validate_release_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def summarize(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std_data_seed": (
            float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        ),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "count": int(array.size),
    }


def extract_train_level(run_dir: Path) -> Dict[str, object]:
    status = _load_json(run_dir / "status.json")
    run_config = _load_json(run_dir / "config.json")
    audit = _load_json(run_dir / "sampled_attribution_audit_H64.json")
    intervention = _load_json(run_dir / "fixed_target_interventions.json")
    mixing = _load_json(run_dir / "coordinate_mixing_audit.json")
    horizon = _load_json(run_dir / "horizon_sensitivity.json")
    gate_report = _load_json(run_dir / "gate_report.json")
    score_arrays = np.load(
        run_dir / "sampled_attribution_objects_H64.npz",
        allow_pickle=False,
    )
    method = status["method"]
    row = {
        "status_complete": status["status"] == "complete",
        "run_id": status["run_id"],
        "run_index": int(status["run_index"]),
        "data_seed": int(status["data_seed"]),
        "train_seed": int(status["train_seed"]),
        "method": method,
        "iterations": int(status["iterations"]),
        "selected_iteration": int(status["selected_iteration"]),
        "fixed_target_prediction_mse": float(
            status["fixed_target_prediction_mse"]
        ),
        "total_nominal_auroc": float(
            audit["total_nominal_metrics"]["auroc"]
        ),
        "total_nominal_auprc": float(
            audit["total_nominal_metrics"]["auprc"]
        ),
        "total_nominal_f1": float(
            audit["total_nominal_metrics"]["f1_exact_topk"]
        ),
        "total_nominal_mcc": float(
            audit["total_nominal_metrics"]["mcc_exact_topk"]
        ),
        "partial_nominal_auroc": float(
            audit["partial_nominal_metrics"]["auroc"]
        ),
        "partial_nominal_auprc": float(
            audit["partial_nominal_metrics"]["auprc"]
        ),
        "partial_nominal_f1": float(
            audit["partial_nominal_metrics"]["f1_exact_topk"]
        ),
        "missing_route_relative_magnitude": float(
            audit["missing_route_relative_magnitude"]
        ),
        "partial_total_nominal_pearson": float(
            audit["partial_total_nominal_pearson"]
        ),
        "partial_total_nominal_topk_jaccard": float(
            audit["partial_total_nominal_topk_jaccard"]
        ),
        "partial_total_nominal_max_abs": float(np.max(np.abs(
            score_arrays["s_total_nominal"]
            - score_arrays["s_partial_nominal"]
        ))),
        "temporal_tail_median": float(
            audit["temporal_tail_statistics"]["median"]
        ),
        "mask_c_fixed_target_mse_delta": (
            0.0
            if method == "baseline"
            else float(
                intervention["fixed_target_prediction_mse_delta"]["mask_c"]
            )
        ),
        "coordinate_entropy_median": (
            0.0
            if method == "baseline"
            else float(mixing["normalized_source_entropy"]["median"])
        ),
        "h64_h128_mass_ratio": (
            1.0
            if method == "baseline"
            else float(
                horizon["offdiagonal_cumulative_mass_ratio_vs_H128"]["64"]
            )
        ),
        "nominal_h64_h128_max_abs": (
            0.0
            if method == "baseline"
            else float(
                horizon["nominal_score_max_abs_difference_vs_H128"]["64"]
            )
        ),
        "per_run_diagnostic_gate_passed": bool(gate_report["passed"]),
        "no_nan_inf": bool(status["no_nan_inf"]),
        "deterministic_algorithms": bool(status["deterministic_algorithms"]),
        "formal_result": bool(status["formal_result"]),
        "manuscript_evidence": bool(status["manuscript_evidence"]),
        "release_commit": status["release_commit"],
        "source_manifest_sha256": status["source_manifest_sha256"],
        "target_indices": tuple(run_config["target_indices"]),
        "dataset_x_sha256": _load_json(
            run_dir / "dataset_metadata.json"
        )["x_sha256"],
        "dataset_graph_sha256": _load_json(
            run_dir / "dataset_metadata.json"
        )["graph_sha256"],
        "predictor_seed": int(run_config["seed_bundle"]["predictor_seed"]),
        "score_window_seed": int(
            run_config["seed_bundle"]["score_window_seed"]
        ),
        "perturbation_seed": int(
            run_config["seed_bundle"]["perturbation_seed"]
        ),
    }
    numeric_values = [
        value
        for value in row.values()
        if isinstance(value, float)
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise RuntimeError(f"Nonfinite train-level record: {run_dir}")
    return row


def validate_run_set(
    *,
    formal_root: Path,
    matrix: Sequence[Mapping[str, object]],
    release_lock: Mapping[str, object],
) -> tuple[list[Dict[str, object]], Dict[str, object]]:
    expected = {
        (int(row["data_seed"]), int(row["train_seed"]), str(row["method"])): row
        for row in matrix
    }
    run_dirs = sorted(
        path.parent
        for path in (formal_root / "runs").glob("*/status.json")
    )
    records = [extract_train_level(path) for path in run_dirs]
    actual = {
        (row["data_seed"], row["train_seed"], row["method"]): row
        for row in records
    }
    checks = {
        "exact_run_count": len(records) == len(matrix) == 20,
        "exact_run_matrix": set(actual) == set(expected),
        "all_complete_finite": all(
            row["status_complete"] and row["no_nan_inf"]
            for row in records
        ),
        "all_deterministic": all(
            row["deterministic_algorithms"] for row in records
        ),
        "all_formal": all(row["formal_result"] for row in records),
        "none_prematurely_manuscript_evidence": all(
            row["manuscript_evidence"] is False for row in records
        ),
        "release_commit": all(
            row["release_commit"] == release_lock["approved_commit"]
            for row in records
        ),
        "source_manifest": all(
            row["source_manifest_sha256"]
            == release_lock["source_manifest_sha256"]
            for row in records
        ),
        "baseline_partial_total_identity": all(
            float(row["partial_total_nominal_max_abs"]) <= 1e-7
            for row in records
            if row["method"] == "baseline"
        ),
        "baseline_missing_route_zero": all(
            float(row["missing_route_relative_magnitude"]) <= 1e-7
            for row in records
            if row["method"] == "baseline"
        ),
    }
    run_index_ok = True
    shared_contract_ok = True
    for key, expected_row in expected.items():
        if key not in actual:
            continue
        row = actual[key]
        run_index_ok &= row["run_index"] == int(expected_row["run_index"])
        run_index_ok &= row["iterations"] == int(expected_row["iterations"])
    for data_seed in sorted({row["data_seed"] for row in records}):
        subset = [row for row in records if row["data_seed"] == data_seed]
        shared_contract_ok &= len({row["target_indices"] for row in subset}) == 1
        shared_contract_ok &= len({row["dataset_x_sha256"] for row in subset}) == 1
        shared_contract_ok &= (
            len({row["dataset_graph_sha256"] for row in subset}) == 1
        )
        shared_contract_ok &= len({row["score_window_seed"] for row in subset}) == 1
        shared_contract_ok &= len({row["perturbation_seed"] for row in subset}) == 1
        for train_seed in sorted({row["train_seed"] for row in subset}):
            paired = [row for row in subset if row["train_seed"] == train_seed]
            shared_contract_ok &= len(paired) == 2
            shared_contract_ok &= len({row["predictor_seed"] for row in paired}) == 1
    checks["run_indices"] = bool(run_index_ok)
    checks["shared_dataset_windows_and_seeds"] = bool(shared_contract_ok)
    return records, {
        "checks": checks,
        "passed": all(checks.values()),
        "expected_run_count": len(matrix),
        "observed_run_count": len(records),
    }


def build_data_seed_rows(
    records: Sequence[Mapping[str, object]],
) -> list[Dict[str, object]]:
    data_seed_rows = []
    metric_names = (
        "fixed_target_prediction_mse",
        "total_nominal_auroc",
        "total_nominal_auprc",
        "total_nominal_f1",
        "total_nominal_mcc",
        "partial_nominal_auroc",
        "partial_nominal_auprc",
        "partial_nominal_f1",
        "missing_route_relative_magnitude",
        "partial_total_nominal_pearson",
        "partial_total_nominal_topk_jaccard",
        "temporal_tail_median",
        "mask_c_fixed_target_mse_delta",
        "coordinate_entropy_median",
        "h64_h128_mass_ratio",
        "nominal_h64_h128_max_abs",
    )
    for data_seed in sorted({int(row["data_seed"]) for row in records}):
        row: Dict[str, object] = {"data_seed": data_seed}
        method_rows = {}
        for method in ("baseline", "mamba_concat"):
            subset = [
                item
                for item in records
                if int(item["data_seed"]) == data_seed
                and item["method"] == method
            ]
            if len(subset) != 2:
                raise RuntimeError(
                    f"Expected two train seeds: data={data_seed}, method={method}"
                )
            method_rows[method] = subset
            for metric in metric_names:
                row[f"{method}_{metric}"] = _mean([
                    float(item[metric]) for item in subset
                ])
        baseline_mse = float(row["baseline_fixed_target_prediction_mse"])
        concat_mse = float(row["mamba_concat_fixed_target_prediction_mse"])
        row["concat_vs_baseline_relative_mse"] = (
            (concat_mse - baseline_mse) / max(abs(baseline_mse), 1e-12)
        )
        row["concat_discrepancy_pass"] = (
            float(row["mamba_concat_partial_total_nominal_pearson"]) <= 0.9
            or float(
                row["mamba_concat_partial_total_nominal_topk_jaccard"]
            ) <= 0.8
        )
        data_seed_rows.append(row)
    return data_seed_rows


def evaluate_aggregate_gates(
    *,
    data_seed_rows: Sequence[Mapping[str, object]],
    run_validation_passed: bool,
    gates: Mapping[str, object],
) -> Dict[str, object]:
    baseline_aurocs = [
        float(row["baseline_total_nominal_auroc"])
        for row in data_seed_rows
    ]
    missing = [
        float(row["mamba_concat_missing_route_relative_magnitude"])
        for row in data_seed_rows
    ]
    mask_c = [
        float(row["mamba_concat_mask_c_fixed_target_mse_delta"])
        for row in data_seed_rows
    ]
    entropy = [
        float(row["mamba_concat_coordinate_entropy_median"])
        for row in data_seed_rows
    ]
    tail = [
        float(row["mamba_concat_temporal_tail_median"])
        for row in data_seed_rows
    ]
    mass_ratio = [
        float(row["mamba_concat_h64_h128_mass_ratio"])
        for row in data_seed_rows
    ]
    nominal_diff = [
        float(row["mamba_concat_nominal_h64_h128_max_abs"])
        for row in data_seed_rows
    ]
    relative_mse = [
        float(row["concat_vs_baseline_relative_mse"])
        for row in data_seed_rows
    ]
    discrepancy_count = sum(
        bool(row["concat_discrepancy_pass"]) for row in data_seed_rows
    )
    counts = {
        "baseline_auroc": sum(
            value >= float(gates["baseline_data_seed_auroc_min"])
            for value in baseline_aurocs
        ),
        "missing_route": sum(
            value >= float(gates["concat_missing_route_data_seed_threshold"])
            for value in missing
        ),
        "mask_c": sum(
            value >= float(gates["concat_mask_c_delta_data_seed_threshold"])
            for value in mask_c
        ),
        "discrepancy": discrepancy_count,
        "coordinate_entropy": sum(
            value >= float(gates["concat_coordinate_entropy_data_seed_threshold"])
            for value in entropy
        ),
        "temporal_tail": sum(
            value >= float(gates["concat_temporal_tail_data_seed_threshold"])
            for value in tail
        ),
        "h64_h128_mass": sum(
            value >= float(gates["concat_h64_h128_ratio_data_seed_threshold"])
            for value in mass_ratio
        ),
        "nominal_horizon": sum(
            value
            <= float(gates["concat_nominal_horizon_max_abs_data_seed_threshold"])
            for value in nominal_diff
        ),
        "prediction_mse": sum(
            value
            <= float(gates["concat_vs_baseline_data_seed_relative_mse_max"])
            for value in relative_mse
        ),
    }
    checks = {
        "run_integrity": bool(run_validation_passed),
        "baseline_mean_auroc": _mean(baseline_aurocs)
        >= float(gates["baseline_mean_total_nominal_auroc_min"]),
        "baseline_seed_coverage": counts["baseline_auroc"]
        >= int(gates["baseline_data_seed_pass_count_min"]),
        "missing_route_mean": _mean(missing)
        >= float(gates["concat_missing_route_data_seed_mean_min"]),
        "missing_route_seed_coverage": counts["missing_route"]
        >= int(gates["concat_missing_route_data_seed_pass_count_min"]),
        "mask_c_mean": _mean(mask_c)
        >= float(gates["concat_mask_c_delta_data_seed_mean_min"]),
        "mask_c_seed_coverage": counts["mask_c"]
        >= int(gates["concat_mask_c_data_seed_pass_count_min"]),
        "discrepancy_seed_coverage": counts["discrepancy"]
        >= int(gates["concat_discrepancy_data_seed_pass_count_min"]),
        "coordinate_entropy_seed_coverage": counts["coordinate_entropy"]
        >= int(gates["concat_coordinate_entropy_data_seed_pass_count_min"]),
        "temporal_tail_seed_coverage": counts["temporal_tail"]
        >= int(gates["concat_temporal_tail_data_seed_pass_count_min"]),
        "h64_h128_seed_coverage": counts["h64_h128_mass"]
        >= int(gates["concat_h64_h128_ratio_data_seed_pass_count_min"]),
        "nominal_horizon_seed_coverage": counts["nominal_horizon"]
        >= int(gates["concat_nominal_horizon_data_seed_pass_count_min"]),
        "prediction_mse_mean": _mean(relative_mse)
        <= float(gates["concat_vs_baseline_mean_relative_mse_max"]),
        "prediction_mse_seed_coverage": counts["prediction_mse"]
        >= int(gates["concat_vs_baseline_mse_data_seed_pass_count_min"]),
    }
    return {
        "checks": checks,
        "counts": counts,
        "summaries": {
            "baseline_total_nominal_auroc": summarize(baseline_aurocs),
            "concat_missing_route_relative_magnitude": summarize(missing),
            "concat_mask_c_fixed_target_mse_delta": summarize(mask_c),
            "concat_partial_total_nominal_pearson": summarize([
                float(row["mamba_concat_partial_total_nominal_pearson"])
                for row in data_seed_rows
            ]),
            "concat_partial_total_nominal_topk_jaccard": summarize([
                float(
                    row[
                        "mamba_concat_partial_total_nominal_topk_jaccard"
                    ]
                )
                for row in data_seed_rows
            ]),
            "concat_coordinate_entropy_median": summarize(entropy),
            "concat_temporal_tail_median": summarize(tail),
            "concat_h64_h128_mass_ratio": summarize(mass_ratio),
            "concat_nominal_h64_h128_max_abs": summarize(nominal_diff),
            "concat_vs_baseline_relative_mse": summarize(relative_mse),
        },
        "passed": all(checks.values()),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    excluded = {
        "target_indices",
        "dataset_x_sha256",
        "dataset_graph_sha256",
    }
    fieldnames = [
        key for key in rows[0].keys() if key not in excluded
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row[key]
                for key in fieldnames
            })


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    formal_root = args.formal_root.resolve()
    output_dir = args.output_dir.resolve()
    config = load_protocol(config_path)
    matrix = load_run_matrix(config)
    release_lock = validate_release_lock(
        config_path=config_path,
        release_manifest_path=args.release_manifest.resolve(),
        config=config,
    )
    dataset_manifest = _load_json(
        formal_root / "formal_dataset_manifest.json"
    )
    if dataset_manifest.get("dataset_count") != 5:
        raise RuntimeError("Formal dataset manifest must contain exactly five seeds")
    records, run_validation = validate_run_set(
        formal_root=formal_root,
        matrix=matrix,
        release_lock=release_lock,
    )
    data_seed_rows = build_data_seed_rows(records)
    gate_report = evaluate_aggregate_gates(
        data_seed_rows=data_seed_rows,
        run_validation_passed=run_validation["passed"],
        gates=config["aggregate_gates"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "train_level_metrics.csv", records)
    write_csv(output_dir / "data_seed_level_metrics.csv", data_seed_rows)
    atomic_json(output_dir / "run_integrity_validation.json", run_validation)
    atomic_json(output_dir / "aggregate_gate_report.json", gate_report)
    summary = {
        "protocol_name": config["protocol_name"],
        "release_lock": release_lock,
        "formal_run_count": len(records),
        "statistical_unit_count": len(data_seed_rows),
        "statistical_unit": config["formal_design"]["statistical_unit"],
        "aggregate_gate_report": gate_report,
        "formal_result": True,
        "manuscript_evidence_eligible": bool(gate_report["passed"]),
        "claim_boundary": config["claim_boundary"],
        "stage1b_authorized": False,
    }
    atomic_json(output_dir / "formal_confirmation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if gate_report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
