"""Frozen Phase 8 result loading, aggregation, and gate evaluation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from phase8_coverage import build_stratified_lag_schedule, schedule_sha256
from phase8_protocol import file_sha256, load_json, load_run_matrix


REPLICATION_BLOCKS = {
    "capacity_replication",
    "fixed_target_interventions",
    "coefficient_replication",
}
PILOT_METHODS = {
    "baseline_jrngc",
    "concat_x_only",
    "full_aux_equal_lambda",
    "full_aux_lc10",
    "coverage_aligned_raw_chain",
}


def _all_finite(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return None if not finite else float(np.mean(finite))


def _max_state_difference(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> float:
    if set(a) != set(b):
        return float("inf")
    maximum = 0.0
    for name in a:
        if tuple(a[name].shape) != tuple(b[name].shape):
            return float("inf")
        maximum = max(maximum, float(torch.max(torch.abs(a[name] - b[name]))))
    return maximum


def _artifact_integrity(run_dir: Path) -> Tuple[bool, List[Dict[str, object]]]:
    manifest_path = run_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        return False, [{"path": str(manifest_path), "reason": "missing_manifest"}]
    manifest = load_json(manifest_path)
    failures = []
    for name, expected in manifest.get("files", {}).items():
        path = run_dir / name
        actual = file_sha256(path) if path.is_file() else None
        if actual != expected:
            failures.append({"path": str(path), "expected": expected, "actual": actual})
    return not failures, failures


def load_completed_run(
    root: Path,
    record_id: str,
    *,
    expected_config_sha256: Optional[str] = None,
    expected_matrix_sha256: Optional[str] = None,
) -> Dict[str, object]:
    run_dir = root / "runs" / record_id
    required = ["status.json", "metrics.json", "resolved_config.json", "runtime.json", "artifact_manifest.json"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Run {record_id} is incomplete; missing {missing}")
    status = load_json(run_dir / "status.json")
    if status.get("status") != "complete" or not status.get("output_complete") or not status.get("no_nan_inf"):
        raise RuntimeError(f"Run {record_id} status is not a finite complete result: {status}")
    integrity, failures = _artifact_integrity(run_dir)
    if not integrity:
        raise RuntimeError(f"Run {record_id} artifact integrity failed: {failures}")
    resolved = load_json(run_dir / "resolved_config.json")
    if expected_config_sha256 is not None and resolved.get("config_sha256") != expected_config_sha256:
        raise RuntimeError(f"Run {record_id} config SHA mismatch")
    if expected_matrix_sha256 is not None and resolved.get("run_matrix_sha256") != expected_matrix_sha256:
        raise RuntimeError(f"Run {record_id} matrix SHA mismatch")
    metrics = load_json(run_dir / "metrics.json")
    runtime = load_json(run_dir / "runtime.json")
    if not _all_finite(metrics) or not _all_finite(runtime):
        raise RuntimeError(f"Run {record_id} contains NaN/Inf")
    return {
        "record_id": record_id,
        "run_dir": run_dir,
        "status": status,
        "resolved": resolved,
        "metrics": metrics,
        "runtime": runtime,
    }


def _graph_metric(run: Mapping[str, object], score_track: str, metric: str) -> float:
    metrics = run["metrics"]  # type: ignore[index]
    value = metrics["graph_metrics"][score_track][metric]  # type: ignore[index]
    return float(value)


def _scalar(run: Mapping[str, object], field: str) -> Optional[float]:
    value = run["metrics"].get(field)  # type: ignore[index]
    return None if value is None else float(value)


def _classification(consistent_pairs: int, config: Mapping[str, object]) -> str:
    gates = config["replication_classification_gates"]  # type: ignore[index]
    if consistent_pairs >= int(gates["replicated_min_consistent_pairs"]):
        return "REPLICATED"
    if consistent_pairs >= int(gates["partial_replication_min_consistent_pairs"]):
        return "PARTIAL_REPLICATION"
    return "NON_REPLICATION"


def aggregate_replication(
    root: Path,
    *,
    config_path: Path,
    matrix_path: Path,
) -> Dict[str, object]:
    config = load_json(config_path)
    rows = [row for row in load_run_matrix(matrix_path) if row["block"] in REPLICATION_BLOCKS]
    config_sha = file_sha256(config_path)
    matrix_sha = file_sha256(matrix_path)
    if len(rows) != 50:
        raise RuntimeError(f"Expected 50 replication records, found {len(rows)}")
    runs = {
        row["record_id"]: load_completed_run(
            root,
            row["record_id"],
            expected_config_sha256=config_sha,
            expected_matrix_sha256=matrix_sha,
        )
        for row in rows
    }
    by_block_seed_method: Dict[Tuple[str, int, str], Dict[str, object]] = {}
    for row in rows:
        run = runs[row["record_id"]]
        if run["resolved"].get("formal_result") is not True:  # type: ignore[index]
            raise RuntimeError(f"Replication run {row['record_id']} is not marked formal")
        by_block_seed_method[(row["block"], int(row["data_seed"]), row["method"])] = run

    capacity_pairs = []
    for data_seed in sorted({int(row["data_seed"]) for row in rows if row["block"] == "capacity_replication"}):
        baseline = by_block_seed_method[("capacity_replication", data_seed, "baseline_jrngc")]
        method_rows = []
        for d_cond in [1, 2, 4, 8, 16]:
            run = by_block_seed_method[("capacity_replication", data_seed, f"concat_dcond_{d_cond}")]
            method_rows.append({
                "d_cond": d_cond,
                "fixed_target_prediction_mse": _scalar(run, "fixed_target_prediction_mse"),
                "partial_nominal_auroc": _graph_metric(run, "partial_nominal", "auroc"),
                "partial_nominal_auprc": _graph_metric(run, "partial_nominal", "auprc"),
                "total_nominal_auroc": _graph_metric(run, "total_nominal", "auroc"),
            })
        high = [item for item in method_rows if item["d_cond"] in {4, 8, 16}]
        baseline_mse = float(_scalar(baseline, "fixed_target_prediction_mse"))
        baseline_partial = _graph_metric(baseline, "partial_nominal", "auroc")
        mse_delta = float(np.mean([item["fixed_target_prediction_mse"] for item in high])) - baseline_mse
        partial_delta = float(np.mean([item["partial_nominal_auroc"] for item in high])) - baseline_partial
        capacity_pairs.append({
            "data_seed": data_seed,
            "model_seed": int(baseline["resolved"]["model_seed"]),  # type: ignore[index]
            "baseline": {
                "fixed_target_prediction_mse": baseline_mse,
                "partial_nominal_auroc": baseline_partial,
                "total_nominal_auroc": _graph_metric(baseline, "total_nominal", "auroc"),
            },
            "concat_by_capacity": method_rows,
            "high_capacity_mean_minus_baseline": {
                "fixed_target_prediction_mse": mse_delta,
                "partial_nominal_auroc": partial_delta,
            },
            "joint_direction_consistent": bool(mse_delta < 0 and partial_delta < 0),
        })
    capacity_consistent = sum(item["joint_direction_consistent"] for item in capacity_pairs)

    coefficient_pairs = []
    for data_seed in sorted({int(row["data_seed"]) for row in rows if row["block"] == "coefficient_replication"}):
        baseline = by_block_seed_method[("coefficient_replication", data_seed, "baseline_jrngc")]
        concat = by_block_seed_method[("coefficient_replication", data_seed, "concat_x_only")]
        auroc_delta = _graph_metric(concat, "partial_nominal", "auroc") - _graph_metric(
            baseline, "partial_nominal", "auroc"
        )
        coefficient_delta = float(_scalar(concat, "coefficient_r_partial_lag1")) - float(
            _scalar(baseline, "coefficient_r_partial_lag1")
        )
        coefficient_pairs.append({
            "data_seed": data_seed,
            "model_seed": int(baseline["resolved"]["model_seed"]),  # type: ignore[index]
            "concat_minus_baseline": {
                "partial_nominal_auroc": auroc_delta,
                "coefficient_r_partial_lag1": coefficient_delta,
                "total_nominal_auroc": _graph_metric(concat, "total_nominal", "auroc")
                - _graph_metric(baseline, "total_nominal", "auroc"),
                "coefficient_r_total_lag1": float(_scalar(concat, "coefficient_r_total_lag1"))
                - float(_scalar(baseline, "coefficient_r_total_lag1")),
            },
            "joint_direction_consistent": bool(auroc_delta < 0 and coefficient_delta < 0),
        })
    coefficient_consistent = sum(item["joint_direction_consistent"] for item in coefficient_pairs)

    intervention_pairs = []
    for data_seed in sorted({int(row["data_seed"]) for row in rows if row["block"] == "fixed_target_interventions"}):
        concat = by_block_seed_method[("fixed_target_interventions", data_seed, "concat_fixed_target_interventions")]
        no_aux = by_block_seed_method[("fixed_target_interventions", data_seed, "no_aux_fixed_target_interventions")]
        deltas = concat["metrics"]["interventions"]["fixed_target_prediction_mse_delta"]  # type: ignore[index]
        contrasts = {
            "mask_c_delta_minus_mask_x_delta": float(deltas["mask_c"] - deltas["mask_x"]),
            "mask_both_delta_minus_mask_x_delta": float(deltas["mask_both"] - deltas["mask_x"]),
            "shuffle_c_only_delta_minus_shuffle_x_only_delta": float(
                deltas["shuffle_c_only"] - deltas["shuffle_x_only"]
            ),
        }
        intervention_pairs.append({
            "data_seed": data_seed,
            "model_seed": int(concat["resolved"]["model_seed"]),  # type: ignore[index]
            "perturbation_seed": int(concat["resolved"]["perturbation_seed"]),  # type: ignore[index]
            "concat_fixed_target_prediction_mse_delta": deltas,
            "concat_legacy_objective_delta": concat["metrics"]["interventions"]["legacy_objective_delta"],  # type: ignore[index]
            "new_no_aux_control": no_aux["metrics"]["interventions"],  # type: ignore[index]
            "route_ordering_contrasts": contrasts,
            "joint_direction_consistent": all(value > 0 for value in contrasts.values()),
        })
    intervention_consistent = sum(item["joint_direction_consistent"] for item in intervention_pairs)

    classifications = {
        "capacity_replication": _classification(capacity_consistent, config),
        "fixed_target_interventions": _classification(intervention_consistent, config),
        "coefficient_replication": _classification(coefficient_consistent, config),
    }
    return {
        "passed_execution_completeness": True,
        "config_sha256": config_sha,
        "run_matrix_sha256": matrix_sha,
        "record_count": len(rows),
        "classification_rule": config["replication_classification_gates"],
        "capacity": {
            "pairs": capacity_pairs,
            "consistent_pair_count": capacity_consistent,
            "classification": classifications["capacity_replication"],
        },
        "fixed_target_interventions": {
            "pairs": intervention_pairs,
            "consistent_pair_count": intervention_consistent,
            "classification": classifications["fixed_target_interventions"],
        },
        "coefficient": {
            "pairs": coefficient_pairs,
            "consistent_pair_count": coefficient_consistent,
            "classification": classifications["coefficient_replication"],
        },
        "classifications": classifications,
        "all_three_replicated": all(value == "REPLICATED" for value in classifications.values()),
        "partial_and_total_tracks_both_retained": True,
    }


def _pilot_compact(run: Mapping[str, object]) -> Dict[str, Optional[float]]:
    return {
        "auroc": _graph_metric(run, "total_nominal", "auroc"),
        "auprc": _graph_metric(run, "total_nominal", "auprc"),
        "coefficient_r": _scalar(run, "coefficient_r_total_lag1"),
        "fixed_target_prediction_mse": _scalar(run, "fixed_target_prediction_mse"),
        "partial_total_pearson": _scalar(run, "nominal_partial_total_pearson"),
        "partial_total_topk_jaccard": _scalar(run, "nominal_partial_total_topk_jaccard"),
        "M_missing": _scalar(run, "M_missing"),
    }


def _average_compact(items: Sequence[Mapping[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    keys = set().union(*(item.keys() for item in items))
    output = {}
    for key in sorted(keys):
        values = [item.get(key) for item in items]
        output[key] = None if any(value is None for value in values) else float(np.mean(values))
    return output


def aggregate_pilot(
    root: Path,
    *,
    config_path: Path,
    matrix_path: Path,
    preflight_report: Mapping[str, object],
    replication_report: Mapping[str, object],
) -> Dict[str, object]:
    config = load_json(config_path)
    rows = [row for row in load_run_matrix(matrix_path) if row["block"] == "repair_pilot"]
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 repair-pilot records, found {len(rows)}")
    config_sha = file_sha256(config_path)
    matrix_sha = file_sha256(matrix_path)
    grouped: Dict[Tuple[int, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        run = load_completed_run(
            root,
            row["record_id"],
            expected_config_sha256=config_sha,
            expected_matrix_sha256=matrix_sha,
        )
        if run["resolved"].get("gating_checkpoint") != "final":  # type: ignore[index]
            raise RuntimeError(f"Pilot run {row['record_id']} did not use final-checkpoint gating")
        grouped[(int(row["data_seed"]), row["method"])].append(run)

    data_seed_rows = []
    for data_seed in sorted({key[0] for key in grouped}):
        method_values = {}
        for method in sorted(PILOT_METHODS):
            runs = grouped[(data_seed, method)]
            if len(runs) != 2:
                raise RuntimeError(f"Expected two model seeds for {data_seed}/{method}, got {len(runs)}")
            method_values[method] = _average_compact([_pilot_compact(run) for run in runs])
        data_seed_rows.append({"data_seed": data_seed, "methods": method_values})

    def effects(comparator: str) -> List[Dict[str, float]]:
        output = []
        for row in data_seed_rows:
            repair = row["methods"]["coverage_aligned_raw_chain"]
            other = row["methods"][comparator]
            output.append({
                "data_seed": int(row["data_seed"]),
                "delta_auroc": float(repair["auroc"] - other["auroc"]),
                "delta_auprc": float(repair["auprc"] - other["auprc"]),
                "delta_coefficient_r": float(repair["coefficient_r"] - other["coefficient_r"]),
                "relative_mse_degradation": float(
                    (repair["fixed_target_prediction_mse"] - other["fixed_target_prediction_mse"])
                    / max(other["fixed_target_prediction_mse"], 1e-12)
                ),
            })
        return output

    all_effects = {method: effects(method) for method in sorted(PILOT_METHODS - {"coverage_aligned_raw_chain"})}
    concat_effects = all_effects["concat_x_only"]
    repair_gate = config["pilot_go_gates"]  # type: ignore[index]
    required_effect = repair_gate["repair_minus_concat"]
    effect_means = {
        key: float(np.mean([row[key] for row in concat_effects]))
        for key in ["delta_auroc", "delta_auprc", "delta_coefficient_r", "relative_mse_degradation"]
    }
    positive_counts = {
        key: sum(row[key] > 0 for row in concat_effects)
        for key in ["delta_auroc", "delta_auprc", "delta_coefficient_r"]
    }
    effect_gate = bool(
        effect_means["delta_auroc"] >= required_effect["mean_delta_auroc_min"]
        and effect_means["delta_auprc"] >= required_effect["mean_delta_auprc_min"]
        and effect_means["delta_coefficient_r"] >= required_effect["mean_delta_coefficient_r_min"]
        and all(count >= required_effect["positive_data_seeds_min"] for count in positive_counts.values())
    )
    mse_gate_cfg = repair_gate["pure_mse"]
    pure_mse_gate = bool(
        effect_means["relative_mse_degradation"] <= mse_gate_cfg["mean_relative_degradation_max"]
        and all(
            row["relative_mse_degradation"] <= mse_gate_cfg["per_data_seed_relative_degradation_max"]
            for row in concat_effects
        )
    )

    safety_cfg = repair_gate["comparator_safety"]
    safety_details = {}
    safety_pass = True
    for comparator in safety_cfg["comparators"]:
        rows_for_comparator = all_effects[comparator]
        means = {
            key: float(np.mean([row[key] for row in rows_for_comparator]))
            for key in ["delta_auroc", "delta_auprc", "delta_coefficient_r"]
        }
        passed = bool(
            means["delta_auroc"] >= safety_cfg["mean_delta_auroc_min"]
            and means["delta_auprc"] >= safety_cfg["mean_delta_auprc_min"]
            and means["delta_coefficient_r"] >= safety_cfg["mean_delta_coefficient_r_min"]
            and min(row["delta_auroc"] for row in rows_for_comparator)
            >= safety_cfg["per_data_seed_delta_auroc_min"]
        )
        safety_details[comparator] = {"effects": rows_for_comparator, "means": means, "passed": passed}
        safety_pass = safety_pass and passed

    direct_cfg = repair_gate["direct_missing_route"]
    historical_cfg = repair_gate["historical_missing_route"]
    mechanism_rows = []
    for row in data_seed_rows:
        concat = row["methods"]["concat_x_only"]
        pearson = concat["partial_total_pearson"]
        jaccard = concat["partial_total_topk_jaccard"]
        direct = bool(
            (pearson is not None and pearson < direct_cfg["pearson_lt"])
            or (jaccard is not None and jaccard < direct_cfg["topk_jaccard_lt"])
        )
        historical = bool(concat["M_missing"] is not None and concat["M_missing"] >= historical_cfg["M_missing_min"])
        mechanism_rows.append({
            "data_seed": row["data_seed"],
            "concat_partial_total_pearson": pearson,
            "concat_partial_total_topk_jaccard": jaccard,
            "concat_M_missing": concat["M_missing"],
            "direct_gate_passed": direct,
            "historical_gate_passed": historical,
        })
    direct_pass_count = sum(row["direct_gate_passed"] for row in mechanism_rows)
    historical_pass_count = sum(row["historical_gate_passed"] for row in mechanism_rows)
    direct_gate = direct_pass_count >= direct_cfg["passing_data_seeds_min"]
    historical_gate = historical_pass_count >= historical_cfg["passing_data_seeds_min"]
    semantic_compute_gate = bool(preflight_report.get("passed"))
    pilot_go = bool(
        semantic_compute_gate
        and effect_gate
        and pure_mse_gate
        and safety_pass
        and direct_gate
        and historical_gate
    )
    replication_required = bool(replication_report.get("all_three_replicated"))
    return {
        "passed_execution_completeness": True,
        "config_sha256": config_sha,
        "run_matrix_sha256": matrix_sha,
        "record_count": len(rows),
        "statistical_unit": "data_seed_after_averaging_two_model_seeds",
        "data_seed_level_values": data_seed_rows,
        "paired_effects": all_effects,
        "repair_minus_concat": {
            "effects": concat_effects,
            "means": effect_means,
            "positive_counts": positive_counts,
            "effect_gate_passed": effect_gate,
            "pure_mse_gate_passed": pure_mse_gate,
        },
        "comparator_safety": {"comparators": safety_details, "passed": safety_pass},
        "missing_route": {
            "data_seed_values": mechanism_rows,
            "direct_pass_count": direct_pass_count,
            "historical_pass_count": historical_pass_count,
            "direct_gate_passed": direct_gate,
            "historical_gate_passed": historical_gate,
        },
        "semantic_compute_gate_passed": semantic_compute_gate,
        "pilot_go_passed": pilot_go,
        "replication_required_for_confirmation_passed": replication_required,
        "confirmation_eligible": bool(pilot_go and replication_required),
        "confirmation_executed": False,
    }


def projected_phase8_hours(preflight_runs: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    per_iter = {
        "baseline": preflight_runs["P8-PRE-001"]["runtime"]["training_seconds"] / 20,  # type: ignore[index]
        "concat": preflight_runs["P8-PRE-002"]["runtime"]["training_seconds"] / 20,  # type: ignore[index]
        "full_aux": preflight_runs["P8-PRE-003"]["runtime"]["training_seconds"] / 20,  # type: ignore[index]
        "repair": preflight_runs["P8-PRE-005"]["runtime"]["training_seconds"] / 100,  # type: ignore[index]
    }
    evaluation = {
        "baseline": preflight_runs["P8-PRE-001"]["runtime"]["evaluation_seconds"],  # type: ignore[index]
        "concat": preflight_runs["P8-PRE-002"]["runtime"]["evaluation_seconds"],  # type: ignore[index]
        "full_aux": preflight_runs["P8-PRE-003"]["runtime"]["evaluation_seconds"],  # type: ignore[index]
        "repair": preflight_runs["P8-PRE-005"]["runtime"]["evaluation_seconds"],  # type: ignore[index]
    }
    seconds = 0.0
    # 50 replication records.
    seconds += 5 * 1500 * per_iter["baseline"] + 25 * 1500 * per_iter["concat"]
    seconds += 5 * 2000 * per_iter["concat"] + 5 * 2000 * per_iter["repair"]
    seconds += 5 * 2000 * per_iter["baseline"] + 5 * 2000 * per_iter["concat"]
    seconds += 10 * evaluation["baseline"] + 30 * evaluation["concat"]
    # 30 repair-pilot records.
    seconds += 6 * 2000 * per_iter["baseline"] + 6 * 2000 * per_iter["concat"]
    seconds += 12 * 2000 * per_iter["full_aux"] + 6 * 2000 * per_iter["repair"]
    seconds += 6 * evaluation["baseline"] + 6 * evaluation["concat"]
    seconds += 12 * evaluation["full_aux"] + 6 * evaluation["repair"]
    # Sealed confirmation is never executed by this release, but its frozen
    # cost still belongs in the protocol-wide 48-hour feasibility audit.
    seconds += 10 * 2000 * per_iter["baseline"] + 10 * 2000 * per_iter["concat"]
    seconds += 20 * 2000 * per_iter["full_aux"] + 10 * 2000 * per_iter["repair"]
    seconds += 10 * evaluation["baseline"] + 10 * evaluation["concat"]
    seconds += 20 * evaluation["full_aux"] + 10 * evaluation["repair"]
    actual_preflight = sum(float(run["runtime"]["total_seconds"]) for run in preflight_runs.values())  # type: ignore[index]
    seconds += actual_preflight
    return {
        "per_iteration_seconds": per_iter,
        "full_evaluation_seconds": evaluation,
        "projected_preflight_replication_pilot_seconds": seconds,
        "projected_preflight_replication_pilot_gpu_hours": seconds / 3600.0,
        "includes_confirmation_cost_projection_only": True,
        "confirmation_execution_authorized": False,
    }


def validate_gpu_preflight(
    root: Path,
    *,
    config_path: Path,
    matrix_path: Path,
    cpu_preflight_summary: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    config = load_json(config_path)
    rows = [row for row in load_run_matrix(matrix_path) if row["phase"] == "preflight"]
    if len(rows) != 5:
        raise RuntimeError(f"Expected five preflight rows, found {len(rows)}")
    config_sha = file_sha256(config_path)
    matrix_sha = file_sha256(matrix_path)
    runs = {
        row["record_id"]: load_completed_run(
            root,
            row["record_id"],
            expected_config_sha256=config_sha,
            expected_matrix_sha256=matrix_sha,
        )
        for row in rows
    }
    failures: List[Dict[str, object]] = []
    source_locks = {
        (
            run["resolved"]["source_release"]["approved_commit"],  # type: ignore[index]
            run["resolved"]["source_release"]["source_manifest_sha256"],  # type: ignore[index]
        )
        for run in runs.values()
    }
    if len(source_locks) != 1:
        failures.append({"gate": "release_lock_consistency", "values": sorted(source_locks)})
    for record_id, run in runs.items():
        resolved = run["resolved"]
        if resolved.get("execution_device") != "cuda":
            failures.append({"gate": "cuda_device", "record_id": record_id, "actual": resolved.get("execution_device")})
        deterministic = resolved.get("deterministic_settings", {})
        if not deterministic.get("torch_deterministic_algorithms"):
            failures.append({"gate": "deterministic_algorithms", "record_id": record_id})

    smoke = torch.load(runs["P8-PRE-004"]["run_dir"] / "checkpoint_iter20.pt", map_location="cpu", weights_only=False)
    benchmark = torch.load(runs["P8-PRE-005"]["run_dir"] / "checkpoint_iter20.pt", map_location="cpu", weights_only=False)
    checkpoint_difference = _max_state_difference(smoke["model_state_dict"], benchmark["model_state_dict"])
    trace_difference = float(np.max(np.abs(
        np.asarray(smoke["objective_trace_first20"], dtype=np.float64)
        - np.asarray(benchmark["objective_trace_first20"], dtype=np.float64)
    )))
    smoke_schedule = load_json(runs["P8-PRE-004"]["run_dir"] / "jacobian_schedule.json")
    benchmark_schedule = load_json(runs["P8-PRE-005"]["run_dir"] / "jacobian_schedule.json")
    schedule_integrity = {}
    for record_id, actual_schedule in [
        ("P8-PRE-004", smoke_schedule),
        ("P8-PRE-005", benchmark_schedule),
    ]:
        resolved = runs[record_id]["resolved"]
        expected_schedule = build_stratified_lag_schedule(
            T=int(resolved["T"]),
            lag=int(resolved["K"]),
            d_out=int(resolved["d"]),
            max_iter=int(resolved["max_iter"]),
            seed=int(resolved["jacobian_seed"]),
        )
        expected_hash = schedule_sha256(expected_schedule)
        checks = {
            "entry_count": len(actual_schedule.get("entries", [])) == len(expected_schedule),
            "entries_exact": actual_schedule.get("entries") == expected_schedule,
            "file_sha_field": actual_schedule.get("sha256") == expected_hash,
            "file_seed_field": int(actual_schedule.get("seed", -1)) == int(resolved["jacobian_seed"]),
            "resolved_sha_field": resolved.get("schedule_sha256") == expected_hash,
            "resolved_seed_field": int(resolved.get("schedule_seed", -1)) == int(resolved["jacobian_seed"]),
        }
        schedule_integrity[record_id] = {
            "passed": all(checks.values()),
            "checks": checks,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_schedule.get("sha256"),
            "expected_entry_count": len(expected_schedule),
            "actual_entry_count": len(actual_schedule.get("entries", [])),
        }
        if not schedule_integrity[record_id]["passed"]:
            failures.append({
                "gate": "frozen_stratified_schedule_integrity",
                "record_id": record_id,
                **schedule_integrity[record_id],
            })
    schedule_prefix_equal = smoke_schedule["entries"] == benchmark_schedule["entries"][:20]
    deterministic_max = float(config["gpu_preflight_gates"]["deterministic_max_abs"])
    if checkpoint_difference > deterministic_max or trace_difference > deterministic_max or not schedule_prefix_equal:
        failures.append({
            "gate": "repair_20_iteration_determinism",
            "checkpoint_max_abs": checkpoint_difference,
            "trace_max_abs": trace_difference,
            "schedule_prefix_equal": schedule_prefix_equal,
        })

    benchmark_metrics = runs["P8-PRE-005"]["metrics"]
    scale = benchmark_metrics["regularizer_scale_preflight"]
    initial = scale["initial"]
    final = scale["final"]
    mse_improvement = (initial["fixed_target_prediction_mse"] - final["fixed_target_prediction_mse"]) / max(
        initial["fixed_target_prediction_mse"], 1e-12
    )
    gates = config["gpu_preflight_gates"]
    expected_points = [str(value) for value in config["gpu_preflight_reporting_completed_iterations"]]
    actual_points = scale.get("by_completed_iteration", {})
    if sorted(actual_points, key=int) != expected_points:
        failures.append({"gate": "scale_reporting_points", "actual": sorted(actual_points, key=int), "expected": expected_points})
    for point, snapshot in actual_points.items():
        if not snapshot.get("gradients_finite"):
            failures.append({"gate": "gradients_finite", "completed_iteration": point, "actual": False})
        for field in [
            "nominal_jacobian_penalty",
            "nominal_regularizer_predictor_gradient_norm",
            "nominal_regularizer_preprocessor_gradient_norm",
        ]:
            value = snapshot.get(field)
            if value is None or not np.isfinite(value) or value <= 0:
                failures.append({"gate": f"{field}_finite_nonzero", "completed_iteration": point, "actual": value})
    stratified_trace = benchmark_metrics.get("stratified_benchmark_trace")
    if not isinstance(stratified_trace, dict):
        failures.append({"gate": "stratified_benchmark_trace", "actual": stratified_trace})
        stratified_trace = {}
    for field in [
        "cumulative_historical_contribution",
        "cumulative_historical_predictor_gradient_norm",
        "cumulative_historical_preprocessor_gradient_norm",
    ]:
        value = stratified_trace.get(field)
        if value is None or not np.isfinite(value) or value <= 0:
            failures.append({"gate": f"{field}_finite_nonzero", "actual": value})
    strata = stratified_trace.get("strata", {})
    iterations = stratified_trace.get("iterations", [])
    expected_benchmark_entries = benchmark_schedule.get("entries", [])
    trace_schedule_mismatches = []
    if len(iterations) != len(expected_benchmark_entries):
        failures.append({
            "gate": "stratified_trace_length",
            "actual": len(iterations),
            "expected": len(expected_benchmark_entries),
        })
    for index, (row, expected_entry) in enumerate(zip(iterations, expected_benchmark_entries)):
        expected_fields = {
            "completed_iteration": index + 1,
            "nominal_lag": 1,
            "historical_stratum": expected_entry["historical_stratum"],
            "historical_lag": expected_entry["historical_lag"],
        }
        mismatched = {
            key: {"actual": row.get(key), "expected": value}
            for key, value in expected_fields.items()
            if row.get(key) != value
        }
        if mismatched:
            trace_schedule_mismatches.append({"iteration": index + 1, "fields": mismatched})
    if trace_schedule_mismatches:
        failures.append({"gate": "stratified_trace_schedule_alignment", "mismatches": trace_schedule_mismatches})
    expected_stratum_counts = {
        name: sum(1 for entry in expected_benchmark_entries if entry.get("historical_stratum") == name)
        for name in ["B1", "B2", "B3"]
    }
    cycle_mismatches = [
        {"iteration": index + 1, "actual": row.get("historical_stratum"), "expected": ["B1", "B2", "B3"][index % 3]}
        for index, row in enumerate(iterations)
        if row.get("historical_stratum") != ["B1", "B2", "B3"][index % 3]
    ]
    if cycle_mismatches:
        failures.append({"gate": "historical_stratum_cycle", "mismatches": cycle_mismatches})
    for name in ["B1", "B2", "B3"]:
        actual_count = strata.get(name, {}).get("sample_count")
        if actual_count != expected_stratum_counts[name]:
            failures.append({
                "gate": "historical_stratum_sampling_frequency",
                "stratum": name,
                "actual": actual_count,
                "expected": expected_stratum_counts[name],
            })
    for name in gates["historical_strata_requiring_nonzero_float32_draw"]:
        count = strata.get(name, {}).get("nonzero_float32_contribution_count", 0)
        if count <= 0:
            failures.append({"gate": "historical_stratum_nonzero_float32_draw", "stratum": name, "actual": count})
    if not stratified_trace.get("all_strata_sampled"):
        failures.append({"gate": "all_historical_strata_sampled", "actual": stratified_trace.get("all_strata_sampled")})
    if mse_improvement < gates["pure_mse_relative_improvement_min"]:
        failures.append({"gate": "pure_mse_learning", "relative_improvement": mse_improvement})
    if final["output_target_variance_ratio"] < gates["output_target_variance_ratio_min"]:
        failures.append({"gate": "output_variance", "ratio": final["output_target_variance_ratio"]})
    benchmark_runtime = runs["P8-PRE-005"]["runtime"]
    if benchmark_runtime["projected_2000_iteration_training_seconds"] > gates["projected_repair_runtime_seconds_max"]:
        failures.append({
            "gate": "projected_repair_runtime",
            "seconds": benchmark_runtime["projected_2000_iteration_training_seconds"],
        })
    if benchmark_runtime["evaluation_seconds"] > gates["full_attribution_evaluation_seconds_max"]:
        failures.append({"gate": "full_attribution_evaluation", "seconds": benchmark_runtime["evaluation_seconds"]})
    peak_fraction = max(float(run["runtime"]["peak_reserved_fraction"]) for run in runs.values())  # type: ignore[index]
    if peak_fraction > gates["peak_reserved_fraction_max"]:
        failures.append({"gate": "peak_vram", "peak_reserved_fraction": peak_fraction})
    projection = projected_phase8_hours(runs)
    if projection["projected_preflight_replication_pilot_gpu_hours"] > gates["projected_total_gpu_hours_max"]:
        failures.append({
            "gate": "projected_total_gpu_hours",
            "hours": projection["projected_preflight_replication_pilot_gpu_hours"],
        })
    if cpu_preflight_summary is not None and not cpu_preflight_summary.get("passed"):
        failures.append({"gate": "cpu_preflight", "actual": cpu_preflight_summary})
    return {
        "passed": not failures,
        "record_count": len(runs),
        "config_sha256": config_sha,
        "run_matrix_sha256": matrix_sha,
        "source_release_locks": [list(value) for value in source_locks],
        "determinism": {
            "checkpoint_state_max_abs_difference": checkpoint_difference,
            "loss_trace_max_abs_difference": trace_difference,
            "schedule_prefix_equal": schedule_prefix_equal,
        },
        "schedule_integrity": schedule_integrity,
        "stratified_trace_schedule_mismatch_count": len(trace_schedule_mismatches),
        "regularizer_scale": scale,
        "stratified_benchmark_trace": stratified_trace,
        "pure_mse_relative_improvement": mse_improvement,
        "peak_reserved_fraction": peak_fraction,
        "compute_projection": projection,
        "failures": failures,
    }
