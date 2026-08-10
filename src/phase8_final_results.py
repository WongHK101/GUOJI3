"""Aggregation and frozen decision rules for the final Phase 8 repair cycle."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from phase8_final_protocol import (
    CONFIRMATION_METHODS,
    PILOT_LAMBDAS,
    validate_final_run_matrix,
)
from phase8_protocol import file_sha256, load_json, load_run_matrix
from phase8_results import load_completed_run


REFERENCE_METHODS = (
    "baseline_jrngc",
    "concat_x_only",
    "full_aux_equal_lambda",
    "full_aux_lc10",
    "coverage_aligned_raw_chain",
)
EPS = 1e-12


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Expected a nonempty finite metric vector")
    return float(np.mean(array))


def select_eligible_lambda(lambda_reports: Sequence[Mapping[str, object]]) -> Optional[Mapping[str, object]]:
    """Apply the frozen MSE, AUROC, then lambda tie-breaking order."""
    eligible = [report for report in lambda_reports if report.get("eligible") is True]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda report: (
            report["repair_minus_concat"]["means"]["relative_mse_degradation"],  # type: ignore[index]
            -report["repair_minus_concat"]["means"]["delta_auroc"],  # type: ignore[index]
            report["lambda"],
        ),
    )


def _scalar(run: Mapping[str, object], field: str) -> Optional[float]:
    value = run["metrics"].get(field)  # type: ignore[index]
    if value is None:
        return None
    output = float(value)
    return output if np.isfinite(output) else None


def _graph_metric(run: Mapping[str, object], metric: str) -> float:
    value = run["metrics"]["graph_metrics"]["total_nominal"][metric]  # type: ignore[index]
    output = float(value)
    if not np.isfinite(output):
        raise ValueError(f"Nonfinite total-nominal {metric}")
    return output


def compact_metrics(run: Mapping[str, object]) -> Dict[str, Optional[float]]:
    return {
        "auroc": _graph_metric(run, "auroc"),
        "auprc": _graph_metric(run, "auprc"),
        "coefficient_r": _scalar(run, "coefficient_r_total_lag1"),
        "fixed_target_prediction_mse": _scalar(run, "fixed_target_prediction_mse"),
        "jacobian_penalty": _scalar(run, "jacobian_penalty"),
        "total_regularized_objective": _scalar(run, "total_regularized_objective"),
        "partial_total_pearson": _scalar(run, "nominal_partial_total_pearson"),
        "partial_total_topk_jaccard": _scalar(run, "nominal_partial_total_topk_jaccard"),
        "M_missing": _scalar(run, "M_missing"),
        "tail_mass_mean": _nested_scalar(run["metrics"], "temporal_tail_statistics", "mean"),  # type: ignore[index]
        "tail_mass_median": _nested_scalar(run["metrics"], "temporal_tail_statistics", "median"),  # type: ignore[index]
    }


def _nested_scalar(payload: Mapping[str, object], outer: str, inner: str) -> Optional[float]:
    nested = payload.get(outer)
    if not isinstance(nested, Mapping) or nested.get(inner) is None:
        return None
    value = float(nested[inner])
    return value if np.isfinite(value) else None


def average_metrics(items: Sequence[Mapping[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    if not items:
        raise ValueError("Cannot average an empty metric list")
    keys = set().union(*(item.keys() for item in items))
    output: Dict[str, Optional[float]] = {}
    for key in sorted(keys):
        values = [item.get(key) for item in items]
        output[key] = None if any(value is None for value in values) else _mean(float(value) for value in values)
    return output


def _run_stability(run: Mapping[str, object], *, expected_schedule_sha256: Optional[str]) -> Dict[str, object]:
    resolved = run["resolved"]  # type: ignore[index]
    metadata = resolved.get("checkpoint_policy_metadata", {})
    metrics = run["metrics"]  # type: ignore[index]
    finite_fields = [
        _scalar(run, "fixed_target_prediction_mse"),
        _scalar(run, "jacobian_penalty"),
        _scalar(run, "total_regularized_objective"),
        _scalar(run, "coefficient_r_total_lag1"),
        _graph_metric(run, "auroc"),
        _graph_metric(run, "auprc"),
    ]
    schedule_match = expected_schedule_sha256 is None or resolved.get("schedule_sha256") == expected_schedule_sha256
    passed = bool(
        all(value is not None and np.isfinite(value) for value in finite_fields)
        and resolved.get("gating_checkpoint") == "final"
        and metadata.get("gating_checkpoint") == "final"
        and int(metadata.get("iterations_completed", -1)) == 2000
        and int(metadata.get("selected_iteration", -1)) == 1999
        and metadata.get("early_stopped") is False
        and schedule_match
        and float(metrics.get("output_variance", 0.0)) > 0.0
    )
    return {
        "passed": passed,
        "iterations_completed": metadata.get("iterations_completed"),
        "selected_iteration": metadata.get("selected_iteration"),
        "early_stopped": metadata.get("early_stopped"),
        "schedule_sha256": resolved.get("schedule_sha256"),
        "expected_schedule_sha256": expected_schedule_sha256,
        "schedule_match": schedule_match,
        "finite_primary_metrics": all(value is not None and np.isfinite(value) for value in finite_fields),
        "output_variance": metrics.get("output_variance"),
    }


def _validate_reference_lock(root: Path, config: Mapping[str, object]) -> Dict[str, object]:
    lock = root / "execution_lock"
    reference = config["frozen_reference"]  # type: ignore[index]
    paths = {
        "source_manifest_sha256": lock / "release_source_manifest.json",
        "config_sha256": lock / "config_snapshot.json",
        "run_matrix_sha256": lock / "run_matrix_snapshot.csv",
        "authorization_sha256": lock / "authorization_snapshot.json",
    }
    actual = {name: file_sha256(path) if path.is_file() else None for name, path in paths.items()}
    failures = {
        name: {"actual": actual[name], "expected": reference[name]}
        for name in paths
        if actual[name] != reference[name]
    }
    commit_path = lock / "approved_phase8_code_commit.txt"
    commit = commit_path.read_text(encoding="utf-8").strip() if commit_path.is_file() else None
    if commit != reference["source_commit"]:
        failures["source_commit"] = {"actual": commit, "expected": reference["source_commit"]}
    if failures:
        raise RuntimeError(f"Frozen comparator release-lock mismatch: {failures}")
    preflight = load_json(root / "gpu_preflight_validation.json")
    if preflight.get("passed") is not True:
        raise RuntimeError("Frozen comparator GPU preflight is not passing")
    return {"passed": True, "source_commit": commit, **actual, "gpu_preflight_passed": True}


def _reference_rows(root: Path, config: Mapping[str, object]) -> Tuple[Dict[Tuple[int, int, str], Dict[str, object]], Dict[str, object]]:
    lock_report = _validate_reference_lock(root, config)
    matrix_path = root / "execution_lock" / "run_matrix_snapshot.csv"
    reference = config["frozen_reference"]  # type: ignore[index]
    rows = [row for row in load_run_matrix(matrix_path) if row["block"] == "repair_pilot"]
    selected = {}
    for row in rows:
        method = row["method"]
        if method not in REFERENCE_METHODS:
            continue
        run = load_completed_run(
            root,
            row["record_id"],
            expected_config_sha256=str(reference["config_sha256"]),
            expected_matrix_sha256=str(reference["run_matrix_sha256"]),
        )
        key = (int(row["data_seed"]), int(row["model_seed"]), method)
        if key in selected:
            raise RuntimeError(f"Duplicate frozen comparator run: {key}")
        selected[key] = run
    expected_count = 3 * 2 * len(REFERENCE_METHODS)
    if len(selected) != expected_count:
        raise RuntimeError(f"Expected {expected_count} frozen pilot records, found {len(selected)}")
    return selected, lock_report


def _effect(repair: Mapping[str, Optional[float]], comparator: Mapping[str, Optional[float]]) -> Dict[str, float]:
    required = ["auroc", "auprc", "coefficient_r", "fixed_target_prediction_mse"]
    if any(repair.get(key) is None or comparator.get(key) is None for key in required):
        raise ValueError("A gated effect contains an unavailable metric")
    return {
        "delta_auroc": float(repair["auroc"] - comparator["auroc"]),  # type: ignore[operator]
        "delta_auprc": float(repair["auprc"] - comparator["auprc"]),  # type: ignore[operator]
        "delta_coefficient_r": float(repair["coefficient_r"] - comparator["coefficient_r"]),  # type: ignore[operator]
        "relative_mse_degradation": float(
            (repair["fixed_target_prediction_mse"] - comparator["fixed_target_prediction_mse"])  # type: ignore[operator]
            / max(float(comparator["fixed_target_prediction_mse"]), EPS)
        ),
    }


def _pilot_gate(
    *,
    data_seed_rows: Sequence[Mapping[str, object]],
    lambda_value: float,
    config: Mapping[str, object],
    semantic_compute_passed: bool,
    mechanism_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    lambda_key = f"lambda_{lambda_value:.10g}"
    comparator_names = ("concat_x_only", "baseline_jrngc", "full_aux_equal_lambda", "full_aux_lc10")
    paired_effects: Dict[str, List[Dict[str, float]]] = {name: [] for name in comparator_names}
    for row in data_seed_rows:
        repair = row["repairs"][lambda_key]  # type: ignore[index]
        for comparator in comparator_names:
            effect = _effect(repair, row["comparators"][comparator])  # type: ignore[index]
            paired_effects[comparator].append({"data_seed": int(row["data_seed"]), **effect})

    gates = config["pilot_go_gates"]  # type: ignore[index]
    concat_effects = paired_effects["concat_x_only"]
    means = {
        field: _mean(row[field] for row in concat_effects)
        for field in ["delta_auroc", "delta_auprc", "delta_coefficient_r", "relative_mse_degradation"]
    }
    positive_counts = {
        field: sum(row[field] > 0 for row in concat_effects)
        for field in ["delta_auroc", "delta_auprc", "delta_coefficient_r"]
    }
    effect_cfg = gates["repair_minus_concat"]
    effect_passed = bool(
        means["delta_auroc"] >= effect_cfg["mean_delta_auroc_min"]
        and means["delta_auprc"] >= effect_cfg["mean_delta_auprc_min"]
        and means["delta_coefficient_r"] >= effect_cfg["mean_delta_coefficient_r_min"]
        and all(count >= effect_cfg["positive_data_seeds_min"] for count in positive_counts.values())
    )
    mse_cfg = gates["pure_mse"]
    mse_passed = bool(
        means["relative_mse_degradation"] <= mse_cfg["mean_relative_degradation_max"]
        and all(
            row["relative_mse_degradation"] <= mse_cfg["per_data_seed_relative_degradation_max"]
            for row in concat_effects
        )
    )

    safety_cfg = gates["comparator_safety"]
    safety_details = {}
    safety_passed = True
    for comparator in safety_cfg["comparators"]:
        effects = paired_effects[comparator]
        comparator_means = {
            field: _mean(row[field] for row in effects)
            for field in ["delta_auroc", "delta_auprc", "delta_coefficient_r"]
        }
        passed = bool(
            comparator_means["delta_auroc"] >= safety_cfg["mean_delta_auroc_min"]
            and comparator_means["delta_auprc"] >= safety_cfg["mean_delta_auprc_min"]
            and comparator_means["delta_coefficient_r"] >= safety_cfg["mean_delta_coefficient_r_min"]
            and min(row["delta_auroc"] for row in effects) >= safety_cfg["per_data_seed_delta_auroc_min"]
        )
        safety_details[comparator] = {"effects": effects, "means": comparator_means, "passed": passed}
        safety_passed = safety_passed and passed

    direct_count = sum(bool(row["direct_gate_passed"]) for row in mechanism_rows)
    historical_count = sum(bool(row["historical_gate_passed"]) for row in mechanism_rows)
    direct_passed = direct_count >= gates["direct_missing_route"]["passing_data_seeds_min"]
    historical_passed = historical_count >= gates["historical_missing_route"]["passing_data_seeds_min"]
    passed = bool(
        semantic_compute_passed
        and effect_passed
        and mse_passed
        and safety_passed
        and direct_passed
        and historical_passed
    )
    return {
        "lambda": lambda_value,
        "eligible": passed,
        "semantic_compute_gate_passed": semantic_compute_passed,
        "repair_minus_concat": {
            "effects": concat_effects,
            "means": means,
            "positive_counts": positive_counts,
            "effect_gate_passed": effect_passed,
            "pure_mse_gate_passed": mse_passed,
        },
        "all_paired_effects": paired_effects,
        "comparator_safety": {"comparators": safety_details, "passed": safety_passed},
        "missing_route": {
            "data_seed_values": list(mechanism_rows),
            "direct_pass_count": direct_count,
            "historical_pass_count": historical_count,
            "direct_gate_passed": direct_passed,
            "historical_gate_passed": historical_passed,
        },
    }


def aggregate_lambda_tradeoff(
    tradeoff_root: Path,
    frozen_reference_root: Path,
    *,
    config_path: Path,
    matrix_path: Path,
    cpu_preflight_report: Mapping[str, object],
) -> Dict[str, object]:
    matrix_report = validate_final_run_matrix(config_path, matrix_path)
    if not matrix_report["passed"]:
        raise RuntimeError(f"Final lambda matrix validation failed: {matrix_report['failures']}")
    config = load_json(config_path)
    reference_runs, reference_lock = _reference_rows(frozen_reference_root, config)
    config_sha = file_sha256(config_path)
    matrix_sha = file_sha256(matrix_path)

    new_runs: Dict[Tuple[float, int, int], Dict[str, object]] = {}
    stability_rows = []
    expected_schedule = None
    for row in load_run_matrix(matrix_path):
        run = load_completed_run(
            tradeoff_root,
            row["record_id"],
            expected_config_sha256=config_sha,
            expected_matrix_sha256=matrix_sha,
        )
        value = float(row["raw_chain_lambda"])
        key = (value, int(row["data_seed"]), int(row["model_seed"]))
        if key in new_runs:
            raise RuntimeError(f"Duplicate new lambda run: {key}")
        new_runs[key] = run
        current_schedule = str(run["resolved"].get("schedule_sha256"))  # type: ignore[index]
        expected_schedule = current_schedule if expected_schedule is None else expected_schedule
        stability = _run_stability(run, expected_schedule_sha256=expected_schedule)
        stability_rows.append({"record_id": row["record_id"], "lambda": value, **stability})
        resolved_lambda = run["resolved"].get("raw_chain_lambda")  # type: ignore[index]
        metric_lambda = run["metrics"].get("raw_chain_lambda")  # type: ignore[index]
        if not math.isclose(float(resolved_lambda), value, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"Resolved lambda mismatch for {row['record_id']}")
        if not math.isclose(float(metric_lambda), value, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"Metric lambda mismatch for {row['record_id']}")

    expected_new = len(PILOT_LAMBDAS) * 3 * 2
    if len(new_runs) != expected_new:
        raise RuntimeError(f"Expected {expected_new} new repair runs, found {len(new_runs)}")
    all_new_stable = all(row["passed"] for row in stability_rows)
    semantic_compute_passed = bool(
        cpu_preflight_report.get("passed")
        and reference_lock["gpu_preflight_passed"]
        and all_new_stable
    )

    data_seed_rows = []
    for data_seed in (12001, 12002, 12003):
        comparators = {}
        for method in REFERENCE_METHODS[:-1]:
            comparators[method] = average_metrics([
                compact_metrics(reference_runs[(data_seed, model_seed, method)])
                for model_seed in (22001, 22002)
            ])
        repairs = {}
        for value in (*PILOT_LAMBDAS, 0.01):
            if value == 0.01:
                items = [
                    compact_metrics(reference_runs[(data_seed, model_seed, "coverage_aligned_raw_chain")])
                    for model_seed in (22001, 22002)
                ]
            else:
                items = [compact_metrics(new_runs[(value, data_seed, model_seed)]) for model_seed in (22001, 22002)]
            repairs[f"lambda_{value:.10g}"] = average_metrics(items)
        data_seed_rows.append({"data_seed": data_seed, "comparators": comparators, "repairs": repairs})

    direct_cfg = config["pilot_go_gates"]["direct_missing_route"]  # type: ignore[index]
    historical_cfg = config["pilot_go_gates"]["historical_missing_route"]  # type: ignore[index]
    mechanism_rows = []
    for row in data_seed_rows:
        concat = row["comparators"]["concat_x_only"]
        pearson = concat["partial_total_pearson"]
        jaccard = concat["partial_total_topk_jaccard"]
        m_missing = concat["M_missing"]
        direct = bool(
            (pearson is not None and pearson < direct_cfg["pearson_lt"])
            or (jaccard is not None and jaccard < direct_cfg["topk_jaccard_lt"])
        )
        historical = bool(m_missing is not None and m_missing >= historical_cfg["M_missing_min"])
        mechanism_rows.append({
            "data_seed": row["data_seed"],
            "concat_partial_total_pearson": pearson,
            "concat_partial_total_topk_jaccard": jaccard,
            "concat_M_missing": m_missing,
            "concat_tail_mass_mean": concat["tail_mass_mean"],
            "concat_tail_mass_median": concat["tail_mass_median"],
            "direct_gate_passed": direct,
            "historical_gate_passed": historical,
        })

    lambda_reports = []
    for value in (*PILOT_LAMBDAS, 0.01):
        report = _pilot_gate(
            data_seed_rows=data_seed_rows,
            lambda_value=value,
            config=config,
            semantic_compute_passed=semantic_compute_passed,
            mechanism_rows=mechanism_rows,
        )
        report["result_source"] = "frozen_recovery_pilot" if value == 0.01 else "new_final_tradeoff"
        lambda_reports.append(report)

    eligible = [report for report in lambda_reports if report["eligible"]]
    selected = select_eligible_lambda(lambda_reports)
    return {
        "passed_execution_completeness": True,
        "config_sha256": config_sha,
        "run_matrix_sha256": matrix_sha,
        "new_run_count": len(new_runs),
        "frozen_reference_run_count": len(reference_runs),
        "frozen_reference_lock": reference_lock,
        "statistical_unit": "data_seed_after_averaging_two_model_seeds",
        "data_seed_level_values": data_seed_rows,
        "new_run_stability": {
            "passed": all_new_stable,
            "expected_schedule_sha256": expected_schedule,
            "records": stability_rows,
        },
        "semantic_compute_gate_passed": semantic_compute_passed,
        "lambda_reports": lambda_reports,
        "eligible_lambdas": [report["lambda"] for report in eligible],
        "selected_lambda": None if selected is None else selected["lambda"],
        "selected_lambda_gate_report": selected,
        "candidate_selection_rule": config["candidate_selection"],
        "pilot_go_passed": selected is not None,
        "confirmation_eligible": selected is not None,
        "confirmation_executed": False,
        "method_decision": (
            "FREEZE_SELECTED_LAMBDA_FOR_HELD_OUT_CONFIRMATION"
            if selected is not None
            else config["candidate_selection"]["no_candidate_decision"]  # type: ignore[index]
        ),
        "prior_intervention_nonreplication_does_not_block_latest_conditional_confirmation": True,
    }


def aggregate_confirmation(
    root: Path,
    *,
    config_path: Path,
    matrix_path: Path,
    pilot_report: Mapping[str, object],
) -> Dict[str, object]:
    matrix_report = validate_final_run_matrix(config_path, matrix_path)
    if not matrix_report["passed"]:
        raise RuntimeError(f"Confirmation matrix validation failed: {matrix_report['failures']}")
    config = load_json(config_path)
    if pilot_report.get("confirmation_eligible") is not True:
        raise PermissionError("Held-out confirmation is not pilot-go eligible")
    selected_lambda = float(config["selected_lambda"])
    if not math.isclose(selected_lambda, float(pilot_report["selected_lambda"]), rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Confirmation selected lambda differs from pilot selection")
    config_sha = file_sha256(config_path)
    matrix_sha = file_sha256(matrix_path)
    grouped: Dict[Tuple[int, str], List[Dict[str, object]]] = defaultdict(list)
    stability_rows = []
    expected_schedule = None
    for row in load_run_matrix(matrix_path):
        run = load_completed_run(
            root,
            row["record_id"],
            expected_config_sha256=config_sha,
            expected_matrix_sha256=matrix_sha,
        )
        grouped[(int(row["data_seed"]), row["method"])].append(run)
        if row["method"] == "coverage_aligned_raw_chain":
            current_schedule = str(run["resolved"].get("schedule_sha256"))  # type: ignore[index]
            expected_schedule = current_schedule if expected_schedule is None else expected_schedule
            stability = _run_stability(run, expected_schedule_sha256=expected_schedule)
        else:
            metadata = run["resolved"].get("checkpoint_policy_metadata", {})  # type: ignore[index]
            stability = {
                "passed": bool(
                    metadata.get("gating_checkpoint") == "final"
                    and int(metadata.get("iterations_completed", -1)) == 2000
                    and int(metadata.get("selected_iteration", -1)) == 1999
                    and metadata.get("early_stopped") is False
                )
            }
        stability_rows.append({"record_id": row["record_id"], **stability})

    data_seed_rows = []
    for data_seed in sorted({key[0] for key in grouped}):
        methods = {}
        for method in CONFIRMATION_METHODS:
            runs = grouped[(data_seed, method)]
            if len(runs) != 2:
                raise RuntimeError(f"Expected two confirmation model seeds for {data_seed}/{method}")
            methods[method] = average_metrics([compact_metrics(run) for run in runs])
        data_seed_rows.append({"data_seed": data_seed, "methods": methods})

    def effects(comparator: str) -> List[Dict[str, float]]:
        output = []
        for row in data_seed_rows:
            effect = _effect(row["methods"]["coverage_aligned_raw_chain"], row["methods"][comparator])
            output.append({"data_seed": row["data_seed"], **effect})
        return output

    concat_effects = effects("concat_x_only")
    baseline_effects = effects("baseline_jrngc")
    lc10_effects = effects("full_aux_lc10")
    effect_means = {
        field: _mean(row[field] for row in concat_effects)
        for field in ["delta_auroc", "delta_auprc", "delta_coefficient_r", "relative_mse_degradation"]
    }
    positive_counts = {
        field: sum(row[field] > 0 for row in concat_effects)
        for field in ["delta_auroc", "delta_auprc", "delta_coefficient_r"]
    }
    gates = config["strong_success_gates"]
    effect_cfg = gates["repair_minus_concat"]
    effect_passed = bool(
        effect_means["delta_auroc"] >= effect_cfg["mean_delta_auroc_min"]
        and effect_means["delta_auprc"] >= effect_cfg["mean_delta_auprc_min"]
        and effect_means["delta_coefficient_r"] >= effect_cfg["mean_delta_coefficient_r_min"]
        and all(count >= effect_cfg["positive_data_seeds_min"] for count in positive_counts.values())
    )
    mse_cfg = gates["pure_mse"]
    mse_passed = bool(
        effect_means["relative_mse_degradation"] <= mse_cfg["mean_relative_degradation_max"]
        and all(
            row["relative_mse_degradation"] <= mse_cfg["per_data_seed_relative_degradation_max"]
            for row in concat_effects
        )
    )
    lc10_means = {
        field: _mean(row[field] for row in lc10_effects)
        for field in ["delta_auroc", "delta_auprc", "delta_coefficient_r"]
    }
    lc10_cfg = gates["full_aux_lc10_safety"]
    lc10_passed = bool(
        lc10_means["delta_auroc"] >= lc10_cfg["mean_delta_auroc_min"]
        and lc10_means["delta_auprc"] >= lc10_cfg["mean_delta_auprc_min"]
        and lc10_means["delta_coefficient_r"] >= lc10_cfg["mean_delta_coefficient_r_min"]
    )
    baseline_mean_auroc = _mean(row["delta_auroc"] for row in baseline_effects)
    baseline_passed = baseline_mean_auroc >= gates["baseline_safety"]["mean_delta_auroc_min"]
    semantic_passed = bool(
        all(row["passed"] for row in stability_rows)
        and all(
            row["methods"]["coverage_aligned_raw_chain"]["M_missing"] is not None
            and row["methods"]["coverage_aligned_raw_chain"]["partial_total_topk_jaccard"] is not None
            for row in data_seed_rows
        )
    )
    strong_passed = bool(semantic_passed and effect_passed and mse_passed and lc10_passed and baseline_passed)
    if strong_passed:
        classification = gates["classification"]["strong_success"]
    elif semantic_passed:
        classification = gates["classification"]["semantic_pass_strong_fail"]
    else:
        classification = gates["classification"]["semantic_or_numerical_fail"]
    return {
        "passed_execution_completeness": True,
        "config_sha256": config_sha,
        "run_matrix_sha256": matrix_sha,
        "record_count": sum(len(value) for value in grouped.values()),
        "selected_lambda": selected_lambda,
        "statistical_unit": "data_seed_after_averaging_two_model_seeds",
        "data_seed_level_values": data_seed_rows,
        "repair_minus_concat": {
            "effects": concat_effects,
            "means": effect_means,
            "positive_counts": positive_counts,
            "strong_effect_gate_passed": effect_passed,
            "pure_mse_gate_passed": mse_passed,
        },
        "full_aux_lc10_safety": {"effects": lc10_effects, "means": lc10_means, "passed": lc10_passed},
        "baseline_safety": {"effects": baseline_effects, "mean_delta_auroc": baseline_mean_auroc, "passed": baseline_passed},
        "semantic_gate": {"passed": semantic_passed, "records": stability_rows},
        "strong_success_passed": strong_passed,
        "classification": classification,
        "no_further_tuning_authorized": True,
    }
