"""Aggregate and mechanically gate Phase 9 Stage B held-out confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from knowledge_metrics import topk_edges_exact  # noqa: E402
from phase9_audit_stageb import (  # noqa: E402
    validate_release_lock,
    validate_smoke_gate,
)


STAGE = "B_AUTODL_CONFIRMATION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def read_matrix(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_number(value) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def mean(values: Iterable[object]) -> Optional[float]:
    numbers = [
        number
        for value in values
        if (number := finite_number(value)) is not None
    ]
    return None if not numbers else float(np.mean(numbers))


def median(values: Iterable[object]) -> Optional[float]:
    numbers = [
        number
        for value in values
        if (number := finite_number(value)) is not None
    ]
    return None if not numbers else float(np.median(numbers))


def max_abs_nested(left: object, right: object) -> float:
    if left is None and right is None:
        return 0.0
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return float("inf")
        return max(
            (max_abs_nested(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return max(
            (max_abs_nested(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def checkpoint_max_abs(left: Path, right: Path) -> float:
    left_state = torch.load(left, map_location="cpu")["model_state"]
    right_state = torch.load(right, map_location="cpu")["model_state"]
    if set(left_state) != set(right_state):
        return float("inf")
    maximum = 0.0
    for name in left_state:
        maximum = max(
            maximum,
            float(torch.max(torch.abs(left_state[name] - right_state[name]))),
        )
    return maximum


def score_objects_max_abs(left: Path, right: Path) -> float:
    with np.load(left, allow_pickle=False) as a, np.load(
        right, allow_pickle=False
    ) as b:
        if set(a.files) != set(b.files):
            return float("inf")
        return max(
            (
                float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key]))))
                for key in a.files
            ),
            default=0.0,
        )


def read_run(root: Path, row: Mapping[str, str]) -> Dict[str, object]:
    run_dir = root / "runs" / row["run_id"]
    required = [
        "status.json",
        "config.json",
        "sampled_attribution_audit.json",
        "sampled_attribution_objects.npz",
        "fixed_target_interventions.json",
        "audit_profile.json",
        "checkpoint.pt",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete run {row['run_id']}: {missing}")
    status = load_json(run_dir / "status.json")
    if (
        status.get("status") != "complete"
        or status.get("smoke")
        or status.get("formal_result") is not True
        or status.get("confirmation_candidate") is not True
    ):
        raise RuntimeError(
            f"Run is not a complete Stage B record: {row['run_id']}"
        )
    audit = load_json(run_dir / "sampled_attribution_audit.json")
    interventions = load_json(run_dir / "fixed_target_interventions.json")
    mixing_path = run_dir / "coordinate_mixing_audit.json"
    mixing = load_json(mixing_path) if mixing_path.is_file() else None
    horizon_path = run_dir / "horizon_sensitivity.json"
    horizon = load_json(horizon_path) if horizon_path.is_file() else None
    profile = load_json(run_dir / "audit_profile.json")
    with np.load(
        run_dir / "sampled_attribution_objects.npz",
        allow_pickle=False,
    ) as arrays:
        partial_total_max_abs = float(
            np.max(
                np.abs(
                    arrays["s_total_nominal"]
                    - arrays["s_partial_nominal"]
                )
            )
        )
    delta = interventions.get("fixed_target_prediction_mse_delta", {})
    total_metrics = audit.get("total_nominal_metrics")
    partial_metrics = audit.get("partial_nominal_metrics")
    record = {
        "run_id": row["run_id"],
        "data_unit": row["data_unit"],
        "dataset_kind": row["dataset_kind"],
        "method": row["method"],
        "replicate": int(row["replicate"]),
        "evidence_role": row["evidence_role"],
        "duplicate_of": row["duplicate_of"] or None,
        "missing_route": audit.get("missing_route_relative_magnitude"),
        "partial_total_pearson": audit.get(
            "partial_total_nominal_pearson"
        ),
        "partial_total_jaccard": audit.get(
            "partial_total_nominal_topk_jaccard"
        ),
        "temporal_tail_median": audit.get(
            "temporal_tail_statistics", {}
        ).get("median"),
        "mask_c_delta": delta.get("mask_c"),
        "coordinate_entropy_median": (
            None
            if mixing is None
            else mixing.get("normalized_source_entropy", {}).get("median")
        ),
        "h64_h128_mass_ratio": (
            None
            if horizon is None
            else horizon.get(
                "offdiagonal_cumulative_mass_ratio_vs_H128", {}
            ).get("64")
        ),
        "horizon_nominal_max_abs": (
            None
            if horizon is None
            else max(
                horizon.get(
                    "nominal_score_max_abs_difference_vs_H128", {}
                ).values(),
                default=float("inf"),
            )
        ),
        "baseline_partial_total_max_abs": partial_total_max_abs,
        "total_auroc": (
            None if total_metrics is None else total_metrics.get("auroc")
        ),
        "partial_auroc": (
            None if partial_metrics is None else partial_metrics.get("auroc")
        ),
        "fixed_target_prediction_mse": status.get(
            "fixed_target_prediction_mse"
        ),
        "predictor_initial_sha256": status.get(
            "predictor_initial_sha256"
        ),
        "wall_seconds": status.get("wall_seconds"),
        "no_nan_inf": status.get("no_nan_inf") is True,
        "deterministic_algorithms": (
            status.get("deterministic_algorithms") is True
        ),
        "formal_result": status.get("formal_result") is True,
        "confirmation_candidate": (
            status.get("confirmation_candidate") is True
        ),
        "approved_commit": status.get("approved_commit"),
        "release_token_sha256": status.get("release_token_sha256"),
        "audit_profile": profile,
        "run_dir": str(run_dir),
    }
    numeric = [
        value
        for key, value in record.items()
        if key
        not in {
            "run_id",
            "data_unit",
            "dataset_kind",
            "method",
            "evidence_role",
            "duplicate_of",
            "predictor_initial_sha256",
            "no_nan_inf",
            "deterministic_algorithms",
            "formal_result",
            "confirmation_candidate",
            "approved_commit",
            "release_token_sha256",
            "audit_profile",
            "run_dir",
        }
        and value is not None
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        record["no_nan_inf"] = False
    return record


def unit_records(
    runs: Sequence[Mapping[str, object]],
    method: str,
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for run in runs:
        if (
            run["evidence_role"] == "conditional_external_confirmation"
            and run["method"] == method
        ):
            grouped[str(run["data_unit"])].append(run)
    output = []
    fields = [
        "missing_route",
        "partial_total_pearson",
        "partial_total_jaccard",
        "temporal_tail_median",
        "mask_c_delta",
        "coordinate_entropy_median",
        "h64_h128_mass_ratio",
        "horizon_nominal_max_abs",
        "total_auroc",
        "partial_auroc",
        "fixed_target_prediction_mse",
    ]
    for data_unit, members in sorted(grouped.items()):
        record: Dict[str, object] = {
            "data_unit": data_unit,
            "dataset_kind": members[0]["dataset_kind"],
            "method": method,
            "replicate_count": len(members),
        }
        for field in fields:
            record[f"{field}_mean"] = mean(
                member[field] for member in members
            )
        record["partial_total_pearson_defined_count"] = sum(
            finite_number(member["partial_total_pearson"]) is not None
            for member in members
        )
        record["horizon_nominal_max_abs_max"] = max(
            (
                float(member["horizon_nominal_max_abs"])
                for member in members
                if finite_number(member["horizon_nominal_max_abs"]) is not None
            ),
            default=float("inf"),
        )
        record["missing_route_seed_pass_count"] = sum(
            finite_number(member["missing_route"]) is not None
            and float(member["missing_route"]) >= 0.20
            for member in members
        )
        record["mask_c_positive_seed_count"] = sum(
            finite_number(member["mask_c_delta"]) is not None
            and float(member["mask_c_delta"]) > 0
            for member in members
        )
        output.append(record)
    return output


def architecture_gate(
    units: Sequence[Mapping[str, object]],
    *,
    gates: Mapping[str, object],
) -> Dict[str, object]:
    required_units = int(gates["qualifying_unit_count"])
    required_seeds = int(gates["qualifying_seed_count"])
    missing_qualifying = [
        unit
        for unit in units
        if finite_number(unit["missing_route_mean"]) is not None
        and float(unit["missing_route_mean"]) >= float(gates["missing_route_min"])
        and int(unit["missing_route_seed_pass_count"]) >= required_seeds
    ]
    mask_qualifying = [
        unit
        for unit in units
        if finite_number(unit["mask_c_delta_mean"]) is not None
        and float(unit["mask_c_delta_mean"]) > 0
        and int(unit["mask_c_positive_seed_count"]) >= required_seeds
    ]
    pearson_median = median(
        unit["partial_total_pearson_mean"] for unit in units
    )
    pearson_all_defined = all(
        int(unit["partial_total_pearson_defined_count"])
        == int(gates["seed_count"])
        for unit in units
    )
    mask_median = median(unit["mask_c_delta_mean"] for unit in units)
    entropy_median = median(
        unit["coordinate_entropy_median_mean"] for unit in units
    )
    tail_median = median(
        unit["temporal_tail_median_mean"] for unit in units
    )
    netsim_units = [
        unit for unit in units if unit["dataset_kind"] == "netsim"
    ]
    netsim_discrepancy = [
        unit
        for unit in netsim_units
        if (
            finite_number(unit["partial_total_pearson_mean"]) is not None
            and float(unit["partial_total_pearson_mean"])
            <= float(gates["partial_total_pearson_max"])
        )
        or (
            finite_number(unit["partial_total_jaccard_mean"]) is not None
            and float(unit["partial_total_jaccard_mean"])
            <= float(gates["topk_jaccard_max"])
        )
    ]
    horizon_qualifying = [
        unit
        for unit in units
        if finite_number(unit["h64_h128_mass_ratio_mean"]) is not None
        and float(unit["h64_h128_mass_ratio_mean"])
        >= float(gates["h64_h128_ratio_qualifying_min"])
    ]
    horizon_min = min(
        (
            float(unit["h64_h128_mass_ratio_mean"])
            for unit in units
            if finite_number(unit["h64_h128_mass_ratio_mean"]) is not None
        ),
        default=float("-inf"),
    )
    nominal_max = max(
        (
            float(unit["horizon_nominal_max_abs_max"])
            for unit in units
            if finite_number(unit["horizon_nominal_max_abs_max"]) is not None
        ),
        default=float("inf"),
    )
    checks = {
        "unit_count": len(units) == int(gates["unit_count"]),
        "missing_route": len(missing_qualifying) >= required_units,
        "mask_c_direction": len(mask_qualifying) >= required_units,
        "mask_c_median": (
            mask_median is not None
            and mask_median >= float(gates["mask_c_delta_median_min"])
        ),
        "partial_total_pearson": (
            pearson_all_defined
            and
            pearson_median is not None
            and pearson_median <= float(gates["partial_total_pearson_max"])
        ),
        "netsim_discrepancy": (
            len(netsim_discrepancy)
            >= int(gates["netsim_discrepancy_subject_count"])
        ),
        "coordinate_entropy": (
            entropy_median is not None
            and entropy_median
            >= float(gates["coordinate_entropy_median_min"])
        ),
        "temporal_tail": (
            tail_median is not None
            and tail_median >= float(gates["temporal_tail_median_min"])
        ),
        "horizon_qualifying": len(horizon_qualifying) >= required_units,
        "horizon_absolute_min": (
            horizon_min >= float(gates["h64_h128_ratio_absolute_min"])
        ),
        "horizon_nominal_invariance": (
            nominal_max <= float(gates["nominal_horizon_max_abs"])
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "missing_route_qualifying_units": len(missing_qualifying),
        "mask_c_qualifying_units": len(mask_qualifying),
        "netsim_discrepancy_units": len(netsim_discrepancy),
        "horizon_qualifying_units": len(horizon_qualifying),
        "mask_c_unit_median": mask_median,
        "partial_total_pearson_unit_median": pearson_median,
        "coordinate_entropy_unit_median": entropy_median,
        "temporal_tail_unit_median": tail_median,
        "h64_h128_unit_minimum": horizon_min,
        "nominal_horizon_max_abs": nominal_max,
    }


def determinism_audit(
    root: Path,
    runs: Sequence[Mapping[str, object]],
    tolerance: float,
) -> Dict[str, object]:
    by_id = {str(run["run_id"]): run for run in runs}
    comparisons = []
    for duplicate in runs:
        original_id = duplicate.get("duplicate_of")
        if not original_id:
            continue
        original = by_id[str(original_id)]
        original_dir = Path(str(original["run_dir"]))
        duplicate_dir = Path(str(duplicate["run_dir"]))
        score_diff = score_objects_max_abs(
            original_dir / "sampled_attribution_objects.npz",
            duplicate_dir / "sampled_attribution_objects.npz",
        )
        checkpoint_diff = checkpoint_max_abs(
            original_dir / "checkpoint.pt",
            duplicate_dir / "checkpoint.pt",
        )
        metric_diff = max_abs_nested(
            load_json(original_dir / "sampled_attribution_audit.json"),
            load_json(duplicate_dir / "sampled_attribution_audit.json"),
        )
        with np.load(
            original_dir / "sampled_attribution_objects.npz",
            allow_pickle=False,
        ) as a, np.load(
            duplicate_dir / "sampled_attribution_objects.npz",
            allow_pickle=False,
        ) as b:
            score_a = np.asarray(a["s_total_nominal"])
            score_b = np.asarray(b["s_total_nominal"])
        audit = load_json(original_dir / "sampled_attribution_audit.json")
        metrics = audit.get("total_nominal_metrics")
        edge_count = 0 if metrics is None else int(metrics.get("n_true_edges", 0))
        topk_equal = (
            True
            if edge_count < 1
            else topk_edges_exact(
                score_a,
                k=edge_count,
                exclude_diag=True,
            )
            == topk_edges_exact(
                score_b,
                k=edge_count,
                exclude_diag=True,
            )
        )
        passed = (
            score_diff <= tolerance
            and checkpoint_diff <= tolerance
            and metric_diff <= tolerance
            and topk_equal
        )
        comparisons.append(
            {
                "original": original_id,
                "duplicate": duplicate["run_id"],
                "method": duplicate["method"],
                "score_max_abs": score_diff,
                "checkpoint_max_abs": checkpoint_diff,
                "metric_max_abs": metric_diff,
                "topk_equal": topk_equal,
                "passed": passed,
            }
        )
    return {
        "passed": len(comparisons) == 2
        and all(item["passed"] for item in comparisons),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }


def report_markdown(result: Mapping[str, object]) -> str:
    lines = [
        "# Phase 9 Audit-Generality Stage B Confirmation",
        "",
        f"**Decision:** `{result['decision']}`",
        "",
        "## Integrity",
        "",
        f"- semantic integrity: `{result['semantic_integrity_passed']}`",
        f"- deterministic execution settings: "
        f"`{result['deterministic_execution']['passed']}`",
        f"- Stage A exact-repeat determinism: "
        f"`{result['stage_a_prerequisite']['determinism_passed']}`",
        f"- predictor initialization parity: `{result['predictor_parity']['passed']}`",
        f"- GPU hours: `{result['gpu_hours']:.4f}`",
        "",
        "## Architecture gates",
        "",
    ]
    for method, gate in result["architecture_gates"].items():
        lines.extend(
            [
                f"### {method}",
                "",
                f"- passed: `{gate['passed']}`",
                f"- missing-route qualifying units: "
                f"`{gate['missing_route_qualifying_units']}/6`",
                f"- mask-c qualifying units: "
                f"`{gate['mask_c_qualifying_units']}/6`",
                f"- median partial-total Pearson: "
                f"`{gate['partial_total_pearson_unit_median']}`",
                f"- median coordinate entropy: "
                f"`{gate['coordinate_entropy_unit_median']}`",
                f"- median temporal tail mass: "
                f"`{gate['temporal_tail_unit_median']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Evidence boundary",
            "",
            "- Stage B uses the separately sealed confirmation subjects and seeds.",
            "- Manuscript use requires this Stage B decision and the frozen Stage A "
            "boundary to be reported together.",
            "- MoCap has no accepted direct graph ground truth.",
            "- Attribution beyond H=128 remains unassessed.",
            "- The result does not establish improved graph recovery or a "
            "successful repair method.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    config_path = root / "config_snapshot.json"
    matrix_path = root / "matrix_snapshot.csv"
    token_path = root / "release_token_snapshot.json"
    stage_a_decision_path = root / "stage_a_decision_snapshot.json"
    config = load_json(config_path)
    matrix = read_matrix(matrix_path)
    stored_release_lock = load_json(root / "release_lock.json")
    actual_release_lock = validate_release_lock(
        config_path=config_path,
        matrix_path=matrix_path,
        token_path=token_path,
        stage_a_decision_path=stage_a_decision_path,
    )
    if stored_release_lock != actual_release_lock:
        raise RuntimeError("Stage B release lock changed after execution")
    smoke_gate = validate_smoke_gate(
        root / "smoke_validation_snapshot.json",
        release_lock=actual_release_lock,
    )
    stage_a_decision = load_json(stage_a_decision_path)
    stage_a_prerequisite = {
        "decision": stage_a_decision.get("decision"),
        "unlock_stage_b": stage_a_decision.get("unlock_stage_b") is True,
        "semantic_integrity_passed": (
            stage_a_decision.get("semantic_integrity_passed") is True
        ),
        "determinism_passed": (
            stage_a_decision.get("determinism", {}).get("passed") is True
        ),
        "canonical_sha256": stored_release_lock[
            "stage_a_decision_canonical_sha256"
        ],
    }
    stage_a_prerequisite["passed"] = all(
        (
            stage_a_prerequisite["decision"] == "UNLOCK_STAGE_B",
            stage_a_prerequisite["unlock_stage_b"],
            stage_a_prerequisite["semantic_integrity_passed"],
            stage_a_prerequisite["determinism_passed"],
        )
    )
    expected_rows = [
        row
        for row in matrix
        if row["stage"] == STAGE
        and row["evidence_role"] == "conditional_external_confirmation"
    ]
    if len(expected_rows) != 36:
        raise RuntimeError("Frozen Stage B matrix cardinality mismatch")
    runs = [read_run(root, row) for row in expected_rows]
    primary = runs
    gates = config["gates"]
    baseline_runs = [run for run in primary if run["method"] == "baseline"]
    baseline_partial_max = max(
        float(run["baseline_partial_total_max_abs"])
        for run in baseline_runs
    )
    baseline_missing_max = max(
        float(run["missing_route"]) for run in baseline_runs
    )
    finite_pass = all(bool(run["no_nan_inf"]) for run in runs)
    deterministic_execution = {
        "passed": all(
            bool(run["deterministic_algorithms"]) for run in runs
        ),
        "record_count": len(runs),
    }
    formal_metadata_pass = all(
        bool(run["formal_result"])
        and bool(run["confirmation_candidate"])
        and run["approved_commit"] == stored_release_lock["approved_commit"]
        and run["release_token_sha256"]
        == stored_release_lock["release_token_sha256"]
        for run in runs
    )
    profile_pass = all(
        (
            run["audit_profile"]["audit_dimensions"][
                "partial_score_route_completeness"
            ]
            == "PARTIALLY COVERED"
        )
        for run in primary
        if run["method"] in {"mamba_concat", "tcn_concat"}
    )
    predictor_groups: Dict[tuple, Dict[str, str]] = defaultdict(dict)
    for run in primary:
        if run["method"] in {"mamba_concat", "tcn_concat"}:
            predictor_groups[
                (run["data_unit"], run["replicate"])
            ][str(run["method"])] = str(run["predictor_initial_sha256"])
    predictor_comparisons = [
        {
            "data_unit": key[0],
            "replicate": key[1],
            "mamba_sha": values.get("mamba_concat"),
            "tcn_sha": values.get("tcn_concat"),
            "equal": (
                set(values) == {"mamba_concat", "tcn_concat"}
                and values["mamba_concat"] == values["tcn_concat"]
            ),
        }
        for key, values in sorted(predictor_groups.items())
    ]
    predictor_parity = {
        "passed": len(predictor_comparisons) == 12
        and all(item["equal"] for item in predictor_comparisons),
        "comparisons": predictor_comparisons,
    }
    semantic_checks = {
        "release_lock": stored_release_lock == actual_release_lock,
        "smoke_gate": smoke_gate is not None,
        "stage_a_prerequisite": stage_a_prerequisite["passed"],
        "finite": finite_pass,
        "deterministic_execution": deterministic_execution["passed"],
        "formal_result_metadata": formal_metadata_pass,
        "baseline_partial_total": (
            baseline_partial_max
            <= float(gates["baseline_partial_total_max_abs"])
        ),
        "baseline_missing_route": (
            baseline_missing_max
            <= float(gates["baseline_missing_route_max"])
        ),
        "auxiliary_profile_labels": profile_pass,
        "predictor_initialization_parity": predictor_parity["passed"],
    }
    semantic_passed = all(semantic_checks.values())
    units = {
        method: unit_records(primary, method)
        for method in ("mamba_concat", "tcn_concat")
    }
    architecture_gates = {
        method: architecture_gate(records, gates=gates)
        for method, records in units.items()
    }
    baseline_netsim_aurocs = [
        run["total_auroc"]
        for run in baseline_runs
        if run["dataset_kind"] == "netsim"
    ]
    baseline_netsim_median = median(baseline_netsim_aurocs)
    graph_context = {
        "baseline_netsim_auroc_median": baseline_netsim_median,
        "strong_operating_point": (
            baseline_netsim_median is not None
            and baseline_netsim_median
            >= float(gates["baseline_netsim_auroc_context_min"])
        ),
        "performance_gate": False,
    }
    gpu_hours = sum(float(run["wall_seconds"]) for run in runs) / 3600.0
    runtime_pass = gpu_hours <= float(gates["autodl_gpu_hour_hard_cap"])
    confirmed = (
        semantic_passed
        and runtime_pass
        and all(gate["passed"] for gate in architecture_gates.values())
    )
    decision = (
        "CONFIRMED_AUDIT_GENERALITY"
        if confirmed
        else (
            "INVALID_STAGE_B"
            if not semantic_passed
            else "STAGE_B_CONFIRMATION_GATE_FAIL"
        )
    )
    result = {
        "decision": decision,
        "confirmed_audit_generality": confirmed,
        "manuscript_evidence_eligible": confirmed,
        "semantic_integrity_passed": semantic_passed,
        "semantic_checks": semantic_checks,
        "stage_a_prerequisite": stage_a_prerequisite,
        "smoke_gate": smoke_gate,
        "baseline_partial_total_max_abs": baseline_partial_max,
        "baseline_missing_route_max": baseline_missing_max,
        "predictor_parity": predictor_parity,
        "deterministic_execution": deterministic_execution,
        "unit_records": units,
        "architecture_gates": architecture_gates,
        "graph_context": graph_context,
        "gpu_hours": gpu_hours,
        "runtime_passed": runtime_pass,
        "primary_run_count": len(primary),
        "duplicate_run_count": 0,
        "stage_b_is_heldout_confirmation": True,
        "mass_beyond_H128_assessed": False,
        "allowed_claim": (
            "Under a preregistered bounded raw-chain audit, omitted-route "
            "attribution, fixed-target auxiliary-route use, and "
            "partial-versus-total score disagreement were reproduced across "
            "held-out NetSim subjects, nonoverlapping MoCap segments, and two "
            "causal auxiliary preprocessors."
        ),
    }
    atomic_json(root / "stageb_confirmation_decision.json", result)
    atomic_text(root / "STAGEB_DECISION.md", report_markdown(result))
    return 0 if semantic_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
