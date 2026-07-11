"""Release-locked protocol for the final bounded Phase 8 repair cycle."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from phase8_coverage import METHOD_IMPLEMENTATION_REGISTRY
from phase8_protocol import file_sha256, load_json, load_run_matrix


PILOT_LAMBDAS = (0.0003, 0.001, 0.003)
PILOT_DATA_SEEDS = (12001, 12002, 12003)
PILOT_MODEL_SEEDS = (22001, 22002)
CONFIRMATION_DATA_SEEDS = (13001, 13002, 13003, 13004, 13005)
CONFIRMATION_MODEL_SEEDS = (23001, 23002)
CONFIRMATION_METHODS = (
    "baseline_jrngc",
    "concat_x_only",
    "full_aux_lc10",
    "coverage_aligned_raw_chain",
)
FORBIDDEN_PHASE7_DATA_SEEDS = frozenset({4, 5, 6, 7, 8})


def _optional_int(row: Mapping[str, str], key: str) -> Optional[int]:
    value = row.get(key, "")
    return None if value in {None, "", "not_applicable"} else int(value)


def _optional_float(row: Mapping[str, str], key: str) -> Optional[float]:
    value = row.get(key, "")
    return None if value in {None, "", "not_applicable"} else float(value)


def _bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Expected lowercase true/false, got {value!r}")


def _float_equal(left: Optional[float], right: float) -> bool:
    return left is not None and math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)


def resolve_final_run_record(
    row: Mapping[str, str],
    config: Mapping[str, object],
    *,
    config_sha256: str,
) -> Dict[str, object]:
    method = row["method"]
    block = row["block"]
    phase = row["phase"]
    method_map = config["method_classes"]  # type: ignore[index]
    blocks = config["blocks"]  # type: ignore[index]
    if method not in method_map:
        raise ValueError(f"Unknown final-cycle method {method!r}")
    if METHOD_IMPLEMENTATION_REGISTRY.get(method) != method_map[method]:  # type: ignore[index]
        raise ValueError(f"Method implementation registry mismatch for {method!r}")
    if block not in blocks:
        raise ValueError(f"Unknown final-cycle block {block!r}")
    root_key = "gated_confirmation" if phase == "gated_confirmation" else "formal"
    root = config["output_roots"][root_key]  # type: ignore[index]
    block_config = blocks[block]  # type: ignore[index]
    return {
        "record_id": row["record_id"],
        "formal_result": _bool(row["formal_result"]),
        "phase": phase,
        "block": block,
        "method": method,
        "method_class": method_map[method],  # type: ignore[index]
        "data_seed": _optional_int(row, "data_seed"),
        "model_seed": _optional_int(row, "model_seed"),
        "perturbation_seed": _optional_int(row, "perturbation_seed"),
        "jacobian_seed": _optional_int(row, "jacobian_seed"),
        "d": _optional_int(row, "d"),
        "T": _optional_int(row, "T"),
        "K": _optional_int(row, "K"),
        "d_cond": _optional_int(row, "d_cond"),
        "max_iter": _optional_int(row, "max_iter"),
        "lambda_x": _optional_float(row, "lambda_x"),
        "lambda_c": _optional_float(row, "lambda_c"),
        "raw_chain_lambda": _optional_float(row, "raw_chain_lambda"),
        "training_attribution_horizon": row["training_attribution_horizon"],
        "primary_graph_score": row["primary_graph_score"],
        "coefficient_metric": row["coefficient_metric"],
        "secondary_history_score": row["secondary_history_score"],
        "n_min": _optional_int(row, "n_min"),
        "notes": row.get("notes", ""),
        "checkpoint_policy": block_config["checkpoint_policy"],  # type: ignore[index]
        "gating_checkpoint": block_config["gating_checkpoint"],  # type: ignore[index]
        "output_root": f"{root}/{config_sha256}/runs/{row['record_id']}",
        "confirmation_sealed": phase == "gated_confirmation",
    }


def _validate_common_row(row: Mapping[str, str], failures: List[Dict[str, object]]) -> None:
    record_id = row.get("record_id")
    expected = {
        "formal_result": "true",
        "d": "8",
        "T": "500",
        "K": "1",
        "d_cond": "4" if row.get("method") != "baseline_jrngc" else "0",
        "max_iter": "2000",
        "primary_graph_score": "S_GC_total_nominal",
        "coefficient_metric": "coefficient_r_total_lag1",
        "secondary_history_score": "S_reliable_history",
        "n_min": "50",
    }
    for key, value in expected.items():
        if row.get(key) != value:
            failures.append({
                "type": "row_field",
                "record_id": record_id,
                "field": key,
                "actual": row.get(key),
                "expected": value,
            })
    data_seed = _optional_int(row, "data_seed")
    if data_seed in FORBIDDEN_PHASE7_DATA_SEEDS:
        failures.append({"type": "forbidden_phase7_seed", "record_id": record_id, "data_seed": data_seed})


def validate_final_run_matrix(config_path: Path, matrix_path: Path) -> Dict[str, object]:
    config = load_json(config_path)
    failures: List[Dict[str, object]] = []
    if config.get("protocol_mode") != "phase8_final":
        failures.append({"type": "protocol_mode", "actual": config.get("protocol_mode")})
    config_sha = file_sha256(config_path)
    matrix_sha = file_sha256(matrix_path)
    configured_matrix = config.get("run_matrix", {})
    if matrix_sha != configured_matrix.get("sha256"):
        failures.append({
            "type": "matrix_sha256",
            "actual": matrix_sha,
            "expected": configured_matrix.get("sha256"),
        })
    rows = load_run_matrix(matrix_path)
    expected_count = int(configured_matrix.get("record_count", -1))
    if len(rows) != expected_count:
        failures.append({"type": "record_count", "actual": len(rows), "expected": expected_count})
    identifiers = [row.get("record_id") for row in rows]
    if len(set(identifiers)) != len(identifiers):
        failures.append({"type": "duplicate_record_ids"})

    stage = config.get("execution_stage")
    resolved: List[Dict[str, object]] = []
    for row in rows:
        _validate_common_row(row, failures)
        try:
            resolved.append(resolve_final_run_record(row, config, config_sha256=config_sha))
        except Exception as exc:
            failures.append({"type": "resolution", "record_id": row.get("record_id"), "error": str(exc)})

    if stage == "lambda_tradeoff":
        expected_combinations = {
            (value, data_seed, model_seed)
            for value in PILOT_LAMBDAS
            for data_seed in PILOT_DATA_SEEDS
            for model_seed in PILOT_MODEL_SEEDS
        }
        actual_combinations = set()
        for row in rows:
            if row.get("phase") != "final_lambda_tradeoff" or row.get("block") != "repair_lambda_tradeoff":
                failures.append({"type": "pilot_phase_or_block", "record_id": row.get("record_id")})
            if row.get("method") != "coverage_aligned_raw_chain":
                failures.append({"type": "pilot_method", "record_id": row.get("record_id")})
            value = _optional_float(row, "raw_chain_lambda")
            actual_combinations.add((value, _optional_int(row, "data_seed"), _optional_int(row, "model_seed")))
            if not any(_float_equal(value, expected) for expected in PILOT_LAMBDAS):
                failures.append({"type": "pilot_lambda", "record_id": row.get("record_id"), "actual": value})
            if _optional_int(row, "jacobian_seed") != 32001:
                failures.append({"type": "pilot_schedule_seed", "record_id": row.get("record_id")})
        if actual_combinations != expected_combinations:
            failures.append({
                "type": "pilot_run_matrix_combinations",
                "missing": sorted(expected_combinations - actual_combinations),
                "unexpected": sorted(actual_combinations - expected_combinations),
            })
    elif stage == "confirmation":
        selected_lambda = float(config["selected_lambda"])
        expected_combinations = {
            (method, data_seed, model_seed)
            for method in CONFIRMATION_METHODS
            for data_seed in CONFIRMATION_DATA_SEEDS
            for model_seed in CONFIRMATION_MODEL_SEEDS
        }
        actual_combinations = set()
        for row in rows:
            if row.get("phase") != "gated_confirmation" or row.get("block") != "repair_confirmation":
                failures.append({"type": "confirmation_phase_or_block", "record_id": row.get("record_id")})
            method = row.get("method")
            actual_combinations.add((method, _optional_int(row, "data_seed"), _optional_int(row, "model_seed")))
            value = _optional_float(row, "raw_chain_lambda")
            if method == "coverage_aligned_raw_chain":
                if not _float_equal(value, selected_lambda):
                    failures.append({"type": "confirmation_lambda", "record_id": row.get("record_id")})
            elif value is not None:
                failures.append({"type": "comparator_raw_chain_lambda", "record_id": row.get("record_id")})
            if _optional_int(row, "jacobian_seed") != 33001:
                failures.append({"type": "confirmation_schedule_seed", "record_id": row.get("record_id")})
        if actual_combinations != expected_combinations:
            failures.append({
                "type": "confirmation_run_matrix_combinations",
                "missing": sorted(expected_combinations - actual_combinations),
                "unexpected": sorted(actual_combinations - expected_combinations),
            })
    else:
        failures.append({"type": "execution_stage", "actual": stage})

    return {
        "passed": not failures,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "run_matrix_path": str(matrix_path),
        "run_matrix_sha256": matrix_sha,
        "execution_stage": stage,
        "record_count": len(rows),
        "resolved_records": resolved,
        "failures": failures,
    }


def validate_final_authorization(
    path: Path,
    *,
    release_commit: str,
    config_sha256: str,
    matrix_sha256: str,
    phase: str,
    block: str,
    confirmation_token_path: Optional[Path] = None,
) -> Dict[str, object]:
    payload = load_json(path)
    is_confirmation = phase == "gated_confirmation" or block == "repair_confirmation"
    expected_authorization = (
        "GPT_APPROVED_PHASE8_FINAL_CONDITIONAL_CONFIRMATION"
        if is_confirmation
        else "GPT_APPROVED_PHASE8_FINAL_BOUNDED_LAMBDA_TRADEOFF"
    )
    expected = {
        "authorization": expected_authorization,
        "release_commit": release_commit,
        "config_sha256": config_sha256,
        "run_matrix_sha256": matrix_sha256,
    }
    failures = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if block not in set(payload.get("allowed_blocks", [])):
        failures["allowed_blocks"] = {"actual": payload.get("allowed_blocks"), "required": block}
    if failures:
        raise PermissionError(f"Final Phase 8 authorization mismatch: {failures}")
    if is_confirmation:
        if confirmation_token_path is None or not confirmation_token_path.is_file():
            raise PermissionError("Conditional confirmation requires a frozen release token")
        token = load_json(confirmation_token_path)
        token_expected = {
            "authorization": "GPT_APPROVED_PHASE8_FINAL_CONDITIONAL_CONFIRMATION",
            "pilot_go_passed": True,
            "source_commit": release_commit,
            "config_sha256": config_sha256,
            "run_matrix_sha256": matrix_sha256,
            "estimator_schedule_sha256": payload.get("estimator_schedule_sha256"),
            "selected_lambda": payload.get("selected_lambda"),
            "pilot_aggregate_sha256": payload.get("pilot_aggregate_sha256"),
        }
        token_failures = {
            key: {"actual": token.get(key), "expected": value}
            for key, value in token_expected.items()
            if token.get(key) != value
        }
        if token_failures:
            raise PermissionError(f"Conditional confirmation token mismatch: {token_failures}")
        payload = {**payload, "confirmation_token_sha256": file_sha256(confirmation_token_path)}
    return payload


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
