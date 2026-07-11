"""Frozen Phase 8 run-matrix, release-lock, and provenance validation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from phase8_coverage import METHOD_IMPLEMENTATION_REGISTRY, build_balanced_lag_schedule, schedule_sha256


EXPECTED_MATRIX_SHA256 = "cc82b4283dfb28f5180891f1d0716d868bc10ddad53e64dbf174b9a36a04ac1a"
EXPECTED_RECORDS = 135
EXPECTED_FORMAL = 130
EXPECTED_NON_EVIDENTIARY = 5
EXPECTED_CONFIRMATION = 50
FORBIDDEN_PHASE7_DATA_SEEDS = frozenset({4, 5, 6, 7, 8})
EXPECTED_GATE_OBJECTS = {
    "capacity_replication": {
        "gate_graph_score": "S_partial_nominal",
        "gate_prediction_metric": "fixed_target_prediction_mse",
        "secondary_graph_score": "S_GC_total_nominal",
        "gate_coefficient_metric": "not_applicable",
    },
    "coefficient_replication": {
        "gate_graph_score": "S_partial_nominal",
        "gate_prediction_metric": "not_applicable",
        "secondary_graph_score": "S_GC_total_nominal",
        "gate_coefficient_metric": "coefficient_r_partial_lag1",
        "secondary_coefficient_metric": "coefficient_r_total_lag1",
    },
    "fixed_target_interventions": {
        "gate_graph_score": "not_applicable",
        "gate_prediction_metric": "fixed_target_prediction_mse_delta",
        "secondary_graph_score": "not_applicable",
        "gate_coefficient_metric": "not_applicable",
    },
    "repair_pilot": {
        "gate_graph_score": "S_GC_total_nominal",
        "gate_prediction_metric": "fixed_target_prediction_mse",
        "secondary_graph_score": "S_reliable_history",
        "gate_coefficient_metric": "coefficient_r_total_lag1",
    },
    "repair_confirmation": {
        "gate_graph_score": "S_GC_total_nominal",
        "gate_prediction_metric": "fixed_target_prediction_mse",
        "secondary_graph_score": "S_reliable_history",
        "gate_coefficient_metric": "coefficient_r_total_lag1",
    },
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_sha256(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_matrix(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _int(row: Mapping[str, str], key: str) -> Optional[int]:
    value = row.get(key, "")
    return None if value in {None, "", "not_applicable"} else int(value)


def _bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Expected lowercase true/false, got {value!r}")


def resolve_run_record(
    row: Mapping[str, str],
    config: Mapping[str, object],
    *,
    config_sha256: str,
) -> Dict[str, object]:
    method_map = config["method_classes"]  # type: ignore[index]
    blocks = config["blocks"]  # type: ignore[index]
    output_roots = config["output_roots"]  # type: ignore[index]
    method = row["method"]
    block = row["block"]
    phase = row["phase"]
    if method not in method_map:
        raise ValueError(f"Unknown method {method!r} in {row['record_id']}")
    if METHOD_IMPLEMENTATION_REGISTRY.get(method) != method_map[method]:  # type: ignore[index]
        raise ValueError(
            f"Method registry mismatch for {method}: "
            f"implementation={METHOD_IMPLEMENTATION_REGISTRY.get(method)!r}, config={method_map[method]!r}"  # type: ignore[index]
        )
    if block not in blocks:
        raise ValueError(f"Unknown block {block!r} in {row['record_id']}")
    phase_root = "preflight" if phase == "preflight" else ("gated_confirmation" if phase == "gated_confirmation" else "formal")
    root = str(output_roots[phase_root])  # type: ignore[index]
    block_cfg = blocks[block]  # type: ignore[index]
    gate_mapping = EXPECTED_GATE_OBJECTS.get(block, {
        "gate_graph_score": "not_applicable_preflight",
        "gate_prediction_metric": "not_applicable_preflight",
        "secondary_graph_score": row["secondary_history_score"],
        "gate_coefficient_metric": "not_applicable_preflight",
    })
    resolved = {
        "record_id": row["record_id"],
        "formal_result": _bool(row["formal_result"]),
        "phase": phase,
        "block": block,
        "method": method,
        "method_class": method_map[method],  # type: ignore[index]
        "data_seed": _int(row, "data_seed"),
        "model_seed": _int(row, "model_seed"),
        "perturbation_seed": _int(row, "perturbation_seed"),
        "jacobian_seed": _int(row, "jacobian_seed"),
        "d": _int(row, "d"),
        "T": _int(row, "T"),
        "K": _int(row, "K"),
        "d_cond": _int(row, "d_cond"),
        "max_iter": _int(row, "max_iter"),
        "primary_graph_score": row["primary_graph_score"],
        "coefficient_metric": row["coefficient_metric"],
        **gate_mapping,
        "checkpoint_policy": block_cfg["checkpoint_policy"],  # type: ignore[index]
        "gating_checkpoint": block_cfg["gating_checkpoint"],  # type: ignore[index]
        "output_root": f"{root}/{config_sha256}/runs/{row['record_id']}",
        "confirmation_sealed": bool(phase == "gated_confirmation"),
    }
    return resolved


def validate_run_matrix(config_path: Path, matrix_path: Path) -> Dict[str, object]:
    config = load_json(config_path)
    config_sha = file_sha256(config_path)
    matrix_sha = file_sha256(matrix_path)
    failures: List[Dict[str, object]] = []
    configured_gate_objects = config.get("gate_objects", {})
    expected_config_projection = {
        "capacity_replication": {
            "replication_claim_graph_score": "S_partial_nominal",
            "replication_claim_prediction_metric": "fixed_target_prediction_mse",
            "semantic_secondary_graph_score": "S_GC_total_nominal",
        },
        "coefficient_replication": {
            "replication_claim_graph_score": "S_partial_nominal",
            "replication_claim_coefficient_metric": "coefficient_r_partial_lag1",
            "semantic_secondary_graph_score": "S_GC_total_nominal",
            "semantic_secondary_coefficient_metric": "coefficient_r_total_lag1",
        },
        "fixed_target_interventions": {
            "replication_claim_graph_score": "not_applicable",
            "replication_claim_prediction_metric": "fixed_target_prediction_mse_delta",
            "legacy_secondary_metric": "legacy_objective_delta",
        },
        "repair_pilot": {
            "method_gate_graph_score": "S_GC_total_nominal",
            "method_gate_coefficient_metric": "coefficient_r_total_lag1",
        },
        "repair_confirmation": {
            "method_gate_graph_score": "S_GC_total_nominal",
            "method_gate_coefficient_metric": "coefficient_r_total_lag1",
        },
    }
    for block_name, expected_mapping in expected_config_projection.items():
        actual_mapping = configured_gate_objects.get(block_name) if isinstance(configured_gate_objects, dict) else None
        if actual_mapping != expected_mapping:
            failures.append({
                "type": "block_gate_object_mapping",
                "block": block_name,
                "actual": actual_mapping,
                "expected": expected_mapping,
            })
    if matrix_sha != EXPECTED_MATRIX_SHA256:
        failures.append({"type": "matrix_sha256", "actual": matrix_sha, "expected": EXPECTED_MATRIX_SHA256})
    rows = load_run_matrix(matrix_path)
    if len(rows) != EXPECTED_RECORDS:
        failures.append({"type": "record_count", "actual": len(rows), "expected": EXPECTED_RECORDS})
    ids = [row["record_id"] for row in rows]
    if len(set(ids)) != len(ids):
        failures.append({"type": "duplicate_record_ids"})

    resolved: List[Dict[str, object]] = []
    for row in rows:
        try:
            item = resolve_run_record(row, config, config_sha256=config_sha)
            resolved.append(item)
        except Exception as exc:
            failures.append({"type": "resolution", "record_id": row.get("record_id"), "error": str(exc)})
            continue
        data_seed = item["data_seed"]
        if data_seed in FORBIDDEN_PHASE7_DATA_SEEDS:
            failures.append({"type": "forbidden_phase7_data_seed", "record_id": item["record_id"], "data_seed": data_seed})
        if item["block"] == "fixed_target_interventions":
            if item["primary_graph_score"] != "not_applicable" or item["coefficient_metric"] != "not_applicable":
                failures.append({"type": "intervention_graph_object", "record_id": item["record_id"]})
        else:
            if item["primary_graph_score"] != "S_GC_total_nominal":
                failures.append({"type": "primary_score", "record_id": item["record_id"], "actual": item["primary_graph_score"]})
            if item["coefficient_metric"] != "coefficient_r_total_lag1":
                failures.append({"type": "coefficient_metric", "record_id": item["record_id"], "actual": item["coefficient_metric"]})
        expected_gate = EXPECTED_GATE_OBJECTS.get(str(item["block"]))
        if expected_gate is not None:
            for key, expected_value in expected_gate.items():
                if item.get(key) != expected_value:
                    failures.append({
                        "type": "resolved_gate_object",
                        "record_id": item["record_id"],
                        "field": key,
                        "actual": item.get(key),
                        "expected": expected_value,
                    })

    formal = [item for item in resolved if item["formal_result"]]
    non_evidentiary = [item for item in resolved if not item["formal_result"]]
    confirmation = [item for item in resolved if item["phase"] == "gated_confirmation"]
    if len(formal) != EXPECTED_FORMAL:
        failures.append({"type": "formal_count", "actual": len(formal), "expected": EXPECTED_FORMAL})
    if len(non_evidentiary) != EXPECTED_NON_EVIDENTIARY:
        failures.append({"type": "non_evidentiary_count", "actual": len(non_evidentiary), "expected": EXPECTED_NON_EVIDENTIARY})
    if len(confirmation) != EXPECTED_CONFIRMATION:
        failures.append({"type": "confirmation_count", "actual": len(confirmation), "expected": EXPECTED_CONFIRMATION})
    if any(not item["confirmation_sealed"] for item in confirmation):
        failures.append({"type": "unsealed_confirmation_record"})

    block_counts: Dict[str, int] = {}
    for item in resolved:
        block = str(item["block"])
        block_counts[block] = block_counts.get(block, 0) + 1
    return {
        "passed": not failures,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "run_matrix_path": str(matrix_path),
        "run_matrix_sha256": matrix_sha,
        "record_count": len(rows),
        "formal_count": len(formal),
        "non_evidentiary_count": len(non_evidentiary),
        "confirmation_count": len(confirmation),
        "block_counts": block_counts,
        "forbidden_phase7_data_seed_matches": [
            item["record_id"] for item in resolved if item["data_seed"] in FORBIDDEN_PHASE7_DATA_SEEDS
        ],
        "all_confirmation_records_sealed": all(item["confirmation_sealed"] for item in confirmation),
        "block_gate_objects": EXPECTED_GATE_OBJECTS,
        "resolved_records": resolved,
        "failures": failures,
    }


def validate_confirmation_release_token(
    token: Mapping[str, object],
    *,
    config_sha256: str,
    run_matrix_sha256: str,
    estimator_schedule_sha256: str,
) -> None:
    expected = {
        "pilot_go_passed": True,
        "authorization": "GPT_APPROVED_PHASE8_CONFIRMATION",
        "config_sha256": config_sha256,
        "run_matrix_sha256": run_matrix_sha256,
        "estimator_schedule_sha256": estimator_schedule_sha256,
    }
    failures = {key: {"actual": token.get(key), "expected": value} for key, value in expected.items() if token.get(key) != value}
    if failures:
        raise PermissionError(f"Invalid or missing Phase 8 confirmation release token fields: {failures}")


def git_commit(project_root: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_status(project_root: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def critical_source_manifest(
    project_root: Path,
    relative_paths: Sequence[str],
    *,
    release_commit: Optional[str] = None,
) -> Dict[str, object]:
    files = {}
    for relative in relative_paths:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Critical source file missing: {relative}")
        files[relative.replace("\\", "/")] = file_sha256(path)
    return {
        "release_commit": release_commit if release_commit is not None else git_commit(project_root),
        "files": files,
    }


def verify_release_lock(project_root: Path, lock_dir: Path, *, require_clean: bool = True) -> Dict[str, object]:
    manifest_path = lock_dir / "release_source_manifest.json"
    approved_commit_path = lock_dir / "approved_phase8_code_commit.txt"
    if not manifest_path.is_file() or not approved_commit_path.is_file():
        raise RuntimeError("Release lock directory is incomplete")
    manifest = load_json(manifest_path)
    approved_commit = approved_commit_path.read_text(encoding="utf-8").strip()
    actual_commit = git_commit(project_root)
    status = git_status(project_root)
    if actual_commit is not None and actual_commit != approved_commit:
        raise RuntimeError(f"Git commit mismatch: {actual_commit} != {approved_commit}")
    if actual_commit is not None and require_clean and status != "":
        raise RuntimeError(f"Git worktree is not clean:\n{status}")
    mismatches = []
    for relative, expected in manifest["files"].items():  # type: ignore[index]
        path = project_root / relative
        actual = file_sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches.append({"path": relative, "actual": actual, "expected": expected})
    if mismatches:
        raise RuntimeError(f"Critical source manifest mismatch: {mismatches}")
    return {
        "approved_commit": approved_commit,
        "actual_commit": actual_commit,
        "clean_worktree": True if actual_commit is None else status == "",
        "source_manifest_sha256": file_sha256(manifest_path),
        "key_file_sha256": manifest["files"],
    }


def environment_snapshot(*, deterministic_settings: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available_reported": bool(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "execution_device": "cpu",
        "deterministic_settings": dict(deterministic_settings or {}),
    }


def formal_estimator_schedule_report() -> Dict[str, object]:
    pilot = build_balanced_lag_schedule(T=500, lag=1, d_out=8, max_iter=2000, seed=32001)
    confirmation = build_balanced_lag_schedule(T=500, lag=1, d_out=8, max_iter=2000, seed=33001)
    return {
        "pilot": {"seed": 32001, "entries": len(pilot), "sha256": schedule_sha256(pilot)},
        "confirmation": {"seed": 33001, "entries": len(confirmation), "sha256": schedule_sha256(confirmation)},
    }
