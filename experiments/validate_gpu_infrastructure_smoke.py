"""Validate the two-root Stage 1a GPU infrastructure smoke.

Inputs are two identical 5-method smoke roots. Root A also serves as the
five-method infrastructure smoke; the CP-depthwise run in root A is the
determinism duplicate A, and the CP-depthwise run in root B is duplicate B.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from knowledge_metrics import topk_edges_exact  # noqa: E402


EXPECTED_RUN_IDS = {
    "stage1a__baseline__NS_Nonlinear__data1__train0": {"role": "formal", "method": "baseline"},
    "stage1a__cp_depthwise__NS_Nonlinear__data1__train0": {"role": "formal", "method": "cp_depthwise"},
    "stage1a__fixed_fir3__NS_Nonlinear__data1__train0": {"role": "formal", "method": "fixed_fir3"},
    "stage1a__fixed_ema__NS_Nonlinear__data1__train0": {"role": "formal", "method": "fixed_ema"},
    "stage1a_limited__raw_chain_mamba__NS_Nonlinear__data1__train0": {
        "role": "limited_ablation",
        "method": "raw_chain_mamba",
    },
}
CP_RUN_ID = "stage1a__cp_depthwise__NS_Nonlinear__data1__train0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root-a", required=True)
    parser.add_argument("--smoke-root-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tol", type=float, default=1e-6)
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def finite_positive(value) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except Exception:
        return False


def run_dir(root: Path, run_id: str) -> Path:
    return root / "runs" / run_id


def validate_smoke_root(root: Path, root_label: str) -> Dict[str, object]:
    failures: List[Dict[str, object]] = []
    manifest_path = root / "run_manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "failures": [{"type": "missing_manifest", "root": str(root)}]}
    manifest = load_json(manifest_path)
    runs = manifest.get("runs", [])
    if manifest.get("formal_result") is not False:
        failures.append({"type": "root_formal_result_not_false", "value": manifest.get("formal_result")})
    if len(runs) != 5 or manifest.get("run_count") != 5:
        failures.append({"type": "run_count_not_5", "run_count": manifest.get("run_count"), "len_runs": len(runs)})
    seen = {r.get("run_id"): r for r in runs}
    if set(seen) != set(EXPECTED_RUN_IDS):
        failures.append({"type": "run_id_set_mismatch", "observed": sorted(seen), "expected": sorted(EXPECTED_RUN_IDS)})
    root_release = load_json(root / "release_lock.json") if (root / "release_lock.json").exists() else None
    for run_id, expected in EXPECTED_RUN_IDS.items():
        r = seen.get(run_id, {})
        if r.get("role") != expected["role"] or r.get("method") != expected["method"]:
            failures.append({"type": "manifest_role_method_mismatch", "run_id": run_id, "manifest": r, "expected": expected})
        rd = run_dir(root, run_id)
        status_path = rd / "status.json"
        runtime_path = rd / "runtime.json"
        env_path = rd / "environment.json"
        diagnostics_path = rd / "diagnostics.json"
        metrics_path = rd / "metrics.json"
        missing = [str(p) for p in [status_path, runtime_path, env_path, diagnostics_path, metrics_path] if not p.exists()]
        if missing:
            failures.append({"type": "missing_run_files", "run_id": run_id, "missing": missing})
            continue
        status = load_json(status_path)
        runtime = load_json(runtime_path)
        env = load_json(env_path)
        diagnostics = load_json(diagnostics_path)
        if status.get("status") != "complete":
            failures.append({"type": "status_not_complete", "run_id": run_id, "status": status.get("status")})
        if status.get("formal_result") is not False:
            failures.append({"type": "run_formal_result_not_false", "run_id": run_id, "value": status.get("formal_result")})
        if status.get("no_nan_inf") is not True or status.get("output_complete") is not True:
            failures.append({"type": "run_not_clean_complete", "run_id": run_id, "status": status})
        if "cuda" not in str(status.get("device", "")).lower():
            failures.append({"type": "run_device_not_cuda", "run_id": run_id, "device": status.get("device")})
        if not finite_positive(runtime.get("cuda_max_memory_allocated_mb")):
            failures.append({"type": "cuda_allocated_not_positive", "run_id": run_id, "value": runtime.get("cuda_max_memory_allocated_mb")})
        if not finite_positive(runtime.get("cuda_max_memory_reserved_mb")):
            failures.append({"type": "cuda_reserved_not_positive", "run_id": run_id, "value": runtime.get("cuda_max_memory_reserved_mb")})
        det = status.get("deterministic_settings") or env.get("deterministic_settings") or {}
        if det.get("torch_use_deterministic_algorithms") is not True or det.get("cudnn_deterministic") is not True:
            failures.append({"type": "determinism_not_enabled", "run_id": run_id, "deterministic_settings": det})
        if root_release is not None:
            if status.get("source_manifest_sha256") != root_release.get("source_manifest_sha256"):
                failures.append({"type": "source_manifest_sha_mismatch", "run_id": run_id})
            if status.get("approved_commit") != root_release.get("approved_commit"):
                failures.append({"type": "approved_commit_mismatch", "run_id": run_id})
        method = expected["method"]
        semantic_passed = bool(status.get("semantic_audit_passed"))
        audit = diagnostics.get("semantic_audit", {})
        if method in {"cp_depthwise", "fixed_fir3", "fixed_ema"} and not semantic_passed:
            failures.append({"type": "semantic_audit_failed", "run_id": run_id, "audit": audit})
        if method == "raw_chain_mamba" and status.get("status") != "complete":
            failures.append({"type": "raw_chain_mamba_limited_incomplete", "run_id": run_id})
    return {"root": str(root), "root_label": root_label, "passed": len(failures) == 0, "failures": failures}


def numeric_loss_trace(trace) -> np.ndarray:
    rows = []
    for item in trace:
        rows.append([
            float(item[key])
            for key in sorted(item)
            if key != "iter" and isinstance(item[key], (int, float))
        ])
    return np.asarray(rows, dtype=np.float64)


def max_abs_diff_dict(a: Dict[str, object], b: Dict[str, object], prefix: str = "") -> Dict[str, float]:
    diffs: Dict[str, float] = {}
    for key, aval in a.items():
        bval = b.get(key)
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(aval, dict) and isinstance(bval, dict):
            diffs.update(max_abs_diff_dict(aval, bval, name))
        elif isinstance(aval, (int, float)) and isinstance(bval, (int, float)):
            diffs[name] = abs(float(aval) - float(bval))
    return diffs


def checkpoint_state_diff(run_a: Path, run_b: Path) -> Dict[str, float]:
    ckpt_a = next((run_a / "checkpoints").glob("iter_*.pt"))
    ckpt_b = next((run_b / "checkpoints").glob("iter_*.pt"))
    a = torch.load(ckpt_a, map_location="cpu")
    b = torch.load(ckpt_b, map_location="cpu")
    out: Dict[str, float] = {}
    for key, aval in a["model_state"].items():
        bval = b["model_state"].get(key)
        out[key] = float("inf") if bval is None else float(torch.max(torch.abs(aval.cpu() - bval.cpu())))
    for key in b["model_state"]:
        if key not in a["model_state"]:
            out[key] = float("inf")
    return out


def validate_cp_duplicate(root_a: Path, root_b: Path, tol: float) -> Dict[str, object]:
    failures: List[Dict[str, object]] = []
    a = run_dir(root_a, CP_RUN_ID)
    b = run_dir(root_b, CP_RUN_ID)
    status_a = load_json(a / "status.json")
    status_b = load_json(b / "status.json")
    for label, status in [("a", status_a), ("b", status_b)]:
        if "cuda" not in str(status.get("device", "")).lower():
            failures.append({"type": "cp_duplicate_not_cuda", "which": label, "device": status.get("device")})
        if status.get("semantic_audit_passed") is not True:
            failures.append({"type": "cp_duplicate_semantic_failed", "which": label})
    for field in ["config_sha256", "schedule_hash", "predictor_seed"]:
        if status_a.get(field) != status_b.get(field):
            failures.append({"type": f"cp_duplicate_{field}_mismatch", "a": status_a.get(field), "b": status_b.get(field)})
    loss_a = numeric_loss_trace(load_json(a / "loss_trace.json"))
    loss_b = numeric_loss_trace(load_json(b / "loss_trace.json"))
    loss_diff = float(np.max(np.abs(loss_a - loss_b))) if loss_a.size or loss_b.size else 0.0
    score_nom_a = np.load(a / "scores" / "score_nominal.npy")
    score_nom_b = np.load(b / "scores" / "score_nominal.npy")
    score_full_a = np.load(a / "scores" / "score_full_H.npy")
    score_full_b = np.load(b / "scores" / "score_full_H.npy")
    score_nom_diff = float(np.max(np.abs(score_nom_a - score_nom_b)))
    score_full_diff = float(np.max(np.abs(score_full_a - score_full_b)))
    metrics_a = load_json(a / "metrics.json")
    metrics_b = load_json(b / "metrics.json")
    metric_diffs = max_abs_diff_dict(metrics_a, metrics_b)
    state_diffs = checkpoint_state_diff(a, b)
    k = int(metrics_a["metrics_nominal"]["n_true_edges"])
    edges_a = sorted(list(topk_edges_exact(score_nom_a, k=k, exclude_diag=True)))
    edges_b = sorted(list(topk_edges_exact(score_nom_b, k=k, exclude_diag=True)))
    if loss_diff > tol:
        failures.append({"type": "loss_trace_diff_exceeds_tol", "diff": loss_diff, "tol": tol})
    if score_nom_diff > tol:
        failures.append({"type": "score_nominal_diff_exceeds_tol", "diff": score_nom_diff, "tol": tol})
    if score_full_diff > tol:
        failures.append({"type": "score_full_H_diff_exceeds_tol", "diff": score_full_diff, "tol": tol})
    if any(v > tol for v in metric_diffs.values()):
        failures.append({"type": "metric_diff_exceeds_tol", "diffs": metric_diffs, "tol": tol})
    if any(v > tol for v in state_diffs.values()):
        failures.append({"type": "checkpoint_state_diff_exceeds_tol", "diffs": state_diffs, "tol": tol})
    if edges_a != edges_b:
        failures.append({"type": "exact_topk_edges_mismatch", "edges_a": edges_a, "edges_b": edges_b})
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "loss_trace_max_abs_diff": loss_diff,
        "score_nominal_max_abs_diff": score_nom_diff,
        "score_full_H_max_abs_diff": score_full_diff,
        "metrics_max_abs_diff": metric_diffs,
        "checkpoint_state_max_abs_diff": state_diffs,
        "exact_topk_edges_equal": edges_a == edges_b,
    }


def main() -> int:
    args = parse_args()
    root_a = Path(args.smoke_root_a)
    root_b = Path(args.smoke_root_b)
    report = {
        "smoke_root_a": validate_smoke_root(root_a, "A"),
        "smoke_root_b": validate_smoke_root(root_b, "B"),
        "cp_duplicate": validate_cp_duplicate(root_a, root_b, float(args.tol)),
    }
    report["passed"] = bool(
        report["smoke_root_a"]["passed"]
        and report["smoke_root_b"]["passed"]
        and report["cp_duplicate"]["passed"]
    )
    save_json(Path(args.output), report)
    print(json.dumps({"output": args.output, "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
