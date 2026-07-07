"""Aggregate frozen Stage 1a results and evaluate go/no-go gates.

Aggregation order is preregistered:
1. average train seeds per cell/method/data_seed;
2. compute paired data-seed effects;
3. evaluate frozen effect-size gates;
4. do not emit strong n=3 significance claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from release_lock import file_sha256, verify_release_lock  # noqa: E402


METRIC_MAP = {
    "auroc": "auroc",
    "auprc": "auprc",
    "mcc": "mcc_exact_topk",
}
FORMAL_METHODS = ["baseline", "cp_depthwise", "fixed_fir3", "fixed_ema"]
APPROVED_FROZEN_CONFIG_SHA_PATH = PROJECT_ROOT / "configs" / "approved_stage1a_frozen_config_sha256.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage1-root", default=None)
    parser.add_argument("--results-json", default=None, help="Synthetic or exported result rows for tests.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def canonical_config_hash(config: Dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_approved_frozen_config_sha() -> str:
    if not APPROVED_FROZEN_CONFIG_SHA_PATH.exists():
        raise RuntimeError(f"Approved frozen config SHA file missing: {APPROVED_FROZEN_CONFIG_SHA_PATH}")
    return APPROVED_FROZEN_CONFIG_SHA_PATH.read_text(encoding="utf-8").strip().split()[0].lower()


def validate_aggregation_inputs(
    config_path: Path,
    config: Dict[str, object],
    stage1_root: Path | None,
    require_release_lock: bool = True,
) -> Dict[str, object]:
    approved_file_sha = read_approved_frozen_config_sha()
    actual_file_sha = file_sha256(config_path).lower()
    if actual_file_sha != approved_file_sha:
        raise RuntimeError(
            f"Frozen aggregation config file SHA mismatch: actual {actual_file_sha}, approved {approved_file_sha}"
        )
    canonical_hash = canonical_config_hash(config)
    snapshot_equivalent = None
    root_hash = None
    if stage1_root is not None:
        root_hash = root_config_hash(stage1_root)
        if canonical_hash != root_hash:
            raise RuntimeError(
                f"Aggregation config canonical hash {canonical_hash} does not match stage root {root_hash}"
            )
        snapshot_path = stage1_root / "config_snapshot.json"
        if not snapshot_path.exists():
            raise RuntimeError(f"Stage root config snapshot is missing: {snapshot_path}")
        snapshot = load_json(snapshot_path)
        snapshot_equivalent = canonical_config_hash(snapshot) == canonical_hash
        if not snapshot_equivalent:
            raise RuntimeError("Stage root config_snapshot.json is not canonically equivalent to aggregation config")
    release_lock = verify_release_lock(require_clean_worktree=False) if require_release_lock else None
    return {
        "approved_config_file_sha256": approved_file_sha,
        "actual_config_file_sha256": actual_file_sha,
        "canonical_config_hash": canonical_hash,
        "stage_root_config_hash": root_hash,
        "root_config_snapshot_equivalent": snapshot_equivalent,
        "source_release_lock": release_lock,
    }


def metric_track_for_method(method: str) -> str:
    return "full_H" if method == "fixed_ema" else "nominal"


def load_rows_from_stage1_root(root: Path) -> List[Dict[str, object]]:
    manifest = load_json(root / "run_manifest.json")
    rows: List[Dict[str, object]] = []
    for run in manifest["runs"]:
        if run["role"] != "formal":
            continue
        run_dir = Path(run["output_path"])
        status_path = run_dir / "status.json"
        metrics_path = run_dir / "metrics.json"
        diagnostics_path = run_dir / "diagnostics.json"
        status = load_json(status_path) if status_path.exists() else {}
        metrics_payload = load_json(metrics_path) if metrics_path.exists() else {}
        diagnostics_payload = load_json(diagnostics_path) if diagnostics_path.exists() else {}
        track = metric_track_for_method(run["method"])
        key = "metrics_full_H" if track == "full_H" else "metrics_nominal"
        rows.append({
            "cell": run["cell"],
            "method": run["method"],
            "data_seed": int(run["data_seed"]),
            "train_seed": int(run["train_seed"]),
            "metric_track": metrics_payload.get("metric_track_for_aggregation", track),
            "metrics": metrics_payload.get(key, {}),
            "status": status.get("status"),
            "formal_result": status.get("formal_result"),
            "config_sha256": status.get("config_sha256"),
            "no_nan_inf": status.get("no_nan_inf"),
            "run_dir": str(run_dir),
            "semantic_audit": diagnostics_payload.get("semantic_audit", {}),
            "run_id": run["run_id"],
        })
    return rows


def root_config_hash(root: Path) -> Optional[str]:
    path = root / "config_sha256.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    manifest = load_json(root / "run_manifest.json")
    return manifest.get("config_sha256")


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def validate_completeness(
    rows: Sequence[Dict[str, object]],
    config: Dict[str, object],
    expected_config_hash: str,
) -> Dict[str, object]:
    cells = list(config["data"]["cells"])
    methods = list(config["methods"]["formal"])
    data_seeds = [int(s) for s in config["data"]["data_seeds"]]
    train_seeds = [int(s) for s in config["data"]["train_seeds"]]
    expected_total = len(cells) * len(methods) * len(data_seeds) * len(train_seeds)
    failures: List[Dict[str, object]] = []
    seen: Dict[tuple, Dict[str, object]] = {}
    for row in rows:
        key = (row.get("cell"), row.get("method"), int(row.get("data_seed")), int(row.get("train_seed")))
        if key in seen:
            failures.append({"type": "duplicate_run", "key": list(key), "run_id": row.get("run_id")})
            continue
        seen[key] = row
        method = str(row.get("method"))
        if row.get("status") != "complete":
            failures.append({"type": "status_not_complete", "key": list(key), "status": row.get("status")})
        if row.get("formal_result") is not True:
            failures.append({"type": "formal_result_not_true", "key": list(key), "formal_result": row.get("formal_result")})
        if row.get("config_sha256") != expected_config_hash:
            failures.append({
                "type": "config_hash_mismatch",
                "key": list(key),
                "config_sha256": row.get("config_sha256"),
                "expected": expected_config_hash,
            })
        expected_track = metric_track_for_method(method)
        if row.get("metric_track") != expected_track:
            failures.append({
                "type": "metric_track_mismatch",
                "key": list(key),
                "metric_track": row.get("metric_track"),
                "expected": expected_track,
            })
        metrics = row.get("metrics") or {}
        for metric in ["auroc", "auprc", "mcc_exact_topk", "f1_exact_topk", "shd_exact_topk", "nshd_exact_topk"]:
            if metric not in metrics or not _is_finite_number(metrics[metric]):
                failures.append({"type": "missing_or_nonfinite_metric", "key": list(key), "metric": metric})
    for cell in cells:
        for method in methods:
            present_data = set()
            for data_seed in data_seeds:
                present_train = {
                    int(k[3]) for k in seen
                    if k[0] == cell and k[1] == method and int(k[2]) == data_seed
                }
                if present_train != set(train_seeds):
                    failures.append({
                        "type": "train_seed_set_mismatch",
                        "cell": cell,
                        "method": method,
                        "data_seed": data_seed,
                        "present_train_seeds": sorted(present_train),
                        "expected_train_seeds": train_seeds,
                    })
                if present_train:
                    present_data.add(data_seed)
            if present_data != set(data_seeds):
                failures.append({
                    "type": "data_seed_set_mismatch",
                    "cell": cell,
                    "method": method,
                    "present_data_seeds": sorted(present_data),
                    "expected_data_seeds": data_seeds,
                })
    passed = len(failures) == 0 and len(rows) == expected_total
    if len(rows) != expected_total:
        failures.append({"type": "row_count_mismatch", "rows": len(rows), "expected": expected_total})
        passed = False
    return {
        "passed": bool(passed),
        "expected_formal_run_count": expected_total,
        "observed_formal_row_count": len(rows),
        "expected_cells": cells,
        "expected_methods": methods,
        "expected_data_seeds": data_seeds,
        "expected_train_seeds": train_seeds,
        "expected_config_sha256": expected_config_hash,
        "failures": failures,
    }


def evaluate_semantic_gates(rows: Sequence[Dict[str, object]], config: Dict[str, object]) -> Dict[str, object]:
    failures: List[Dict[str, object]] = []
    by_method = defaultdict(lambda: {"passed": 0, "failed": 0})
    for row in rows:
        method = str(row["method"])
        audit = row.get("semantic_audit") or {}
        if method == "baseline":
            continue
        required = method in {"cp_depthwise", "fixed_fir3", "fixed_ema"}
        if not audit:
            if required:
                failures.append({"type": "missing_semantic_audit", "row": _row_id(row)})
                by_method[method]["failed"] += 1
            continue
        if bool(audit.get("passed")):
            by_method[method]["passed"] += 1
        else:
            by_method[method]["failed"] += 1
            failures.append({
                "type": "semantic_gate_failed",
                "row": _row_id(row),
                "failures": audit.get("failures", []),
            })
        if method == "fixed_ema" and audit.get("metric_track_required") != "full_H":
            failures.append({"type": "ema_missing_full_H_track_marker", "row": _row_id(row)})
    cp_failures = [f for f in failures if f.get("row", {}).get("method") == "cp_depthwise"]
    return {
        "passed": len(failures) == 0,
        "by_method": dict(by_method),
        "failures": failures,
        "cp_failed_runs": cp_failures,
    }


def _row_id(row: Dict[str, object]) -> Dict[str, object]:
    return {
        "cell": row.get("cell"),
        "method": row.get("method"),
        "data_seed": row.get("data_seed"),
        "train_seed": row.get("train_seed"),
        "run_id": row.get("run_id"),
    }


def average_train_seeds(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["method"] not in FORMAL_METHODS:
            continue
        grouped[(row["cell"], row["method"], int(row["data_seed"]))].append(row)
    averaged: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(lambda: defaultdict(dict))
    for (cell, method, data_seed), group in grouped.items():
        out_metrics = {}
        for public_key, metric_key in METRIC_MAP.items():
            vals = [float(g["metrics"][metric_key]) for g in group]  # type: ignore[index]
            out_metrics[public_key] = _mean(vals)
        averaged[cell][method][str(data_seed)] = {
            "n_train_seeds": len(group),
            "train_seeds": sorted(int(g["train_seed"]) for g in group),
            "metric_track": metric_track_for_method(method),
            "metrics": out_metrics,
        }
    return averaged


def _diffs(
    averaged: Dict[str, object],
    cell: str,
    left: str,
    right: str,
    data_seeds: Sequence[int],
) -> Dict[str, object]:
    per_seed = {}
    for seed in data_seeds:
        seed_key = str(seed)
        if (
            cell not in averaged
            or left not in averaged[cell]
            or right not in averaged[cell]
            or seed_key not in averaged[cell][left]
            or seed_key not in averaged[cell][right]
        ):
            continue
        lm = averaged[cell][left][seed_key]["metrics"]
        rm = averaged[cell][right][seed_key]["metrics"]
        per_seed[seed_key] = {
            "delta_auroc": float(lm["auroc"] - rm["auroc"]),
            "delta_auprc": float(lm["auprc"] - rm["auprc"]),
            "delta_mcc": float(lm["mcc"] - rm["mcc"]),
        }
    means = {}
    for key in ["delta_auroc", "delta_auprc", "delta_mcc"]:
        vals = [v[key] for v in per_seed.values()]
        means[key] = _mean(vals) if vals else None
    return {
        "left": left,
        "right": right,
        "per_data_seed": per_seed,
        "mean": means,
        "positive_data_seed_count_delta_auroc": int(sum(v["delta_auroc"] > 0 for v in per_seed.values())),
        "n_data_seeds": len(per_seed),
    }


def compute_paired_effects(averaged: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    cells = config["data"]["cells"]
    data_seeds = config["data"]["data_seeds"]
    effects = {}
    for cell in cells:
        effects[cell] = {
            "cp_vs_baseline": _diffs(averaged, cell, "cp_depthwise", "baseline", data_seeds),
            "cp_vs_fir3": _diffs(averaged, cell, "cp_depthwise", "fixed_fir3", data_seeds),
            "ema_vs_cp": _diffs(averaged, cell, "fixed_ema", "cp_depthwise", data_seeds),
        }
    return effects


def is_nonstationary_cell(cell: str) -> bool:
    return cell.startswith("NS+")


def _cell_passes_delta_gate(effect: Dict[str, object], gate: Dict[str, object]) -> bool:
    mean = effect["mean"]
    return bool(
        mean["delta_auroc"] is not None
        and mean["delta_auroc"] >= gate["mean_delta_auroc_min"]
        and effect["positive_data_seed_count_delta_auroc"] >= gate["min_positive_data_seeds"]
        and mean["delta_auprc"] >= gate["mean_delta_auprc_min"]
        and mean["delta_mcc"] >= gate["mean_delta_mcc_min"]
    )


def evaluate_go_no_go(
    effects: Dict[str, object],
    config: Dict[str, object],
    completeness: Optional[Dict[str, object]] = None,
    semantic: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    gates_cfg = config["stage1a_go_no_go_gates"]
    cp_gate_cfg = gates_cfg["cp_vs_baseline"]
    cp_cell_gate = {
        "mean_delta_auroc_min": 0.03,
        "min_positive_data_seeds": cp_gate_cfg["per_qualifying_cell_min_positive_data_seeds"],
        "mean_delta_auprc_min": cp_gate_cfg["mean_delta_auprc_min"],
        "mean_delta_mcc_min": cp_gate_cfg["mean_delta_mcc_min"],
    }
    qualifying_cells = [
        cell for cell, payload in effects.items()
        if _cell_passes_delta_gate(payload["cp_vs_baseline"], cp_cell_gate)
    ]
    cp_vs_baseline_pass = (
        len(qualifying_cells) >= cp_gate_cfg["min_qualifying_cells_delta_auroc_ge_0_03"]
        and any(is_nonstationary_cell(cell) for cell in qualifying_cells)
    )

    fir_cfg = gates_cfg["fixed_fir3_novelty"]
    fir_cell_gate = {
        "mean_delta_auroc_min": fir_cfg["mean_delta_auroc_cp_minus_fir3_min"],
        "min_positive_data_seeds": fir_cfg["min_positive_data_seeds"],
        "mean_delta_auprc_min": fir_cfg["mean_delta_auprc_min"],
        "mean_delta_mcc_min": fir_cfg["mean_delta_mcc_min"],
    }
    fir_novelty_cells = [
        cell for cell in qualifying_cells
        if _cell_passes_delta_gate(effects[cell]["cp_vs_fir3"], fir_cell_gate)
    ]
    fixed_fir3_novelty_pass = len(fir_novelty_cells) >= 1

    ema_cfg = gates_cfg["ema_reference_dominance_no_go"]
    ema_mean_ge_cp_all = True
    ema_delta_ge_002 = 0
    ema_noninferior_all = True
    for cell, payload in effects.items():
        mean = payload["ema_vs_cp"]["mean"]
        if mean["delta_auroc"] is None or mean["delta_auroc"] < 0:
            ema_mean_ge_cp_all = False
        if mean["delta_auroc"] is not None and mean["delta_auroc"] >= 0.02:
            ema_delta_ge_002 += 1
        if (
            mean["delta_auprc"] is None
            or mean["delta_mcc"] is None
            or mean["delta_auprc"] < ema_cfg["ema_minus_cp_delta_auprc_min"]
            or mean["delta_mcc"] < ema_cfg["ema_minus_cp_delta_mcc_min"]
        ):
            ema_noninferior_all = False
    ema_reference_dominance_no_go = bool(
        ema_mean_ge_cp_all
        and ema_delta_ge_002 >= ema_cfg["min_cells_ema_minus_cp_delta_auroc_ge_0_02"]
        and ema_noninferior_all
    )

    performance_go = bool(cp_vs_baseline_pass and fixed_fir3_novelty_pass and not ema_reference_dominance_no_go)
    completeness_pass = True if completeness is None else bool(completeness.get("passed"))
    semantic_pass = True if semantic is None else bool(semantic.get("passed"))
    final_go = bool(completeness_pass and semantic_pass and performance_go)
    return {
        "final_go": final_go,
        "performance_go": performance_go,
        "completeness_passed": completeness_pass,
        "semantic_passed": semantic_pass,
        "stage1a_is_effect_size_triage_only": True,
        "no_strong_n3_significance_claim": True,
        "cp_vs_baseline": {
            "passed": cp_vs_baseline_pass,
            "qualifying_cells": qualifying_cells,
        },
        "fixed_fir3_novelty": {
            "passed": fixed_fir3_novelty_pass,
            "qualifying_cells": fir_novelty_cells,
            "failure_interpretation": "FixedFIR3 has basically matched CP if this gate fails",
        },
        "ema_reference_dominance": {
            "no_go_triggered": ema_reference_dominance_no_go,
            "ema_track": "full_H_reference_not_nominal_lag_causal_method",
            "ema_mean_auroc_not_lower_than_cp_all_cells": ema_mean_ge_cp_all,
            "cells_ema_minus_cp_delta_auroc_ge_0_02": ema_delta_ge_002,
            "ema_auprc_mcc_noninferior_all_cells": ema_noninferior_all,
        },
    }


def aggregate(
    rows: Sequence[Dict[str, object]],
    config: Dict[str, object],
    expected_config_hash: Optional[str] = None,
) -> Dict[str, object]:
    if expected_config_hash is None:
        expected_config_hash = canonical_config_hash(config)
    completeness = validate_completeness(rows, config, expected_config_hash)
    if not completeness["passed"]:
        return {
            "formal_result": bool(config["formal_result"]),
            "aggregation_status": "failed_completeness",
            "completeness_gate": completeness,
        }
    semantic = evaluate_semantic_gates(rows, config)
    averaged = average_train_seeds(rows)
    effects = compute_paired_effects(averaged, config)
    gates = evaluate_go_no_go(effects, config, completeness=completeness, semantic=semantic)
    return {
        "formal_result": bool(config["formal_result"]),
        "aggregation_status": "complete",
        "completeness_gate": completeness,
        "semantic_gate": semantic,
        "averaged_by_train_seed": averaged,
        "paired_effects_by_data_seed": effects,
        "go_no_go": gates,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_json(config_path)
    if bool(config.get("formal_result")) is False:
        raise ValueError("Aggregation is intended for formal Stage 1a results; smoke config is not accepted.")
    stage_root_for_guard = Path(args.stage1_root) if args.stage1_root else None
    aggregation_guard = validate_aggregation_inputs(config_path, config, stage_root_for_guard)
    if args.results_json:
        payload = load_json(Path(args.results_json))
        rows = payload["rows"]
        expected_hash = payload.get("root_config_sha256") or canonical_config_hash(config)
    elif args.stage1_root:
        stage_root = Path(args.stage1_root)
        rows = load_rows_from_stage1_root(stage_root)
        expected_hash = root_config_hash(stage_root) or canonical_config_hash(config)
    else:
        raise ValueError("Either --stage1-root or --results-json is required")
    if aggregation_guard["canonical_config_hash"] != expected_hash:
        raise RuntimeError(
            f"Aggregation expected hash {expected_hash} does not match canonical frozen config "
            f"{aggregation_guard['canonical_config_hash']}"
        )
    completeness = validate_completeness(rows, config, expected_hash)
    completeness_path = Path(args.output).with_name("completeness_report.json")
    save_json(completeness_path, completeness)
    if not completeness["passed"]:
        out = {
            "formal_result": True,
            "aggregation_status": "failed_completeness",
            "completeness_gate": completeness,
            "aggregation_release_lock": aggregation_guard,
        }
        save_json(Path(args.output), out)
        print(json.dumps({
            "output": args.output,
            "completeness_report": str(completeness_path),
            "rows": len(rows),
            "aggregation_status": "failed_completeness",
        }, indent=2))
        return 1
    out = aggregate(rows, config, expected_config_hash=expected_hash)
    out["aggregation_release_lock"] = aggregation_guard
    save_json(Path(args.output), out)
    print(json.dumps({
        "output": args.output,
        "rows": len(rows),
        "final_go": out["go_no_go"]["final_go"],
        "semantic_passed": out["semantic_gate"]["passed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
