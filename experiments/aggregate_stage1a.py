"""Aggregate frozen Stage 1a results and evaluate go/no-go gates.

Aggregation order is preregistered:
1. average train seeds per cell/method/data_seed;
2. compute paired data-seed effects;
3. evaluate frozen effect-size gates;
4. do not emit strong n=3 significance claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


METRIC_MAP = {
    "auroc": "auroc",
    "auprc": "auprc",
    "mcc": "mcc_exact_topk",
}
FORMAL_METHODS = ["baseline", "cp_depthwise", "fixed_fir3", "fixed_ema"]


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
        if not status_path.exists() or not metrics_path.exists():
            continue
        status = load_json(status_path)
        if status.get("status") != "complete":
            continue
        metrics_payload = load_json(metrics_path)
        track = metric_track_for_method(run["method"])
        key = "metrics_full_H" if track == "full_H" else "metrics_nominal"
        metrics = metrics_payload[key]
        rows.append({
            "cell": run["cell"],
            "method": run["method"],
            "data_seed": int(run["data_seed"]),
            "train_seed": int(run["train_seed"]),
            "metric_track": track,
            "metrics": metrics,
            "run_id": run["run_id"],
        })
    return rows


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


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


def evaluate_go_no_go(effects: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
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

    final_go = bool(cp_vs_baseline_pass and fixed_fir3_novelty_pass and not ema_reference_dominance_no_go)
    return {
        "final_go": final_go,
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


def aggregate(rows: Sequence[Dict[str, object]], config: Dict[str, object]) -> Dict[str, object]:
    averaged = average_train_seeds(rows)
    effects = compute_paired_effects(averaged, config)
    gates = evaluate_go_no_go(effects, config)
    return {
        "formal_result": bool(config["formal_result"]),
        "averaged_by_train_seed": averaged,
        "paired_effects_by_data_seed": effects,
        "go_no_go": gates,
    }


def main() -> int:
    args = parse_args()
    config = load_json(Path(args.config))
    if bool(config.get("formal_result")) is False:
        raise ValueError("Aggregation is intended for formal Stage 1a results; smoke config is not accepted.")
    if args.results_json:
        rows = load_json(Path(args.results_json))["rows"]
    elif args.stage1_root:
        rows = load_rows_from_stage1_root(Path(args.stage1_root))
    else:
        raise ValueError("Either --stage1-root or --results-json is required")
    out = aggregate(rows, config)
    save_json(Path(args.output), out)
    print(json.dumps({
        "output": args.output,
        "rows": len(rows),
        "final_go": out["go_no_go"]["final_go"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
