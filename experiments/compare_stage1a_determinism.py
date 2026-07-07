"""Compare two Stage 1a run directories for deterministic GPU smoke.

The intended use is the P0.3d CP-depthwise duplicate smoke:
same config, NS+Nonlinear, data_seed=1, train_seed=0, 20 iterations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from knowledge_metrics import topk_edges_exact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tol", type=float, default=1e-6)
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


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


def tensor_state_max_abs_diff(a_path: Path, b_path: Path) -> Dict[str, float]:
    a = torch.load(a_path, map_location="cpu")
    b = torch.load(b_path, map_location="cpu")
    out: Dict[str, float] = {}
    a_state = a["model_state"]
    b_state = b["model_state"]
    for key, aval in a_state.items():
        if key not in b_state:
            out[key] = float("inf")
            continue
        out[key] = float(torch.max(torch.abs(aval.detach().cpu() - b_state[key].detach().cpu())))
    for key in b_state:
        if key not in a_state:
            out[key] = float("inf")
    return out


def main() -> int:
    args = parse_args()
    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    tol = float(args.tol)
    ckpt_a = next((run_a / "checkpoints").glob("iter_*.pt"))
    ckpt_b = next((run_b / "checkpoints").glob("iter_*.pt"))
    loss_a = numeric_loss_trace(load_json(run_a / "loss_trace.json"))
    loss_b = numeric_loss_trace(load_json(run_b / "loss_trace.json"))
    score_a = np.load(run_a / "scores" / "score_nominal.npy")
    score_b = np.load(run_b / "scores" / "score_nominal.npy")
    metrics_a = load_json(run_a / "metrics.json")
    metrics_b = load_json(run_b / "metrics.json")
    k = int(metrics_a["metrics_nominal"]["n_true_edges"])
    edges_a = sorted(list(topk_edges_exact(score_a, k=k, exclude_diag=True)))
    edges_b = sorted(list(topk_edges_exact(score_b, k=k, exclude_diag=True)))
    loss_diff = float(np.max(np.abs(loss_a - loss_b))) if loss_a.size or loss_b.size else 0.0
    score_diff = float(np.max(np.abs(score_a - score_b)))
    metric_diffs = max_abs_diff_dict(metrics_a, metrics_b)
    state_diffs = tensor_state_max_abs_diff(ckpt_a, ckpt_b)
    passed = (
        loss_diff <= tol
        and score_diff <= tol
        and edges_a == edges_b
        and all(v <= tol for v in metric_diffs.values())
        and all(v <= tol for v in state_diffs.values())
    )
    report = {
        "passed": bool(passed),
        "tolerance": tol,
        "loss_trace_max_abs_diff": loss_diff,
        "score_nominal_max_abs_diff": score_diff,
        "metrics_max_abs_diff": metric_diffs,
        "exact_topk_edges_equal": edges_a == edges_b,
        "exact_topk_edges_a": edges_a,
        "exact_topk_edges_b": edges_b,
        "state_dict_max_abs_diff": state_diffs,
        "run_a": str(run_a),
        "run_b": str(run_b),
    }
    save_json(Path(args.output), report)
    print(json.dumps({"output": args.output, "passed": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
