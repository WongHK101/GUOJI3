"""P0 factorial smoke comparing cross-channel and depthwise ISTF.

Runs a small CPU-friendly subset of the existing factorial generator:
{stationary, non-stationary} x {linear, nonlinear}. This is intended to test
whether the coordinate-preserving depthwise repair remains viable in the
settings where ISTF is supposed to help.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable

import numpy as np
import torch


PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJ_ROOT, "src")
for _path in (SRC_DIR, PROJ_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mamba_jrngc_pilot import (  # noqa: E402
    BaselineJRNGC,
    MambaFilterJRNGC,
    compute_metrics,
    train_model,
)
from p0_jacobian_semantics_audit import (  # noqa: E402
    choose_device,
    filtered_coordinate_gc,
    raw_chain_gc_for_filter,
    select_windows,
    set_seed,
    summarize_pair,
)
from src.factorial_data import (  # noqa: E402
    FACTORIAL_CELLS,
    FACTORIAL_SETTINGS,
    generate_all_factorial_cells,
)


def numeric_summary(values: Iterable[float | None]) -> Dict[str, float | None]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def summary_max(gc_true: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    if gc_true.ndim == 3:
        gt = (gc_true.sum(axis=2) > 0).astype(np.int32)
    else:
        gt = gc_true.astype(np.int32)
    if score.ndim == 3:
        pr = np.max(np.abs(score), axis=2)
    else:
        pr = np.abs(score)
    metrics = compute_metrics(gt, pr)
    keep = ("auroc", "auprc", "f1", "shd_topk", "nshd_topk", "mcc_topk")
    return {k: float(metrics[k]) for k in keep}


def train_one(
    model,
    x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[object, float]:
    return train_model(
        model,
        x,
        max_iter=args.max_iter,
        lr=args.lr,
        lookback=args.lookback,
        check_every=args.check_every,
        verbose=False,
    )


def run_baseline(
    x: np.ndarray,
    gc_true: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(seed)
    model = BaselineJRNGC(
        d=args.d,
        lag=args.lag,
        layers=args.layers,
        hidden=args.hidden,
        jacobian_lam=args.jacobian_lam,
    ).to(device)
    t0 = time.time()
    model, loss = train_one(model, x, args)
    score = model.get_gc_matrix(x)
    return {
        "train_time_s": float(time.time() - t0),
        "train_loss": float(loss),
        "summary_max": summary_max(gc_true, score),
    }


def run_filter(
    x: np.ndarray,
    gc_true: np.ndarray,
    seed: int,
    filter_type: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(seed)
    model = MambaFilterJRNGC(
        d=args.d,
        lag=args.lag,
        layers=args.layers,
        hidden=args.hidden,
        jacobian_lam=args.jacobian_lam,
        d_state=args.d_state,
        ortho_lam=args.ortho_lam,
        residual_scale=args.residual_scale,
        filter_type=filter_type,
    ).to(device)
    t0 = time.time()
    model, loss = train_one(model, x, args)
    score = model.get_gc_matrix(x)

    out: Dict[str, object] = {
        "train_time_s": float(time.time() - t0),
        "train_loss": float(loss),
        "summary_max": summary_max(gc_true, score),
    }
    if args.semantic:
        window_idx = select_windows(x.shape[1] - args.lag, args.max_windows)
        current = filtered_coordinate_gc(model, x, window_idx)
        raw, raw_stats = raw_chain_gc_for_filter(model, x, window_idx)
        out["semantic_alignment"] = summarize_pair(
            f"{filter_type} filtered-coordinate vs raw-chain derivative",
            current,
            raw,
            gc_true,
            raw_stats,
        )
    return out


def run_seed(seed: int, args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    cells = generate_all_factorial_cells(
        setting=args.setting,
        d=args.d,
        T=args.t_steps,
        lag=args.lag,
        seed=seed,
        sparsity=args.sparsity,
    )
    out: Dict[str, object] = {}
    for cell_name, _, _ in FACTORIAL_CELLS:
        x, gc_true = cells[cell_name]
        out[cell_name] = {
            "setup": {
                "n_edges": int(gc_true.sum()),
                "x_std": float(np.std(x)),
            },
            "baseline": run_baseline(x, gc_true, seed, args, device),
            "istf_mamba": run_filter(x, gc_true, seed, "mamba", args, device),
            "istf_depthwise": run_filter(x, gc_true, seed, "depthwise", args, device),
            "istf_depthwise_gated": run_filter(
                x, gc_true, seed, "depthwise_gated", args, device
            ),
        }
    return out


def aggregate(results: Dict[str, object]) -> Dict[str, object]:
    seeds = results["seeds"]
    agg: Dict[str, object] = {}
    for cell_name, _, _ in FACTORIAL_CELLS:
        agg[cell_name] = {}
        for method in ("baseline", "istf_mamba", "istf_depthwise", "istf_depthwise_gated"):
            rows = [seeds[s][cell_name][method] for s in seeds]
            entry: Dict[str, object] = {
                "auroc": numeric_summary(r["summary_max"]["auroc"] for r in rows),
                "auprc": numeric_summary(r["summary_max"]["auprc"] for r in rows),
                "mcc_topk": numeric_summary(r["summary_max"]["mcc_topk"] for r in rows),
            }
            if method != "baseline" and "semantic_alignment" in rows[0]:
                entry["semantic_corr"] = numeric_summary(
                    r["semantic_alignment"]["offdiag_score_correlation"] for r in rows
                )
                entry["semantic_topk_jaccard"] = numeric_summary(
                    r["semantic_alignment"]["topk_jaccard"] for r in rows
                )
                entry["leakage"] = numeric_summary(
                    r["semantic_alignment"]["raw_chain_leakage_mean"] for r in rows
                )
            agg[cell_name][method] = entry
    return agg


def print_summary(results: Dict[str, object]) -> None:
    print("\nP0 factorial depthwise smoke")
    print(json.dumps(results["setup"], indent=2))
    for cell_name, methods in results["aggregate"].items():
        print(f"\n[{cell_name}]")
        base = methods["baseline"]["auroc"]["mean"]
        for method in ("baseline", "istf_mamba", "istf_depthwise", "istf_depthwise_gated"):
            au = methods[method]["auroc"]
            line = f"  {method:14s} AUROC={au['mean']:.4f}+/-{au['std']:.4f}"
            if method != "baseline":
                delta = au["mean"] - base
                corr = methods[method].get("semantic_corr", {}).get("mean")
                leak = methods[method].get("leakage", {}).get("mean")
                line += f" delta={delta:+.4f}"
                if corr is not None:
                    line += f" corr={corr:.4f} leak={leak:.4f}"
            print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0 factorial depthwise smoke")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--setting", choices=sorted(FACTORIAL_SETTINGS), default="D2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--d", type=int, default=6)
    parser.add_argument("--t-steps", type=int, default=180)
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument("--sparsity", type=float, default=0.25)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--d-state", type=int, default=4)
    parser.add_argument("--jacobian-lam", type=float, default=0.01)
    parser.add_argument("--ortho-lam", type=float, default=0.05)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--semantic", action="store_true", help="Also run selected-window raw-chain semantic checks.")
    parser.add_argument(
        "--out",
        default=os.path.join(PROJ_ROOT, "results", "p0_audit", "p0_factorial_depthwise_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    device = choose_device(args.device)
    results: Dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "setup": {
            "device": str(device),
            "setting": args.setting,
            "setting_params": FACTORIAL_SETTINGS[args.setting],
            "seeds": [int(s) for s in args.seeds],
            "d": args.d,
            "T": args.t_steps,
            "lag": args.lag,
            "max_iter": args.max_iter,
            "semantic": bool(args.semantic),
        },
        "seeds": {},
    }
    for seed in args.seeds:
        print(f"[seed {seed}] running", flush=True)
        results["seeds"][str(seed)] = run_seed(int(seed), args, device)
    results["aggregate"] = aggregate(results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print_summary(results)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
