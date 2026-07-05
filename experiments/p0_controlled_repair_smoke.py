"""P0 controlled repair smoke for ISTF score semantics.

This script is intentionally CPU-friendly. It checks whether a
coordinate-preserving ISTF filter can repair the raw-input Jacobian semantics
without immediately destroying graph-recovery quality on a small synthetic
process.

Default run:
    python experiments/p0_controlled_repair_smoke.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List

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
    compute_metrics_multimode,
    train_model,
)
from p0_jacobian_semantics_audit import (  # noqa: E402
    choose_device,
    filtered_coordinate_gc,
    generate_stable_var,
    raw_chain_gc_for_filter,
    select_windows,
    set_seed,
    summarize_pair,
)


def numeric_summary(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def summary_max_metrics(gc_true: np.ndarray, gc_pred: np.ndarray) -> Dict[str, float]:
    metrics = compute_metrics_multimode(gc_true, gc_pred)["summary_max"]
    keep = ("auroc", "auprc", "f1", "shd_topk", "nshd_topk", "mcc_topk")
    return {k: float(metrics[k]) for k in keep}


def train_baseline(
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
    model, loss = train_model(
        model,
        x,
        max_iter=args.max_iter,
        lr=args.lr,
        lookback=args.lookback,
        check_every=args.check_every,
        verbose=False,
    )
    gc_pred = model.get_gc_matrix(x)
    return {
        "train_time_s": float(time.time() - t0),
        "train_loss": float(loss),
        "current_summary_max": summary_max_metrics(gc_true, gc_pred),
    }


def train_filter(
    x: np.ndarray,
    gc_true: np.ndarray,
    seed: int,
    filter_type: str,
    window_idx: np.ndarray,
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
    model, loss = train_model(
        model,
        x,
        max_iter=args.max_iter,
        lr=args.lr,
        lookback=args.lookback,
        check_every=args.check_every,
        verbose=False,
    )
    train_time = float(time.time() - t0)
    current = filtered_coordinate_gc(model, x, window_idx)
    raw, raw_stats = raw_chain_gc_for_filter(model, x, window_idx)
    semantic = summarize_pair(
        f"{filter_type} filtered-coordinate score vs raw-chain score",
        current,
        raw,
        gc_true,
        raw_stats,
    )
    return {
        "train_time_s": train_time,
        "train_loss": float(loss),
        "current_summary_max": summary_max_metrics(gc_true, current),
        "raw_chain_summary_max": summary_max_metrics(gc_true, raw),
        "semantic_alignment": semantic,
    }


def run_seed(seed: int, args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    x, gc_true, _ = generate_stable_var(
        d=args.d,
        t_steps=args.t_steps,
        lag=args.lag,
        seed=seed,
        sparsity=args.sparsity,
        noise_scale=args.noise_scale,
    )
    n_windows = x.shape[1] - args.lag
    window_idx = select_windows(n_windows, args.max_windows)

    seed_result: Dict[str, object] = {
        "setup": {
            "seed": seed,
            "n_true_lagged_edges": int(gc_true.sum()),
            "n_windows": int(n_windows),
            "window_indices": [int(i) for i in window_idx],
        },
        "baseline": train_baseline(x, gc_true, seed, args, device),
    }
    for filter_type in args.filter_types:
        seed_result[filter_type] = train_filter(
            x=x,
            gc_true=gc_true,
            seed=seed,
            filter_type=filter_type,
            window_idx=window_idx,
            args=args,
            device=device,
        )
    return seed_result


def aggregate(results: Dict[str, object], filter_types: List[str]) -> Dict[str, object]:
    seed_items = results["seeds"]
    methods = ["baseline", *filter_types]
    out: Dict[str, object] = {}
    for method in methods:
        current_auroc = [
            seed_items[s][method]["current_summary_max"]["auroc"]
            for s in seed_items
        ]
        out[method] = {
            "current_auroc": numeric_summary(current_auroc),
            "current_auprc": numeric_summary(
                seed_items[s][method]["current_summary_max"]["auprc"]
                for s in seed_items
            ),
            "current_mcc_topk": numeric_summary(
                seed_items[s][method]["current_summary_max"]["mcc_topk"]
                for s in seed_items
            ),
        }
        if method != "baseline":
            out[method]["raw_chain_auroc"] = numeric_summary(
                seed_items[s][method]["raw_chain_summary_max"]["auroc"]
                for s in seed_items
            )
            out[method]["semantic_corr"] = numeric_summary(
                seed_items[s][method]["semantic_alignment"]["offdiag_score_correlation"]
                for s in seed_items
            )
            out[method]["semantic_topk_jaccard"] = numeric_summary(
                seed_items[s][method]["semantic_alignment"]["topk_jaccard"]
                for s in seed_items
            )
            out[method]["raw_chain_leakage_mean"] = numeric_summary(
                seed_items[s][method]["semantic_alignment"]["raw_chain_leakage_mean"]
                for s in seed_items
            )
    return out


def print_compact_summary(results: Dict[str, object]) -> None:
    print("\nP0 controlled repair smoke")
    print(json.dumps(results["setup"], indent=2))
    print("\nAggregate")
    for method, metrics in results["aggregate"].items():
        cur = metrics["current_auroc"]
        line = f"  {method:10s} current_AUROC={cur['mean']:.4f}+/-{cur['std']:.4f}"
        if "raw_chain_auroc" in metrics:
            raw = metrics["raw_chain_auroc"]
            corr = metrics["semantic_corr"]
            jac = metrics["semantic_topk_jaccard"]
            leak = metrics["raw_chain_leakage_mean"]
            line += (
                f" raw_AUROC={raw['mean']:.4f}+/-{raw['std']:.4f}"
                f" corr={corr['mean']:.4f}"
                f" topkJ={jac['mean']:.4f}"
                f" leakage={leak['mean']:.4f}"
            )
        print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0 controlled ISTF repair smoke")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--d", type=int, default=6)
    parser.add_argument("--t-steps", type=int, default=100)
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument("--sparsity", type=float, default=0.35)
    parser.add_argument("--noise-scale", type=float, default=0.15)
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
    parser.add_argument(
        "--filter-types",
        nargs="+",
        choices=["mamba", "tcn", "depthwise", "depthwise_gated"],
        default=["mamba", "depthwise"],
    )
    parser.add_argument(
        "--out",
        default=os.path.join(PROJ_ROOT, "results", "p0_audit", "p0_controlled_repair_smoke.json"),
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
            "seeds": [int(s) for s in args.seeds],
            "d": args.d,
            "T": args.t_steps,
            "lag": args.lag,
            "max_iter": args.max_iter,
            "filter_types": args.filter_types,
            "max_windows_for_raw_chain": args.max_windows,
        },
        "seeds": {},
    }
    for seed in args.seeds:
        print(f"[seed {seed}] running", flush=True)
        results["seeds"][str(seed)] = run_seed(int(seed), args, device)
    results["aggregate"] = aggregate(results, args.filter_types)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print_compact_summary(results)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
