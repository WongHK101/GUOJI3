"""P0 unified diagnostic smoke for repair-route selection.

This CPU-friendly script compares candidate repair directions on the same
small controlled VAR(1) setting:

- Baseline JRNGC
- Concat JRNGC, reported with partial and total-derivative graph scores
- Cross-channel ISTF-Mamba
- Coordinate-preserving depthwise ISTF
- Simple causal smoothers (EMA and moving average) as non-ISTF references

It is a local smoke test, not a replacement for the canonical benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJ_ROOT, "src")
for _path in (SRC_DIR, PROJ_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mamba_jrngc_pilot import (  # noqa: E402
    BaselineJRNGC,
    MambaFilterJRNGC,
    MambaJRNGC,
    compute_metrics,
    train_model,
)
from p0_jacobian_semantics_audit import (  # noqa: E402
    choose_device,
    concat_partial_gc,
    concat_total_raw_gc,
    filtered_coordinate_gc,
    raw_chain_gc_for_filter,
    select_windows,
    set_seed,
    summarize_pair,
)


def generate_var1_data(
    d: int,
    t_steps: int,
    seed: int,
    sparsity: float,
    noise_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coeff = rng.normal(0.0, 1.0, size=(d, d))
    coeff *= rng.random(size=(d, d)) < sparsity
    np.fill_diagonal(coeff, 0.0)
    if np.sum(np.abs(coeff) > 1e-8) < max(2, d):
        for target in range(d):
            source = (target + 1) % d
            coeff[target, source] = 1.0
    # Use a conservative row-sum bound rather than spectral-radius scaling.
    # Small smoke diagnostics should not be dominated by non-normal transient
    # growth from an otherwise asymptotically stable matrix.
    row_sum = float(np.max(np.sum(np.abs(coeff), axis=1)))
    coeff = coeff * (0.55 / max(row_sum, 1e-6))

    x = np.zeros((d, t_steps), dtype=np.float32)
    x[:, 0] = rng.normal(0.0, 0.2, size=d)
    for t in range(1, t_steps):
        x[:, t] = coeff @ x[:, t - 1] + rng.normal(0.0, noise_scale, size=d)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
    gc = (np.abs(coeff) > 1e-10).astype(np.int32)
    return x.astype(np.float32), gc, coeff.astype(np.float32)


def causal_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    for t in range(x.shape[1]):
        start = max(0, t - window + 1)
        y[:, t] = x[:, start:t + 1].mean(axis=1)
    return y


def causal_ema(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    y[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        y[:, t] = alpha * y[:, t - 1] + (1.0 - alpha) * x[:, t]
    return y


class SmoothedJRNGC(nn.Module):
    """Fixed causal smoother followed by BaselineJRNGC."""

    def __init__(
        self,
        smoothing_fn: Callable[[np.ndarray], np.ndarray],
        d: int,
        lag: int,
        layers: int,
        hidden: int,
        jacobian_lam: float,
    ) -> None:
        super().__init__()
        self.smoothing_fn = smoothing_fn
        self.d = d
        self.lag = lag
        self.model = BaselineJRNGC(
            d=d,
            lag=lag,
            layers=layers,
            hidden=hidden,
            jacobian_lam=jacobian_lam,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def compute_loss(self, x_full: np.ndarray) -> torch.Tensor:
        return self.model.compute_loss(self.smoothing_fn(x_full))

    def get_gc_matrix(self, x_full: np.ndarray) -> np.ndarray:
        return self.model.get_gc_matrix(self.smoothing_fn(x_full))


def graph_metrics(gc_true: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    metrics = compute_metrics(gc_true, score)
    keep = ("auroc", "auprc", "f1", "shd_topk", "nshd_topk", "mcc_topk")
    return {k: float(metrics[k]) for k in keep}


def coefficient_metrics(score: np.ndarray, coeff: np.ndarray) -> Dict[str, float]:
    if score.ndim == 3:
        score_2d = np.max(np.abs(score), axis=2)
    else:
        score_2d = np.abs(score)
    truth = np.abs(coeff)
    mask = ~np.eye(score_2d.shape[0], dtype=bool)
    a = score_2d[mask].astype(np.float64)
    b = truth[mask].astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    corr = None
    if a.size > 1:
        a_scale = max(float(np.max(np.abs(a))), 1e-12)
        b_scale = max(float(np.max(np.abs(b))), 1e-12)
        a_scaled = a / a_scale
        b_scaled = b / b_scale
        if np.std(a_scaled) > 1e-12 and np.std(b_scaled) > 1e-12:
            corr = float(np.corrcoef(a_scaled, b_scaled)[0, 1])
    return {
        "pearson_abs_coeff": corr,
        "norm_ratio_to_true": float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-12)),
    }


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


def train_current_model(model: nn.Module, x: np.ndarray, args: argparse.Namespace) -> Tuple[nn.Module, float]:
    return train_model(
        model,
        x,
        max_iter=args.max_iter,
        lr=args.lr,
        lookback=args.lookback,
        check_every=args.check_every,
        verbose=False,
    )


def raw_windows(x_np: np.ndarray, lag: int, device: torch.device) -> torch.Tensor:
    x = torch.tensor(x_np, device=device, dtype=torch.float32).unsqueeze(0)
    x_t = x.transpose(1, 2)
    return x_t.unfold(1, lag + 1, 1).squeeze(0)


def concat_pred_loss_with_inputs(
    model: MambaJRNGC,
    x_input_np: np.ndarray,
    x_target_np: np.ndarray,
    c_override: torch.Tensor | None = None,
) -> float:
    """Prediction loss with perturbed input windows but clean raw targets."""
    device = next(model.parameters()).device
    if c_override is None:
        xz_win, _ = model.preprocess_and_windowing(x_input_np)
        xz_in = xz_win[:, :, : model.lag]
    else:
        x_win = raw_windows(x_input_np, model.lag, device)
        z_win = c_override.unfold(1, model.lag + 1, 1).squeeze(0)
        xz_in = torch.cat([x_win, z_win], dim=1)[:, :, : model.lag]
    target_win = raw_windows(x_target_np, model.lag, device)
    target = target_win[:, : model.d, -1]
    with torch.no_grad():
        pred = model(xz_in.flatten(start_dim=1))
        loss = torch.mean((pred - target) ** 2)
    return float(loss.detach().cpu())


def concat_side_channel_diagnostic(model: MambaJRNGC, x: np.ndarray) -> Dict[str, float]:
    device = next(model.parameters()).device
    x_clean = x.astype(np.float32)
    x_zero = np.zeros_like(x_clean, dtype=np.float32)
    with torch.no_grad():
        c_clean, _ = model.preprocessor(torch.tensor(x_clean, device=device).unsqueeze(0))
    c_zero = torch.zeros_like(c_clean)

    clean = concat_pred_loss_with_inputs(model, x_clean, x_clean, c_clean)
    zero_x_keep_c = concat_pred_loss_with_inputs(model, x_zero, x_clean, c_clean)
    keep_x_zero_c = concat_pred_loss_with_inputs(model, x_clean, x_clean, c_zero)
    zero_both = concat_pred_loss_with_inputs(model, x_zero, x_clean, c_zero)
    return {
        "clean_pred_loss": clean,
        "zero_x_keep_c_delta": zero_x_keep_c - clean,
        "keep_x_zero_c_delta": keep_x_zero_c - clean,
        "zero_both_delta": zero_both - clean,
        "zero_x_keep_c_over_keep_x_zero_c": (zero_x_keep_c - clean)
        / max(keep_x_zero_c - clean, 1e-12),
    }


def run_seed(seed: int, args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    x, gc, coeff = generate_var1_data(
        d=args.d,
        t_steps=args.t_steps,
        seed=seed,
        sparsity=args.sparsity,
        noise_scale=args.noise_scale,
    )
    window_idx = select_windows(x.shape[1] - args.lag, args.max_windows)
    out: Dict[str, object] = {
        "setup": {
            "seed": seed,
            "n_edges": int(gc.sum()),
            "window_indices": [int(i) for i in window_idx],
        }
    }

    set_seed(seed)
    baseline = BaselineJRNGC(args.d, args.lag, args.layers, args.hidden, jacobian_lam=args.jacobian_lam).to(device)
    baseline, loss = train_current_model(baseline, x, args)
    score = baseline.get_gc_matrix(x)
    out["baseline"] = {
        "train_loss": float(loss),
        "current": graph_metrics(gc, score),
        "coeff": coefficient_metrics(score, coeff),
    }

    for name, smoothing_fn in [
        ("ma3", lambda arr: causal_moving_average(arr, 3)),
        ("ema07", lambda arr: causal_ema(arr, 0.7)),
    ]:
        set_seed(seed)
        model = SmoothedJRNGC(
            smoothing_fn=smoothing_fn,
            d=args.d,
            lag=args.lag,
            layers=args.layers,
            hidden=args.hidden,
            jacobian_lam=args.jacobian_lam,
        ).to(device)
        model, loss = train_current_model(model, x, args)
        score = model.get_gc_matrix(x)
        out[name] = {
            "train_loss": float(loss),
            "current": graph_metrics(gc, score),
            "coeff": coefficient_metrics(score, coeff),
        }

    set_seed(seed)
    concat = MambaJRNGC(
        d=args.d,
        lag=args.lag,
        layers=args.layers,
        hidden=args.hidden,
        jacobian_lam=args.jacobian_lam,
        d_state=args.d_state,
        d_cond=args.d_cond,
        use_time_weight_loss=False,
    ).to(device)
    concat, loss = train_current_model(concat, x, args)
    partial = concat_partial_gc(concat, x, window_idx)
    total, total_stats = concat_total_raw_gc(concat, x, window_idx)
    out["concat"] = {
        "train_loss": float(loss),
        "partial": graph_metrics(gc, partial),
        "total_raw": graph_metrics(gc, total),
        "coeff_partial": coefficient_metrics(partial, coeff),
        "coeff_total_raw": coefficient_metrics(total, coeff),
        "semantic_alignment": summarize_pair(
            "concat partial vs total raw derivative",
            partial,
            total,
            gc,
            total_stats,
        ),
        "side_channel": concat_side_channel_diagnostic(concat, x),
    }

    for filter_type in ("mamba", "depthwise"):
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
        model, loss = train_current_model(model, x, args)
        current = filtered_coordinate_gc(model, x, window_idx)
        raw, raw_stats = raw_chain_gc_for_filter(model, x, window_idx)
        out[f"istf_{filter_type}"] = {
            "train_loss": float(loss),
            "current": graph_metrics(gc, current),
            "raw_chain": graph_metrics(gc, raw),
            "coeff_current": coefficient_metrics(current, coeff),
            "coeff_raw_chain": coefficient_metrics(raw, coeff),
            "semantic_alignment": summarize_pair(
                f"{filter_type} filtered-coordinate vs raw-chain derivative",
                current,
                raw,
                gc,
                raw_stats,
            ),
        }

    return out


def aggregate(results: Dict[str, object]) -> Dict[str, object]:
    seeds = results["seeds"]
    methods = ["baseline", "ma3", "ema07", "concat", "istf_mamba", "istf_depthwise"]
    agg: Dict[str, object] = {}
    for method in methods:
        rows = [seeds[s][method] for s in seeds]
        if method == "concat":
            agg[method] = {
                "partial_auroc": numeric_summary(r["partial"]["auroc"] for r in rows),
                "total_raw_auroc": numeric_summary(r["total_raw"]["auroc"] for r in rows),
                "semantic_corr": numeric_summary(
                    r["semantic_alignment"]["offdiag_score_correlation"] for r in rows
                ),
                "topk_jaccard": numeric_summary(r["semantic_alignment"]["topk_jaccard"] for r in rows),
                "zero_x_keep_c_delta": numeric_summary(
                    r["side_channel"]["zero_x_keep_c_delta"] for r in rows
                ),
                "keep_x_zero_c_delta": numeric_summary(
                    r["side_channel"]["keep_x_zero_c_delta"] for r in rows
                ),
                "side_channel_delta_ratio": numeric_summary(
                    r["side_channel"]["zero_x_keep_c_over_keep_x_zero_c"] for r in rows
                ),
            }
        elif method.startswith("istf_"):
            agg[method] = {
                "current_auroc": numeric_summary(r["current"]["auroc"] for r in rows),
                "raw_chain_auroc": numeric_summary(r["raw_chain"]["auroc"] for r in rows),
                "semantic_corr": numeric_summary(
                    r["semantic_alignment"]["offdiag_score_correlation"] for r in rows
                ),
                "topk_jaccard": numeric_summary(r["semantic_alignment"]["topk_jaccard"] for r in rows),
                "leakage": numeric_summary(
                    r["semantic_alignment"]["raw_chain_leakage_mean"] for r in rows
                ),
            }
        else:
            agg[method] = {
                "current_auroc": numeric_summary(r["current"]["auroc"] for r in rows),
                "coeff_corr": numeric_summary(r["coeff"]["pearson_abs_coeff"] for r in rows),
            }
    return agg


def print_summary(results: Dict[str, object]) -> None:
    print("\nP0 unified diagnostic smoke")
    print(json.dumps(results["setup"], indent=2))
    print("\nAggregate")
    for method, metrics in results["aggregate"].items():
        if method == "concat":
            print(
                f"  {method:14s} partial_AUROC={metrics['partial_auroc']['mean']:.4f} "
                f"total_AUROC={metrics['total_raw_auroc']['mean']:.4f} "
                f"corr={metrics['semantic_corr']['mean']:.4f} "
                f"side_ratio={metrics['side_channel_delta_ratio']['mean']:.4f}"
            )
        elif method.startswith("istf_"):
            print(
                f"  {method:14s} current_AUROC={metrics['current_auroc']['mean']:.4f} "
                f"raw_AUROC={metrics['raw_chain_auroc']['mean']:.4f} "
                f"corr={metrics['semantic_corr']['mean']:.4f} "
                f"leak={metrics['leakage']['mean']:.4f}"
            )
        else:
            print(
                f"  {method:14s} current_AUROC={metrics['current_auroc']['mean']:.4f} "
                f"coeff_corr={metrics['coeff_corr']['mean']:.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0 unified diagnostic smoke")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--d", type=int, default=6)
    parser.add_argument("--t-steps", type=int, default=140)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--sparsity", type=float, default=0.35)
    parser.add_argument("--noise-scale", type=float, default=0.12)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--d-state", type=int, default=4)
    parser.add_argument("--d-cond", type=int, default=4)
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
        "--out",
        default=os.path.join(PROJ_ROOT, "results", "p0_audit", "p0_unified_diagnostic_smoke.json"),
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
            "max_windows_for_raw_chain": args.max_windows,
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
