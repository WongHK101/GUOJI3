"""P0 audit: Jacobian score semantics for ISTF/JRNGC.

This script is intentionally small and CPU-friendly by default. It answers
three pre-submission questions before any expensive rerun:

1. Does current ISTF scoring d y / d x' match the full chain-rule score
   d y / d x with respect to the raw input?
2. Does concat partial scoring d y / d x differ from total-derivative scoring
   through the auxiliary channel z(x)?
3. Does a coordinate-preserving depthwise filter reduce the semantic gap?

Default run:
    python experiments/p0_jacobian_semantics_audit.py --mode all --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJ_ROOT, "src")
for _path in (SRC_DIR, PROJ_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mamba_jrngc_pilot import (  # noqa: E402
    MambaFilterJRNGC,
    MambaJRNGC,
    train_model,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def generate_stable_var(
    d: int,
    t_steps: int,
    lag: int,
    seed: int,
    sparsity: float = 0.35,
    noise_scale: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a small stable VAR-like process with known directed lags.

    Returns:
        x: (d, T)
        gc_true: (d, d, lag), where [target, source, k] is source -> target.
        coeff: real-valued coefficient tensor with the same convention.
    """
    rng = np.random.default_rng(seed)
    coeff = rng.normal(0.0, 0.25, size=(d, d, lag))
    mask = rng.random(size=(d, d, lag)) < sparsity
    coeff *= mask
    for k in range(lag):
        np.fill_diagonal(coeff[:, :, k], 0.0)

    # Keep the process conservative. This is not an exact companion-matrix
    # stability proof, but is enough for small smoke diagnostics.
    coeff *= 0.45 / max(np.sum(np.abs(coeff)), 1e-8) * d

    # Ensure at least a few non-self edges.
    if np.sum(np.abs(coeff) > 1e-8) < max(2, d):
        for target in range(d):
            source = (target + 1) % d
            coeff[target, source, 0] = 0.18

    x = np.zeros((d, t_steps), dtype=np.float32)
    x[:, :lag] = rng.normal(0.0, 0.2, size=(d, lag))
    for t in range(lag, t_steps):
        value = np.zeros(d, dtype=np.float64)
        for k in range(lag):
            value += coeff[:, :, k] @ x[:, t - k - 1]
        x[:, t] = value + rng.normal(0.0, noise_scale, size=d)

    # Standardize per variable for training stability.
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
    gc_true = (np.abs(coeff) > 1e-10).astype(np.int32)
    return x.astype(np.float32), gc_true, coeff.astype(np.float32)


def select_windows(n_windows: int, max_windows: int) -> np.ndarray:
    if n_windows <= 0:
        raise ValueError("n_windows must be positive")
    n = min(n_windows, max_windows)
    if n == n_windows:
        return np.arange(n_windows, dtype=np.int64)
    return np.unique(np.linspace(0, n_windows - 1, n, dtype=np.int64))


def filtered_coordinate_gc(
    model: MambaFilterJRNGC,
    x_np: np.ndarray,
    window_idx: Iterable[int],
) -> np.ndarray:
    """Current ISTF score: d y / d x' on detached filtered windows."""
    model.eval()
    windows, _, _ = model.make_filtered_windows(x_np)
    idx = torch.as_tensor(list(window_idx), device=windows.device, dtype=torch.long)
    x_input = windows.index_select(0, idx)[:, :, : model.lag].detach().clone()
    x_input.requires_grad_(True)
    y = model(x_input)
    jac = torch.zeros(
        (x_input.shape[0], model.d, model.d, model.lag),
        device=x_input.device,
        dtype=x_input.dtype,
    )
    for target in range(model.d):
        grad = torch.autograd.grad(
            y[:, target].sum(),
            x_input,
            retain_graph=True,
            create_graph=False,
        )[0]
        jac[:, target] = grad
    return jac.abs().mean(dim=0).detach().cpu().numpy()


def raw_chain_gc_for_filter(
    model: MambaFilterJRNGC,
    x_np: np.ndarray,
    window_idx: Iterable[int],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Full chain-rule local score: d(f(F(x))) / d x on raw input.

    The local score keeps the nominal raw lag window x[t-K:t]. We also report
    leakage, the fraction of gradient mass outside that nominal lag window.
    """
    model.eval()
    device = next(model.parameters()).device
    x_raw = torch.tensor(x_np, device=device, dtype=torch.float32).unsqueeze(0)
    x_raw.requires_grad_(True)

    x_t = x_raw.transpose(1, 2)
    x_filtered = model.filter_mamba(x_t)
    windows = x_filtered.unfold(1, model.lag + 1, 1).reshape(-1, model.d, model.lag + 1)

    idx_list = [int(i) for i in window_idx]
    idx = torch.as_tensor(idx_list, device=device, dtype=torch.long)
    x_input = windows.index_select(0, idx)[:, :, : model.lag]
    y = model(x_input)

    jac_local = torch.zeros(
        (len(idx_list), model.d, model.d, model.lag),
        device=device,
        dtype=torch.float32,
    )
    leakage_values: List[float] = []

    for row, start in enumerate(idx_list):
        for target in range(model.d):
            grad = torch.autograd.grad(
                y[row, target],
                x_raw,
                retain_graph=True,
                create_graph=False,
            )[0][0]  # (d, T)
            local = grad[:, start : start + model.lag]
            jac_local[row, target] = local
            total_abs = float(grad.abs().sum().detach().cpu())
            local_abs = float(local.abs().sum().detach().cpu())
            if total_abs > 1e-12:
                leakage_values.append(max(total_abs - local_abs, 0.0) / total_abs)

    stats = {
        "raw_chain_leakage_mean": float(np.mean(leakage_values)) if leakage_values else 0.0,
        "raw_chain_leakage_max": float(np.max(leakage_values)) if leakage_values else 0.0,
    }
    return jac_local.abs().mean(dim=0).detach().cpu().numpy(), stats


def concat_partial_gc(
    model: MambaJRNGC,
    x_np: np.ndarray,
    window_idx: Iterable[int],
) -> np.ndarray:
    """Current concat score: partial d y / d x with z detached."""
    model.eval()
    windows, _ = model.preprocess_and_windowing(x_np)
    idx = torch.as_tensor(list(window_idx), device=windows.device, dtype=torch.long)
    xz = windows.index_select(0, idx)[:, :, : model.lag]
    x_orig = xz[:, : model.d, :].detach().clone().requires_grad_(True)
    x_cond = xz[:, model.d :, :].detach()
    x_cat = torch.cat([x_orig, x_cond], dim=1).flatten(start_dim=1)
    y = model(x_cat)
    jac = torch.zeros(
        (x_orig.shape[0], model.d, model.d, model.lag),
        device=x_orig.device,
        dtype=x_orig.dtype,
    )
    for target in range(model.d):
        grad = torch.autograd.grad(
            y[:, target].sum(),
            x_orig,
            retain_graph=True,
            create_graph=False,
        )[0]
        jac[:, target] = grad
    return jac.abs().mean(dim=0).detach().cpu().numpy()


def concat_total_raw_gc(
    model: MambaJRNGC,
    x_np: np.ndarray,
    window_idx: Iterable[int],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Total derivative score for concat: d f([x, z(x)]) / d x."""
    model.eval()
    device = next(model.parameters()).device
    x_raw = torch.tensor(x_np, device=device, dtype=torch.float32).unsqueeze(0)
    x_raw.requires_grad_(True)

    z_t, _ = model.preprocessor(x_raw)
    x_t = x_raw.transpose(1, 2)
    x_win = x_t.unfold(1, model.lag + 1, 1).squeeze(0)
    z_win = z_t.unfold(1, model.lag + 1, 1).squeeze(0)
    xz_win = torch.cat([x_win, z_win], dim=1)

    idx_list = [int(i) for i in window_idx]
    idx = torch.as_tensor(idx_list, device=device, dtype=torch.long)
    x_input = xz_win.index_select(0, idx)[:, :, : model.lag].flatten(start_dim=1)
    y = model(x_input)

    jac_local = torch.zeros(
        (len(idx_list), model.d, model.d, model.lag),
        device=device,
        dtype=torch.float32,
    )
    leakage_values: List[float] = []

    for row, start in enumerate(idx_list):
        for target in range(model.d):
            grad = torch.autograd.grad(
                y[row, target],
                x_raw,
                retain_graph=True,
                create_graph=False,
            )[0][0]
            local = grad[:, start : start + model.lag]
            jac_local[row, target] = local
            total_abs = float(grad.abs().sum().detach().cpu())
            local_abs = float(local.abs().sum().detach().cpu())
            if total_abs > 1e-12:
                leakage_values.append(max(total_abs - local_abs, 0.0) / total_abs)

    stats = {
        "total_derivative_leakage_mean": float(np.mean(leakage_values)) if leakage_values else 0.0,
        "total_derivative_leakage_max": float(np.max(leakage_values)) if leakage_values else 0.0,
    }
    return jac_local.abs().mean(dim=0).detach().cpu().numpy(), stats


def aggregate_scores(jac: np.ndarray) -> np.ndarray:
    if jac.ndim == 3:
        return np.max(np.abs(jac), axis=2)
    return np.abs(jac)


def offdiag_values(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    mask = ~np.eye(arr.shape[0], dtype=bool)
    return arr[mask]


def offdiag_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    va = offdiag_values(aggregate_scores(a))
    vb = offdiag_values(aggregate_scores(b))
    if va.size == 0 or np.std(va) < 1e-12 or np.std(vb) < 1e-12:
        return None
    return float(np.corrcoef(va, vb)[0, 1])


def simple_auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    s = offdiag_values(scores)
    y = offdiag_values(labels).astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(s)
    ranks = np.empty_like(s, dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos = float(ranks[y].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def topk_edges(scores: np.ndarray, k: int) -> set[Tuple[int, int]]:
    scores = np.asarray(scores, dtype=np.float64)
    candidates = []
    d = scores.shape[0]
    for target in range(d):
        for source in range(d):
            if target == source:
                continue
            candidates.append((-float(scores[target, source]), target, source))
    candidates.sort()
    return {(source, target) for _, target, source in candidates[:k]}


def summarize_pair(
    name: str,
    reference: np.ndarray,
    alternative: np.ndarray,
    gc_true: np.ndarray,
    extra: Dict[str, float] | None = None,
) -> Dict[str, object]:
    ref_2d = aggregate_scores(reference)
    alt_2d = aggregate_scores(alternative)
    gt_2d = (np.sum(gc_true, axis=2) > 0).astype(np.int32) if gc_true.ndim == 3 else gc_true
    k = int(gt_2d.sum())
    ref_edges = topk_edges(ref_2d, k)
    alt_edges = topk_edges(alt_2d, k)
    union = ref_edges | alt_edges
    jaccard = len(ref_edges & alt_edges) / len(union) if union else 1.0
    ratio = float(np.mean(np.abs(alt_2d)) / (np.mean(np.abs(ref_2d)) + 1e-12))

    out: Dict[str, object] = {
        "name": name,
        "offdiag_score_correlation": offdiag_corr(reference, alternative),
        "mean_abs_alternative_over_reference": ratio,
        "topk_jaccard": float(jaccard),
        "auroc_reference": simple_auroc(ref_2d, gt_2d),
        "auroc_alternative": simple_auroc(alt_2d, gt_2d),
        "n_true_edges": k,
    }
    if extra:
        out.update(extra)
    return out


def train_small_model(model: nn.Module, x: np.ndarray, args: argparse.Namespace) -> nn.Module:
    model, _ = train_model(
        model,
        x,
        max_iter=args.max_iter,
        lr=args.lr,
        lookback=4,
        check_every=max(1, min(10, args.max_iter // 2 if args.max_iter > 1 else 1)),
        verbose=False,
    )
    return model


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = choose_device(args.device)
    x, gc_true, coeff = generate_stable_var(args.d, args.t_steps, args.lag, args.seed)
    n_windows = x.shape[1] - args.lag
    window_idx = select_windows(n_windows, args.max_windows)

    results: Dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "setup": {
            "seed": args.seed,
            "device": str(device),
            "d": args.d,
            "T": args.t_steps,
            "lag": args.lag,
            "max_iter": args.max_iter,
            "window_indices": [int(i) for i in window_idx],
            "n_true_edges": int(gc_true.sum()),
        },
    }

    if args.mode in ("all", "istf"):
        set_seed(args.seed)
        model = MambaFilterJRNGC(
            d=args.d,
            lag=args.lag,
            layers=args.layers,
            hidden=args.hidden,
            jacobian_lam=0.01,
            d_state=4,
            ortho_lam=0.05,
            residual_scale=0.1,
            filter_type="mamba",
        ).to(device)
        model = train_small_model(model, x, args)
        current = filtered_coordinate_gc(model, x, window_idx)
        raw, raw_stats = raw_chain_gc_for_filter(model, x, window_idx)
        results["istf_mamba_filtered_vs_raw_chain"] = summarize_pair(
            "ISTF-Mamba current dY/dx_prime vs full chain dY/dx",
            current,
            raw,
            gc_true,
            raw_stats,
        )

    if args.mode in ("all", "concat"):
        set_seed(args.seed)
        concat = MambaJRNGC(
            d=args.d,
            lag=args.lag,
            layers=args.layers,
            hidden=args.hidden,
            jacobian_lam=0.01,
            d_state=4,
            d_cond=args.d_cond,
            use_time_weight_loss=False,
        ).to(device)
        concat = train_small_model(concat, x, args)
        partial = concat_partial_gc(concat, x, window_idx)
        total, total_stats = concat_total_raw_gc(concat, x, window_idx)
        results["concat_partial_vs_total_derivative"] = summarize_pair(
            "Concat partial dY/dx vs total dY/dx through z(x)",
            partial,
            total,
            gc_true,
            total_stats,
        )

    if args.mode in ("all", "depthwise"):
        set_seed(args.seed)
        depthwise = MambaFilterJRNGC(
            d=args.d,
            lag=args.lag,
            layers=args.layers,
            hidden=args.hidden,
            jacobian_lam=0.01,
            d_state=4,
            ortho_lam=0.05,
            residual_scale=0.1,
            filter_type="depthwise",
        ).to(device)
        depthwise = train_small_model(depthwise, x, args)
        current = filtered_coordinate_gc(depthwise, x, window_idx)
        raw, raw_stats = raw_chain_gc_for_filter(depthwise, x, window_idx)
        results["depthwise_filtered_vs_raw_chain"] = summarize_pair(
            "Depthwise current dY/dx_prime vs full chain dY/dx",
            current,
            raw,
            gc_true,
            raw_stats,
        )

    return results


def print_summary(results: Dict[str, object]) -> None:
    print("\nP0 Jacobian semantics audit summary")
    print(json.dumps(results["setup"], indent=2))
    for key, value in results.items():
        if key in {"setup", "created_at"}:
            continue
        if not isinstance(value, dict):
            continue
        print(f"\n[{key}]")
        for metric in (
            "offdiag_score_correlation",
            "mean_abs_alternative_over_reference",
            "topk_jaccard",
            "auroc_reference",
            "auroc_alternative",
            "raw_chain_leakage_mean",
            "raw_chain_leakage_max",
            "total_derivative_leakage_mean",
            "total_derivative_leakage_max",
        ):
            if metric in value:
                print(f"  {metric}: {value[metric]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0 Jacobian semantics audit")
    parser.add_argument("--mode", choices=["all", "istf", "concat", "depthwise"], default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--t-steps", type=int, default=80)
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--d-cond", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--out",
        default=os.path.join(PROJ_ROOT, "results", "p0_audit", "p0_jacobian_semantics_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_audit(args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print_summary(results)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
