"""Focused NS+Linear diagnostics across 5 seeds (0-4).

Per-expert request: computes Jacobian ratio, Delta-S true/false, and selectivity index
to test whether Mamba's score shift is symmetric (SI ~ 0 = non-selective rescaling).

Usage:
    python diagnostics_nslinear_5seed.py --setting D2 --seeds 0 1 2 3 4 --max-iter 2000
"""

import torch
import numpy as np
import sys, os, json, argparse
from datetime import datetime

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_device
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

torch.backends.cudnn.enabled = False
device = resolve_device()


def generate_nslinear(setting_params, seed, d=10, T=600, lag=3):
    """Generate NS+Linear data for a given setting and seed."""
    from src.factorial_data import generate_factorial_cell
    params = setting_params.copy()
    x, gc = generate_factorial_cell(
        d=d, T=T, lag=lag, seed=seed,
        stationary=False, linear=True,
        coeff_scale=params["coeff_scale"],
        noise_scale=params["noise_scale"],
        regime_shift_strength=params["regime_shift_strength"],
        nonlinear_strength=0.0,
        sparsity=0.2,
    )
    return x, gc


def train_and_get_gc(model_cls, model_kwargs, x, max_iter, seed):
    """Train a model and return its GC matrix."""
    from mamba_jrngc_pilot import train_model
    torch.manual_seed(seed)
    np.random.seed(seed)
    d, lag = model_kwargs["d"], model_kwargs["lag"]
    model = model_cls(**model_kwargs).to(device)
    model, _ = train_model(model, x, max_iter=max_iter, lr=1e-3, verbose=False)
    gc = model.get_gc_matrix(x)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return gc


def compute_jacobian_ratio(model_cls, model_kwargs_train, x, max_iter, seed):
    """Train baseline and mamba, return Jacobian norm ratio."""
    from mamba_jrngc_pilot import BaselineJRNGC, MambaFilterJRNGC, train_model

    d, lag = model_kwargs_train["d"], model_kwargs_train["lag"]

    # Train baseline
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = BaselineJRNGC(d=d, lag=lag, layers=5, hidden=50, jacobian_lam=0.01).to(device)
    base, _ = train_model(base, x, max_iter=max_iter, lr=1e-3, verbose=False)

    # Train mamba
    torch.manual_seed(seed)
    np.random.seed(seed)
    mamba = MambaFilterJRNGC(
        d=d, lag=lag, layers=5, hidden=50,
        jacobian_lam=0.01, d_state=4, ortho_lam=0.05,
        residual_scale=0.1, filter_type="mamba"
    ).to(device)
    mamba, _ = train_model(mamba, x, max_iter=max_iter, lr=1e-3, verbose=False)

    # Compute Jacobian norms (per-window Frobenius)
    jac_base = _compute_jacobian_norms(base, x, lag)
    jac_mamba = _compute_jacobian_norms(mamba, x, lag)

    # Get GC matrices for score shift
    gc_base = base.get_gc_matrix(x)
    gc_mamba = mamba.get_gc_matrix(x)

    del base, mamba
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return jac_base, jac_mamba, gc_base, gc_mamba


def _compute_jacobian_norms(model, x_full, lag):
    """Per-window Jacobian Frobenius norms."""
    x = torch.tensor(x_full, device=device, dtype=torch.float32)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    x_t = x.transpose(1, 2)
    windows = x_t.unfold(1, lag + 1, 1)
    windows = windows.reshape(-1, windows.shape[2], windows.shape[3])

    N = windows.shape[0]
    d = windows.shape[1]
    jac_norms = np.zeros(N, dtype=np.float32)

    for i in range(N):
        x_input = windows[i:i+1, :, :lag].detach().clone().requires_grad_(True)
        y = model(x_input)
        jac_rows = []
        for j in range(d):
            grad = torch.autograd.grad(y[0, j], x_input, retain_graph=True, create_graph=False)[0]
            jac_rows.append(grad.reshape(-1))
        jac = torch.stack(jac_rows)
        jac_norms[i] = torch.norm(jac, p='fro').item()

    return jac_norms


def compute_selectivity_index(gc_base, gc_mamba, gc_true):
    """Compute Delta-S and selectivity index."""
    score_base = np.max(np.abs(gc_base), axis=2)
    score_mamba = np.max(np.abs(gc_mamba), axis=2)
    gt_summary = (gc_true.sum(axis=2) > 0).astype(np.float32)

    d = gc_true.shape[0]
    for i in range(d):
        score_base[i, i] = 0
        score_mamba[i, i] = 0
        gt_summary[i, i] = 0

    delta = score_mamba - score_base
    true_mask = gt_summary > 0
    false_mask = gt_summary == 0

    delta_true = delta[true_mask]
    delta_false = delta[false_mask]

    ds_true_mean = float(np.mean(delta_true)) if len(delta_true) > 0 else 0.0
    ds_true_std = float(np.std(delta_true)) if len(delta_true) > 0 else 0.0
    ds_false_mean = float(np.mean(delta_false)) if len(delta_false) > 0 else 0.0
    ds_false_std = float(np.std(delta_false)) if len(delta_false) > 0 else 0.0

    numerator = ds_true_mean - ds_false_mean
    denominator = abs(ds_true_mean) + abs(ds_false_mean) + 1e-8
    si = numerator / denominator

    return {
        "delta_true_mean": ds_true_mean,
        "delta_true_std": ds_true_std,
        "delta_false_mean": ds_false_mean,
        "delta_false_std": ds_false_std,
        "selectivity_index": round(float(si), 6),
        "n_true_edges": int(np.sum(true_mask)),
        "n_false_edges": int(np.sum(false_mask)),
        "delta_true_values": [round(float(v), 6) for v in delta_true],
        "delta_false_values": [round(float(v), 6) for v in delta_false],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default="D2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--T", type=int, default=600)
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()
    global device
    device = resolve_device(args.cpu)

    SETTINGS = {
        "D2": {"coeff_scale": 0.40, "noise_scale": 0.15,
               "regime_shift_strength": 0.20, "nonlinear_strength": 0.50},
    }
    params = SETTINGS[args.setting]

    print(f"NS+Linear 5-seed diagnostics: setting={args.setting}, seeds={args.seeds}, max_iter={args.max_iter}")
    print(f"Device: {device}")
    print()

    per_seed = {}
    si_values = []
    jac_ratios = []

    for seed in args.seeds:
        print(f"--- Seed {seed} ---")
        x, gc_true = generate_nslinear(params, seed, d=args.d, T=args.T, lag=args.lag)
        n_edges = int(gc_true.sum())
        print(f"  NS+Linear: T={x.shape[1]}, edges={n_edges}")

        jac_base, jac_mamba, gc_base, gc_mamba = compute_jacobian_ratio(
            None, {"d": args.d, "lag": args.lag}, x, args.max_iter, seed
        )

        si_result = compute_selectivity_index(gc_base, gc_mamba, gc_true)

        jac_ratio = float(np.mean(jac_mamba)) / max(float(np.mean(jac_base)), 1e-8)
        jac_ratios.append(jac_ratio)
        si_values.append(si_result["selectivity_index"])

        per_seed[f"seed_{seed}"] = {
            "edges": n_edges,
            "jacobian_base_mean": round(float(np.mean(jac_base)), 4),
            "jacobian_base_std": round(float(np.std(jac_base)), 4),
            "jacobian_mamba_mean": round(float(np.mean(jac_mamba)), 4),
            "jacobian_mamba_std": round(float(np.std(jac_mamba)), 4),
            "jacobian_ratio": round(jac_ratio, 4),
            **{k: si_result[k] for k in ["delta_true_mean", "delta_true_std",
                                          "delta_false_mean", "delta_false_std",
                                          "selectivity_index", "n_true_edges", "n_false_edges"]},
        }

        print(f"  JacR={jac_ratio:.3f}  "
              f"ΔS_true={si_result['delta_true_mean']:+.4f}  "
              f"ΔS_false={si_result['delta_false_mean']:+.4f}  "
              f"SI={si_result['selectivity_index']:+.3f}")

    # Summary across seeds
    summary = {
        "jacobian_ratio_mean": round(float(np.mean(jac_ratios)), 4),
        "jacobian_ratio_std": round(float(np.std(jac_ratios)), 4),
        "selectivity_index_mean": round(float(np.mean(si_values)), 6),
        "selectivity_index_std": round(float(np.std(si_values)), 6),
        "selectivity_index_values": [round(float(v), 6) for v in si_values],
    }

    output = {
        "_meta": {
            "script": "experiments/diagnostics_nslinear_5seed.py",
            "setting": args.setting,
            "cell": "NS+Linear",
            "n_seeds": len(args.seeds),
            "max_iter": args.max_iter,
            "date": str(datetime.now()),
        },
        "summary": summary,
        "per_seed": per_seed,
    }

    out_path = args.output or f"diagnostics_nslinear_5seed_{args.setting}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Interpretation
    print(f"\n--- Summary ---")
    print(f"Jacobian ratio (Mamba/Base): {summary['jacobian_ratio_mean']:.3f} ± {summary['jacobian_ratio_std']:.3f}")
    print(f"Selectivity index (SI):       {summary['selectivity_index_mean']:.3f} ± {summary['selectivity_index_std']:.3f}")
    if abs(summary["selectivity_index_mean"]) < 0.15:
        print("SI ≈ 0 → Mamba score shift is nearly symmetric (true ≈ false).")
        print("Interpretation: global sensitivity rescaling, not structure-aware denoising.")
    else:
        direction = "selectively enhances true edges" if summary["selectivity_index_mean"] > 0 else "selectively suppresses true edges"
        print(f"SI ≠ 0 → Mamba {direction}.")


if __name__ == "__main__":
    main()
