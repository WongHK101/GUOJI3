"""Post-hoc diagnostics for factorial experiment.

Computes three diagnostic metrics (per expert specification):
1. Filter deviation by regime: |x' - x| / |x|
2. Jacobian norm by regime (timestep-level)
3. True-edge vs false-edge GC score shift (Mamba vs Baseline)

Usage:
    python factorial_diagnostics.py --results factorial_D2_10seed_iter2000.json
                                     --setting D2 --seed 0 --cell "NS+Nonlinear"
"""

import torch
import numpy as np
import sys, os, json, argparse

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path, resolve_device
_jrngc = resolve_jrngc_path()
if _jrngc and _jrngc not in sys.path:
    sys.path.insert(0, _jrngc)

torch.backends.cudnn.enabled = False
device = resolve_device()


def generate_data_for_diagnostics(setting="D2", seed=0, d=10, T=600, lag=3):
    """Regenerate data for a specific factorial cell."""
    from run_factorial_pilot_cloud import (FACTORIAL_SETTINGS, FACTORIAL_CELLS,
                                           generate_factorial_cell)
    params = FACTORIAL_SETTINGS[setting].copy()
    cells = {}
    for name, stationary, linear in FACTORIAL_CELLS:
        regime = params["regime_shift_strength"] if not stationary else 0.0
        nl = params["nonlinear_strength"] if not linear else 0.0
        x, gc = generate_factorial_cell(
            d=d, T=T, lag=lag, seed=seed,
            stationary=stationary, linear=linear,
            coeff_scale=params["coeff_scale"],
            noise_scale=params["noise_scale"],
            regime_shift_strength=regime,
            nonlinear_strength=nl,
            sparsity=0.2,
        )
        cells[name] = (x, gc)
    return cells


def compute_filter_deviation(model, x, lag):
    """Compute |x_filtered - x| / |x| for each timestep.

    Returns:
        deviation_per_t: (T,) array of relative deviations
        mean_deviation: float
    """
    from mamba_jrngc_pilot import MambaFilterJRNGC
    if not isinstance(model, MambaFilterJRNGC):
        return None, None

    x_tensor = torch.tensor(x, device=device, dtype=torch.float32)
    if x_tensor.dim() == 2:
        x_tensor = x_tensor.unsqueeze(0)  # (1, d, T)

    x_t = x_tensor.transpose(1, 2)  # (1, T, d)
    with torch.no_grad():
        x_filtered = model.filter_mamba(x_t)  # (1, T, d)

    x_t_np = x_t.squeeze(0).cpu().numpy()  # (T, d)
    x_filt_np = x_filtered.squeeze(0).cpu().numpy()  # (T, d)

    diff_norm = np.linalg.norm(x_filt_np - x_t_np, axis=1)  # (T,)
    orig_norm = np.linalg.norm(x_t_np, axis=1) + 1e-8  # (T,)

    deviation_per_t = diff_norm / orig_norm
    return deviation_per_t, float(np.mean(deviation_per_t))


def compute_jacobian_norm_by_timestep(model, x_full, lag):
    """Compute per-window Jacobian Frobenius norm.

    For each time window t, compute |J_t|_F where J_t = ∂y_t/∂x_{t-lag:t-1}.

    Returns:
        jac_norms: (T-lag,) array of Frobenius norms per window
    """
    device = next(model.parameters()).device
    x = torch.tensor(x_full, device=device, dtype=torch.float32)
    if x.dim() == 2:
        x = x.unsqueeze(0)

    # Create windows: (N, d, lag+1)
    if hasattr(model, 'make_filtered_windows'):
        windows, _, _ = model.make_filtered_windows(x_full)
    elif hasattr(model, 'make_windows'):
        windows = model.make_windows(x_full)
    else:
        x_t = x.transpose(1, 2)
        windows = x_t.unfold(1, lag + 1, 1)
        windows = windows.reshape(-1, windows.shape[2], windows.shape[3])

    N = windows.shape[0]
    d = windows.shape[1]
    jac_norms = np.zeros(N, dtype=np.float32)

    for i in range(N):
        x_input = windows[i:i+1, :, :lag].detach().clone().requires_grad_(True)
        y = model(x_input)  # (1, d)
        jac_rows = []
        for j in range(d):
            grad = torch.autograd.grad(y[0, j], x_input, retain_graph=True,
                                       create_graph=False)[0]
            jac_rows.append(grad.reshape(-1))
        jac = torch.stack(jac_rows)  # (d, d*lag)
        jac_norms[i] = torch.norm(jac, p='fro').item()

    return jac_norms


def compute_edge_score_shift(gc_baseline, gc_mamba, gc_true):
    """Compute true-edge vs false-edge GC score shift.

    Args:
        gc_baseline: (d, d, lag) from baseline model
        gc_mamba: (d, d, lag) from mamba model
        gc_true: (d, d, lag) ground truth binary

    Returns:
        dict with delta_true_mean, delta_false_mean, delta_true_std, delta_false_std
    """
    # Use summary-max scores: max absolute score over lags
    score_base = np.max(np.abs(gc_baseline), axis=2)  # (d, d)
    score_mamba = np.max(np.abs(gc_mamba), axis=2)  # (d, d)
    gt_summary = (gc_true.sum(axis=2) > 0).astype(np.float32)  # (d, d)

    # Remove diagonal
    d = gc_true.shape[0]
    for i in range(d):
        score_base[i, i] = 0
        score_mamba[i, i] = 0
        gt_summary[i, i] = 0

    delta = score_mamba - score_base  # positive = mamba higher

    true_mask = gt_summary > 0
    false_mask = gt_summary == 0

    delta_true = delta[true_mask]
    delta_false = delta[false_mask]

    return {
        "delta_true_mean": float(np.mean(delta_true)) if len(delta_true) > 0 else 0.0,
        "delta_true_std": float(np.std(delta_true)) if len(delta_true) > 0 else 0.0,
        "delta_false_mean": float(np.mean(delta_false)) if len(delta_false) > 0 else 0.0,
        "delta_false_std": float(np.std(delta_false)) if len(delta_false) > 0 else 0.0,
        "n_true_edges": int(np.sum(true_mask)),
        "n_false_edges": int(np.sum(false_mask)),
        "delta_true_positive_fraction": float(np.mean(delta_true > 0)) if len(delta_true) > 0 else 0.0,
    }


def run_full_diagnostics(setting="D2", seed=0, max_iter=2000):
    """Run all diagnostics for all 4 cells of a given setting/seed.

    Returns nested dict: cell_name -> diagnostic_name -> values
    """
    from mamba_jrngc_pilot import BaselineJRNGC, MambaFilterJRNGC, train_model

    cells = generate_data_for_diagnostics(setting=setting, seed=seed)
    results = {}

    for cell_name, (x, gc_true) in cells.items():
        d = gc_true.shape[0]
        lag = gc_true.shape[2]
        print(f"\n  {cell_name}: T={x.shape[1]}, edges={int(gc_true.sum())}")

        # Train baseline
        torch.manual_seed(seed)
        np.random.seed(seed)
        base = BaselineJRNGC(d=d, lag=lag, layers=5, hidden=50, jacobian_lam=0.01).to(device)
        base, _ = train_model(base, x, max_iter=max_iter, lr=1e-3, verbose=False)
        gc_base = base.get_gc_matrix(x)

        # Train Mamba
        torch.manual_seed(seed)
        np.random.seed(seed)
        mamba = MambaFilterJRNGC(
            d=d, lag=lag, layers=5, hidden=50,
            jacobian_lam=0.01, d_state=4, ortho_lam=0.05,
            residual_scale=0.1, filter_type="mamba"
        ).to(device)
        mamba, _ = train_model(mamba, x, max_iter=max_iter, lr=1e-3, verbose=False)
        gc_mamba = mamba.get_gc_matrix(x)

        # Diagnostic 1: Filter deviation
        dev_per_t, mean_dev = compute_filter_deviation(mamba, x, lag)

        # Diagnostic 2: Jacobian norm per window
        jac_norms_base = compute_jacobian_norm_by_timestep(base, x, lag)
        jac_norms_mamba = compute_jacobian_norm_by_timestep(mamba, x, lag)

        # Diagnostic 3: Edge score shift
        edge_shift = compute_edge_score_shift(gc_base, gc_mamba, gc_true)

        results[cell_name] = {
            "filter_deviation_mean": mean_dev,
            "filter_deviation_per_t_mean": float(np.mean(dev_per_t)) if dev_per_t is not None else None,
            "filter_deviation_per_t_std": float(np.std(dev_per_t)) if dev_per_t is not None else None,
            "jacobian_norm_base_mean": float(np.mean(jac_norms_base)),
            "jacobian_norm_base_std": float(np.std(jac_norms_base)),
            "jacobian_norm_mamba_mean": float(np.mean(jac_norms_mamba)),
            "jacobian_norm_mamba_std": float(np.std(jac_norms_mamba)),
            "jacobian_norm_ratio": float(np.mean(jac_norms_mamba) / max(np.mean(jac_norms_base), 1e-8)),
            **edge_shift,
        }

        print(f"    Dev: {results[cell_name]['filter_deviation_mean']:.4f}" if mean_dev else "    Dev: N/A")
        print(f"    JacB: {results[cell_name]['jacobian_norm_base_mean']:.4f} "
              f"JacM: {results[cell_name]['jacobian_norm_mamba_mean']:.4f} "
              f"Ratio: {results[cell_name]['jacobian_norm_ratio']:.3f}")
        print(f"    ΔS_true: {edge_shift['delta_true_mean']:+.4f}±{edge_shift['delta_true_std']:.4f}  "
              f"ΔS_false: {edge_shift['delta_false_mean']:+.4f}±{edge_shift['delta_false_std']:.4f}  "
              f"(true>0: {edge_shift['delta_true_positive_fraction']:.2f})")

        del base, mamba
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    global device
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default="D2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()
    device = resolve_device(args.cpu)

    print(f"Factorial Diagnostics: setting={args.setting}, seed={args.seed}, max_iter={args.max_iter}")
    results = run_full_diagnostics(
        setting=args.setting, seed=args.seed, max_iter=args.max_iter
    )

    out_path = args.output or f"diagnostics_{args.setting}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
