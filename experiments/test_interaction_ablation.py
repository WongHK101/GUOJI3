"""P2 #4 v2: Nonlinear/Nonstationary Interaction Ablation.

Uses EXISTING proven data loaders for reliable results.
4 cells: Stat+Linear (VAR), Stat+Nonlinear (Lorenz), NS+Linear (NSVAR), NS+Nonlinear (NSVAR+tanh)
3 methods: Baseline, TCN, Mamba. 3 seeds each.
d=10 for all cells.
"""
import torch
import torch.nn as nn
import numpy as np
import sys, os, json, time, inspect

torch.backends.cudnn.enabled = False

# os.chdir removed — paths resolved via config
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path
_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)
# project root now resolved via _PROJ_ROOT

from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                train_model, compute_metrics)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JRNGC_DATA = resolve_data_dir()  # resolved via config


def log(msg):
    log_fh.write(msg + "\n")
    log_fh.flush()
    print(msg, flush=True)

log_fh = open("interaction_ablation.log", "w", buffering=1)


# ============================================================
# Data Loaders (using existing data)
# ============================================================

def load_stat_linear(d=10, lag=3, noise_scale=1, seed=0):
    """Cell 1: Stationary Linear VAR. Existing JRNGC data."""
    from tgc.data.var import var_stable
    x, _, gc = var_stable(d=d, t=500, t_eval=0, lag=lag,
                           sparsity=0.1, beta_value=0.5, sd=noise_scale, seed=seed)
    if x.ndim == 3:
        x = x[0]
    return x, gc


def load_stat_nonlinear(d=10, F=40, seed=0):
    """Cell 2: Stationary Nonlinear (Lorenz-96). Use JRNGC data loader."""
    from tgc.data.lorenz import lorenz_96
    x, _, gc = lorenz_96(d=d, t=500, t_eval=0, f=F, seed=seed)
    if x.ndim == 3:
        x = x[0]
    return x, gc


def load_ns_linear(d=10, seed=0):
    """Cell 3: Non-stationary Linear VAR. Existing NSVAR data."""
    p = f"" + os.path.join(_PROJ_ROOT, "data", "nonstationary_var/num_nodes_{d}/true_lag_7/noise_scale_1/seed_{seed}"
    x = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))
    return x, gc


def load_ns_nonlinear(d=10, seed=0):
    """Cell 4: Non-stationary Nonlinear. Load NSVAR data, apply tanh link."""
    # Load NSVAR data (linear generation)
    p = f"" + os.path.join(_PROJ_ROOT, "data", "nonstationary_var/num_nodes_{d}/true_lag_7/noise_scale_1/seed_{seed}"
    x_lin = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))

    # Apply tanh to make it nonlinear while preserving GC structure
    # tanh is monotonic, so zero→zero and non-zero→non-zero in the GC sense
    x_nl = np.tanh(x_lin * 1.5)  # scale before tanh to get nonlinear regime
    return x_nl.astype(np.float32), gc


# ============================================================
# Single Experiment Runner
# ============================================================

def run_one(x, gc_true, filter_type, lag, max_iter=5000, seed=0):
    """Run one method on one dataset."""
    d = x.shape[0]
    torch.manual_seed(seed)
    np.random.seed(seed)

    if filter_type == "baseline":
        model = BaselineJRNGC(d=d, lag=lag, layers=5, hidden=50,
                               jacobian_lam=0.01).to(device)
    else:
        model = MambaFilterJRNGC(d=d, lag=lag, layers=5, hidden=50,
                                  jacobian_lam=0.01, filter_type=filter_type).to(device)

    t0 = time.time()
    model, loss = train_model(model, x, max_iter=max_iter)
    gc_pred = model.get_gc_matrix(x)
    train_time = time.time() - t0
    metrics = compute_metrics(gc_true, gc_pred)
    metrics["train_time"] = train_time
    metrics["train_loss"] = float(loss)
    del model
    torch.cuda.empty_cache()
    return metrics


# ============================================================
# Main Experiment
# ============================================================

CELLS = [
    ("Stat+Linear", load_stat_linear, {"lag": 3}),
    ("Stat+Nonlinear", load_stat_nonlinear, {"lag": 5}),
    ("NS+Linear", load_ns_linear, {"lag": 7}),
    ("NS+Nonlinear", load_ns_nonlinear, {"lag": 7}),
]

METHODS = ["baseline", "tcn", "mamba"]
N_SEEDS = 3
MAX_ITER = 5000


def main():
    log("=" * 60)
    log("P2 #4 v2: NONLINEAR/NONSTATIONARY INTERACTION ABLATION")
    log("=" * 60)
    log(f"  Device: {device}, seeds: {N_SEEDS}, max_iter: {MAX_ITER}")
    log(f"  Using existing proven data loaders")
    log(f"  Methods: Baseline, TCN, Mamba")

    all_results = {}

    for cell_name, loader, loader_opts in CELLS:
        log(f"\n{'='*60}")
        log(f"  CELL: {cell_name}")
        log(f"{'='*60}")
        all_results[cell_name] = {}

        lag = loader_opts["lag"]

        for seed in range(N_SEEDS):
            seed_key = f"seed_{seed}"
            all_results[cell_name][seed_key] = {}

            # Load data — only pass parameters the loader accepts
            kwargs = {}
            loader_params = inspect.signature(loader).parameters
            if "seed" in loader_params:
                kwargs["seed"] = seed
            if "d" in loader_params:
                kwargs["d"] = 10
            if "lag" in loader_params:
                kwargs["lag"] = loader_opts.get("lag", 5)
            try:
                x, gc = loader(**kwargs)
            except Exception as e:
                log(f"  {seed_key} data load FAILED: {e}")
                continue

            n_edges = int(gc.sum() if gc.ndim == 2 else gc[..., 0].sum())
            d = x.shape[0]
            T = x.shape[1]
            log(f"  {seed_key}: d={d}, T={T}, edges={n_edges}")

            for method in METHODS:
                log(f"    [{method}]...")
                try:
                    metrics = run_one(x, gc, filter_type=method,
                                      lag=lag, max_iter=MAX_ITER, seed=seed)
                    all_results[cell_name][seed_key][method] = metrics
                    log(f"      AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, "
                        f"SHD={metrics['shd']}, nSHD={metrics.get('nshd','N/A'):.3f}, "
                        f"time={metrics['train_time']:.0f}s")
                except Exception as e:
                    import traceback
                    log(f"      FAILED: {e}")
                    traceback.print_exc()

        # Checkpoint after each cell
        with open("interaction_ablation_checkpoint.json", "w") as f:
            json.dump({"per_seed": all_results}, f, indent=2, default=str)

    # ---- 2x2 Interaction Analysis ----
    log(f"\n{'='*60}")
    log(f"2x2 INTERACTION TABLE (mean AUROC ± std over {N_SEEDS} seeds)")
    log(f"{'='*60}")

    cell_mean_auroc = {}
    for cell_name, _, _ in CELLS:
        cell_mean_auroc[cell_name] = {}
        for method in METHODS:
            aurocs = []
            for seed in range(N_SEEDS):
                sk = f"seed_{seed}"
                if sk in all_results[cell_name] and method in all_results[cell_name][sk]:
                    aurocs.append(all_results[cell_name][sk][method]["auroc"])
            if aurocs:
                cell_mean_auroc[cell_name][method] = {
                    "mean": np.mean(aurocs),
                    "std": np.std(aurocs, ddof=1) if len(aurocs) > 1 else 0.0,
                    "n": len(aurocs),
                }

    # Print 2x2 grid for each method
    for method in METHODS:
        log(f"\n  [{method.upper()}] AUROC by cell:")
        log(f"  {'':>18} {'Linear':>14} {'Nonlinear':>14}")
        for s_label in ["Stat", "NS"]:
            row = f"  {s_label:>18}"
            for n_label in ["Linear", "Nonlinear"]:
                cell_key = f"{s_label}+{n_label}"
                if cell_key in cell_mean_auroc and method in cell_mean_auroc[cell_key]:
                    m = cell_mean_auroc[cell_key][method]
                    row += f" {m['mean']:>8.4f}±{m['std']:.4f}"
                else:
                    row += f" {'N/A':>13}"
            log(row)

    # Delta over baseline
    log(f"\n{'='*60}")
    log("FILTER GAIN OVER BASELINE (Δ AUROC)")
    log(f"{'='*60}")
    log(f"  {'':>18} {'Mamba Δ':>12} {'TCN Δ':>12}")
    log(f"  {'-'*18} {'-'*12} {'-'*12}")

    interaction_scores = {}
    for cell_name, _, _ in CELLS:
        deltas = {}
        for method in ["mamba", "tcn"]:
            if (cell_name in cell_mean_auroc and "baseline" in cell_mean_auroc[cell_name]
                and method in cell_mean_auroc[cell_name]):
                delta = (cell_mean_auroc[cell_name][method]["mean"] -
                         cell_mean_auroc[cell_name]["baseline"]["mean"])
                deltas[method] = delta
            else:
                deltas[method] = float('nan')
        log(f"  {cell_name:<18} {deltas['mamba']:>+10.4f}   {deltas['tcn']:>+10.4f}")
        interaction_scores[cell_name] = deltas

    # Attribution analysis
    log(f"\n{'='*60}")
    log("ATTRIBUTION ANALYSIS (2x2 factorial)")
    log(f"{'='*60}")

    for method in ["mamba", "tcn"]:
        sl = interaction_scores.get("Stat+Linear", {}).get(method, float('nan'))
        sn = interaction_scores.get("Stat+Nonlinear", {}).get(method, float('nan'))
        nl = interaction_scores.get("NS+Linear", {}).get(method, float('nan'))
        nn = interaction_scores.get("NS+Nonlinear", {}).get(method, float('nan'))

        if any(np.isnan(x) for x in [sl, sn, nl, nn]):
            log(f"  [{method.upper()}] Incomplete data, skipping")
            continue

        ns_effect = (nl + nn) / 2
        stat_effect = (sl + sn) / 2
        linear_effect = (sl + nl) / 2
        nonlinear_effect = (sn + nn) / 2

        ns_contrib = ns_effect - stat_effect
        nl_contrib = nonlinear_effect - linear_effect
        interaction = nn - nl - sn + sl

        log(f"\n  [{method.upper()}]")
        log(f"    S main effect (stationary→NS):        {ns_contrib:+.4f}")
        log(f"    N main effect (linear→nonlinear):      {nl_contrib:+.4f}")
        log(f"    S×N interaction:                       {interaction:+.4f}")

        best_cells = []
        best_delta = max(sl, sn, nl, nn)
        for name, d in [("Stat+Linear", sl), ("Stat+Nonlinear", sn),
                         ("NS+Linear", nl), ("NS+Nonlinear", nn)]:
            if abs(d - best_delta) < 1e-10:
                best_cells.append(name)
        log(f"    Best cell(s): {', '.join(best_cells)} (Δ={best_delta:+.4f})")

        total_contrib = abs(ns_contrib) + abs(nl_contrib) + 0.001
        log(f"    %% non-stationarity contribution:       {abs(ns_contrib)/total_contrib*100:.0f}%")
        log(f"    %% nonlinearity contribution:            {abs(nl_contrib)/total_contrib*100:.0f}%")

    # Save final results
    with open("interaction_ablation_results.json", "w") as f:
        json.dump({
            "per_seed": all_results,
            "summary": {c: {m: cell_mean_auroc[c][m] for m in METHODS if m in cell_mean_auroc[c]}
                       for c in cell_mean_auroc},
            "interaction": interaction_scores,
        }, f, indent=2, default=str)
    log(f"\nSaved to interaction_ablation_results.json")

    # Paper-ready summary
    log(f"\n{'='*60}")
    log("PAPER-READY TABLE")
    log(f"{'='*60}")
    log(f"{'Cell':<18} {'Baseline':>10} {'TCN':>10} {'Mamba':>10} {'Mamba Δ':>10}")
    log(f"{'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for cell_name, _, _ in CELLS:
        row = f"{cell_name:<18}"
        for method in ["baseline", "tcn", "mamba"]:
            if cell_name in cell_mean_auroc and method in cell_mean_auroc[cell_name]:
                row += f" {cell_mean_auroc[cell_name][method]['mean']:>8.4f} "
            else:
                row += f" {'N/A':>8} "
        if cell_name in interaction_scores:
            row += f" {interaction_scores[cell_name].get('mamba', float('nan')):>+8.4f}"
        log(row)

    log_fh.close()


if __name__ == "__main__":
    main()
