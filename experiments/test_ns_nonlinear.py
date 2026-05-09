"""Complete the 2x2 interaction table: NS+Nonlinear cell.

Uses exact same config as test_filter_5seed.py (max_iter=2000, lag=7, same models).
Loads NSVAR d=10 data, applies tanh for nonlinear variant.
Runs Baseline + Mamba filter, 5 seeds.
"""
import torch
import torch.nn as nn
import numpy as np
import sys, os, json, time

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

device = torch.device("cuda")
N_SEEDS = 5

log_fh = open("ns_nonlinear_results.log", "w", buffering=1)


def log(msg):
    log_fh.write(msg + "\n")
    log_fh.flush()
    print(msg, flush=True)


def run_one(model_cls, kwargs, x, gc_true, seed, label, max_iter=2000, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_cls(**kwargs).to(device)
    t0 = time.time()
    model, loss = train_model(model, x, max_iter=max_iter, lr=lr)
    gc_pred = model.get_gc_matrix(x)
    train_time = time.time() - t0
    metrics = compute_metrics(gc_true, gc_pred)
    metrics["train_time"] = train_time
    metrics["train_loss"] = float(loss)
    del model
    torch.cuda.empty_cache()
    log(f"  [{label}] AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, "
        f"SHD={metrics['shd']}, nSHD={metrics.get('nshd','N/A'):.3f}, time={train_time:.0f}s")
    return metrics


def main():
    log("=" * 60)
    log("NS+NONLINEAR CELL: NSVAR d=10 + tanh")
    log(f"  Config: lag=7, layers=5, hidden=50, jacobian_lam=0.01, max_iter=2000")
    log(f"  Seeds: {N_SEEDS}, Methods: Baseline + Mamba filter")
    log("=" * 60)

    all_results = {}

    for seed in range(N_SEEDS):
        seed_key = f"seed_{seed}"
        all_results[seed_key] = {}

        # Load NSVAR data
        p = os.path.join(_PROJ_ROOT, "data", "nonstationary_var", "num_nodes_10", "true_lag_7", "noise_scale_1", "seed_{seed}")
        x_lin = np.load(os.path.join(p, "_x.npy"))
        gc = np.load(os.path.join(p, "_gc.npy"))

        # Apply tanh for nonlinear variant
        x_nl = np.tanh(x_lin * 1.5).astype(np.float32)

        n_edges = int(gc.sum() if gc.ndim == 2 else gc.max(axis=2).sum())
        log(f"\n{seed_key}: d={x_nl.shape[0]}, T={x_nl.shape[1]}, edges={n_edges}")

        # Baseline JRNGC
        all_results[seed_key]["baseline"] = run_one(
            BaselineJRNGC,
            {"d": 10, "lag": 7, "layers": 5, "hidden": 50, "jacobian_lam": 0.01},
            x_nl, gc, seed, "Baseline")

        # Mamba Filter
        all_results[seed_key]["mamba"] = run_one(
            MambaFilterJRNGC,
            {"d": 10, "lag": 7, "layers": 5, "hidden": 50,
             "jacobian_lam": 0.01, "d_state": 4, "filter_type": "mamba"},
            x_nl, gc, seed, "MambaFilter")

    # Summary
    log(f"\n{'='*60}")
    log("NS+NONLINEAR SUMMARY")
    log(f"{'='*60}")

    for method in ["baseline", "mamba"]:
        aurocs = [all_results[f"seed_{s}"][method]["auroc"] for s in range(N_SEEDS)]
        auprcs = [all_results[f"seed_{s}"][method]["auprc"] for s in range(N_SEEDS)]
        log(f"  {method}: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs,ddof=1):.4f}, "
            f"AUPRC={np.mean(auprcs):.4f}±{np.std(auprcs,ddof=1):.4f}")

    # Delta
    base_aurocs = [all_results[f"seed_{s}"]["baseline"]["auroc"] for s in range(N_SEEDS)]
    mamba_aurocs = [all_results[f"seed_{s}"]["mamba"]["auroc"] for s in range(N_SEEDS)]
    deltas = [m - b for m, b in zip(mamba_aurocs, base_aurocs)]
    log(f"\n  Mamba Δ over baseline: {np.mean(deltas):+.4f}±{np.std(deltas,ddof=1):.4f}")

    with open("ns_nonlinear_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nSaved to ns_nonlinear_results.json")
    log_fh.close()


if __name__ == "__main__":
    main()
