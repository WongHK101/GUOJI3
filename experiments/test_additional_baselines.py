"""P1 #2: Additional Baselines — cMLP, PCMCI+GPDC, TCDF.

Runs 3 additional baselines on representative datasets:
  - cMLP (Neural-GC): Group-lasso regularized MLP
  - PCMCI+GPDC: tigramite with Gaussian Process conditional independence test
  - TCDF: Temporal Causal Discovery Framework (attention-based TCN)

Datasets: VAR d=50 stat, Lorenz-96 F=40, NSVAR d=10, DREAM3 d=10

Run on cloud: /root/autodl-tmp/GUOJI/mamba_enhanced/
"""
import torch
import torch.nn as nn
import numpy as np
import sys, os, json, time, tempfile, traceback

torch.backends.cudnn.enabled = False

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path
_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)
# project root now resolved via _PROJ_ROOT
# Neural-GC resolved via sibling directory
# baselines resolved via sibling directory

from tgc.metrics import two_classify_metrics, remove_self_connection
from mamba_jrngc_pilot import compute_metrics

device = torch.device("cuda")
JRNGC_DATA = resolve_data_dir()  # resolved via config


def log(msg):
    print(msg, flush=True)


# ============================================================
# Data Loaders
# ============================================================
def load_var_data(d=50, lag=5, seed=0):
    p = os.path.join(JRNGC_DATA, "var", f"num_nodes_{d}",
                     f"true_lag_{lag}", "noise_scale_1", f"seed_{seed}")
    x = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))
    return x, gc

def load_lorenz_data(d=10, F=40, seed=0):
    p = os.path.join(JRNGC_DATA, "lorenz", f"num_nodes_{d}", f"F_{F}", f"seed_{seed}")
    x = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))
    return x, gc

def load_nsvar_data(d=10, seed=0):
    p = os.path.join(_PROJ_ROOT, "data", "nonstationary_var", "num_nodes_{d}", "true_lag_7", "noise_scale_1", "seed_{seed}")
    x = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))
    return x, gc

def load_dream3_data(d=10, subject=0):
    os.chdir(resolve_jrngc_path() or ".")  # resolved via config
    from tgc.data.dream3 import dream3_trajectories
    x, _, gc = dream3_trajectories(d=d, subject=subject)
    if x.ndim == 3:
        x = x[0]
    return x, gc


# ============================================================
# Baseline 1: cMLP (Neural-GC)
# ============================================================
def run_cmlp(x, gc_true, lag=5, hidden=None, lam=0.002, max_iter=10000, verbose=False):
    """Run cMLP from Neural-GC on given data.

    Args:
        x: np.ndarray (d, T)
        gc_true: np.ndarray ground truth
        lag: number of lags
        hidden: list of hidden units
        lam: regularization strength
        max_iter: max training iterations

    Returns:
        dict: metrics from compute_metrics
    """
    from models.cmlp import cMLP, train_model_ista

    d, T = x.shape
    if hidden is None:
        hidden = [100]

    # Convert: (d, T) → (T, d) → (1, T, d)
    X = torch.tensor(x.T[np.newaxis], dtype=torch.float32, device=device)

    model = cMLP(d, lag=lag, hidden=hidden).to(device)

    t0 = time.time()
    train_model_ista(model, X, lr=5e-2, max_iter=max_iter, lam=lam,
                     lam_ridge=1e-2, penalty='H', lookback=5,
                     check_every=100, verbose=1 if verbose else 0)

    gc_pred = model.GC(threshold=False, ignore_lag=True).cpu().data.numpy()
    # gc_pred shape: (d, d), continuous scores

    metrics = compute_metrics(gc_true, gc_pred)
    metrics["train_time"] = time.time() - t0

    del model
    torch.cuda.empty_cache()
    return metrics


# ============================================================
# Baseline 2: PCMCI+ with GPDC
# ============================================================
def run_pcmci_gpdc(x, pc_alpha=0.05, tau_max=1):
    """Run PCMCI+ with GPDC conditional independence test.

    Args:
        x: np.ndarray (d, T)
        pc_alpha: significance level
        tau_max: max lag

    Returns:
        np.ndarray (d, d) continuous GC score matrix
    """
    from tigramite import data_processing as pp
    from tigramite.independence_tests.gpdc import GPDC
    from tigramite.pcmci import PCMCI

    d, T = x.shape
    df = pp.DataFrame(x.T)
    gpdc = GPDC(significance='analytic')
    pcmci = PCMCI(dataframe=df, cond_ind_test=gpdc, verbosity=0)

    res = pcmci.run_pcmci(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
    p_mat = res['p_matrix'][:, :, 1]  # lag=1 only
    gc_pred = (1 - p_mat)[:, :, np.newaxis]
    return np.clip(gc_pred, 0, 1)


# ============================================================
# Baseline 3: TCDF (using existing runner)
# ============================================================
def run_tcdf_single(x, epochs=500, kernel_size=4, significance=0.8):
    """Run TCDF on a single dataset. x: (d, T) numpy array."""
    import pandas as pd
    # tcdf resolved via sibling directory
    import TCDF

    d, T = x.shape

    # Write temp CSV
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df = pd.DataFrame(x.T)
    df.columns = [f"V{i}" for i in range(d)]
    df.to_csv(tmpfile, index=False)
    tmpfile.close()

    use_cuda = torch.cuda.is_available()
    gc_mat = np.zeros((d, d))
    columns = list(df.columns)

    for effect_idx in range(d):
        target = columns[effect_idx]
        try:
            causes, _, _, scores = TCDF.findcauses(
                target=target, cuda=use_cuda, epochs=epochs,
                kernel_size=kernel_size, layers=1, log_interval=500,
                lr=0.01, optimizername="Adam", seed=1111,
                dilation_c=kernel_size, significance=significance,
                file=tmpfile.name,
            )
            if scores is not None and len(scores) == d:
                gc_mat[effect_idx, :] = np.array(scores)
            for cause_idx in causes:
                if cause_idx != effect_idx:
                    gc_mat[effect_idx, cause_idx] = max(gc_mat[effect_idx, cause_idx], 1.0)
        except Exception as e:
            log(f"    TCDF target {target}: FAILED - {e}")

    os.unlink(tmpfile.name)
    return gc_mat


# ============================================================
# Main Experiment Runner
# ============================================================
DATASETS = [
    ("VAR_d50_stat", load_var_data, {"d": 50, "lag": 5}, 3,
     {"cmlp_lam": 0.002, "lag": 5, "max_iter": 10000}),
    ("Lorenz_F40", load_lorenz_data, {"d": 10, "F": 40}, 3,
     {"cmlp_lam": 10.0, "lag": 5, "max_iter": 10000}),
    ("NSVAR_d10", load_nsvar_data, {"d": 10}, 5,
     {"cmlp_lam": 0.002, "lag": 7, "max_iter": 10000}),
    ("DREAM3_d10", load_dream3_data, {"d": 10}, 3,
     {"cmlp_lam": 0.002, "lag": 1, "max_iter": 5000}),
]

# Methods to run per dataset
# cMLP: all datasets
# PCMCI+GPDC: all datasets (fast)
# TCDF: VAR d=50 and Lorenz-96 only (slow — per-variable training)
METHODS = ["cmlp", "pcmci_gpdc", "tcdf"]


def main():
    log("ADDITIONAL BASELINES: cMLP + PCMCI+GPDC + TCDF")
    log("=" * 60)
    log(f"Device: {device}")

    all_results = {}

    for ds_name, loader, loader_args, n_seeds, cmlp_cfg in DATASETS:
        log(f"\n{'='*60}")
        log(f"  {ds_name} ({n_seeds} seeds)")
        log(f"{'='*60}")
        all_results[ds_name] = {}

        for seed in range(n_seeds):
            seed_key = f"seed_{seed}"
            all_results[ds_name][seed_key] = {}

            # Load data
            kwargs = dict(loader_args)
            if "seed" in loader.__code__.co_varnames:
                kwargs["seed"] = seed
            if "subject" in loader.__code__.co_varnames:
                kwargs["subject"] = seed

            try:
                x, gc = loader(**kwargs)
            except Exception as e:
                log(f"  {seed_key} data load FAILED: {e}")
                continue

            log(f"  {seed_key}: d={x.shape[0]}, T={x.shape[1]}, "
                 f"edges={int(gc.sum() if gc.ndim == 2 else gc[:,:,0].sum())}")

            # --- cMLP ---
            if "cmlp" in METHODS:
                log(f"    [cMLP] training (lam={cmlp_cfg['cmlp_lam']})...")
                t0 = time.time()
                try:
                    metrics = run_cmlp(x, gc,
                                       lag=cmlp_cfg["lag"],
                                       lam=cmlp_cfg["cmlp_lam"],
                                       max_iter=cmlp_cfg["max_iter"])
                    metrics["train_time"] = time.time() - t0
                    all_results[ds_name][seed_key]["cmlp"] = metrics
                    log(f"      AUROC={metrics['auroc']:.4f}, "
                         f"AUPRC={metrics['auprc']:.4f}, "
                         f"SHD={metrics['shd']}, "
                         f"nSHD={metrics.get('nshd', 'N/A'):.3f}, "
                         f"time={metrics['train_time']:.0f}s")
                except Exception as e:
                    log(f"      FAILED: {e}")
                    traceback.print_exc()

            # --- PCMCI+GPDC ---
            if "pcmci_gpdc" in METHODS:
                log(f"    [PCMCI+GPDC] running...")
                t0 = time.time()
                try:
                    gc_pred = run_pcmci_gpdc(x)
                    metrics = compute_metrics(gc, gc_pred)
                    metrics["train_time"] = time.time() - t0
                    all_results[ds_name][seed_key]["pcmci_gpdc"] = metrics
                    log(f"      AUROC={metrics['auroc']:.4f}, "
                         f"AUPRC={metrics['auprc']:.4f}, "
                         f"SHD={metrics['shd']}, "
                         f"time={metrics['train_time']:.0f}s")
                except Exception as e:
                    log(f"      FAILED: {e}")
                    traceback.print_exc()

            # --- TCDF (selected datasets only) ---
            if "tcdf" in METHODS and ds_name in ["VAR_d50_stat", "Lorenz_F40"]:
                log(f"    [TCDF] running (500 epochs per var)...")
                t0 = time.time()
                try:
                    gc_pred = run_tcdf_single(x, epochs=500)
                    metrics = compute_metrics(gc, gc_pred)
                    metrics["train_time"] = time.time() - t0
                    all_results[ds_name][seed_key]["tcdf"] = metrics
                    log(f"      AUROC={metrics['auroc']:.4f}, "
                         f"AUPRC={metrics['auprc']:.4f}, "
                         f"SHD={metrics['shd']}, "
                         f"time={metrics['train_time']:.0f}s")
                except Exception as e:
                    log(f"      FAILED: {e}")
                    traceback.print_exc()

    # ---- Summary ----
    log(f"\n{'='*60}")
    log("BASELINE SUMMARY")
    log(f"{'='*60}")

    # Aggregate per-method per-dataset
    method_agg = {}
    for ds_name in all_results:
        for method in ["cmlp", "pcmci_gpdc", "tcdf"]:
            aurocs = []
            auprcs = []
            shds = []
            for seed_key in all_results[ds_name]:
                if method in all_results[ds_name][seed_key]:
                    m = all_results[ds_name][seed_key][method]
                    aurocs.append(m["auroc"])
                    auprcs.append(m["auprc"])
                    shds.append(m["shd"])
            if aurocs:
                key = f"{ds_name}/{method}"
                method_agg[key] = {
                    "auroc_mean": float(np.mean(aurocs)),
                    "auroc_std": float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0,
                    "auprc_mean": float(np.mean(auprcs)),
                    "auprc_std": float(np.std(auprcs, ddof=1)) if len(auprcs) > 1 else 0.0,
                    "shd_mean": float(np.mean(shds)),
                    "shd_std": float(np.std(shds, ddof=1)) if len(shds) > 1 else 0.0,
                    "n": len(aurocs),
                }

    # Print summary table
    print(f"\n{'Method':<25} {'AUROC':>16} {'AUPRC':>16} {'SHD':>10}  n")
    print(f"{'-'*25} {'-'*16} {'-'*16} {'-'*10}  -")
    for key, agg in sorted(method_agg.items()):
        print(f"  {key:<25} {agg['auroc_mean']:>8.4f}±{agg['auroc_std']:<7.4f} "
              f"{agg['auprc_mean']:>8.4f}±{agg['auprc_std']:<7.4f} "
              f"{agg['shd_mean']:>8.1f}±{agg['shd_std']:<5.1f}  {agg['n']}")

    # Compare with our method (hardcoded from previous results)
    our_results = {
        "VAR_d50_stat/Filter": (0.6963, 0.035),
        "VAR_d50_stat/Baseline": (0.7145, 0.011),
        "Lorenz_F40/Filter": (0.9374, 0.0),
        "Lorenz_F40/Baseline": (0.9350, 0.0),
        "NSVAR_d10/Filter": (0.9457, 0.028),
        "NSVAR_d10/Baseline": (0.9296, 0.024),
        "DREAM3_d10/Filter": (0.5442, 0.008),
        "DREAM3_d10/Baseline": (0.5113, 0.047),
    }

    method_names = {
        "cmlp": "cMLP (Neural-GC)",
        "pcmci_gpdc": "PCMCI+GPDC",
        "tcdf": "TCDF",
    }

    log(f"\n{'='*60}")
    log("COMPARISON WITH OUR METHOD")
    log(f"{'='*60}")
    log(f"{'Dataset':<20} {'cMLP':>10} {'PCMCI+GPDC':>12} {'TCDF':>10} "
         f"{'Ours (Filter)':>14} {'Ours (Base)':>13}")
    log(f"{'-'*20} {'-'*10} {'-'*12} {'-'*10} {'-'*14} {'-'*13}")

    for ds_name in ["VAR_d50_stat", "Lorenz_F40", "NSVAR_d10", "DREAM3_d10"]:
        row = f"  {ds_name:<20}"
        for method in ["cmlp", "pcmci_gpdc", "tcdf"]:
            key = f"{ds_name}/{method}"
            if key in method_agg:
                row += f" {method_agg[key]['auroc_mean']:>8.4f}  "
            else:
                row += f" {'N/A':>8}  "
        our_key_f = f"{ds_name}/Filter"
        our_key_b = f"{ds_name}/Baseline"
        if our_key_f in our_results:
            row += f" {our_results[our_key_f][0]:>8.4f}     "
        else:
            row += f" {'N/A':>8}     "
        if our_key_b in our_results:
            row += f" {our_results[our_key_b][0]:>8.4f}"
        log(row)

    with open("additional_baselines_results.json", "w") as f:
        json.dump({"per_seed": all_results, "summary": method_agg}, f, indent=2, default=str)
    log(f"\nSaved to additional_baselines_results.json")


if __name__ == "__main__":
    main()
