"""P2 S3: TCN backfill for 11-config main table.

Runs TCN filter on the 4 missing configs:
  1. Lorenz-96 F=40 (3 seeds)
  2. NSVAR d=50 PlanA (3 seeds)
  3. CausalTime traffic (1 config)
  4. fMRI d=15 (3 subjects x 1 seed each)

Also generates the final 11-config table at end.
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

log_fh = open("tcn_backfill_results.log", "w", buffering=1)


def log(msg):
    log_fh.write(msg + "\n")
    log_fh.flush()
    print(msg, flush=True)


def run_one(x, gc_true, d, lag, seed, label, max_iter=5000, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MambaFilterJRNGC(
        d=d, lag=lag, layers=5, hidden=50,
        jacobian_lam=0.01, d_state=4, ortho_lam=0.05,
        residual_scale=0.1, filter_type="tcn"
    ).to(device)
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
        f"SHD={metrics['shd']}, nSHD={metrics['nshd']:.3f}, time={train_time:.0f}s")
    return metrics


def main():
    log("=" * 60)
    log("S3: TCN BACKFILL FOR 11-CONFIG MAIN TABLE")
    log("=" * 60)

    all_results = {}

    # ============================================================
    # 1. Lorenz-96 F=40 (3 seeds)
    # ============================================================
    log("\n--- LORENZ-96 F=40 (TCN, 3 seeds) ---")
    all_results["Lorenz_F40"] = {}
    os.chdir(resolve_jrngc_path() or ".")  # resolved via config
    from tgc.data.lorenz import lorenz_96
    # os.chdir removed — paths resolved via config
    for seed in range(3):
        x, _, gc = lorenz_96(d=10, t=500, t_eval=0, f=40, seed=seed)
        if x.ndim == 3:
            x = x[0]
        all_results["Lorenz_F40"][f"seed_{seed}"] = {
            "tcn": run_one(x, gc, d=10, lag=5, seed=seed,
                          label=f"TCN Lorenz seed_{seed}")
        }

    # ============================================================
    # 2. NSVAR d=50 PlanA (3 seeds)
    # ============================================================
    log("\n--- NSVAR d=50 PlanA (TCN, 3 seeds) ---")
    all_results["NSVAR_d50_PlanA"] = {}
    for seed in range(3):
        p = f"" + os.path.join(_PROJ_ROOT, "data", "nonstationary_var_planA/num_nodes_50/true_lag_14/noise_scale_1/seed_{seed}"
        x = np.load(os.path.join(p, "_x.npy"))
        gc = np.load(os.path.join(p, "_gc.npy"))
        all_results["NSVAR_d50_PlanA"][f"seed_{seed}"] = {
            "tcn": run_one(x, gc, d=50, lag=14, seed=seed,
                          label=f"TCN PlanA seed_{seed}")
        }

    # ============================================================
    # 3. CausalTime traffic
    # ============================================================
    log("\n--- CausalTime traffic (TCN) ---")
    p = "" + os.path.join(JRNGC_DATA, "causaltime", "traffic"
    x = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))
    d = x.shape[0]
    all_results["CT_traffic"] = {
        "tcn": run_one(x, gc, d=d, lag=1, seed=0,
                      label="TCN traffic")
    }

    # ============================================================
    # 4. fMRI d=15 (3 subjects, seed_0)
    # ============================================================
    log("\n--- fMRI d=15 (TCN, 3 subjects) ---")
    all_results["fMRI_d15"] = {}
    fmri_base = "" + os.path.join(JRNGC_DATA, "fmri", "num_nodes_15"
    for subj in range(3):
        p = os.path.join(fmri_base, f"subject_{subj}", "seed_0")
        x = np.load(os.path.join(p, "_x.npy"))
        gc = np.load(os.path.join(p, "_gc.npy"))
        all_results["fMRI_d15"][f"subj_{subj}"] = {
            "tcn": run_one(x, gc, d=15, lag=1, seed=subj,
                          label=f"TCN fMRI subj_{subj}")
        }

    # Save
    with open("tcn_backfill_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nSaved to tcn_backfill_results.json")

    # ============================================================
    # Summary
    # ============================================================
    log(f"\n{'='*60}")
    log("TCN BACKFILL SUMMARY")
    log(f"{'='*60}")
    for config, seeds in all_results.items():
        for seed_key, methods in seeds.items():
            if "tcn" in methods:
                log(f"  {config}/{seed_key}: TCN AUROC={methods['tcn']['auroc']:.4f}")

    log_fh.close()


if __name__ == "__main__":
    main()
