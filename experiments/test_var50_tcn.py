"""S3: VAR d=50 TCN backfill (stationary, 3 seeds)."""
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
JRNGC_DATA = resolve_data_dir()  # resolved via config

log_fh = open("var50_tcn_results.log", "w", buffering=1)

def log(msg):
    log_fh.write(msg + "\n")
    log_fh.flush()
    print(msg, flush=True)

def run_one(x, gc_true, d, lag, seed, label, filter_type=None, max_iter=5000, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if filter_type is None:
        model = BaselineJRNGC(
            d=d, lag=lag, layers=5, hidden=50,
            jacobian_lam=0.01
        ).to(device)
    else:
        model = MambaFilterJRNGC(
            d=d, lag=lag, layers=5, hidden=50,
            jacobian_lam=0.01, d_state=4, ortho_lam=0.05,
            residual_scale=0.1, filter_type=filter_type
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
    log("S3: VAR d=50 TCN BACKFILL (3 seeds)")
    log("=" * 60)

    all_results = {}

    for seed in range(3):
        data_path = os.path.join(JRNGC_DATA,
            f"var/num_nodes_50/true_lag_5/noise_scale_1/seed_{seed}")
        x = np.load(os.path.join(data_path, "_x.npy"))
        gc = np.load(os.path.join(data_path, "_gc.npy"))
        log(f"\nVAR d=50 seed_{seed}: x.shape={x.shape}, gc.shape={gc.shape}, edges={int(gc.sum())}")

        seed_key = f"seed_{seed}"
        all_results[seed_key] = {}

        all_results[seed_key]["tcn"] = run_one(
            x, gc, d=50, lag=5, seed=seed,
            label=f"TCN VAR50 seed_{seed}",
            filter_type="tcn", max_iter=5000)

    # Save
    with open("var50_tcn_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nSaved to var50_tcn_results.json")

    # Summary
    aurocs = [all_results[f"seed_{s}"]["tcn"]["auroc"] for s in range(3)]
    log(f"\nVAR_d50 TCN: AUROC={np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")

    log_fh.close()

if __name__ == "__main__":
    main()
