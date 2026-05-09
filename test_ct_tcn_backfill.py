"""S3: CausalTime medical + pm25 TCN backfill."""
import torch
import torch.nn as nn
import numpy as np
import sys, os, json, time

torch.backends.cudnn.enabled = False

os.chdir("/root/autodl-tmp/GUOJI/mamba_enhanced")
sys.path.insert(0, "/root/autodl-tmp/GUOJI/JRNGC")
sys.path.insert(0, "/root/autodl-tmp/GUOJI/mamba_enhanced")

from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                train_model, compute_metrics)

device = torch.device("cuda")

log_fh = open("ct_tcn_backfill.log", "w", buffering=1)

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
    log("S3: CT medical + pm25 TCN BACKFILL")
    log("=" * 60)

    all_results = {}
    base = "/root/autodl-tmp/GUOJI/JRNGC/data/causaltime"

    for ds in ["medical", "pm25"]:
        log(f"\n--- CT_{ds} (TCN) ---")
        p = os.path.join(base, ds)
        x = np.load(os.path.join(p, "_x.npy"))
        gc = np.load(os.path.join(p, "_gc.npy"))
        d = x.shape[0]
        log(f"  x.shape={x.shape}, gc.shape={gc.shape}, d={d}")
        all_results[f"CT_{ds}"] = {
            "tcn": run_one(x, gc, d=d, lag=1, seed=0,
                          label=f"TCN {ds}", max_iter=5000)
        }

    with open("ct_tcn_backfill_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nSaved to ct_tcn_backfill_results.json")

    for cfg, methods in all_results.items():
        if "tcn" in methods:
            log(f"  {cfg}: TCN AUROC={methods['tcn']['auroc']:.4f}")

    log_fh.close()

if __name__ == "__main__":
    main()
