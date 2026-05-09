"""S3: DREAM3 backfill — baseline + mamba + tcn on d=10/50/100, 3 subjects each."""
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

log_fh = open("dream3_backfill.log", "w", buffering=1)

def log(msg):
    log_fh.write(msg + "\n")
    log_fh.flush()
    print(msg, flush=True)

def run_one(x, gc_true, d, lag, seed, label, filter_type=None, max_iter=2000, lr=1e-3):
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

# Import DREAM3 loader
os.chdir("/root/autodl-tmp/GUOJI/JRNGC")
from tgc.data.dream3 import dream3_trajectories

def main():
    log("=" * 60)
    log("S3: DREAM3 BACKFILL (baseline + mamba + tcn)")
    log("=" * 60)

    all_results = {}

    for d in [10, 50, 100]:
        cfg_name = f"DREAM3_d{d}"
        all_results[cfg_name] = {}
        for subj in range(3):  # 3 subjects per size
            x, _, gc = dream3_trajectories(d=d, subject=subj)
            # x is (reps, d, T), 3D — make_windows handles this automatically
            log(f"\n{'='*50}")
            log(f"DREAM3 d={d} subj={subj}: x.shape={x.shape}, T=21, edges={int(gc.sum())}")
            log(f"{'='*50}")

            seed_key = f"subj_{subj}"
            all_results[cfg_name][seed_key] = {}

            # Baseline
            all_results[cfg_name][seed_key]["baseline"] = run_one(
                x, gc, d=d, lag=1, seed=subj,
                label=f"baseline DREAM3 d={d} s={subj}",
                filter_type=None, max_iter=2000)

            # Mamba ISTF
            all_results[cfg_name][seed_key]["mamba"] = run_one(
                x, gc, d=d, lag=1, seed=subj,
                label=f"mamba DREAM3 d={d} s={subj}",
                filter_type="mamba", max_iter=2000)

            # TCN
            all_results[cfg_name][seed_key]["tcn"] = run_one(
                x, gc, d=d, lag=1, seed=subj,
                label=f"tcn DREAM3 d={d} s={subj}",
                filter_type="tcn", max_iter=2000)

    # Save
    with open("dream3_backfill_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nSaved to dream3_backfill_results.json")

    # Summary
    log(f"\n{'='*60}")
    log("DREAM3 BACKFILL SUMMARY")
    log(f"{'='*60}")
    for cfg in sorted(all_results.keys()):
        for method in ["baseline", "mamba", "tcn"]:
            aurocs = []
            for seed_key in all_results[cfg]:
                if method in all_results[cfg][seed_key]:
                    aurocs.append(all_results[cfg][seed_key][method]["auroc"])
            if aurocs:
                log(f"  {cfg} {method}: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f} (n={len(aurocs)})")

    log_fh.close()

if __name__ == "__main__":
    main()
