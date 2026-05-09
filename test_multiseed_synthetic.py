"""P1-3: Multi-seed synthetic validation (all available seeds).

Runs baseline + ISTF-Mamba on:
  - VAR_d50 (5 seeds: 0-4)
  - Lorenz_F40 (5 seeds: 0-4)
  - NSVAR_d10 (5 seeds: 0-4)
  - NSVAR_d50_PlanA (3 seeds: 0-2)

Uses max_iter=5000 for consistency with main experiments.
"""
import torch
import numpy as np
import sys, os, json, time

torch.backends.cudnn.enabled = False

os.chdir("/root/autodl-tmp/GUOJI/mamba_enhanced")
sys.path.insert(0, "/root/autodl-tmp/GUOJI/JRNGC")
sys.path.insert(0, "/root/autodl-tmp/GUOJI/mamba_enhanced")

from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                train_model, compute_metrics)

device = torch.device("cuda")
JRNGC_DATA = "/root/autodl-tmp/GUOJI/JRNGC/data"
OUR_DATA = "/root/autodl-tmp/GUOJI/mamba_enhanced/data"

log_fh = open("multiseed_synthetic.log", "w", buffering=1)

def log(msg):
    log_fh.write(msg + "\n")
    log_fh.flush()
    print(msg, flush=True)

def run_one(x, gc_true, d, lag, seed, model_type, max_iter=5000, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if model_type == "baseline":
        model = BaselineJRNGC(
            d=d, lag=lag, layers=5, hidden=50, jacobian_lam=0.01
        ).to(device)
    else:
        model = MambaFilterJRNGC(
            d=d, lag=lag, layers=5, hidden=50,
            jacobian_lam=0.01, d_state=4, ortho_lam=0.05,
            residual_scale=0.1, filter_type="mamba"
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
    return metrics

def main():
    log("=" * 60)
    log("P1-3: MULTI-SEED SYNTHETIC VALIDATION (all available seeds)")
    log("=" * 60)

    configs = [
        {
            "name": "VAR_d50",
            "path": f"{JRNGC_DATA}/var/num_nodes_50/true_lag_5/noise_scale_1",
            "n_seeds": 5, "lag": 5, "d": 50,
        },
        {
            "name": "Lorenz_F40",
            "path": f"{JRNGC_DATA}/lorenz/num_nodes_10/F_40",
            "n_seeds": 5, "lag": 1, "d": 10,
        },
        {
            "name": "NSVAR_d10",
            "path": f"{OUR_DATA}/nonstationary_var/num_nodes_10/true_lag_7/noise_scale_1",
            "n_seeds": 5, "lag": 7, "d": 10,
        },
        {
            "name": "NSVAR_d50_PlanA",
            "path": f"{OUR_DATA}/nonstationary_var_planA/num_nodes_50/true_lag_14/noise_scale_1",
            "n_seeds": 3, "lag": 14, "d": 50,
        },
    ]

    all_results = {}

    for cfg in configs:
        name = cfg["name"]
        log(f"\n{'='*60}")
        log(f"CONFIG: {name} ({cfg['n_seeds']} seeds)")
        log(f"{'='*60}")
        all_results[name] = {}

        for seed in range(cfg["n_seeds"]):
            data_path = os.path.join(cfg["path"], f"seed_{seed}")
            x = np.load(os.path.join(data_path, "_x.npy"))
            gc = np.load(os.path.join(data_path, "_gc.npy"))
            log(f"\n  seed_{seed}: T={x.shape[1]}, edges={int(gc.sum())}")

            # Baseline
            t0 = time.time()
            base_m = run_one(x, gc, cfg["d"], cfg["lag"], seed, "baseline")
            log(f"    Baseline: AUROC={base_m['auroc']:.4f}, AUPRC={base_m['auprc']:.4f}, "
                f"SHD={base_m['shd']}, time={time.time()-t0:.0f}s")

            # Mamba
            t0 = time.time()
            mamba_m = run_one(x, gc, cfg["d"], cfg["lag"], seed, "mamba")
            log(f"    Mamba:    AUROC={mamba_m['auroc']:.4f}, AUPRC={mamba_m['auprc']:.4f}, "
                f"SHD={mamba_m['shd']}, time={time.time()-t0:.0f}s")

            all_results[name][f"seed_{seed}"] = {
                "baseline": base_m,
                "mamba": mamba_m,
            }

    # Summary
    log(f"\n{'='*60}")
    log("MULTI-SEED SUMMARY")
    log(f"{'='*60}")
    for cfg in configs:
        name = cfg["name"]
        seeds_data = all_results[name]
        b_au = [seeds_data[f"seed_{s}"]["baseline"]["auroc"] for s in range(cfg["n_seeds"])]
        m_au = [seeds_data[f"seed_{s}"]["mamba"]["auroc"] for s in range(cfg["n_seeds"])]
        log(f"  {name}:")
        log(f"    Baseline: {np.mean(b_au):.4f} ± {np.std(b_au):.4f}  [{', '.join(f'{v:.4f}' for v in b_au)}]")
        log(f"    Mamba:    {np.mean(m_au):.4f} ± {np.std(m_au):.4f}  [{', '.join(f'{v:.4f}' for v in m_au)}]")
        log(f"    Δ AUROC:  {np.mean(m_au) - np.mean(b_au):+.4f}")

    with open("multiseed_synthetic_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nSaved to multiseed_synthetic_results.json")
    log_fh.close()

if __name__ == "__main__":
    main()
