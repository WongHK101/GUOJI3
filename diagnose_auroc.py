"""Mamba-JRNGC concat-architecture diagnostic on seed 0.

Tests (all using CONCATENATION-based architecture, not multiplicative modulation):
1. Baseline JRNGC
2. Concat Mamba with varying d_cond (4 vs 8 vs d//2)
3. Varying d_state (4 vs 8 vs 16)
4. Varying learning rate (1e-3 vs 5e-4 vs 1e-4)
5. Varying Jacobian lambda (0.01 vs 0.05 vs 0.1)
6. With/without time-weighted loss

Upload to /root/autodl-tmp/GUOJI/mamba_enhanced/ and run on cloud.
"""
import torch
import torch.nn as nn
import numpy as np
import sys, os, json, time, copy

sys.path.insert(0, "/root/autodl-tmp/GUOJI/JRNGC")
sys.path.insert(0, "/root/autodl-tmp/GUOJI/mamba_enhanced")

from mamba_jrngc_pilot import (BaselineJRNGC, MambaJRNGC, train_model,
                                compute_metrics, ResidualBlock)


def run_diagnostic(data_path, seed=0, max_iter=3000):
    """Test multiple Mamba concat configs on a single dataset."""
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    d = x.shape[0]
    device = torch.device('cuda')

    configs = {
        "A_baseline": {
            "type": "baseline",
            "lr": 1e-3,
        },
        "C_concat_default": {
            "type": "mamba",
            "lr": 1e-3,
            "d_state": 16,
            "d_cond": 4,
            "jac_lam": 0.01,
            "time_weight": True,
        },
        "C_small_cond": {
            "type": "mamba",
            "lr": 1e-3,
            "d_state": 8,
            "d_cond": 4,
            "jac_lam": 0.01,
            "time_weight": True,
        },
        "C_large_cond": {
            "type": "mamba",
            "lr": 1e-3,
            "d_state": 16,
            "d_cond": 8,
            "jac_lam": 0.01,
            "time_weight": True,
        },
        "C_low_lr": {
            "type": "mamba",
            "lr": 1e-4,
            "d_state": 16,
            "d_cond": 4,
            "jac_lam": 0.01,
            "time_weight": True,
        },
        "C_mid_lr": {
            "type": "mamba",
            "lr": 5e-4,
            "d_state": 8,
            "d_cond": 4,
            "jac_lam": 0.01,
            "time_weight": True,
        },
        "C_high_jac": {
            "type": "mamba",
            "lr": 1e-3,
            "d_state": 16,
            "d_cond": 4,
            "jac_lam": 0.05,
            "time_weight": True,
        },
        "C_small_state_low_lr": {
            "type": "mamba",
            "lr": 1e-4,
            "d_state": 4,
            "d_cond": 4,
            "jac_lam": 0.01,
            "time_weight": True,
        },
        "C_no_tw": {
            "type": "mamba",
            "lr": 1e-3,
            "d_state": 16,
            "d_cond": 4,
            "jac_lam": 0.01,
            "time_weight": False,
        },
    }

    results = {}
    for name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"  {name}: {cfg}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        if cfg["type"] == "baseline":
            model = BaselineJRNGC(d=d, lag=7, layers=5, hidden=50,
                                  jacobian_lam=0.01).to(device)
        else:
            model = MambaJRNGC(d=d, lag=7, layers=5, hidden=50,
                              jacobian_lam=cfg["jac_lam"],
                              d_state=cfg["d_state"],
                              d_cond=cfg["d_cond"],
                              use_time_weight_loss=cfg["time_weight"]).to(device)

        t0 = time.time()
        model, best_loss = train_model(model, x, max_iter=max_iter,
                                       lr=cfg["lr"], verbose=True)
        t_train = time.time() - t0

        gc_pred = model.get_gc_matrix(x)
        metrics = compute_metrics(gc, gc_pred)
        metrics["train_loss"] = float(best_loss)
        metrics["train_time"] = t_train

        results[name] = metrics
        print(f"  → AUROC={metrics['auroc']:.4f}  SHD={metrics['shd']}  "
              f"F1={metrics['f1']:.4f}  time={t_train:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    header = f"{'Config':30s} {'AUROC':>8s} {'SHD':>6s} {'F1':>8s} {'Loss':>10s} {'Time':>6s}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:30s} {m['auroc']:8.4f} {m['shd']:6d} {m['f1']:8.4f} {m['train_loss']:10.6f} {m['train_time']:6.0f}s")

    # Save
    output = "/root/autodl-tmp/GUOJI/mamba_enhanced/diagnostic_results.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/root/autodl-tmp/GUOJI/mamba_enhanced/data/nonstationary_var")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_iter", type=int, default=3000)
    args = parser.parse_args()

    data_path = os.path.join(args.data_dir, "num_nodes_10", "true_lag_7",
                            "noise_scale_1", f"seed_{args.seed}")
    run_diagnostic(data_path, seed=args.seed, max_iter=args.max_iter)
