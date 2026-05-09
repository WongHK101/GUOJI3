"""5-seed verification of Mamba input-filter JRNGC."""
import torch, numpy as np, sys, os, time, json
sys.path.insert(0, "/root/autodl-tmp/GUOJI/JRNGC")
sys.path.insert(0, ".")
from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                train_model, compute_metrics)

device = torch.device("cuda")
seeds = [0, 1, 2, 3, 4]
results = {}

for seed in seeds:
    data_path = f"data/nonstationary_var/num_nodes_10/true_lag_7/noise_scale_1/seed_{seed}"
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    results[f"seed_{seed}"] = {}

    for name, ModelClass, kwargs, lr in [
        ("A_baseline", BaselineJRNGC,
         {"d":10, "lag":7, "layers":5, "hidden":50, "jacobian_lam":0.01}, 1e-3),
        ("F_ds4", MambaFilterJRNGC,
         {"d":10, "lag":7, "layers":5, "hidden":50, "jacobian_lam":0.01, "d_state":4}, 1e-3),
        ("F_ds8", MambaFilterJRNGC,
         {"d":10, "lag":7, "layers":5, "hidden":50, "jacobian_lam":0.01, "d_state":8}, 1e-3),
    ]:
        print(f"\n{'='*50}")
        print(f"Seed {seed}, {name}")
        print(f"{'='*50}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        m = ModelClass(**kwargs).to(device)
        t0 = time.time()
        m, loss = train_model(m, x, max_iter=2000, lr=lr, verbose=True)
        gc_pred = m.get_gc_matrix(x)
        met = compute_metrics(gc, gc_pred)
        met["train_loss"] = float(loss)
        met["train_time"] = time.time() - t0
        results[f"seed_{seed}"][name] = met
        print(f"  => AUROC={met['auroc']:.4f} SHD={met['shd']} F1={met['f1']:.4f}")
        del m
        torch.cuda.empty_cache()

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY (5 seeds)")
print(f"{'='*60}")
for name in ["A_baseline", "F_ds4", "F_ds8"]:
    aurocs = [results[f"seed_{s}"][name]["auroc"] for s in range(5)]
    shds = [results[f"seed_{s}"][name]["shd"] for s in range(5)]
    losses = [results[f"seed_{s}"][name]["train_loss"] for s in range(5)]
    print(f"\n{name}:")
    print(f"  AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  SHD:   {np.mean(shds):.1f} ± {np.std(shds):.1f}")
    print(f"  Loss:  {np.mean(losses):.6f} ± {np.std(losses):.6f}")
    print(f"  Seeds: ", end="")
    for s in range(5):
        print(f"{aurocs[s]:.4f} ", end="")
    print()

with open("filter_5seed_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to filter_5seed_results.json")
