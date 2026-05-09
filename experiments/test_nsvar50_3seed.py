"""Plan A non-stationary VAR d=50 3-seed full comparison."""
import torch, numpy as np, sys, os, time, json
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path
_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)
sys.path.insert(0, ".")
from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                train_model, compute_metrics)

device = torch.device("cuda")
FILTER_KWARGS = {"d_state": 8, "ortho_lam": 0.05, "residual_scale": 0.1}
results = {}

for seed in [0, 1, 2]:
    data_path = f"data/nonstationary_var_planA/num_nodes_50/true_lag_14/noise_scale_1/seed_{seed}"
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    print(f"\nSeed {seed}: x={x.shape}, gc={gc.shape}, edges={int(gc.sum())}")

    results[f"seed_{seed}"] = {}

    for name, ModelClass, kwargs in [
        ("baseline", BaselineJRNGC,
         {"d": 50, "lag": 14, "layers": 5, "hidden": 50, "jacobian_lam": 0.01}),
        ("F_ds8_fix", MambaFilterJRNGC,
         {"d": 50, "lag": 14, "layers": 5, "hidden": 50, "jacobian_lam": 0.01,
          **FILTER_KWARGS}),
    ]:
        print(f"  [{name}] training...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        m = ModelClass(**kwargs).to(device)
        t0 = time.time()
        m, loss = train_model(m, x, max_iter=2000, lr=1e-3, verbose=True)
        gc_pred = m.get_gc_matrix(x)
        met = compute_metrics(gc, gc_pred)
        met["train_loss"] = float(loss)
        met["train_time"] = time.time() - t0
        results[f"seed_{seed}"][name] = met
        print(f"  => AUROC={met['auroc']:.4f} SHD={met['shd']} loss={loss:.6f} time={met['train_time']:.0f}s")
        del m
        torch.cuda.empty_cache()

# Summary
print(f"\n{'='*60}")
print("Plan A NSVAR d=50 3-seed SUMMARY")
for name in ["baseline", "F_ds8_fix"]:
    aurocs = [results[f"seed_{s}"][name]["auroc"] for s in range(3)]
    shds = [results[f"seed_{s}"][name]["shd"] for s in range(3)]
    print(f"\n{name}:")
    print(f"  AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  SHD:   {np.mean(shds):.1f} ± {np.std(shds):.1f}")
    print(f"  Seeds: {[f'{v:.4f}' for v in aurocs]}")

bl_mean = np.mean([results[f"seed_{s}"]["baseline"]["auroc"] for s in range(3)])
f_mean = np.mean([results[f"seed_{s}"]["F_ds8_fix"]["auroc"] for s in range(3)])
print(f"\nΔ AUROC: {f_mean-bl_mean:+.4f} ({(f_mean-bl_mean)/(bl_mean+1e-8)*100:+.1f}%)")

with open("nsvar50_3seed_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved to nsvar50_3seed_results.json")
