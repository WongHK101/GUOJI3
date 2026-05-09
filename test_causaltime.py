"""CausalTime test: baseline vs F_ds8_fix on traffic/medical/pm25."""
import torch, numpy as np, sys, os, time, json
sys.path.insert(0, "/root/autodl-tmp/GUOJI/JRNGC")
sys.path.insert(0, ".")
from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                train_model, compute_metrics)

device = torch.device("cuda")
FILTER_KWARGS = {"d_state": 8, "ortho_lam": 0.05, "residual_scale": 0.1}
JRNGC_DATA = "/root/autodl-tmp/GUOJI/JRNGC/data"
results = {}

for ds_name, max_iter in [("traffic", 2000), ("medical", 2000), ("pm25", 1000)]:
    data_path = os.path.join(JRNGC_DATA, f"causaltime/{ds_name}")
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    d = x.shape[0]
    print(f"\n{'='*50}")
    print(f"CausalTime/{ds_name}: d={d}, T={x.shape[1]}, gc={gc.shape}, edges={int(gc.sum())}")
    print(f"{'='*50}")

    results[ds_name] = {}

    for model_name, ModelClass, kwargs in [
        ("baseline", BaselineJRNGC,
         {"d": d, "lag": 1, "layers": 5, "hidden": 50, "jacobian_lam": 0.01}),
        ("F_ds8_fix", MambaFilterJRNGC,
         {"d": d, "lag": 1, "layers": 5, "hidden": 50, "jacobian_lam": 0.01,
          **FILTER_KWARGS}),
    ]:
        print(f"  [{model_name}] training max_iter={max_iter}...")
        torch.manual_seed(0)
        np.random.seed(0)

        m = ModelClass(**kwargs).to(device)
        t0 = time.time()
        m, loss = train_model(m, x, max_iter=max_iter, lr=1e-3, verbose=True)
        gc_pred = m.get_gc_matrix(x)
        met = compute_metrics(gc, gc_pred)
        met["train_loss"] = float(loss)
        met["train_time"] = time.time() - t0
        results[ds_name][model_name] = met
        print(f"  => AUROC={met['auroc']:.4f} SHD={met['shd']} loss={loss:.6f} time={met['train_time']:.0f}s")
        del m
        torch.cuda.empty_cache()

# Summary
print(f"\n{'='*60}")
print("CausalTime SUMMARY")
for ds_name in ["traffic", "medical", "pm25"]:
    if ds_name not in results:
        continue
    bl = results[ds_name]["baseline"]
    fx = results[ds_name]["F_ds8_fix"]
    delta = fx["auroc"] - bl["auroc"]
    print(f"\n{ds_name}:")
    print(f"  baseline AUROC={bl['auroc']:.4f} SHD={bl['shd']}")
    print(f"  F_ds8_fix AUROC={fx['auroc']:.4f} SHD={fx['shd']}")
    print(f"  Δ AUROC: {delta:+.4f} ({delta/(bl['auroc']+1e-8)*100:+.1f}%)")

with open("causaltime_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved to causaltime_results.json")
