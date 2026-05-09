"""Priority 1: validate Mamba filter on diverse datasets.

Per advisor: 2 datasets × (baseline + F_ds8_fixAB), 3 seeds each.
Using F_ds8 with variance fixes (residual_scale=0.1 + ortho_lam=0.01).
Pass criteria: no dataset shows AUROC degradation >5% vs baseline.
"""
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
JRNGC_DATA = resolve_data_dir()  # resolved via config

# ---- Dataset configurations ----
datasets = []

# 1. Non-stationary VAR d=50, lag=14 (high-dimensional)
for seed in [0, 1, 2]:
    datasets.append({
        "name": f"NSVAR_d50_seed{seed}",
        "data_path": f"data/nonstationary_var/num_nodes_50/true_lag_14/noise_scale_1/seed_{seed}",
        "d": 50, "lag": 14, "seed": seed,
        "max_iter": 3000,  # longer for d=50
    })

# 2. Lorenz-96 d=10, F=40 (nonlinear chaotic)
for seed in [0, 1, 2]:
    datasets.append({
        "name": f"Lorenz_F40_seed{seed}",
        "data_path": os.path.join(JRNGC_DATA, f"lorenz/num_nodes_10/F_40/seed_{seed}"),
        "d": 10, "lag": 1, "seed": seed,  # Lorenz gc has shape (d,d,1)
        "max_iter": 2000,
    })

results = {}

for ds in datasets:
    name = ds["name"]
    d = ds["d"]
    lag = ds["lag"]
    seed = ds["seed"]
    data_path = ds["data_path"]
    max_iter = ds["max_iter"]

    print(f"\n{'='*60}")
    print(f"{name}: d={d}, lag={lag}, seed={seed}")
    print(f"{'='*60}")

    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    print(f"  x shape: {x.shape}, gc shape: {gc.shape}")

    results[name] = {}

    for model_name, ModelClass, kwargs, lr in [
        ("baseline", BaselineJRNGC,
         {"d": d, "lag": lag, "layers": 5, "hidden": 50, "jacobian_lam": 0.01}, 1e-3),
        ("F_ds8_fixAB", MambaFilterJRNGC,
         {"d": d, "lag": lag, "layers": 5, "hidden": 50, "jacobian_lam": 0.01,
          **FILTER_KWARGS}, 1e-3),
    ]:
        print(f"  [{model_name}] training max_iter={max_iter}...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        m = ModelClass(**kwargs).to(device)
        t0 = time.time()
        m, loss = train_model(m, x, max_iter=max_iter, lr=lr, verbose=True)
        gc_pred = m.get_gc_matrix(x)
        met = compute_metrics(gc, gc_pred)
        met["train_loss"] = float(loss)
        met["train_time"] = time.time() - t0
        results[name][model_name] = met
        print(f"  => AUROC={met['auroc']:.4f} SHD={met['shd']} F1={met['f1']:.4f} loss={loss:.6f} time={met['train_time']:.0f}s")
        del m
        torch.cuda.empty_cache()

# ---- Summary ----
print(f"\n{'='*60}")
print("PRIORITY 1 SUMMARY")
print(f"{'='*60}")

for group_name, ds_names in [
    ("Non-stationary VAR d=50, lag=14", ["NSVAR_d50_seed0", "NSVAR_d50_seed1", "NSVAR_d50_seed2"]),
    ("Lorenz-96 d=10, F=40", ["Lorenz_F40_seed0", "Lorenz_F40_seed1", "Lorenz_F40_seed2"]),
]:
    print(f"\n--- {group_name} ---")
    for model_name in ["baseline", "F_ds8_fixAB"]:
        aurocs = [results[n][model_name]["auroc"] for n in ds_names]
        shds = [results[n][model_name]["shd"] for n in ds_names]
        losses = [results[n][model_name]["train_loss"] for n in ds_names]
        times = [results[n][model_name]["train_time"] for n in ds_names]
        print(f"  {model_name}:")
        print(f"    AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
        print(f"    SHD:   {np.mean(shds):.1f} ± {np.std(shds):.1f}")
        print(f"    Loss:  {np.mean(losses):.6f} ± {np.std(losses):.6f}")
        print(f"    Time:  {np.mean(times):.0f}s ± {np.std(times):.0f}s")
        print(f"    Seeds: {[f'{v:.4f}' for v in aurocs]}")

    bl_auroc = np.mean([results[n]["baseline"]["auroc"] for n in ds_names])
    f_auroc = np.mean([results[n]["F_ds8_fixAB"]["auroc"] for n in ds_names])
    delta = f_auroc - bl_auroc
    pct = (delta / (bl_auroc + 1e-8)) * 100
    status = "PASS" if delta > -0.05 * bl_auroc else "FAIL (>5% degradation)"
    print(f"  Δ AUROC: {delta:+.4f} ({pct:+.1f}%) — {status}")

with open("priority1_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved to priority1_results.json")
