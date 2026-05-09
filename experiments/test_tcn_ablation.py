"""TCN filter ablation: baseline vs MambaFilter vs TCNFilter on key datasets."""
import torch, numpy as np, sys, os, time, json
torch.backends.cudnn.enabled = False  # fix Conv1d incompatibility on CUDA 13 + cuDNN 9.1
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
FILTER_KWARGS = {"d_state": 8, "ortho_lam": 0.05, "residual_scale": 0.1}
JRNGC_DATA = resolve_data_dir()  # resolved via config
results = {}

# ====== 1. NSVAR d=10 P=7 (5 seeds) — Mamba's best synthetic performance ======
print("=" * 60)
print("TCN ABLATION: NSVAR d=10 P=7 (5 seeds)")
nsvar_aurocs = {}
for seed in range(5):
    data_path = os.path.join(_PROJ_ROOT, "data", "nonstationary_var", "num_nodes_10", "true_lag_7", "noise_scale_1", "seed_{seed}")
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    d = x.shape[0]
    if seed == 0:
        print(f"  d={d}, T={x.shape[1]}, edges={int(gc.sum())}")

    key = f"nsvar_seed{seed}"
    results[key] = {}

    for model_name, ModelClass, kwargs in [
        ("baseline", BaselineJRNGC,
         {"d": d, "lag": 7, "layers": 5, "hidden": 50, "jacobian_lam": 0.01}),
        ("F_ds8_fix", MambaFilterJRNGC,
         {"d": d, "lag": 7, "layers": 5, "hidden": 50, "jacobian_lam": 0.01,
          "filter_type": "mamba", **FILTER_KWARGS}),
        ("F_tcn", MambaFilterJRNGC,
         {"d": d, "lag": 7, "layers": 5, "hidden": 50, "jacobian_lam": 0.01,
          "filter_type": "tcn", **FILTER_KWARGS}),
    ]:
        print(f"  seed {seed} [{model_name}] training...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        m = ModelClass(**kwargs).to(device)
        t0 = time.time()
        m, loss = train_model(m, x, max_iter=2000, lr=1e-3, verbose=False)
        gc_pred = m.get_gc_matrix(x)
        met = compute_metrics(gc, gc_pred)
        met["train_loss"] = float(loss)
        met["train_time"] = time.time() - t0
        results[key][model_name] = met
        print(f"    AUROC={met['auroc']:.4f} SHD={met['shd']} loss={loss:.6f} time={met['train_time']:.0f}s")
        if model_name not in nsvar_aurocs:
            nsvar_aurocs[model_name] = []
        nsvar_aurocs[model_name].append(met['auroc'])
        del m
        torch.cuda.empty_cache()

print("\nNSVAR d=10 summary:")
for name in ["baseline", "F_ds8_fix", "F_tcn"]:
    a = nsvar_aurocs[name]
    print(f"  {name}: AUROC={np.mean(a):.4f}±{np.std(a):.4f}")

# ====== 2. DREAM3 d=50 (3 subjects) — Mamba's best SHD reduction ======
print("\n" + "=" * 60)
print("TCN ABLATION: DREAM3 d=50 (3 subjects)")
os.chdir(resolve_jrngc_path() or ".")  # resolved via config
from tgc.data.dream3 import dream3_trajectories

dream3_aurocs = {}
for subject in range(3):
    x, _, gc = dream3_trajectories(d=50, subject=subject)
    if x.ndim == 3:
        x = x[0]  # use first trajectory
    d = x.shape[0]
    print(f"  subject {subject}: d={d}, T={x.shape[1]}, edges={int(gc.sum())}")

    key = f"dream3_d50_s{subject}"
    results[key] = {}

    for model_name, ModelClass, kwargs in [
        ("baseline", BaselineJRNGC,
         {"d": d, "lag": 1, "layers": 5, "hidden": 50, "jacobian_lam": 0.01}),
        ("F_ds8_fix", MambaFilterJRNGC,
         {"d": d, "lag": 1, "layers": 5, "hidden": 50, "jacobian_lam": 0.01,
          "filter_type": "mamba", **FILTER_KWARGS}),
        ("F_tcn", MambaFilterJRNGC,
         {"d": d, "lag": 1, "layers": 5, "hidden": 50, "jacobian_lam": 0.01,
          "filter_type": "tcn", **FILTER_KWARGS}),
    ]:
        print(f"  [{model_name}] training...")
        torch.manual_seed(0)
        np.random.seed(0)

        m = ModelClass(**kwargs).to(device)
        t0 = time.time()
        m, loss = train_model(m, x, max_iter=2000, lr=1e-3, verbose=False)
        gc_pred = m.get_gc_matrix(x)
        met = compute_metrics(gc, gc_pred)
        met["train_loss"] = float(loss)
        met["train_time"] = time.time() - t0
        results[key][model_name] = met
        print(f"    AUROC={met['auroc']:.4f} SHD={met['shd']} loss={loss:.6f} time={met['train_time']:.0f}s")
        if model_name not in dream3_aurocs:
            dream3_aurocs[model_name] = []
        dream3_aurocs[model_name].append(met['auroc'])
        del m
        torch.cuda.empty_cache()

print("\nDREAM3 d=50 summary:")
for name in ["baseline", "F_ds8_fix", "F_tcn"]:
    a = dream3_aurocs[name]
    print(f"  {name}: AUROC={np.mean(a):.4f}±{np.std(a):.4f}")

# ====== 3. VAR d=50 stationary (3 seeds) — neutral scenario ======
print("\n" + "=" * 60)
print("TCN ABLATION: VAR d=50 stationary (3 seeds)")
var50_aurocs = {}
for seed in range(3):
    data_path = os.path.join(JRNGC_DATA, "var", "num_nodes_50", "true_lag_5",
                             "noise_scale_1", f"seed_{seed}")
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    d = x.shape[0]
    if seed == 0:
        print(f"  d={d}, T={x.shape[1]}, edges={int(gc.sum())}")

    key = f"var50_seed{seed}"
    results[key] = {}

    for model_name, ModelClass, kwargs in [
        ("baseline", BaselineJRNGC,
         {"d": d, "lag": 5, "layers": 5, "hidden": 50, "jacobian_lam": 0.01}),
        ("F_ds8_fix", MambaFilterJRNGC,
         {"d": d, "lag": 5, "layers": 5, "hidden": 50, "jacobian_lam": 0.01,
          "filter_type": "mamba", **FILTER_KWARGS}),
        ("F_tcn", MambaFilterJRNGC,
         {"d": d, "lag": 5, "layers": 5, "hidden": 50, "jacobian_lam": 0.01,
          "filter_type": "tcn", **FILTER_KWARGS}),
    ]:
        print(f"  seed {seed} [{model_name}] training...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        m = ModelClass(**kwargs).to(device)
        t0 = time.time()
        m, loss = train_model(m, x, max_iter=2000, lr=1e-3, verbose=False)
        gc_pred = m.get_gc_matrix(x)
        met = compute_metrics(gc, gc_pred)
        met["train_loss"] = float(loss)
        met["train_time"] = time.time() - t0
        results[key][model_name] = met
        print(f"    AUROC={met['auroc']:.4f} SHD={met['shd']} loss={loss:.6f} time={met['train_time']:.0f}s")
        if model_name not in var50_aurocs:
            var50_aurocs[model_name] = []
        var50_aurocs[model_name].append(met['auroc'])
        del m
        torch.cuda.empty_cache()

print("\nVAR d=50 stationary summary:")
for name in ["baseline", "F_ds8_fix", "F_tcn"]:
    a = var50_aurocs[name]
    print(f"  {name}: AUROC={np.mean(a):.4f}±{np.std(a):.4f}")

# ====== Final summary ======
print(f"\n{'='*60}")
print("TCN ABLATION FINAL SUMMARY")
print(f"{'='*60}")
for dataset, d in [("NSVAR d=10 P=7", nsvar_aurocs),
                    ("DREAM3 d=50", dream3_aurocs),
                    ("VAR d=50 stat", var50_aurocs)]:
    print(f"\n{dataset}:")
    for name in ["baseline", "F_ds8_fix", "F_tcn"]:
        if name in d:
            a = d[name]
            print(f"  {name:<15} AUROC={np.mean(a):.4f}±{np.std(a):.4f}")

with open("tcn_ablation_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved to tcn_ablation_results.json")
