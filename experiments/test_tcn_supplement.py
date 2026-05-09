"""TCN ablation supplement: DREAM3 d=10/100 + CausalTime medical/pm25."""
import torch, numpy as np, sys, os, time, json
torch.backends.cudnn.enabled = False
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

# ====== DREAM3 d=10 + d=100 (3 subjects each) ======
os.chdir(resolve_jrngc_path() or ".")  # resolved via config
from tgc.data.dream3 import dream3_trajectories

for d in [10, 100]:
    print(f"\n{'='*60}")
    print(f"TCN SUPP: DREAM3 d={d}")
    for subject in range(3):
        try:
            x, _, gc = dream3_trajectories(d=d, subject=subject)
        except Exception as e:
            print(f"  SKIP ({e})")
            continue
        if x.ndim == 3:
            x = x[0]
        print(f"  subject {subject}: d={x.shape[0]}, T={x.shape[1]}, edges={int(gc.sum())}")

        key = f"dream3_d{d}_s{subject}"
        results[key] = {}

        for model_name, ModelClass, kwargs in [
            ("baseline", BaselineJRNGC,
             {"d": d, "lag": 1, "layers": 5, "hidden": 50, "jacobian_lam": 0.01}),
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
            del m
            torch.cuda.empty_cache()

# ====== CausalTime medical + pm25 ======
for ds_name, max_iter in [("medical", 2000), ("pm25", 1000)]:
    print(f"\n{'='*60}")
    print(f"TCN SUPP: CausalTime {ds_name}")
    data_path = os.path.join(JRNGC_DATA, f"causaltime/{ds_name}")
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    d = x.shape[0]
    print(f"  d={d}, T={x.shape[1]}, edges={int(gc.sum())}")

    key = f"ct_{ds_name}"
    results[key] = {}

    for model_name, ModelClass, kwargs in [
        ("baseline", BaselineJRNGC,
         {"d": d, "lag": 1, "layers": 5, "hidden": 50, "jacobian_lam": 0.01}),
        ("F_tcn", MambaFilterJRNGC,
         {"d": d, "lag": 1, "layers": 5, "hidden": 50, "jacobian_lam": 0.01,
          "filter_type": "tcn", **FILTER_KWARGS}),
    ]:
        print(f"  [{model_name}] training max_iter={max_iter}...")
        torch.manual_seed(0)
        np.random.seed(0)
        m = ModelClass(**kwargs).to(device)
        t0 = time.time()
        m, loss = train_model(m, x, max_iter=max_iter, lr=1e-3, verbose=False)
        gc_pred = m.get_gc_matrix(x)
        met = compute_metrics(gc, gc_pred)
        met["train_loss"] = float(loss)
        met["train_time"] = time.time() - t0
        results[key][model_name] = met
        print(f"    AUROC={met['auroc']:.4f} SHD={met['shd']} loss={loss:.6f} time={met['train_time']:.0f}s")
        del m
        torch.cuda.empty_cache()

# ====== Summary ======
print(f"\n{'='*60}")
print("TCN SUPPLEMENT SUMMARY")
print(f"{'='*60}")
for group_name, keys, existing_bl, existing_mb in [
    ("DREAM3 d=10", ["dream3_d10_s0", "dream3_d10_s1", "dream3_d10_s2"],
     0.5113, 0.5442),
    ("DREAM3 d=100", ["dream3_d100_s0", "dream3_d100_s1", "dream3_d100_s2"],
     0.5305, 0.5233),
    ("CT medical", ["ct_medical"], 0.4766, 0.5596),
    ("CT pm25", ["ct_pm25"], 0.4288, 0.4668),
]:
    bl_vals = [results[k]["baseline"]["auroc"] for k in keys if k in results]
    tc_vals = [results[k]["F_tcn"]["auroc"] for k in keys if k in results]
    bl_mean = np.mean(bl_vals) if bl_vals else 0
    tc_mean = np.mean(tc_vals) if tc_vals else 0
    print(f"\n{group_name}:")
    print(f"  baseline (prev): {existing_bl:.4f}  baseline (now): {bl_mean:.4f}")
    print(f"  Mamba (prev):    {existing_mb:.4f}")
    print(f"  TCN   (now):     {tc_mean:.4f}")
    print(f"  Δ TCN-baseline:  {tc_mean-bl_mean:+.4f}  Δ Mamba-baseline: {existing_mb-existing_bl:+.4f}")

with open("tcn_supplement_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to tcn_supplement_results.json")
