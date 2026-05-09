"""DREAM3 test: baseline vs F_ds8_fix on d=10/50/100, subjects 0-2.

Uses JRNGC's dream3_trajectories data loader directly.
"""
import torch, numpy as np, sys, os, time, json
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

# Import DREAM3 loader (requires working dir = JRNGC root for relative data paths)
os.chdir(resolve_jrngc_path() or ".")  # resolved via config
from tgc.data.dream3 import dream3_trajectories

results = {}

for d in [10, 50, 100]:
    for subject in range(min(3, 5)):  # 3 subjects for d=10, subject range may vary
        try:
            x, _, gc = dream3_trajectories(d=d, subject=subject)
        except Exception as e:
            print(f"  DREAM3 d={d} subj={subject}: SKIP ({e})")
            continue

        name = f"d{d}_s{subject}"
        print(f"\n{'='*50}")
        print(f"DREAM3 {name}: x={x.shape}, gc={gc.shape}, edges={int(gc.sum())}")
        print(f"{'='*50}")

        # x is (d, T_total), gc is (d, d, 1), lag=1
        results[name] = {}

        for model_name, ModelClass, kwargs in [
            ("baseline", BaselineJRNGC,
             {"d": d, "lag": 1, "layers": 5, "hidden": 50, "jacobian_lam": 0.01}),
            ("F_ds8_fix", MambaFilterJRNGC,
             {"d": d, "lag": 1, "layers": 5, "hidden": 50, "jacobian_lam": 0.01,
              **FILTER_KWARGS}),
        ]:
            print(f"  [{model_name}] training...")
            torch.manual_seed(0)
            np.random.seed(0)

            m = ModelClass(**kwargs).to(device)
            t0 = time.time()
            m, loss = train_model(m, x, max_iter=2000, lr=1e-3, verbose=True)
            gc_pred = m.get_gc_matrix(x)
            met = compute_metrics(gc, gc_pred)
            met["train_loss"] = float(loss)
            met["train_time"] = time.time() - t0
            results[name][model_name] = met
            print(f"  => AUROC={met['auroc']:.4f} SHD={met['shd']} loss={loss:.6f} time={met['train_time']:.0f}s")
            del m
            torch.cuda.empty_cache()

# Summary by dimension
print(f"\n{'='*60}")
print("DREAM3 SUMMARY")
for d in [10, 50, 100]:
    d_names = [n for n in results if n.startswith(f"d{d}_")]
    if not d_names:
        continue
    print(f"\n--- d={d} ({len(d_names)} subjects) ---")
    for mn in ["baseline", "F_ds8_fix"]:
        try:
            aurocs = [results[n][mn]["auroc"] for n in d_names]
            shds = [results[n][mn]["shd"] for n in d_names]
            print(f"  {mn}: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}, SHD={np.mean(shds):.1f}")
        except:
            pass

with open("dream3_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved to dream3_results.json")
