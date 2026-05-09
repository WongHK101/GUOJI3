"""fMRI d=15 3-subject confirmation test."""
import torch, numpy as np, sys, os, time, json
sys.path.insert(0, "/root/autodl-tmp/GUOJI/JRNGC")
sys.path.insert(0, "/root/autodl-tmp/GUOJI/mamba_enhanced")
from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                train_model, compute_metrics)

device = torch.device("cuda")
FILTER_KWARGS = {"d_state": 8, "ortho_lam": 0.05, "residual_scale": 0.1}
JRNGC_DATA = "/root/autodl-tmp/GUOJI/JRNGC/data"
results = {}

for subject in [0, 1, 2]:
    data_path = os.path.join(JRNGC_DATA,
        f"fmri/num_nodes_15/subject_{subject}/seed_0")
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    print(f"\nSubject {subject}: x={x.shape}, gc={gc.shape}, edges={int(gc.sum())}")

    results[f"subj_{subject}"] = {}

    for name, ModelClass, kwargs in [
        ("baseline", BaselineJRNGC,
         {"d": 15, "lag": 1, "layers": 5, "hidden": 50, "jacobian_lam": 0.01}),
        ("F_ds8_fix", MambaFilterJRNGC,
         {"d": 15, "lag": 1, "layers": 5, "hidden": 50, "jacobian_lam": 0.01,
          **FILTER_KWARGS}),
    ]:
        print(f"  [{name}] training...")
        torch.manual_seed(0)
        np.random.seed(0)

        m = ModelClass(**kwargs).to(device)
        t0 = time.time()
        m, loss = train_model(m, x, max_iter=2000, lr=1e-3, verbose=True)
        gc_pred = m.get_gc_matrix(x)
        met = compute_metrics(gc, gc_pred)
        met["train_loss"] = float(loss)
        met["train_time"] = time.time() - t0
        results[f"subj_{subject}"][name] = met
        print(f"  => AUROC={met['auroc']:.4f} SHD={met['shd']} loss={loss:.6f} time={met['train_time']:.0f}s")
        del m
        torch.cuda.empty_cache()

# Summary
print(f"\n{'='*60}")
print("fMRI 3-subject SUMMARY")
for name in ["baseline", "F_ds8_fix"]:
    aurocs = [results[f"subj_{s}"][name]["auroc"] for s in range(3)]
    shds = [results[f"subj_{s}"][name]["shd"] for s in range(3)]
    print(f"\n{name}:")
    print(f"  AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  SHD:   {np.mean(shds):.1f} ± {np.std(shds):.1f}")
    print(f"  Seeds: {[f'{v:.4f}' for v in aurocs]}")

bl_mean = np.mean([results[f"subj_{s}"]["baseline"]["auroc"] for s in range(3)])
f_mean = np.mean([results[f"subj_{s}"]["F_ds8_fix"]["auroc"] for s in range(3)])
print(f"\nΔ AUROC: {f_mean-bl_mean:+.4f} ({(f_mean-bl_mean)/bl_mean*100:+.1f}%)")

with open("fmri_3subj_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved to fmri_3subj_results.json")
