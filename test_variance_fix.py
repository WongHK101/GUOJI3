"""Diagnose and fix variance issue: seeds 0, 3 underperforming for F_ds8.

Fixes applied:
  A: Residual init — MambaBlock.out_proj zero-init + residual_scale=0.1
  B: Orthogonality reg — penalizes ||x_filt - x_orig||^2
"""
import torch, numpy as np, sys, os, time
sys.path.insert(0, "/root/autodl-tmp/GUOJI/JRNGC")
sys.path.insert(0, ".")
from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                train_model, compute_metrics)

device = torch.device("cuda")
seeds = [0, 3]  # The two problematic seeds

for seed in seeds:
    data_path = f"data/nonstationary_var/num_nodes_10/true_lag_7/noise_scale_1/seed_{seed}"
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))

    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")

    for name, ModelClass, kwargs, lr in [
        # Reference
        ("A_baseline", BaselineJRNGC,
         {"d":10, "lag":7, "layers":5, "hidden":50, "jacobian_lam":0.01}, 1e-3),
        # Old F_ds8 (no fixes)
        ("F_ds8_OLD", MambaFilterJRNGC,
         {"d":10, "lag":7, "layers":5, "hidden":50, "jacobian_lam":0.01,
          "d_state":8, "ortho_lam":0.0, "residual_scale":1.0}, 1e-3),
        # Fix A only: residual_scale=0.1 (zero-init already in MambaBlock)
        ("F_ds8_fixA", MambaFilterJRNGC,
         {"d":10, "lag":7, "layers":5, "hidden":50, "jacobian_lam":0.01,
          "d_state":8, "ortho_lam":0.0, "residual_scale":0.1}, 1e-3),
        # Fix A+B: residual_scale=0.1 + ortho_lam=0.01
        ("F_ds8_fixAB", MambaFilterJRNGC,
         {"d":10, "lag":7, "layers":5, "hidden":50, "jacobian_lam":0.01,
          "d_state":8, "ortho_lam":0.01, "residual_scale":0.1}, 1e-3),
        # Fix A+B stronger: residual_scale=0.1 + ortho_lam=0.05
        ("F_ds8_fixAB05", MambaFilterJRNGC,
         {"d":10, "lag":7, "layers":5, "hidden":50, "jacobian_lam":0.01,
          "d_state":8, "ortho_lam":0.05, "residual_scale":0.1}, 1e-3),
    ]:
        print(f"\n  --- {name} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        m = ModelClass(**kwargs).to(device)
        t0 = time.time()
        m, loss = train_model(m, x, max_iter=2000, lr=lr, verbose=True)
        gc_pred = m.get_gc_matrix(x)
        met = compute_metrics(gc, gc_pred)
        met["train_loss"] = float(loss)
        met["train_time"] = time.time() - t0
        print(f"  => AUROC={met['auroc']:.4f} SHD={met['shd']} F1={met['f1']:.4f} loss={loss:.6f} time={met['train_time']:.0f}s")
        del m
        torch.cuda.empty_cache()

print("\nDone. Compare F_ds8_OLD vs F_ds8_fixAB — seed 0 and 3 should improve.")
