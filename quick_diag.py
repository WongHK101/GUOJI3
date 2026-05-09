"""Quick diagnostic: test Mamba-as-input-filter architecture.

Mamba transforms x → x' within SAME d-dim space (no new channels).
Jacobian ∂y/∂x' is directly the GC score.
"""
import torch, numpy as np, sys, os, time
sys.path.insert(0, "/root/autodl-tmp/GUOJI/JRNGC")
sys.path.insert(0, ".")
from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                train_model, compute_metrics)

x = np.load("data/nonstationary_var/num_nodes_10/true_lag_7/noise_scale_1/seed_0/_x.npy")
gc = np.load("data/nonstationary_var/num_nodes_10/true_lag_7/noise_scale_1/seed_0/_gc.npy")
device = torch.device("cuda")

for name, lr, d_state, jac_lam in [
    ("A_baseline",      1e-3, None, 0.01),
    ("F_default",       1e-3, 4,    0.01),
    ("F_small_state",   1e-3, 2,    0.01),
    ("F_large_state",   1e-3, 8,    0.01),
    ("F_jac05",         1e-3, 4,    0.05),
    ("F_lowlr",         5e-4, 4,    0.01),
    ("F_lowlr_jac05",   5e-4, 4,    0.05),
]:
    print(f"\n=== {name}: lr={lr}, d_state={d_state}, jac_lam={jac_lam} ===")
    torch.manual_seed(0)
    np.random.seed(0)

    if d_state is None:
        m = BaselineJRNGC(d=10, lag=7, layers=5, hidden=50, jacobian_lam=0.01).to(device)
    else:
        m = MambaFilterJRNGC(d=10, lag=7, layers=5, hidden=50,
                             jacobian_lam=jac_lam, d_state=d_state).to(device)

    t0 = time.time()
    m, loss = train_model(m, x, max_iter=2000, lr=lr, verbose=True)
    gc_pred = m.get_gc_matrix(x)
    met = compute_metrics(gc, gc_pred)
    print(f"  => AUROC={met['auroc']:.4f} SHD={met['shd']} F1={met['f1']:.4f} loss={loss:.6f} time={time.time()-t0:.0f}s")
    del m
    torch.cuda.empty_cache()
