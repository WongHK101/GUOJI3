"""Post-hoc Jacobian diagnostic for Concat x-only (P0 supplement).

Trains MambaJRNGC (Concat x-only) and records |J_x|, |J_c| by
computing the full (d+d_cond) Jacobian post-training.
Matches P0 config exactly: VAR(1) d=8 T=500, 5 seeds, max_iter=2000.
"""
import torch
import numpy as np
import sys, os, json, time

torch.backends.cudnn.enabled = False

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, os.path.join(_PROJ_ROOT, "src"))

from src.minimal_mamba import MambaBlock
from src.mamba_jrngc_pilot import (
    BaselineJRNGC, MambaJRNGC,
    train_model, compute_metrics
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(_PROJ_ROOT, "risk_mitigation_results")
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = 5
MAX_ITER = 2000
LR = 1e-3
LAMBDA = 0.01
D = 8
T = 500
LAG = 1
D_COND = 4
D_STATE = 4


def log(msg):
    print(msg, flush=True)


def generate_var1_data(d=D, T=T, noise_scale=0.1, sparsity=0.3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed + 1)
    A = torch.randn(d, d) * (torch.rand(d, d) < sparsity).float()
    sr = torch.linalg.eigvals(A).abs().max()
    A = A * (0.8 / max(float(sr), 0.01))
    A_np = A.numpy()
    x = np.zeros((d, T))
    x[:, 0] = np.random.randn(d) * 0.1
    for t in range(1, T):
        x[:, t] = A_np @ x[:, t - 1] + np.random.randn(d) * noise_scale
    gc = (np.abs(A_np) > 0.01).astype(np.float64)
    return x, gc, A_np


def compute_posthoc_jacobian(model, x):
    """Post-training one-pass: compute |J_x| and |J_c| on full concat input."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        xz_window, _ = model.preprocess_and_windowing(x_tensor)  # returns (windows, t_weights)
    xz_full = xz_window[:, :, :model.lag]  # (M, d+d_cond, lag)
    xz_combined = xz_full.detach().clone().requires_grad_(True)
    x_cat = xz_combined.flatten(start_dim=1)  # (M, (d+d_cond)*lag)

    # Forward through the SIRD network (same as compute_jacobian_loss but on full input)
    if hasattr(model, 'sird'):
        x_cat_in = x_cat
        for layer in model.sird:
            x_cat_in = layer(x_cat_in)
        y = model.predictor(x_cat_in)
    else:
        y = model(x_cat)

    d_out = y.shape[1]
    jac = torch.zeros((xz_combined.shape[0], d_out,
                       xz_combined.shape[1], xz_combined.shape[2]),
                      device=xz_combined.device)
    for j in range(d_out):
        grad = torch.autograd.grad(y[:, j], xz_combined,
                                   grad_outputs=torch.ones_like(y[:, j]),
                                   retain_graph=True)[0]
        jac[:, j] = grad

    jac_abs = torch.abs(jac)
    jac_x = jac_abs[:, :, :model.d, :]
    jac_c = jac_abs[:, :, model.d:, :]

    return float(torch.mean(jac_x).detach()), float(torch.mean(jac_c).detach())


def main():
    log("=" * 70)
    log("POST-HOC JACOBIAN: Concat x-only |J_x| and |J_c|")
    log(f"  VAR(1) d={D}, T={T}, {N_SEEDS} seeds, max_iter={MAX_ITER}")
    log("=" * 70)

    results = {}
    for seed in range(N_SEEDS):
        seed_key = f"seed_{seed}"
        x, gc, A_true = generate_var1_data(d=D, T=T, seed=seed * 100 + 42)
        n_edges = int(gc.sum())
        log(f"\n--- Seed {seed}: d={D}, T={T}, edges={n_edges} ---")

        model = MambaJRNGC(
            d=D, lag=LAG, layers=3, hidden=32,
            jacobian_lam=LAMBDA, d_state=D_STATE, d_cond=D_COND,
            use_time_weight_loss=False
        ).to(device)

        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        model, loss = train_model(model, x, max_iter=MAX_ITER, lr=LR, verbose=False)
        train_time = time.time() - t0
        log(f"  train done, loss={loss:.6f}, time={train_time:.1f}s")

        gc_pred = model.get_gc_matrix(x)
        m = compute_metrics(gc, gc_pred)
        log(f"  AUROC={m['auroc']:.4f}, SHD={int(m['shd_topk'])}")

        jx, jc = compute_posthoc_jacobian(model, x)
        log(f"  |J_x|={jx:.6f}, |J_c|={jc:.6f}, ratio |J_c|/|J_x|={jc/jx:.4f}")

        results[seed_key] = {
            "auroc": float(m["auroc"]),
            "shd": int(m["shd_topk"]),
            "jx_norm": jx,
            "jc_norm": jc,
            "jc_over_jx_ratio": jc / jx if jx > 0 else None,
            "pred_loss": float(loss),
            "train_time": train_time,
        }
        del model
        torch.cuda.empty_cache()

    # Summary
    jx_vals = [r["jx_norm"] for r in results.values()]
    jc_vals = [r["jc_norm"] for r in results.values()]
    ratio_vals = [r["jc_over_jx_ratio"] for r in results.values() if r["jc_over_jx_ratio"] is not None]

    log(f"\n{'='*70}")
    log("POST-HOC SUMMARY")
    log(f"  |J_x| = {np.mean(jx_vals):.6f} ± {np.std(jx_vals, ddof=0):.6f}")
    log(f"  |J_c| = {np.mean(jc_vals):.6f} ± {np.std(jc_vals, ddof=0):.6f}")
    log(f"  |J_c|/|J_x| = {np.mean(ratio_vals):.4f} ± {np.std(ratio_vals, ddof=0):.4f}")

    if np.mean(jc_vals) > np.mean(jx_vals) * 0.5:
        log("  → c-channel sensitivity is substantial relative to x-channel")
        log("    Confirms: without c-penalty, model routes through auxiliary channel")
    else:
        log("  → c-channel sensitivity is low relative to x-channel")

    output = {
        "experiment": "concat_posthoc_jacobian",
        "description": "Post-hoc |J_x| and |J_c| for Concat x-only (trained without c-penalty)",
        "per_seed": results,
        "summary": {
            "jx_norm_mean": float(np.mean(jx_vals)),
            "jx_norm_std": float(np.std(jx_vals, ddof=0)),
            "jc_norm_mean": float(np.mean(jc_vals)),
            "jc_norm_std": float(np.std(jc_vals, ddof=0)),
            "jc_over_jx_mean": float(np.mean(ratio_vals)),
            "jc_over_jx_std": float(np.std(ratio_vals, ddof=0)),
        }
    }

    json_path = os.path.join(OUT_DIR, "concat_posthoc_jacobian.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {json_path}")


if __name__ == "__main__":
    main()
