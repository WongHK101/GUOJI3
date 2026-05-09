"""P2 S2: Side-Channel Mask/Shuffle — Prove concat depends on auxiliary channels.

Trains Concat JRNGC and ISTF on NSVAR d=10. After training, applies:
  - mask_x: zero out original input
  - shuffle_x: permute time dimension per variable

Prediction: if concat learned the shortcut, mask_x should have SMALLER impact
on concat than on ISTF (concat can fall back on condition c).

Uses model.compute_loss() to avoid manual preprocessing complexity.
"""
import torch
import torch.nn as nn
import numpy as np
import sys, os, json, time

torch.backends.cudnn.enabled = False

os.chdir("/root/autodl-tmp/GUOJI/mamba_enhanced")
sys.path.insert(0, "/root/autodl-tmp/GUOJI/JRNGC")
sys.path.insert(0, "/root/autodl-tmp/GUOJI/mamba_enhanced")

from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                MambaJRNGC, train_model, compute_metrics)

device = torch.device("cuda")
torch.manual_seed(0)
np.random.seed(0)

log_fh = open("mask_shuffle_results.log", "w", buffering=1)


def log(msg):
    log_fh.write(msg + "\n")
    log_fh.flush()
    print(msg, flush=True)


def main():
    log("=" * 60)
    log("S2: SIDE-CHANNEL MASK/SHUFFLE — Prove concat dependency")
    log("=" * 60)

    # Load NSVAR d=10 seed_0
    p = "/root/autodl-tmp/GUOJI/mamba_enhanced/data/nonstationary_var/num_nodes_10/true_lag_7/noise_scale_1/seed_0"
    x_orig = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))
    d, T = x_orig.shape
    log(f"  Data: d={d}, T={T}")

    # ---- Train Concat model ----
    log(f"\n  Training Concat JRNGC...")
    torch.manual_seed(0)
    np.random.seed(0)
    m_concat = MambaJRNGC(
        d=d, lag=7, layers=5, hidden=50,
        jacobian_lam=0.01, d_state=4, d_cond=4,
        use_time_weight_loss=False
    ).to(device)
    m_concat, _ = train_model(m_concat, x_orig, max_iter=2000, lr=1e-3, verbose=False)

    # ---- Train ISTF model (control) ----
    log(f"  Training ISTF JRNGC (control)...")
    torch.manual_seed(0)
    np.random.seed(0)
    m_istf = MambaFilterJRNGC(
        d=d, lag=7, layers=5, hidden=50,
        jacobian_lam=0.01, d_state=4, ortho_lam=0.05, residual_scale=0.1,
        filter_type="mamba"
    ).to(device)
    m_istf, _ = train_model(m_istf, x_orig, max_iter=2000, lr=1e-3, verbose=False)

    # ---- Baseline loss (clean input) ----
    log(f"\n  --- Baseline prediction loss (clean input) ---")
    m_concat.eval()
    m_istf.eval()
    base_concat = m_concat.compute_loss(x_orig)
    base_istf = m_istf.compute_loss(x_orig)
    log(f"    Concat baseline loss: {base_concat:.6f}")
    log(f"    ISTF   baseline loss: {base_istf:.6f}")

    # ---- Mask: zero out input ----
    log(f"\n  --- mask_x (zero input) ---")
    x_zero = np.zeros_like(x_orig).astype(np.float32)
    mask_concat = m_concat.compute_loss(x_zero)
    mask_istf = m_istf.compute_loss(x_zero)
    log(f"    Concat loss: {mask_concat:.6f}  (Δ = {mask_concat - base_concat:+.6f})")
    log(f"    ISTF   loss: {mask_istf:.6f}  (Δ = {mask_istf - base_istf:+.6f})")

    concat_mask_delta = mask_concat - base_concat
    istf_mask_delta = mask_istf - base_istf

    # ---- Shuffle: permute time ----
    log(f"\n  --- shuffle_x (permuted time per variable) ---")
    np.random.seed(42)
    x_shuffled = x_orig.copy()
    for i in range(d):
        perm = np.random.permutation(T)
        x_shuffled[i] = x_orig[i, perm]
    shuf_concat = m_concat.compute_loss(x_shuffled)
    shuf_istf = m_istf.compute_loss(x_shuffled)
    log(f"    Concat loss: {shuf_concat:.6f}  (Δ = {shuf_concat - base_concat:+.6f})")
    log(f"    ISTF   loss: {shuf_istf:.6f}  (Δ = {shuf_istf - base_istf:+.6f})")

    concat_shuf_delta = shuf_concat - base_concat
    istf_shuf_delta = shuf_istf - base_istf

    # ---- GC quality ----
    log(f"\n  --- GC quality ---")
    gc_concat = m_concat.get_gc_matrix(x_orig)
    gc_istf = m_istf.get_gc_matrix(x_orig)
    met_concat = compute_metrics(gc, gc_concat)
    met_istf = compute_metrics(gc, gc_istf)
    log(f"    Concat: AUROC={met_concat['auroc']:.4f}, SHD={met_concat['shd']}")
    log(f"    ISTF:   AUROC={met_istf['auroc']:.4f}, SHD={met_istf['shd']}")

    # ---- Interpretation ----
    log(f"\n{'='*60}")
    log("INTERPRETATION")
    log(f"{'='*60}")

    log(f"  Concat mask_x sensitivity:     {concat_mask_delta:+.6f}")
    log(f"  ISTF   mask_x sensitivity:     {istf_mask_delta:+.6f}")
    log(f"")

    if concat_mask_delta < istf_mask_delta:
        log(f"  ✓ SHORTCUT EVIDENCE: Concat LESS sensitive to x-masking than ISTF")
        log(f"    Ratio concat/ISTF mask sensitivity: {concat_mask_delta/istf_mask_delta:.3f}")
        log(f"    → Concat has auxiliary channel to fall back on when x is zeroed")
    else:
        log(f"  ~ Shortcut not confirmed via mask test")
        log(f"    (Both models similarly affected by zeroing input)")

    log(f"")
    log(f"  Concat shuffle_x sensitivity:   {concat_shuf_delta:+.6f}")
    log(f"  ISTF   shuffle_x sensitivity:   {istf_shuf_delta:+.6f}")

    log(f"")
    log(f"  GC quality comparison:")
    log(f"    Concat AUROC: {met_concat['auroc']:.4f} (lower = shortcut collapse)")
    log(f"    ISTF   AUROC: {met_istf['auroc']:.4f}")

    del m_concat, m_istf
    torch.cuda.empty_cache()

    with open("mask_shuffle_results.json", "w") as f:
        json.dump({
            "concat_baseline_loss": base_concat,
            "istf_baseline_loss": base_istf,
            "concat_mask_x_loss": mask_concat,
            "concat_mask_delta": concat_mask_delta,
            "istf_mask_x_loss": mask_istf,
            "istf_mask_delta": istf_mask_delta,
            "concat_shuffle_x_loss": shuf_concat,
            "concat_shuffle_delta": concat_shuf_delta,
            "istf_shuffle_x_loss": shuf_istf,
            "istf_shuffle_delta": istf_shuf_delta,
            "concat_gc_auroc": met_concat['auroc'],
            "istf_gc_auroc": met_istf['auroc'],
        }, f, indent=2)
    log(f"\nSaved to mask_shuffle_results.json")
    log_fh.close()


if __name__ == "__main__":
    main()
