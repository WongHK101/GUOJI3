"""P2: Controlled Three-Principle Ablation.

Tests whether ISTF's three design principles are individually necessary:
  1. Input-space confinement
  2. Near-identity initialization
  3. Orthogonality regularization

Controlled VAR(1) d=8 T=500, 5 seeds.

Output: risk_mitigation_results/istf_principle_ablation.{json,csv}
"""
import torch
import torch.nn as nn
import numpy as np
import sys, os, json, csv, time
from collections import defaultdict

torch.backends.cudnn.enabled = False

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, os.path.join(_PROJ_ROOT, "src"))

from src.minimal_mamba import MambaBlock, TCNBlock
from src.mamba_jrngc_pilot import (
    BaselineJRNGC, MambaJRNGC, MambaFilterJRNGC,
    train_model, compute_metrics
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(_PROJ_ROOT, "risk_mitigation_results")
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = 5
MAX_ITER = 2000
LR = 1e-3
LAMBDA = 0.01


# ============================================================
# Ablation variants
# ============================================================

class MambaFilterNoNearIdentity(MambaFilterJRNGC):
    """ISTF without near-identity init: filter output projections randomized.

    Near-identity relies on zero-initialized out_proj weights (both in
    MambaBlock and its inner SelectiveSSM). We reinitialize those with
    kaiming uniform to break the near-identity property while keeping
    other weights (in_proj, x_proj, dt_proj, norm, etc.) as-is.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import torch.nn.init as init
        # MambaBlock.out_proj is zero-initialized (near-identity)
        if hasattr(self.filter_mamba, 'out_proj'):
            init.kaiming_uniform_(self.filter_mamba.out_proj.weight, a=5**0.5)
        # SelectiveSSM.out_proj is also zero-initialized
        if hasattr(self.filter_mamba, 'ssm') and hasattr(self.filter_mamba.ssm, 'out_proj'):
            init.kaiming_uniform_(self.filter_mamba.ssm.out_proj.weight, a=5**0.5)


class MambaFilterNoOrtho(MambaFilterJRNGC):
    """ISTF without orthogonality regularization: ortho_lam=0 enforced."""
    def __init__(self, *args, **kwargs):
        kwargs["ortho_lam"] = 0.0
        super().__init__(*args, **kwargs)


class MambaFilterNoReg(MambaFilterJRNGC):
    """ISTF without any regularization: ortho_lam=0, near-identity init removed."""
    def __init__(self, *args, **kwargs):
        kwargs["ortho_lam"] = 0.0
        super().__init__(*args, **kwargs)
        import torch.nn.init as init
        if hasattr(self.filter_mamba, 'out_proj'):
            init.kaiming_uniform_(self.filter_mamba.out_proj.weight, a=5**0.5)
        if hasattr(self.filter_mamba, 'ssm') and hasattr(self.filter_mamba.ssm, 'out_proj'):
            init.kaiming_uniform_(self.filter_mamba.ssm.out_proj.weight, a=5**0.5)


def get_filter_drift(model, x):
    """Compute ||x' - x|| for ISTF filter drift. Returns None for non-filter models."""
    with torch.no_grad():
        if hasattr(model, 'make_filtered_windows'):
            # MambaFilterJRNGC path
            x_win, x_orig_t, x_filt_t = model.make_filtered_windows(x)
            drift = torch.norm(x_filt_t - x_orig_t, p=2).item()
            return drift
    return None


def generate_var1_data(d=8, T=500, noise_scale=0.1, sparsity=0.3, seed=42):
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


def coef_metrics(gc_pred, A_true, d):
    if gc_pred.ndim == 3:
        gc_pred = gc_pred[:, :, 0]
    recovered = gc_pred.copy()
    np.fill_diagonal(recovered, 0)
    true_vals = np.abs(A_true).copy()
    np.fill_diagonal(true_vals, 0)
    mask = ~np.eye(d, dtype=bool).flatten()
    r_flat = recovered.flatten()
    t_flat = true_vals.flatten()
    corr = np.corrcoef(r_flat[mask], t_flat[mask])[0, 1]
    shrinkage = np.linalg.norm(r_flat[mask]) / max(np.linalg.norm(t_flat[mask]), 1e-10)
    return float(corr), float(shrinkage)


def run_one(model, x, gc, A_true, seed, label, max_iter=MAX_ITER, lr=LR):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    log(f"    [{label}] training...")
    model, loss = train_model(model, x, max_iter=max_iter, lr=lr, verbose=False)
    log(f"    [{label}] train done, loss={loss:.6f}")
    gc_pred = model.get_gc_matrix(x)
    train_time = time.time() - t0
    m = compute_metrics(gc, gc_pred)
    d = x.shape[0]
    cr, cs = coef_metrics(gc_pred, A_true, d)
    try:
        drift = get_filter_drift(model, x)
    except Exception as e:
        log(f"    [{label}] WARNING: filter drift failed: {e}")
        drift = None

    log(f"    [{label}] AUROC={m['auroc']:.4f}, SHD={int(m['shd_topk'])}" +
        (f", drift={drift:.4f}" if drift is not None else ""))

    result = {
        "auroc": float(m["auroc"]), "auprc": float(m["auprc"]),
        "shd": int(m["shd_topk"]), "coefficient_correlation": cr,
        "coefficient_shrinkage": cs, "pred_loss": float(loss),
        "filter_drift": drift, "train_time": train_time,
    }
    del model
    torch.cuda.empty_cache()
    return result


def log(msg):
    print(msg, flush=True)


def main():
    log("=" * 70)
    log("P2: THREE-PRINCIPLE ABLATION")
    log("  VAR(1) d=8, T=500, 5 seeds, max_iter=2000")
    log("  Variants: Full ISTF, No near-identity, No ortho, No reg, Concat")
    log("=" * 70)

    d, T, lag, d_cond = 8, 500, 1, 4
    all_results = {}

    for seed in range(N_SEEDS):
        seed_key = f"seed_{seed}"
        all_results[seed_key] = {}

        x, gc, A_true = generate_var1_data(d=d, T=T, seed=seed * 100 + 42)
        n_edges = int(gc.sum())
        log(f"\n--- Seed {seed}: d={d}, T={T}, edges={n_edges} ---")

        # 1. JRNGC Baseline
        all_results[seed_key]["baseline"] = run_one(
            BaselineJRNGC(d=d, lag=lag, layers=3, hidden=32, jacobian_lam=LAMBDA).to(device),
            x, gc, A_true, seed, "Baseline"
        )

        # 2. Full ISTF
        all_results[seed_key]["full_istf"] = run_one(
            MambaFilterJRNGC(d=d, lag=lag, layers=3, hidden=32,
                             jacobian_lam=LAMBDA, d_state=4,
                             ortho_lam=0.05, residual_scale=0.1,
                             filter_type="mamba").to(device),
            x, gc, A_true, seed, "Full ISTF"
        )

        # 3. No near-identity init
        all_results[seed_key]["no_near_identity"] = run_one(
            MambaFilterNoNearIdentity(
                d=d, lag=lag, layers=3, hidden=32,
                jacobian_lam=LAMBDA, d_state=4,
                ortho_lam=0.05, residual_scale=0.1,
                filter_type="mamba").to(device),
            x, gc, A_true, seed, "No near-id init"
        )

        # 4. No orthogonality
        all_results[seed_key]["no_ortho"] = run_one(
            MambaFilterNoOrtho(
                d=d, lag=lag, layers=3, hidden=32,
                jacobian_lam=LAMBDA, d_state=4,
                ortho_lam=0.0, residual_scale=0.1,
                filter_type="mamba").to(device),
            x, gc, A_true, seed, "No ortho"
        )

        # 5. No regularization at all
        all_results[seed_key]["no_reg"] = run_one(
            MambaFilterNoReg(
                d=d, lag=lag, layers=3, hidden=32,
                jacobian_lam=LAMBDA, d_state=4,
                ortho_lam=0.0, residual_scale=0.1,
                filter_type="mamba").to(device),
            x, gc, A_true, seed, "No reg"
        )

        # 6. Concat side-channel (breaks input-space confinement)
        all_results[seed_key]["concat"] = run_one(
            MambaJRNGC(d=d, lag=lag, layers=3, hidden=32,
                       jacobian_lam=LAMBDA, d_state=4, d_cond=d_cond,
                       use_time_weight_loss=False).to(device),
            x, gc, A_true, seed, "Concat"
        )

    # Summary
    methods = ["baseline", "full_istf", "no_near_identity", "no_ortho", "no_reg", "concat"]
    labels = ["Baseline", "Full ISTF", "No near-id init", "No ortho", "No reg", "Concat"]

    summary = {}
    for method, label in zip(methods, labels):
        vals = defaultdict(list)
        for seed in range(N_SEEDS):
            for k, v in all_results[f"seed_{seed}"][method].items():
                if v is not None:
                    vals[k].append(v)
        summary[method] = {
            "label": label,
            "mean": {k: np.mean(v) for k, v in vals.items()},
            "std": {k: np.std(v, ddof=0) for k, v in vals.items()},
        }

    log(f"\n{'='*90}")
    log(f"{'Variant':<22} {'AUROC':>8} {'AUPRC':>8} {'SHD':>5} {'Coef r':>7} {'Shrink':>7} {'Drift':>8} {'PredLoss':>9}")
    log(f"{'-'*22} {'-'*8} {'-'*8} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*9}")

    for method, label in zip(methods, labels):
        s = summary[method]
        drift_str = f"{s['mean'].get('filter_drift', float('nan')):.4f}" if s['mean'].get('filter_drift') else "   --"
        log(f"  {label:<20} {s['mean']['auroc']:7.4f}±{s['std']['auroc']:.4f} "
            f"{s['mean']['auprc']:7.4f}±{s['std']['auprc']:.4f} "
            f"{s['mean']['shd']:4.1f}±{s['std']['shd']:.1f} "
            f"{s['mean']['coefficient_correlation']:6.4f} "
            f"{s['mean']['coefficient_shrinkage']:6.4f} "
            f"{drift_str}  "
            f"{s['mean']['pred_loss']:8.6f}")

    # Interpretation
    full_auroc = summary["full_istf"]["mean"]["auroc"]
    no_ni_auroc = summary["no_near_identity"]["mean"]["auroc"]
    no_ortho_auroc = summary["no_ortho"]["mean"]["auroc"]
    no_reg_auroc = summary["no_reg"]["mean"]["auroc"]

    log(f"\n{'='*70}")
    log("ABLATION INTERPRETATION")
    log(f"  Full ISTF AUROC:        {full_auroc:.4f}")
    log(f"  No near-identity:       {no_ni_auroc:.4f}  (Δ={no_ni_auroc - full_auroc:+.4f})")
    log(f"  No ortho:               {no_ortho_auroc:.4f}  (Δ={no_ortho_auroc - full_auroc:+.4f})")
    log(f"  No regularization:      {no_reg_auroc:.4f}  (Δ={no_reg_auroc - full_auroc:+.4f})")

    if no_ni_auroc < full_auroc - 0.01 and no_ortho_auroc < full_auroc - 0.01:
        log(f"  → Both principles matter: near-identity AND ortho contribute")
    elif no_ni_auroc < full_auroc - 0.01:
        log(f"  → Near-identity init is important; ortho less critical")
    elif no_ortho_auroc < full_auroc - 0.01:
        log(f"  → Ortho regularization is important; near-identity less critical")
    else:
        log(f"  → Neither individually large → principles are conservative design choices")

    # Save
    output = {
        "experiment": "istf_principle_ablation",
        "setting": "VAR(1) d=8 T=500, 5 seeds, max_iter=2000",
        "per_seed": {
            f"seed_{s}": {k: v for k, v in all_results[f"seed_{s}"].items()}
            for s in range(N_SEEDS)
        },
        "summary": {
            k: {
                "label": v["label"],
                "mean": {mk: float(mv) if isinstance(mv, (np.floating, np.integer))
                         else mv for mk, mv in v["mean"].items()},
                "std": {mk: float(sv) if isinstance(sv, (np.floating, np.integer))
                        else sv for mk, sv in v["std"].items()},
            }
            for k, v in summary.items()
        }
    }

    json_path = os.path.join(OUT_DIR, "istf_principle_ablation.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nSaved to {json_path}")

    csv_path = os.path.join(OUT_DIR, "istf_principle_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std",
                         "shd_mean", "shd_std", "coef_corr_mean", "coef_shrink_mean",
                         "filter_drift_mean", "pred_loss_mean"])
        for method in methods:
            s = summary[method]
            writer.writerow([
                s["label"],
                f"{s['mean']['auroc']:.4f}", f"{s['std']['auroc']:.4f}",
                f"{s['mean']['auprc']:.4f}", f"{s['std']['auprc']:.4f}",
                f"{s['mean']['shd']:.1f}", f"{s['std']['shd']:.1f}",
                f"{s['mean']['coefficient_correlation']:.4f}",
                f"{s['mean']['coefficient_shrinkage']:.4f}",
                f"{s['mean'].get('filter_drift', 'N/A')}",
                f"{s['mean']['pred_loss']:.6f}",
            ])
    log(f"Saved to {csv_path}")
    return output


if __name__ == "__main__":
    main()
