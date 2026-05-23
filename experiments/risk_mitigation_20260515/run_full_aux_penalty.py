"""P0: Full Auxiliary-Jacobian Penalty Baseline.

Fixes the MambaConcatFullPenaltyJRNGC bug (jacobian_penalty was dead code,
training called inherited compute_jacobian_loss which only penalizes x_orig).

Runs controlled diagnostic on VAR(1) d=8 T=500, 5 seeds.
Compares: baseline, concat x-only, full penalty (same-lambda, budget-normalized,
lambda_c/lambda_x in {0.1, 1, 10}), ISTF-Mamba.

Output: risk_mitigation_results/full_aux_jacobian_penalty.{json,csv}
"""
import torch
import torch.nn as nn
import numpy as np
import sys, os, json, csv, time
from collections import defaultdict

torch.backends.cudnn.enabled = False

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, os.path.join(_PROJ_ROOT, "src"))  # for minimal_mamba
from src.config import resolve_jrngc_path
_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)

from src.minimal_mamba import MambaBlock
from src.mamba_jrngc_pilot import (
    BaselineJRNGC, MambaJRNGC, MambaFilterJRNGC,
    train_model, compute_metrics
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(_PROJ_ROOT, "risk_mitigation_results")
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = 5  # restored to 5 seeds for final run
DEFAULT_LAMBDA = 0.01


# ============================================================
# Fixed MambaConcatFullPenaltyJRNGC
# ============================================================
class MambaConcatFullPenaltyJRNGC(MambaJRNGC):
    """Concat JRNGC with Jacobian penalty on ALL input dimensions [x, c].

    Overrides compute_jacobian_loss to apply autograd to the full (d+d_cond)
    input, closing the auxiliary-channel loophole.

    Supports separate lambda_x and lambda_c via lam_x and lam_c attributes.
    """
    def __init__(self, *args, lam_x=None, lam_c=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam_x = lam_x if lam_x is not None else self.jacobian_lam
        self.lam_c = lam_c if lam_c is not None else self.jacobian_lam

    def compute_jacobian_loss(self, xz_window):
        """Jacobian on FULL concatenated input (d+d_cond dims).

        Splits input into x-portion and c-portion, computes separate Jacobian
        norms, and weights by lam_x / lam_c respectively.
        Also stores |J_x| and |J_c| on self for post-training retrieval.
        """
        xz_full = xz_window[:, :, :self.lag]  # (M, d+d_cond, lag)
        xz_combined = xz_full.detach().clone().requires_grad_(True)

        x_cat = xz_combined.flatten(start_dim=1)  # (M, (d+d_cond)*lag)
        y = self(x_cat)

        jac = torch.zeros((xz_combined.shape[0], y.shape[1],
                           xz_combined.shape[1], xz_combined.shape[2]),
                          device=xz_combined.device)
        for j in range(y.shape[1]):
            grad = torch.autograd.grad(y[:, j], xz_combined,
                                       grad_outputs=torch.ones_like(y[:, j]),
                                       create_graph=True)[0]
            jac[:, j] = grad

        jac_abs = torch.abs(jac)  # (M, d_out, d+d_cond, lag)

        jac_x = jac_abs[:, :, :self.d, :]      # (M, d_out, d, lag)
        jac_c = jac_abs[:, :, self.d:, :]      # (M, d_out, d_cond, lag)

        # Store for post-training retrieval (detached, no graph)
        self._last_jx_norm = float(torch.mean(jac_x).detach())
        self._last_jc_norm = float(torch.mean(jac_c).detach())

        penalty = self.lam_x * torch.mean(jac_x) + self.lam_c * torch.mean(jac_c)
        return penalty

    def get_jacobian_norms(self, x_full):
        """Return |J_x| and |J_c| stored during last compute_jacobian_loss call."""
        return (getattr(self, '_last_jx_norm', None),
                getattr(self, '_last_jc_norm', None))


# ============================================================
# Data: VAR(1) with known A (reuse existing generator pattern)
# ============================================================
def generate_var1_data(d=8, T=500, noise_scale=0.1, sparsity=0.3, seed=42):
    """VAR(1): x_t = A x_{t-1} + eps_t, known sparse A."""
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


# ============================================================
# Coefficient-level metrics
# ============================================================
def compute_coef_metrics(gc_pred, A_true, d):
    """Coefficient correlation and shrinkage on off-diagonal entries."""
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


# ============================================================
# Run one configuration
# ============================================================
def run_one(model, x, gc_true, A_true, seed, label, max_iter=2000, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    log(f"    [{label}] training...")
    model, loss = train_model(model, x, max_iter=max_iter, lr=lr, verbose=False)
    log(f"    [{label}] train done, loss={loss:.6f}, getting GC matrix...")
    gc_pred = model.get_gc_matrix(x)
    train_time = time.time() - t0
    log(f"    [{label}] GC matrix done, computing metrics...")

    metrics = compute_metrics(gc_true, gc_pred)
    d = x.shape[0]
    coef_corr, coef_shrink = compute_coef_metrics(gc_pred, A_true, d)

    # Get Jacobian norms for full-penalty models
    jx_norm, jc_norm = None, None
    if hasattr(model, 'get_jacobian_norms'):
        jx_norm, jc_norm = model.get_jacobian_norms(x)
        log(f"    [{label}] |Jx|={jx_norm:.6f}, |Jc|={jc_norm:.6f}")

    result = {
        "auroc": float(metrics["auroc"]),
        "auprc": float(metrics["auprc"]),
        "shd": int(metrics["shd_topk"]),
        "coefficient_correlation": coef_corr,
        "coefficient_shrinkage": coef_shrink,
        "pred_loss": float(loss),
        "jx_norm": jx_norm,
        "jc_norm": jc_norm,
        "train_time": train_time,
    }
    log(f"    [{label}] AUROC={metrics['auroc']:.4f}, SHD={int(metrics['shd_topk'])}")
    del model
    torch.cuda.empty_cache()
    return result


def log(msg):
    print(msg, flush=True)


# ============================================================
# Main
# ============================================================
def main():
    log("=" * 70)
    log("P0: FULL AUXILIARY-JACOBIAN PENALTY BASELINE")
    log("  VAR(1) d=8, T=500, 5 seeds, max_iter=2000")
    log("  Fixes MambaConcatFullPenaltyJRNGC bug: compute_jacobian_loss")
    log("  now penalizes FULL (d+d_cond) input with separate lam_x, lam_c")
    log("=" * 70)

    d, T, lag = 8, 500, 1
    lam_base = DEFAULT_LAMBDA
    d_cond = 4

    all_results = {}

    for seed in range(N_SEEDS):
        seed_key = f"seed_{seed}"
        all_results[seed_key] = {}

        x, gc, A_true = generate_var1_data(d=d, T=T, seed=seed * 100 + 42)
        n_edges = int(gc.sum())
        log(f"\n--- Seed {seed}: d={d}, T={T}, edges={n_edges} ---")

        # 1. JRNGC Baseline
        all_results[seed_key]["baseline"] = run_one(
            BaselineJRNGC(d=d, lag=lag, layers=3, hidden=32, jacobian_lam=lam_base).to(device),
            x, gc, A_true, seed, "Baseline"
        )

        # 2. Concat x-only (shortcut collapse)
        all_results[seed_key]["concat_x_only"] = run_one(
            MambaJRNGC(d=d, lag=lag, layers=3, hidden=32,
                       jacobian_lam=lam_base, d_state=4, d_cond=d_cond,
                       use_time_weight_loss=False).to(device),
            x, gc, A_true, seed, "Concat x-only"
        )

        # 3. Full penalty: same-lambda (direct alternative)
        all_results[seed_key]["full_same_lambda"] = run_one(
            MambaConcatFullPenaltyJRNGC(
                d=d, lag=lag, layers=3, hidden=32,
                jacobian_lam=lam_base, d_state=4, d_cond=d_cond,
                lam_x=lam_base, lam_c=lam_base,
                use_time_weight_loss=False).to(device),
            x, gc, A_true, seed, "Full same-λ"
        )

        # 4. Full penalty: budget-normalized
        lam_normalized = lam_base * d / (d + d_cond)
        all_results[seed_key]["full_budget_norm"] = run_one(
            MambaConcatFullPenaltyJRNGC(
                d=d, lag=lag, layers=3, hidden=32,
                jacobian_lam=lam_base, d_state=4, d_cond=d_cond,
                lam_x=lam_normalized, lam_c=lam_normalized,
                use_time_weight_loss=False).to(device),
            x, gc, A_true, seed, "Full budget-norm"
        )

        # 5. Full penalty: lambda_c/lambda_x = 0.1 (weak c penalty)
        all_results[seed_key]["full_lc_01"] = run_one(
            MambaConcatFullPenaltyJRNGC(
                d=d, lag=lag, layers=3, hidden=32,
                jacobian_lam=lam_base, d_state=4, d_cond=d_cond,
                lam_x=lam_base, lam_c=lam_base * 0.1,
                use_time_weight_loss=False).to(device),
            x, gc, A_true, seed, "Full λc/λx=0.1"
        )

        # 6. Full penalty: lambda_c/lambda_x = 10 (strong c penalty)
        all_results[seed_key]["full_lc_10"] = run_one(
            MambaConcatFullPenaltyJRNGC(
                d=d, lag=lag, layers=3, hidden=32,
                jacobian_lam=lam_base, d_state=4, d_cond=d_cond,
                lam_x=lam_base, lam_c=lam_base * 10,
                use_time_weight_loss=False).to(device),
            x, gc, A_true, seed, "Full λc/λx=10"
        )

        # 7. ISTF-Mamba
        all_results[seed_key]["istf"] = run_one(
            MambaFilterJRNGC(d=d, lag=lag, layers=3, hidden=32,
                             jacobian_lam=lam_base, d_state=4,
                             ortho_lam=0.05, residual_scale=0.1,
                             filter_type="mamba").to(device),
            x, gc, A_true, seed, "ISTF-Mamba"
        )

    # ============================================================
    # Summary
    # ============================================================
    methods = [
        ("baseline", "JRNGC Baseline"),
        ("concat_x_only", "Concat x-only"),
        ("full_same_lambda", "Full same-λ"),
        ("full_budget_norm", "Full budget-norm"),
        ("full_lc_01", "Full λc/λx=0.1"),
        ("full_lc_10", "Full λc/λx=10"),
        ("istf", "ISTF-Mamba"),
    ]

    log(f"\n{'='*90}")
    log(f"{'Method':<24} {'AUROC':>8} {'AUPRC':>8} {'SHD':>5} {'Coef r':>7} {'Shrink':>7} {'|Jx|':>8} {'|Jc|':>8} {'PredLoss':>9}")
    log(f"{'-'*24} {'-'*8} {'-'*8} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*9}")

    summary = {}
    for key, label in methods:
        vals = defaultdict(list)
        for seed in range(N_SEEDS):
            for k, v in all_results[f"seed_{seed}"][key].items():
                if v is not None:
                    vals[k].append(v)
        mean_vals = {k: np.mean(v) for k, v in vals.items()}
        std_vals = {k: np.std(v, ddof=0) for k, v in vals.items()}
        summary[key] = {"label": label, "mean": mean_vals, "std": std_vals}

        jx_str = f"{mean_vals.get('jx_norm', float('nan')):.4f}" if 'jx_norm' in mean_vals else "   --"
        jc_str = f"{mean_vals.get('jc_norm', float('nan')):.4f}" if 'jc_norm' in mean_vals else "   --"
        log(f"  {label:<22} {mean_vals['auroc']:7.4f}±{std_vals['auroc']:.4f} "
            f"{mean_vals['auprc']:7.4f}±{std_vals['auprc']:.4f} "
            f"{mean_vals['shd']:4.1f}±{std_vals['shd']:.1f} "
            f"{mean_vals['coefficient_correlation']:6.4f} "
            f"{mean_vals['coefficient_shrinkage']:6.4f} "
            f"{jx_str}  {jc_str}  "
            f"{mean_vals['pred_loss']:8.6f}")

    # Interpretation
    baseline_auroc = summary["baseline"]["mean"]["auroc"]
    concat_auroc = summary["concat_x_only"]["mean"]["auroc"]
    full_auroc = summary["full_same_lambda"]["mean"]["auroc"]
    istf_auroc = summary["istf"]["mean"]["auroc"]

    log(f"\n{'='*70}")
    log("INTERPRETATION")
    log(f"  Baseline AUROC:           {baseline_auroc:.4f}")
    log(f"  Concat x-only AUROC:      {concat_auroc:.4f}  (shortcut collapse)")
    log(f"  Full same-λ AUROC:        {full_auroc:.4f}  (penalize c too)")
    log(f"  ISTF-Mamba AUROC:         {istf_auroc:.4f}  (input-space repair)")
    log(f"  Full penalty recovery:    {full_auroc - concat_auroc:+.4f} vs concat")
    log(f"  ISTF vs full penalty:     {istf_auroc - full_auroc:+.4f}")

    if full_auroc < istf_auroc:
        log(f"  → Full penalty mitigates but ISTF remains cleaner")
    elif full_auroc >= istf_auroc:
        log(f"  → WARNING: full penalty >= ISTF → may need to adjust repair claim")

    # Save
    output = {
        "experiment": "full_aux_jacobian_penalty",
        "setting": "VAR(1) d=8 T=500, 5 seeds, max_iter=2000",
        "lambda_base": lam_base,
        "d_cond": d_cond,
        "per_seed": {
            f"seed_{s}": {
                k: v for k, v in all_results[f"seed_{s}"].items()
            }
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

    json_path = os.path.join(OUT_DIR, "full_aux_jacobian_penalty.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nSaved to {json_path}")

    # CSV
    csv_path = os.path.join(OUT_DIR, "full_aux_jacobian_penalty.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "lam_x", "lam_c",
                         "auroc_mean", "auroc_std",
                         "auprc_mean", "auprc_std",
                         "shd_mean", "shd_std",
                         "coef_corr_mean", "coef_shrink_mean",
                         "jx_norm_mean", "jc_norm_mean",
                         "pred_loss_mean"])
        for key, label in methods:
            s = summary[key]
            writer.writerow([
                label,
                lam_base,  # approximate
                lam_base,
                f"{s['mean']['auroc']:.4f}", f"{s['std']['auroc']:.4f}",
                f"{s['mean']['auprc']:.4f}", f"{s['std']['auprc']:.4f}",
                f"{s['mean']['shd']:.1f}", f"{s['std']['shd']:.1f}",
                f"{s['mean']['coefficient_correlation']:.4f}",
                f"{s['mean']['coefficient_shrinkage']:.4f}",
                f"{s['mean'].get('jx_norm', 'N/A')}",
                f"{s['mean'].get('jc_norm', 'N/A')}",
                f"{s['mean']['pred_loss']:.6f}",
            ])
    log(f"Saved to {csv_path}")

    # Sanity check: full penalty should have non-zero jc_norm
    jc_vals = [all_results[f"seed_{s}"]["full_same_lambda"].get("jc_norm", 0)
               for s in range(N_SEEDS)]
    if all(v is not None and v > 0 for v in jc_vals):
        log(f"\n  SANITY CHECK PASSED: |J_c| > 0 in full penalty ({np.mean(jc_vals):.4f})")
    else:
        log(f"\n  SANITY CHECK FAILED: |J_c| missing or zero in full penalty")

    return output


if __name__ == "__main__":
    main()
