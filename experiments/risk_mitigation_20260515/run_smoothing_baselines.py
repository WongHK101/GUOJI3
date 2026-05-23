"""P1: Simple Causal Smoothing Baselines.

Compares ISTF-Mamba against simple causal MA and EMA filters.
All smoothing is strictly causal (past-only). Grid search over
window/alpha parameters, reporting best validation-loss config.

Runs on: controlled VAR(1) d=8 T=500, 5 seeds.

Output: risk_mitigation_results/smoothing_baselines.{json,csv}
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

from src.mamba_jrngc_pilot import (
    BaselineJRNGC, MambaFilterJRNGC,
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
# Causal smoothing filters
# ============================================================
def causal_moving_average(x, window):
    """Causal MA: x_tilde[t] = mean(x[t-w+1:t+1])."""
    d, T = x.shape
    x_smoothed = np.zeros_like(x)
    for t in range(T):
        start = max(0, t - window + 1)
        x_smoothed[:, t] = x[:, start:t+1].mean(axis=1)
    return x_smoothed


def causal_ema(x, alpha):
    """Causal EMA: x_tilde[t] = alpha * x[t] + (1-alpha) * x_tilde[t-1]."""
    d, T = x.shape
    x_smoothed = np.zeros_like(x)
    x_smoothed[:, 0] = x[:, 0]
    for t in range(1, T):
        x_smoothed[:, t] = alpha * x[:, t] + (1 - alpha) * x_smoothed[:, t - 1]
    return x_smoothed


# ============================================================
# JRNGC wrapper that takes pre-smoothed data
# ============================================================
class SmoothingJRNGCWrapper:
    """Wraps a BaselineJRNGC but feeds pre-smoothed data as input."""
    def __init__(self, smoothing_fn, d, lag, layers=3, hidden=32, jacobian_lam=0.01):
        self.smoothing_fn = smoothing_fn
        self.model = BaselineJRNGC(
            d=d, lag=lag, layers=layers, hidden=hidden, jacobian_lam=jacobian_lam
        ).to(device)
        self.d = d

    def to(self, dev):
        self.model = self.model.to(dev)
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)

    def get_gc_matrix(self, x):
        x_smoothed = self.smoothing_fn(x)
        return self.model.get_gc_matrix(x_smoothed)

    def compute_loss(self, x):
        x_smoothed = self.smoothing_fn(x)
        return self.model.compute_loss(x_smoothed)


# ============================================================
# Data generator (same as full_aux_penalty)
# ============================================================
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


def run_one_model(model, x, gc, A_true, seed, label, max_iter=MAX_ITER, lr=LR):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    model, loss = train_model(model, x, max_iter=max_iter, lr=lr, verbose=False)
    gc_pred = model.get_gc_matrix(x)
    train_time = time.time() - t0
    m = compute_metrics(gc, gc_pred)
    d = x.shape[0]
    cr, cs = coef_metrics(gc_pred, A_true, d)
    result = {
        "auroc": float(m["auroc"]), "auprc": float(m["auprc"]),
        "shd": int(m["shd_topk"]), "coefficient_correlation": cr,
        "coefficient_shrinkage": cs, "pred_loss": float(loss),
        "train_time": train_time,
    }
    del model
    torch.cuda.empty_cache()
    return result


def log(msg):
    print(msg, flush=True)


def main():
    log("=" * 70)
    log("P1: CAUSAL SMOOTHING BASELINES")
    log("  VAR(1) d=8, T=500, 5 seeds, max_iter=2000")
    log("  Grid: MA windows {2,3,5}, EMA alpha {0.3,0.5,0.7,0.9}")
    log("=" * 70)

    d, T, lag = 8, 500, 1
    ma_windows = [2, 3, 5]
    ema_alphas = [0.3, 0.5, 0.7, 0.9]

    all_results = {}

    for seed in range(N_SEEDS):
        seed_key = f"seed_{seed}"
        all_results[seed_key] = {}

        x, gc, A_true = generate_var1_data(d=d, T=T, seed=seed * 100 + 42)
        n_edges = int(gc.sum())
        log(f"\n--- Seed {seed}: d={d}, T={T}, edges={n_edges} ---")

        # Baseline (no smoothing)
        log("  Baseline (no smoothing)...")
        all_results[seed_key]["baseline"] = run_one_model(
            BaselineJRNGC(d=d, lag=lag, layers=3, hidden=32, jacobian_lam=LAMBDA).to(device),
            x, gc, A_true, seed, "Baseline"
        )

        # Causal MA
        for w in ma_windows:
            log(f"  MA window={w}...")
            smoothing_fn = lambda x_arr, w=w: causal_moving_average(x_arr, w)
            wrapper = SmoothingJRNGCWrapper(smoothing_fn, d=d, lag=lag,
                                            layers=3, hidden=32, jacobian_lam=LAMBDA)
            all_results[seed_key][f"ma_w{w}"] = run_one_model(
                wrapper, x, gc, A_true, seed, f"MA w={w}"
            )

        # Causal EMA
        for alpha in ema_alphas:
            log(f"  EMA alpha={alpha}...")
            smoothing_fn = lambda x_arr, a=alpha: causal_ema(x_arr, a)
            wrapper = SmoothingJRNGCWrapper(smoothing_fn, d=d, lag=lag,
                                            layers=3, hidden=32, jacobian_lam=LAMBDA)
            all_results[seed_key][f"ema_a{alpha}"] = run_one_model(
                wrapper, x, gc, A_true, seed, f"EMA a={alpha}"
            )

        # ISTF-Mamba
        log("  ISTF-Mamba...")
        all_results[seed_key]["istf"] = run_one_model(
            MambaFilterJRNGC(d=d, lag=lag, layers=3, hidden=32,
                             jacobian_lam=LAMBDA, d_state=4,
                             ortho_lam=0.05, residual_scale=0.1,
                             filter_type="mamba").to(device),
            x, gc, A_true, seed, "ISTF-Mamba"
        )

    # Select best MA and best EMA by validation pred loss
    methods = ["baseline", "istf"] + [f"ma_w{w}" for w in ma_windows] + [f"ema_a{a}" for a in ema_alphas]

    summary = {}
    for method in methods:
        vals = defaultdict(list)
        for seed in range(N_SEEDS):
            for k, v in all_results[f"seed_{seed}"][method].items():
                if v is not None:
                    vals[k].append(v)
        mean_v = {k: np.mean(v) for k, v in vals.items()}
        std_v = {k: np.std(v, ddof=0) for k, v in vals.items()}
        summary[method] = {"mean": mean_v, "std": std_v}

    # Find best MA and best EMA
    ma_methods = [f"ma_w{w}" for w in ma_windows]
    ema_methods = [f"ema_a{a}" for a in ema_alphas]
    best_ma = min(ma_methods, key=lambda m: summary[m]["mean"]["pred_loss"])
    best_ema = min(ema_methods, key=lambda m: summary[m]["mean"]["pred_loss"])

    log(f"\n{'='*70}")
    log("RESULTS: Causal Smoothing Baselines")
    log(f"{'Method':<16} {'AUROC':>8} {'AUPRC':>8} {'SHD':>5} {'Coef r':>7} {'Shrink':>7} {'PredLoss':>9}")
    log(f"{'-'*16} {'-'*8} {'-'*8} {'-'*5} {'-'*7} {'-'*7} {'-'*9}")

    for method in methods:
        s = summary[method]
        marker = ""
        if method == best_ma:
            marker = " ← best MA"
        elif method == best_ema:
            marker = " ← best EMA"
        log(f"  {method:<14} {s['mean']['auroc']:7.4f}±{s['std']['auroc']:.4f} "
            f"{s['mean']['auprc']:7.4f}±{s['std']['auprc']:.4f} "
            f"{s['mean']['shd']:4.1f}±{s['std']['shd']:.1f} "
            f"{s['mean']['coefficient_correlation']:6.4f} "
            f"{s['mean']['coefficient_shrinkage']:6.4f} "
            f"{s['mean']['pred_loss']:8.6f}{marker}")

    istf_auroc = summary["istf"]["mean"]["auroc"]
    log(f"\n  ISTF vs best MA ({best_ma}): {istf_auroc - summary[best_ma]['mean']['auroc']:+.4f} AUROC")
    log(f"  ISTF vs best EMA ({best_ema}): {istf_auroc - summary[best_ema]['mean']['auroc']:+.4f} AUROC")

    if istf_auroc > summary[best_ma]["mean"]["auroc"] and istf_auroc > summary[best_ema]["mean"]["auroc"]:
        log(f"  → ISTF-Mamba outperforms simple causal smoothing")
    elif istf_auroc >= summary[best_ma]["mean"]["auroc"]:
        log(f"  → ISTF-Mamba comparable to best MA; weakens Mamba-specific claim")
    else:
        log(f"  → WARNING: Smoothing >= ISTF → need to rewrite method contribution")

    # Save
    output = {
        "experiment": "causal_smoothing_baselines",
        "setting": "VAR(1) d=8 T=500, 5 seeds, max_iter=2000",
        "ma_windows": ma_windows, "ema_alphas": ema_alphas,
        "best_ma": best_ma, "best_ema": best_ema,
        "per_seed": {
            f"seed_{s}": {k: v for k, v in all_results[f"seed_{s}"].items()}
            for s in range(N_SEEDS)
        },
        "summary": {
            k: {
                "mean": {mk: float(mv) if isinstance(mv, (np.floating, np.integer))
                         else mv for mk, mv in v["mean"].items()},
                "std": {mk: float(sv) if isinstance(sv, (np.floating, np.integer))
                        else sv for mk, sv in v["std"].items()},
            }
            for k, v in summary.items()
        }
    }

    json_path = os.path.join(OUT_DIR, "smoothing_baselines.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nSaved to {json_path}")

    csv_path = os.path.join(OUT_DIR, "smoothing_baselines.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std",
                         "shd_mean", "shd_std", "coef_corr_mean", "coef_shrink_mean",
                         "pred_loss_mean"])
        for method in methods:
            s = summary[method]
            writer.writerow([
                method,
                f"{s['mean']['auroc']:.4f}", f"{s['std']['auroc']:.4f}",
                f"{s['mean']['auprc']:.4f}", f"{s['std']['auprc']:.4f}",
                f"{s['mean']['shd']:.1f}", f"{s['std']['shd']:.1f}",
                f"{s['mean']['coefficient_correlation']:.4f}",
                f"{s['mean']['coefficient_shrinkage']:.4f}",
                f"{s['mean']['pred_loss']:.6f}",
            ])
    log(f"Saved to {csv_path}")
    return output


if __name__ == "__main__":
    main()
