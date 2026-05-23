"""Self-contained factorial pilot for AutoDL cloud.

Works with flat directory structure (pre-P0-C) or new src/experiments/ structure.
All config resolution is inlined — no dependency on src.config.

Upload to /root/autodl-tmp/GUOJI/mamba_enhanced/ and run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate jrngc_bw
    python run_factorial_pilot_cloud.py
"""
import torch
import numpy as np
import sys, os, json, time, argparse

torch.backends.cudnn.enabled = False

# ---- Inlined config resolution (no dependency on src.config) ----
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

def resolve_jrngc_path():
    env = os.environ.get("JRNGC_PATH", "")
    if env and os.path.isdir(env):
        return env
    candidate = os.path.join(os.path.dirname(_PROJ_ROOT), "JRNGC")
    if os.path.isdir(candidate):
        return candidate
    return ""

_jrngc = resolve_jrngc_path()
if _jrngc and _jrngc not in sys.path:
    sys.path.insert(0, _jrngc)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# ---- Imports (works with flat or src/ structure) ----
from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                train_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ============================================================
# Multimode Metrics (inlined — relies on JRNGC tgc.metrics)
# ============================================================
from tgc.metrics import two_classify_metrics, remove_self_connection


def _compute_metrics_core(gt_2d, pr_2d, gc_true_full=None, gc_pred_full=None):
    """Core metric computation given collapsed 2D ground-truth and prediction."""
    (f1, f1_trd), (acc, acc_trd), (auroc, _, _), (auprc, _, _) = two_classify_metrics(pr_2d, gt_2d)

    gt_int = gt_2d.astype(np.int32)
    n_edges_true = int(np.sum(gt_int))
    if n_edges_true > 0:
        thr = np.sort(pr_2d.ravel())[-n_edges_true]
        pred_binary = (pr_2d >= thr).astype(np.int32)
    else:
        pred_binary = np.zeros_like(gt_int, dtype=np.int32)
    shd = int(np.sum(np.abs(gt_int - pred_binary)))
    nshd = shd / max(n_edges_true, 1)

    tp = int(np.sum((pred_binary == 1) & (gt_int == 1)))
    tn = int(np.sum((pred_binary == 0) & (gt_int == 0)))
    fp = int(np.sum((pred_binary == 1) & (gt_int == 0)))
    fn = int(np.sum((pred_binary == 0) & (gt_int == 1)))
    mcc_denom = np.sqrt(float(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0)))
    mcc = float(tp * tn - fp * fn) / max(mcc_denom, 1e-8)

    lag_metrics = {}
    if gc_true_full is not None and gc_pred_full is not None \
            and gc_true_full.ndim == 3 and gc_pred_full.ndim == 3:
        max_lag = min(gc_true_full.shape[2], gc_pred_full.shape[2])
        for start, end in [(0, 2), (2, 4), (4, 6), (6, max_lag)]:
            if start >= max_lag:
                break
            end = min(end, max_lag)
            gt_slice = gc_true_full[:, :, start:end]
            pr_slice = gc_pred_full[:, :, start:end]
            gt_bin = (gt_slice.sum(axis=2) > 0).astype(np.int32)
            pr_score = pr_slice.max(axis=2)
            gt_bin = remove_self_connection(gt_bin)
            pr_score = remove_self_connection(pr_score.astype(np.float64))
            (_, _), (_, _), (auroc_lag, _, _), (_, _, _) = two_classify_metrics(pr_score, gt_bin)
            lag_metrics[f"lag_{start}-{end}"] = float(auroc_lag)

    return {
        "auroc": float(auroc), "auprc": float(auprc),
        "f1": float(f1), "acc": float(acc),
        "shd_topk": shd, "nshd_topk": float(nshd), "mcc_topk": float(mcc),
        "n_edges_true": n_edges_true,
        **lag_metrics
    }


def compute_metrics_multimode(gc_true, gc_pred):
    """Compute metrics in three summary modes: lag0, summary_max, summary_mean.

    - lag0: ground truth = gc_true[:,:,0]. Prediction = max(|pred|) over lags.
    - summary_max: ground truth = any edge at any lag. Prediction = max_k |pred_{ij}^{(k)}|.
    - summary_mean: same gt as summary_max. Prediction = (1/K) Σ_k |pred_{ij}^{(k)}|.
    """
    if gc_true.ndim != 3 or gc_pred.ndim != 3:
        # Fallback: all modes identical for 2D inputs
        gt_2d = remove_self_connection(gc_true.astype(np.int32)) if gc_true.ndim == 2 else remove_self_connection(gc_true[:, :, 0].astype(np.int32))
        pr_2d = remove_self_connection(np.max(np.abs(gc_pred), axis=2).astype(np.float64)) if gc_pred.ndim == 3 else remove_self_connection(gc_pred.astype(np.float64))
        single = _compute_metrics_core(gt_2d, pr_2d)
        return {"lag0": single, "summary_max": single, "summary_mean": single}

    result = {}

    # lag0
    gt_lag0 = remove_self_connection(gc_true[:, :, 0].astype(np.int32))
    pr_lag0 = remove_self_connection(np.max(np.abs(gc_pred), axis=2).astype(np.float64))
    result["lag0"] = _compute_metrics_core(gt_lag0, pr_lag0, gc_true, gc_pred)

    # summary (shared ground truth)
    gt_summary = remove_self_connection((gc_true.sum(axis=2) > 0).astype(np.int32))

    pr_max = remove_self_connection(np.max(np.abs(gc_pred), axis=2).astype(np.float64))
    result["summary_max"] = _compute_metrics_core(gt_summary, pr_max, gc_true, gc_pred)

    pr_mean = remove_self_connection(np.mean(np.abs(gc_pred), axis=2).astype(np.float64))
    result["summary_mean"] = _compute_metrics_core(gt_summary, pr_mean, gc_true, gc_pred)

    return result


# ============================================================
# Unified Factorial Data Generator (inlined in this script)
# ============================================================

FACTORIAL_SETTINGS = {
    "A": {"coeff_scale": 0.25, "noise_scale": 0.20, "regime_shift_strength": 0.30, "nonlinear_strength": 0.50},
    "B": {"coeff_scale": 0.20, "noise_scale": 0.30, "regime_shift_strength": 0.40, "nonlinear_strength": 0.75},
    "C": {"coeff_scale": 0.22, "noise_scale": 0.25, "regime_shift_strength": 0.60, "nonlinear_strength": 0.50},
    # Round 2 (expert-adjusted, only if A/B/C still too hard after summary_max fix):
    "D": {"coeff_scale": 0.40, "noise_scale": 0.15, "regime_shift_strength": 0.30, "nonlinear_strength": 0.50},
    "D2": {"coeff_scale": 0.40, "noise_scale": 0.15, "regime_shift_strength": 0.20, "nonlinear_strength": 0.50},
    "E": {"coeff_scale": 0.50, "noise_scale": 0.12, "regime_shift_strength": 0.40, "nonlinear_strength": 0.65},
    "F": {"coeff_scale": 0.60, "noise_scale": 0.10, "regime_shift_strength": 0.50, "nonlinear_strength": 0.50},
}

FACTORIAL_CELLS = [
    ("Stat+Linear",     True,   True),
    ("Stat+Nonlinear",  True,   False),
    ("NS+Linear",       False,  True),
    ("NS+Nonlinear",    False,  False),
]


def generate_factorial_cell(
    d=10, T=600, lag=3, seed=0,
    stationary=True, linear=True,
    coeff_scale=0.25, noise_scale=0.20,
    regime_shift_strength=0.0, nonlinear_strength=0.0,
    sparsity=0.2,
):
    rng = np.random.RandomState(seed * 1000 + 42)

    # 1. Shared ground-truth GC graph
    gc = np.zeros((d, d, lag), dtype=np.float32)
    for i in range(d):
        for j in range(d):
            if i != j and rng.rand() < sparsity:
                k = rng.randint(0, lag)
                gc[i, j, k] = 1.0

    # 2. Base coefficient matrices from gc
    A_base = []
    for k in range(lag):
        A_k = np.zeros((d, d), dtype=np.float32)
        for i in range(d):
            for j in range(d):
                if gc[i, j, k] > 0:
                    A_k[i, j] = coeff_scale * rng.uniform(0.3, 1.0) * rng.choice([-1, 1])
        A_base.append(A_k)

    # 3. Coefficient drift for non-stationary cells
    if not stationary and regime_shift_strength > 0:
        drift_scale = regime_shift_strength * coeff_scale
        A_drift = []
        for k in range(lag):
            raw = np.cumsum(rng.randn(T, d, d) * drift_scale / np.sqrt(T), axis=0)
            window = max(5, T // 30)
            smoothed = np.zeros_like(raw)
            for t in range(T):
                lo = max(0, t - window)
                smoothed[t] = raw[lo:t + 1].mean(axis=0)
            A_drift.append(smoothed)
    else:
        A_drift = [np.zeros((T, d, d), dtype=np.float32) for _ in range(lag)]

    # 4. Generate time series
    x = np.zeros((d, T), dtype=np.float32)
    noise = rng.randn(d, T).astype(np.float32) * noise_scale

    for t in range(lag):
        x[:, t] = noise[:, t]

    for t in range(lag, T):
        pred = np.zeros(d, dtype=np.float32)
        for k in range(lag):
            A_k_t = A_base[k] + A_drift[k][t]
            pred += A_k_t @ x[:, t - k - 1]

        if not linear and nonlinear_strength > 0:
            # pred_nl = (1-α)·pred + α·s·tanh(pred/s)  where s = std(pred)
            # This smoothly interpolates: identity for small pred, saturation for large pred
            s = float(np.std(pred)) + 1e-8
            pred = (1.0 - nonlinear_strength) * pred + nonlinear_strength * s * np.tanh(pred / s)

        x[:, t] = pred + noise[:, t]

    return x.astype(np.float32), gc


def generate_all_cells(setting="A", d=10, T=600, lag=3, seed=0, sparsity=0.2):
    params = FACTORIAL_SETTINGS[setting].copy()
    cells = {}
    for name, stationary, linear in FACTORIAL_CELLS:
        regime = params["regime_shift_strength"] if not stationary else 0.0
        nl = params["nonlinear_strength"] if not linear else 0.0
        x, gc = generate_factorial_cell(
            d=d, T=T, lag=lag, seed=seed,
            stationary=stationary, linear=linear,
            coeff_scale=params["coeff_scale"],
            noise_scale=params["noise_scale"],
            regime_shift_strength=regime,
            nonlinear_strength=nl,
            sparsity=sparsity,
        )
        cells[name] = (x, gc)
    return cells


# ============================================================
# Experiment runner
# ============================================================

def run_one_cell(x, gc_true, d, lag, seed, model_type, max_iter=2000, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_type == "baseline":
        model = BaselineJRNGC(
            d=d, lag=lag, layers=5, hidden=50, jacobian_lam=0.01
        ).to(device)
    elif model_type == "tcn":
        model = MambaFilterJRNGC(
            d=d, lag=lag, layers=5, hidden=50,
            jacobian_lam=0.01, d_state=4, ortho_lam=0.05,
            residual_scale=0.1, filter_type="tcn"
        ).to(device)
    else:
        model = MambaFilterJRNGC(
            d=d, lag=lag, layers=5, hidden=50,
            jacobian_lam=0.01, d_state=4, ortho_lam=0.05,
            residual_scale=0.1, filter_type="mamba"
        ).to(device)

    t0 = time.time()
    model, loss = train_model(model, x, max_iter=max_iter, lr=lr, verbose=False)
    gc_pred = model.get_gc_matrix(x)
    train_time = time.time() - t0
    try:
        multimode = compute_metrics_multimode(gc_true, gc_pred)
        metrics = {
            "lag0": multimode["lag0"],
            "summary_max": multimode["summary_max"],
            "summary_mean": multimode["summary_mean"],
        }
    except ZeroDivisionError:
        # Edge case: too few edges causes division by zero in JRNGC metrics
        empty = {"auroc": 0.5, "auprc": 0.0, "shd": 0, "nshd": 0.0,
                 "f1": 0.0, "acc": 0.0, "mcc": 0.0, "n_edges_true": 0}
        metrics = {"lag0": empty, "summary_max": empty, "summary_mean": empty}
    metrics["train_time"] = train_time
    metrics["train_loss"] = float(loss)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str, default="A,B,C")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--models", type=str, default="both",
                        choices=["baseline", "mamba", "tcn", "both", "all"],
                        help="Which models to run (default: both = baseline+mamba)")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--T", type=int, default=600)
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    settings = [s.strip() for s in args.settings.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    out_path = args.output or os.path.join(_PROJ_ROOT, "factorial_pilot_results.json")

    print("=" * 70)
    print("FACTORIAL PILOT CALIBRATION (cloud)")
    print(f"  Settings: {settings}")
    print(f"  Seeds: {seeds}")
    print(f"  d={args.d}, T={args.T}, lag={args.lag}, max_iter={args.max_iter}")
    print(f"  Device: {device}")
    print(f"  Output: {out_path}")
    print("=" * 70)

    all_results = {}

    for setting_name in settings:
        params = FACTORIAL_SETTINGS[setting_name]
        print(f"\n{'='*70}")
        print(f"SETTING {setting_name}: {params}")
        print(f"{'='*70}")
        all_results[setting_name] = {"params": params, "seeds": {}}

        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")
            cells = generate_all_cells(
                setting=setting_name, d=args.d, T=args.T, lag=args.lag, seed=seed
            )
            all_results[setting_name]["seeds"][str(seed)] = {}

            for cell_name, stationary, linear in FACTORIAL_CELLS:
                x, gc_true = cells[cell_name]
                n_edges = int(gc_true.sum())
                print(f"    {cell_name}: T={x.shape[1]}, edges={n_edges}")

                cell_result = {}

                # Resolve which models to run
                if args.models == "all":
                    run_list = ["baseline", "mamba", "tcn"]
                elif args.models == "both":
                    run_list = ["baseline", "mamba"]
                else:
                    run_list = [args.models]

                model_labels = {"baseline": "Baseline", "mamba": "Mamba", "tcn": "TCN"}
                for mdl in run_list:
                    t0 = time.time()
                    metrics = run_one_cell(x, gc_true, args.d, args.lag, seed, mdl,
                                           max_iter=args.max_iter, lr=args.lr)
                    sm = metrics["summary_max"]
                    l0 = metrics["lag0"]
                    print(f"      {model_labels[mdl]:<9} AUROC_sm={sm['auroc']:.4f} AUPRC_sm={sm['auprc']:.4f} "
                          f"SHD_sm={sm['shd']} edges_sm={sm['n_edges_true']}  "
                          f"[lag0 AUROC={l0['auroc']:.4f} edges={l0['n_edges_true']}] "
                          f"time={time.time()-t0:.0f}s")
                    cell_result[mdl] = metrics

                all_results[setting_name]["seeds"][str(seed)][cell_name] = cell_result

    # Summary
    print(f"\n{'='*70}")
    print("PILOT SUMMARY")
    print(f"{'='*70}")
    run_baseline = args.models in ("baseline", "both", "all")
    run_mamba = args.models in ("mamba", "both", "all")
    run_tcn = args.models in ("tcn", "all")
    model_keys = []
    if run_baseline: model_keys.append("baseline")
    if run_mamba: model_keys.append("mamba")
    if run_tcn: model_keys.append("tcn")
    model_labels = {"baseline": "Baseline", "mamba": "Mamba", "tcn": "TCN"}

    for setting_name in settings:
        print(f"\nSetting {setting_name} ({FACTORIAL_SETTINGS[setting_name]}):")
        # Dynamic header
        header = f"  {'Cell':<18}"
        sep = f"  {'-'*18}"
        for mk in model_keys:
            header += f" {model_labels[mk]:>10}"
            sep += f" {'-'*10}"
        if len(model_keys) >= 2:
            header += f" {'Δ(m-b)':>9}"
            sep += f" {'-'*9}"
        header += "  |  lag0 AUROC"
        sep += "  |  ----------"
        print(header)
        print(sep)

        for cell_name, _, _ in FACTORIAL_CELLS:
            # Collect per-model values across seeds
            sm_vals = {mk: [] for mk in model_keys}
            l0_vals = {mk: [] for mk in model_keys}
            for seed in seeds:
                sd = all_results[setting_name]["seeds"][str(seed)]
                if cell_name not in sd:
                    continue
                for mk in model_keys:
                    if mk in sd[cell_name]:
                        sm_vals[mk].append(sd[cell_name][mk]["summary_max"]["auroc"])
                        l0_vals[mk].append(sd[cell_name][mk]["lag0"]["auroc"])
            # Build row
            row = f"  {cell_name:<18}"
            sm_means = {}
            l0_mean = None
            for mk in model_keys:
                if sm_vals[mk]:
                    m, s = np.mean(sm_vals[mk]), np.std(sm_vals[mk])
                    sm_means[mk] = m
                    row += f" {m:>8.4f}±{s:.3f}"
                    if l0_mean is None:
                        l0_mean = np.mean(l0_vals[mk])
                else:
                    row += f" {'N/A':>10}"
            if len(model_keys) >= 2 and "baseline" in sm_means:
                # Show mamba minus baseline delta (if mamba present) or tcn minus baseline
                for mk in model_keys:
                    if mk != "baseline" and mk in sm_means:
                        delta = sm_means[mk] - sm_means["baseline"]
                        row += f" {delta:>+8.4f}"
                        break
            elif len(model_keys) >= 2:
                row += f" {'N/A':>9}"
            # lag0 column
            row += "  | "
            if l0_mean is not None:
                row += f" {l0_mean:>8.4f}"
            else:
                row += f" {'N/A':>8}"
            # IN RANGE flag (use baseline)
            if "baseline" in sm_means:
                b_sm_m = sm_means["baseline"]
                if 0.75 <= b_sm_m <= 0.90:
                    row += "  <<< IN RANGE (0.75-0.90)"
            print(row)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
