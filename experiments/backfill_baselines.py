"""P1 #2 backfill: PCMCI+GPDC for Lorenz-96, PCMCI+ParCorr for VAR d=50, cMLP lambda tuning for Lorenz-96.

GPDC is too slow for d=50 (GP-based CI). Use ParCorr for VAR (linear data) and GPDC for Lorenz-96 (nonlinear).
"""
import torch, numpy as np, sys, os, json, time

torch.backends.cudnn.enabled = False

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path, resolve_data_dir, resolve_results_dir, resolve_device

_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)
# Neural-GC path (sibling directory)
_neural_gc = os.path.join(os.path.dirname(_PROJ_ROOT), "Neural-GC")
if os.path.isdir(_neural_gc):
    sys.path.insert(0, _neural_gc)

from mamba_jrngc_pilot import compute_metrics

device = resolve_device()


def log(msg):
    print(msg, flush=True)


def run_pcmci(x, ci_test="parcorr", pc_alpha=0.05, tau_max=1):
    """Run PCMCI with specified conditional independence test.

    ci_test: 'parcorr' (fast, linear) or 'gpdc' (slow, nonlinear GP-based)
    """
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    d, T = x.shape
    df = pp.DataFrame(x.T)

    if ci_test == "gpdc":
        from tigramite.independence_tests.gpdc import GPDC
        cit = GPDC(significance='analytic')
    else:
        from tigramite.independence_tests.parcorr import ParCorr
        cit = ParCorr(significance='analytic')

    pcmci = PCMCI(dataframe=df, cond_ind_test=cit, verbosity=0)
    res = pcmci.run_pcmci(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
    p_mat = res['p_matrix'][:, :, 1]
    gc_pred = (1 - p_mat)[:, :, np.newaxis]
    return np.clip(gc_pred, 0, 1)


def run_cmlp(x, gc_true, lag=5, hidden=None, lam=0.002, max_iter=10000):
    from models.cmlp import cMLP, train_model_ista

    d, T = x.shape
    if hidden is None:
        hidden = [100]

    X = torch.tensor(x.T[np.newaxis], dtype=torch.float32, device=device)
    model = cMLP(d, lag=lag, hidden=hidden).to(device)
    t0 = time.time()
    train_model_ista(model, X, lr=5e-2, max_iter=max_iter, lam=lam,
                     lam_ridge=1e-2, penalty='H', lookback=5,
                     check_every=100, verbose=0)
    gc_pred = model.GC(threshold=False, ignore_lag=True).cpu().data.numpy()
    metrics = compute_metrics(gc_true, gc_pred)
    metrics["train_time"] = time.time() - t0
    del model; torch.cuda.empty_cache()
    return metrics


def main():
    data_dir = resolve_data_dir()
    results = {}

    # ---- 1. PCMCI+ParCorr for VAR d=50 (3 seeds) ----
    log("=" * 60)
    log("Backfill: PCMCI+ParCorr for VAR d=50 (3 seeds)")
    log("=" * 60)
    results["VAR_d50_stat"] = {}
    for seed in range(3):
        p = os.path.join(data_dir,"var", "num_nodes_50", "true_lag_5", "noise_scale_1", f"seed_{seed}")
        x = np.load(os.path.join(p, "_x.npy"))
        gc = np.load(os.path.join(p, "_gc.npy"))
        t0 = time.time()
        gc_pred = run_pcmci(x, ci_test="parcorr")
        metrics = compute_metrics(gc, gc_pred)
        metrics["train_time"] = time.time() - t0
        results["VAR_d50_stat"][f"seed_{seed}"] = {"pcmci_parcorr": metrics}
        log(f"  seed_{seed}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, "
            f"SHD={metrics['shd']}, nSHD={metrics.get('nshd','N/A'):.3f}, time={metrics['train_time']:.0f}s")

    # ---- 2. PCMCI+GPDC for Lorenz-96 F=40 (3 seeds) ----
    log("\n" + "=" * 60)
    log("Backfill: PCMCI+GPDC for Lorenz-96 F=40 (3 seeds)")
    log("=" * 60)
    results["Lorenz_F40"] = {}
    for seed in range(3):
        p = os.path.join(data_dir,"lorenz", "num_nodes_10", "F_40", f"seed_{seed}")
        x = np.load(os.path.join(p, "_x.npy"))
        gc = np.load(os.path.join(p, "_gc.npy"))
        t0 = time.time()
        gc_pred = run_pcmci(x, ci_test="gpdc")
        metrics = compute_metrics(gc, gc_pred)
        metrics["train_time"] = time.time() - t0
        results["Lorenz_F40"][f"seed_{seed}"] = {"pcmci_gpdc": metrics}
        log(f"  seed_{seed}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, "
            f"SHD={metrics['shd']}, nSHD={metrics.get('nshd','N/A'):.3f}, time={metrics['train_time']:.0f}s")

    # ---- 3. cMLP lambda tuning for Lorenz-96 (seed 0) ----
    log("\n" + "=" * 60)
    log("cMLP lambda tuning for Lorenz-96 F=40 (seed 0)")
    log("=" * 60)
    p = os.path.join(data_dir,"lorenz", "num_nodes_10", "F_40", "seed_0")
    x = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))

    lorenz_cmlp_tuning = {}
    for lam in [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]:
        log(f"  lam={lam}...")
        try:
            metrics = run_cmlp(x, gc, lag=5, lam=lam, max_iter=10000)
            lorenz_cmlp_tuning[str(lam)] = metrics
            log(f"    AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, SHD={metrics['shd']}")
        except Exception as e:
            log(f"    FAILED: {e}")

    # Also try the best lambda on all 3 seeds
    # Determine best lambda
    best_lam = None
    best_auroc = -1
    for lam_str, m in lorenz_cmlp_tuning.items():
        if m['auroc'] > best_auroc:
            best_auroc = m['auroc']
            best_lam = float(lam_str)
    log(f"\n  Best lambda on seed 0: {best_lam} (AUROC={best_auroc:.4f})")

    if best_lam is not None:
        log(f"\n  Running best lambda {best_lam} on all 3 seeds...")
        for seed in range(1, 3):
            p = os.path.join(data_dir,"lorenz", "num_nodes_10", "F_40", f"seed_{seed}")
            x = np.load(os.path.join(p, "_x.npy"))
            gc = np.load(os.path.join(p, "_gc.npy"))
            metrics = run_cmlp(x, gc, lag=5, lam=best_lam, max_iter=10000)
            lorenz_cmlp_tuning[f"best_lam_seed{seed}"] = metrics
            log(f"  seed_{seed}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, SHD={metrics['shd']}")

    results["Lorenz_F40"]["cmlp_tuning"] = lorenz_cmlp_tuning

    # Save
    out = {"backfill": results}
    with open("backfill_baselines_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"\nSaved to backfill_baselines_results.json")


if __name__ == "__main__":
    main()
