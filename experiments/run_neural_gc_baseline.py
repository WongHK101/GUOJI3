"""cMLP/cLSTM baseline runner (Neural-GC models). GPU required.

Saves to results/raw/neural_gc_results.json in canonical format.

Usage:
    python experiments/run_neural_gc_baseline.py --method cmlp --datasets VAR_d50
    python experiments/run_neural_gc_baseline.py --method clstm --datasets Lorenz_F40
    python experiments/run_neural_gc_baseline.py --method all --datasets all
"""

import numpy as np
import sys, os, json, time, argparse

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path, resolve_data_dir, resolve_device
from src.schema import (make_result_entry, make_collection, save_collection,
                        make_provenance, load_collection, expected_results_dir)

# Neural-GC path
_NEURAL_GC = os.path.join(os.path.dirname(_PROJ_ROOT), "Neural-GC")
if _NEURAL_GC not in sys.path:
    sys.path.insert(0, _NEURAL_GC)

# JRNGC path (for metrics)
_jrngc = resolve_jrngc_path()
if _jrngc and _jrngc not in sys.path:
    sys.path.insert(0, _jrngc)

import torch
from models.cmlp import cMLP, train_model_ista as train_cmlp_ista
from models.cmlp import train_model_adam as train_cmlp_adam
from models.clstm import cLSTM, train_model_ista as train_clstm_ista
from models.clstm import train_model_adam as train_clstm_adam

DATA_DIR = resolve_data_dir()
RESULTS_DIR = expected_results_dir()
os.makedirs(RESULTS_DIR, exist_ok=True)


def _compute_metrics(gc_true, gc_pred):
    """Self-contained metric computation. Returns dict with auroc, auprc, shd, nshd, mcc."""
    from tgc.metrics.causal import two_classify_metrics, remove_self_connection
    if gc_true.ndim == 3:
        gc_true_2d = gc_true[:, :, 0]
    else:
        gc_true_2d = gc_true
    if gc_pred.ndim == 3:
        gc_pred_summary = np.max(np.abs(gc_pred), axis=2)
    else:
        gc_pred_summary = gc_pred
    gt = remove_self_connection(gc_true_2d.astype(np.int32))
    pr = remove_self_connection(gc_pred_summary.astype(np.float64))
    (f1, _), (acc, _), (auroc, _, _), (auprc, _, _) = two_classify_metrics(pr, gt)
    n_edges = int(gt.sum())
    if n_edges > 0:
        thr = np.sort(pr.ravel())[-n_edges]
        pr_bin = (pr >= thr).astype(np.int32)
        shd = int(np.sum(np.abs(gt - pr_bin)))
    else:
        shd = 0
        pr_bin = np.zeros_like(gt, dtype=np.int32)
    nshd = shd / max(n_edges, 1)
    tp = int(np.sum((pr_bin == 1) & (gt == 1)))
    tn = int(np.sum((pr_bin == 0) & (gt == 0)))
    fp = int(np.sum((pr_bin == 1) & (gt == 0)))
    fn = int(np.sum((pr_bin == 0) & (gt == 1)))
    mcc_denom = np.sqrt(float(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0)))
    mcc = float(tp * tn - fp * fn) / max(mcc_denom, 1e-8)
    return {"auroc": float(auroc), "auprc": float(auprc), "shd_topk": shd,
            "nshd_topk": float(nshd), "mcc_topk": float(mcc),
            "f1": float(f1), "acc": float(acc), "n_edges_true": n_edges}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_var_d50(seed, data_dir):
    base = os.path.join(data_dir, "var", "num_nodes_50", "true_lag_5", "noise_scale_1")
    x = np.load(os.path.join(base, f"seed_{seed}", "_x.npy"))
    gc = np.load(os.path.join(base, f"seed_{seed}", "_gc.npy"))
    return x, gc


def load_lorenz_f40(seed, data_dir):
    base = os.path.join(data_dir, "lorenz", "num_nodes_10", "F_40")
    x = np.load(os.path.join(base, f"seed_{seed}", "_x.npy"))
    gc = np.load(os.path.join(base, f"seed_{seed}", "_gc.npy"))
    return x, gc


def load_dream3(d, subject):
    _cwd = os.getcwd()
    if _jrngc:
        os.chdir(_jrngc)
    try:
        from tgc.data.dream3 import dream3_trajectories
        x, _, gc = dream3_trajectories(d=d, subject=subject)
    finally:
        os.chdir(_cwd)
    if x.ndim == 3:
        x = x[0]
    return x.astype(np.float64), gc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(x, gc_true, method, device, **kwargs):
    """Train a Neural-GC model and return (gc_pred, metrics_dict)."""
    d, T = x.shape
    # Convert to PyTorch: (batch=1, T, d)
    X_t = torch.tensor(x.T, dtype=torch.float32, device=device).unsqueeze(0)

    if method == "cmlp":
        lag = kwargs.get("lag", 5)
        hidden = kwargs.get("hidden", [100])
        lr = kwargs.get("lr", 5e-2)
        lam = kwargs.get("lam", 1e-4)
        lam_ridge = kwargs.get("lam_ridge", 1e-2)
        max_iter = kwargs.get("max_iter", 50000)
        penalty = kwargs.get("penalty", "H")

        model = cMLP(num_series=d, lag=lag, hidden=hidden).to(device)
        # Remove bias from first Conv1d layers so the model MUST use input features.
        # Without this, bias can absorb prediction signal, gradients to weights vanish,
        # and prox_update zeros out all weights → AUROC=0.5.
        for net in model.networks:
            net.layers[0].bias = None
        train_loss = train_cmlp_ista(
            model, X_t, lam=lam, lam_ridge=lam_ridge, lr=lr,
            penalty=penalty, max_iter=max_iter, check_every=100, verbose=0)

    elif method == "clstm":
        context = kwargs.get("context", 10)
        hidden = kwargs.get("hidden", 100)
        lr = kwargs.get("lr", 1e-3)
        lam = kwargs.get("lam", 1.0)
        lam_ridge = kwargs.get("lam_ridge", 1e-2)
        max_iter = kwargs.get("max_iter", 20000)

        model = cLSTM(num_series=d, hidden=hidden).to(device)
        # LSTM uses nn.LSTM internally. Unlike cMLP's Conv1d, the LSTM's
        # recurrent structure means the input bias is less able to "free-ride"
        # — the gate structure distributes the bias effect. We rely on
        # GC(threshold=False) + appropriate lam for continuous GC scores.
        train_loss = train_clstm_ista(
            model, X_t, context=context, lam=lam, lam_ridge=lam_ridge,
            lr=lr, max_iter=max_iter, check_every=50, verbose=0)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Use threshold=False to get raw L2 norms as continuous scores for AUROC.
    # threshold=True would return binary 0/1, which gives AUROC=0.5 when all weights
    # are near zero (a single point in ROC space).
    gc_est = model.GC(threshold=False).cpu().data.numpy()  # shape (d, d), continuous

    # Expand to (d, d, 1) for metric computation
    gc_pred = gc_est[:, :, np.newaxis]
    metrics = _compute_metrics(gc_true, gc_pred)

    # Add train_loss to metrics
    metrics["train_loss"] = float(train_loss[-1]) if train_loss else float("nan")

    return gc_pred, metrics


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

# Default hyperparameters per method x dataset size
HP_DEFAULTS = {
    "cmlp": {
        # Effective ISTA threshold = lr * lam. With lr=5e-2:
        #   lam=1e-4 → threshold=5e-6   lam=1e-3 → threshold=5e-5
        "small":  {"lag": 5, "hidden": [100], "lr": 5e-2, "lam": 1e-4,
                    "lam_ridge": 1e-2, "max_iter": 30000, "penalty": "H"},
        "medium": {"lag": 5, "hidden": [100], "lr": 5e-2, "lam": 5e-5,
                    "lam_ridge": 1e-2, "max_iter": 30000, "penalty": "H"},
        "large":  {"lag": 5, "hidden": [200], "lr": 5e-2, "lam": 1e-5,
                    "lam_ridge": 1e-2, "max_iter": 30000, "penalty": "H"},
    },
    "clstm": {
        "small":  {"context": 10, "hidden": 100, "lr": 1e-3, "lam": 0.1,
                    "lam_ridge": 1e-2, "max_iter": 20000},
        "medium": {"context": 10, "hidden": 100, "lr": 1e-3, "lam": 0.05,
                    "lam_ridge": 1e-2, "max_iter": 20000},
        "large":  {"context": 10, "hidden": 200, "lr": 1e-3, "lam": 0.01,
                    "lam_ridge": 1e-2, "max_iter": 20000},
    },
}


def get_hp(method, d):
    if d <= 15:
        size = "small"
    elif d <= 50:
        size = "medium"
    else:
        size = "large"
    return HP_DEFAULTS[method][size].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="cmlp",
                        help="cmlp, clstm, or all")
    parser.add_argument("--datasets", type=str, default="all",
                        help="comma-separated or 'all'")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="comma-separated seeds")
    parser.add_argument("--lam", type=float, default=None,
                        help="Override regularization strength")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Override max iterations")
    parser.add_argument("--hidden", type=int, default=None,
                        help="Override hidden size")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or DATA_DIR

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")

    methods = ["cmlp", "clstm"] if args.method == "all" else [args.method]
    datasets = ["VAR_d50", "Lorenz_F40", "DREAM3_d10", "DREAM3_d50", "DREAM3_d100"] \
        if args.datasets == "all" else [s.strip() for s in args.datasets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    all_entries = []

    for method in methods:
        for ds in datasets:
            print(f"\n=== {method} on {ds} ===")
            for seed in seeds:
                t0 = time.time()
                hp = get_hp(method, 50)  # default medium; overridden per ds below
                if args.lam is not None:
                    hp["lam"] = args.lam
                if args.max_iter is not None:
                    hp["max_iter"] = args.max_iter
                if args.hidden is not None:
                    hp["hidden"] = args.hidden if method == "clstm" else [args.hidden]

                try:
                    if ds == "VAR_d50":
                        x, gc = load_var_d50(seed, data_dir)
                        hp = get_hp(method, 50)
                    elif ds == "Lorenz_F40":
                        x, gc = load_lorenz_f40(seed, data_dir)
                        hp = get_hp(method, 10)
                    elif ds.startswith("DREAM3_d"):
                        d_val = int(ds.split("d")[1])
                        x, gc = load_dream3(d_val, seed)
                        hp = get_hp(method, d_val)
                    else:
                        print(f"  Unknown dataset: {ds}")
                        continue

                    gc_pred, metrics = train_and_evaluate(x, gc, method, device, **hp)
                    dt = time.time() - t0

                    entry = make_result_entry(
                        dataset=ds, method=method, seed=seed,
                        metrics=metrics,
                        config={"method": method, "data": ds, **hp},
                        runtime={"train_time_s": round(dt, 1), "device": str(device)},
                        provenance=make_provenance(
                            "experiments/run_neural_gc_baseline.py", source="rerun"),
                    )
                    all_entries.append(entry)
                    print(f"  seed={seed}: AUROC={metrics['auroc']:.4f} "
                          f"AUPRC={metrics['auprc']:.4f} time={dt:.1f}s")

                except Exception as e:
                    print(f"  FAIL seed={seed}: {e}")
                    import traceback
                    traceback.print_exc()

    if not all_entries:
        print("\nNo entries produced.")
        return

    # Merge with existing
    out_path = args.output or os.path.join(RESULTS_DIR, "neural_gc_results.json")
    existing_entries = []
    if os.path.exists(out_path):
        try:
            existing_entries = load_collection(out_path).get("results", [])
        except Exception:
            pass

    existing_keys = {(e["dataset"], e["method"], e["seed"]) for e in existing_entries}
    for e in all_entries:
        key = (e["dataset"], e["method"], e["seed"])
        if key in existing_keys:
            for i, old in enumerate(existing_entries):
                if (old["dataset"], old["method"], old["seed"]) == key:
                    existing_entries[i] = e
                    break
        else:
            existing_entries.append(e)

    collection = make_collection(
        existing_entries,
        description="Neural-GC (cMLP/cLSTM) baseline on available datasets",
    )
    collection["_audit"] = make_provenance(
        "experiments/run_neural_gc_baseline.py", source="rerun",
        extra={"datasets_covered": sorted(set(e["dataset"] for e in existing_entries))}
    )
    save_collection(collection, out_path)
    print(f"\nSaved {len(existing_entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
