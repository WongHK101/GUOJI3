"""PCMCI+ baseline: run on all available datasets, output unified schema.

Saves to results/raw/pcmci_results.json in canonical format.

Usage:
    python experiments/run_pcmci_baseline.py
    python experiments/run_pcmci_baseline.py --datasets var_d50,lorenz_f40,dream3
"""

import numpy as np
import sys, os, json, time, argparse

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path, resolve_data_dir
from src.schema import (make_result_entry, make_collection, save_collection,
                        make_provenance, load_collection, expected_results_dir)

_jrngc = resolve_jrngc_path()
if _jrngc and _jrngc not in sys.path:
    sys.path.insert(0, _jrngc)

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

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
    # MCC
    tp = int(np.sum((pr_bin == 1) & (gt == 1)))
    tn = int(np.sum((pr_bin == 0) & (gt == 0)))
    fp = int(np.sum((pr_bin == 1) & (gt == 0)))
    fn = int(np.sum((pr_bin == 0) & (gt == 1)))
    mcc_denom = np.sqrt(float(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0)))
    mcc = float(tp * tn - fp * fn) / max(mcc_denom, 1e-8)
    return {"auroc": float(auroc), "auprc": float(auprc), "shd_topk": shd,
            "nshd_topk": float(nshd), "mcc_topk": float(mcc),
            "f1": float(f1), "acc": float(acc), "n_edges_true": n_edges}


def run_pcmci(x, pc_alpha=0.05, tau_max=1):
    """x: (d, T) numpy array. Returns gc_pred: (d, d, 1), metrics dict."""
    d, T = x.shape
    df = pp.DataFrame(x.T)
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=df, cond_ind_test=parcorr, verbosity=0)
    res = pcmci.run_pcmci(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
    p_mat = res['p_matrix'][:, :, 1]
    gc_pred = (1 - p_mat)[:, :, np.newaxis]
    gc_pred = np.clip(gc_pred, 0, 1)
    return gc_pred


# ==================================================================
# Dataset runners
# ==================================================================

def run_var_d50():
    """Stationary VAR d=50 (3 seeds)."""
    entries = []
    config = {"data": "VAR_d50", "pc_alpha": 0.05, "tau_max": 1}
    base = os.path.join(DATA_DIR, "var", "num_nodes_50", "true_lag_5", "noise_scale_1")
    for seed in range(3):
        x = np.load(os.path.join(base, f"seed_{seed}", "_x.npy"))
        gc = np.load(os.path.join(base, f"seed_{seed}", "_gc.npy"))
        gc_pred = run_pcmci(x)
        met = _compute_metrics(gc, gc_pred)
        entries.append(make_result_entry(
            dataset="VAR_d50", method="pcmci", seed=seed,
            metrics=met, config=config, runtime={},
            provenance=make_provenance("experiments/run_pcmci_baseline.py", source="rerun"),
        ))
    return entries


def run_lorenz_f40():
    """Lorenz-96 F=40 (3 seeds)."""
    entries = []
    config = {"data": "Lorenz_F40", "pc_alpha": 0.05, "tau_max": 1}
    base = os.path.join(DATA_DIR, "lorenz", "num_nodes_10", "F_40")
    for seed in range(3):
        x = np.load(os.path.join(base, f"seed_{seed}", "_x.npy"))
        gc = np.load(os.path.join(base, f"seed_{seed}", "_gc.npy"))
        gc_pred = run_pcmci(x)
        met = _compute_metrics(gc, gc_pred)
        entries.append(make_result_entry(
            dataset="Lorenz_F40", method="pcmci", seed=seed,
            metrics=met, config=config, runtime={},
            provenance=make_provenance("experiments/run_pcmci_baseline.py", source="rerun"),
        ))
    return entries


def run_dream3():
    """DREAM3 d=10/50/100 (3 subjects each)."""
    # dream3_trajectories expects data/ relative to JRNGC root
    _cwd = os.getcwd()
    if _jrngc:
        os.chdir(_jrngc)
    try:
        from tgc.data.dream3 import dream3_trajectories
        entries = []
        for d in [10, 50, 100]:
            config = {"data": f"DREAM3_d{d}", "pc_alpha": 0.05, "tau_max": 1}
            for subject in range(3):
                try:
                    x, _, gc = dream3_trajectories(d=d, subject=subject)
                except Exception as e:
                    print(f"  SKIP DREAM3 d={d} s={subject}: {e}")
                    continue
                if x.ndim == 3:
                    x = x[0]
                gc_pred = run_pcmci(x)
                met = _compute_metrics(gc, gc_pred)
                entries.append(make_result_entry(
                    dataset=f"DREAM3_d{d}", method="pcmci", seed=subject,
                    metrics=met, config=config, runtime={},
                    provenance=make_provenance("experiments/run_pcmci_baseline.py", source="rerun"),
                ))
    finally:
        os.chdir(_cwd)
    return entries


# ==================================================================
# Additional datasets (cloud)
# ==================================================================

def run_causaltime(ds_name):
    """CausalTime single-seed datasets: medical, traffic, pm25."""
    subset = ds_name.replace("causaltime_", "")
    entries = []
    config = {"data": ds_name, "pc_alpha": 0.05, "tau_max": 1}
    p = os.path.join(DATA_DIR, "causaltime", subset)
    if not os.path.isdir(p):
        print(f"  SKIP {ds_name}: data not found at {p}")
        return entries
    try:
        x = np.load(os.path.join(p, "_x.npy"))
        gc = np.load(os.path.join(p, "_gc.npy"))
        gc_pred = run_pcmci(x)
        met = _compute_metrics(gc, gc_pred)
        entries.append(make_result_entry(
            dataset=ds_name, method="pcmci", seed=0,
            metrics=met, config=config, runtime={},
            provenance=make_provenance("experiments/run_pcmci_baseline.py", source="rerun",
                                       extra={"note": "single seed"}),
        ))
    except Exception as e:
        print(f"  SKIP {ds_name}: {e}")
    return entries


def run_fmri():
    """fMRI d=15 (5 subjects)."""
    entries = []
    config = {"data": "fMRI_d15", "pc_alpha": 0.05, "tau_max": 1}
    base = os.path.join(DATA_DIR, "fmri", "num_nodes_15")
    if not os.path.isdir(base):
        print(f"  SKIP fMRI: data not found at {base}")
        return entries
    for subject in range(5):
        p = os.path.join(base, f"subject_{subject}", "seed_0")
        try:
            x = np.load(os.path.join(p, "_x.npy"))
            gc = np.load(os.path.join(p, "_gc.npy"))
            gc_pred = run_pcmci(x)
            met = _compute_metrics(gc, gc_pred)
            entries.append(make_result_entry(
                dataset="fMRI_d15", method="pcmci", seed=subject,
                metrics=met, config=config, runtime={},
                provenance=make_provenance("experiments/run_pcmci_baseline.py", source="rerun"),
            ))
        except Exception as e:
            print(f"  SKIP fMRI subject_{subject}: {e}")
    return entries


# ==================================================================
# Main
# ==================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="all",
                        help="comma-separated or 'all'")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    if args.data_dir:
        global DATA_DIR
        DATA_DIR = args.data_dir

    if args.datasets == "all":
        targets = ["var_d50", "lorenz_f40", "dream3",
                   "causaltime_medical", "causaltime_traffic", "causaltime_pm25",
                   "fmri"]
    else:
        targets = [s.strip() for s in args.datasets.split(",")]

    all_entries = []

    for ds in targets:
        print(f"--- PCMCI+ {ds} ---")
        t0 = time.time()
        if ds == "var_d50":
            entries = run_var_d50()
        elif ds == "lorenz_f40":
            entries = run_lorenz_f40()
        elif ds == "dream3":
            entries = run_dream3()
        elif ds in ("causaltime_medical", "causaltime_traffic", "causaltime_pm25"):
            entries = run_causaltime(ds)
        elif ds == "fmri":
            entries = run_fmri()
        else:
            print(f"Unknown dataset: {ds}")
            continue
        for e in entries:
            print(f"  {e['dataset']} seed={e['seed']}: AUROC={e['metrics']['auroc']:.4f} "
                  f"AUPRC={e['metrics'].get('auprc','?'):.4f}")
        all_entries.extend(entries)
        print(f"  Time: {time.time()-t0:.1f}s")

    # Load existing collection if any, merge
    out_path = args.output or os.path.join(RESULTS_DIR, "pcmci_results.json")
    existing_entries = []
    if os.path.exists(out_path):
        try:
            existing = load_collection(out_path)
            existing_entries = existing.get("results", [])
        except Exception:
            pass
    # Merge: update entries for same (dataset, method, seed), append others
    existing_keys = {(e["dataset"], e["method"], e["seed"]) for e in existing_entries}
    for e in all_entries:
        key = (e["dataset"], e["method"], e["seed"])
        if key in existing_keys:
            # Replace old entry
            for i, old in enumerate(existing_entries):
                if (old["dataset"], old["method"], old["seed"]) == key:
                    existing_entries[i] = e
                    break
        else:
            existing_entries.append(e)
    collection = make_collection(
        existing_entries,
        description="PCMCI+ baseline on available datasets",
    )
    collection["_audit"] = make_provenance(
        "experiments/run_pcmci_baseline.py", source="rerun",
        extra={"datasets_covered": sorted(set(e["dataset"] for e in existing_entries))}
    )
    save_collection(collection, out_path)
    print(f"\nSaved {len(existing_entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
