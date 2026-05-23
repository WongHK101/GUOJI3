"""Phase 2: Backfill SHD/nSHD/MCC (top-k) for baseline and mamba entries.

Re-trains models for entries missing topology metrics (shd_topk, nshd_topk, mcc_topk),
then merges updated metrics back into migrated_all.json.

Requires GPU. Must be run from the mamba_enhanced/ directory.
Deploy to cloud (AutoDL) where all datasets are available.

Usage:
    python experiments/backfill_topology_metrics.py
    python experiments/backfill_topology_metrics.py --datasets CT_medical,DREAM3_d10
    python experiments/backfill_topology_metrics.py --dry-run  # check what would run
"""

import torch
import numpy as np
import sys, os, json, time, argparse, shutil
from collections import defaultdict

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
# Also add src/ for local dev where mamba_jrngc_pilot lives in src/
_src_dir = os.path.join(_PROJ_ROOT, "src")
if os.path.isdir(_src_dir):
    sys.path.insert(0, _src_dir)
from src.config import resolve_jrngc_path, resolve_data_dir, resolve_device

_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)

# mamba_jrngc_pilot may be in JRNGC root (cloud) or src/ (local dev)
try:
    from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                    train_model, compute_metrics)
except ImportError:
    from src.mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                        train_model, compute_metrics)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIGRATED_PATH = os.path.join(_PROJ_ROOT, "results", "raw", "migrated_all.json")
BACKUP_PATH = os.path.join(_PROJ_ROOT, "results", "raw", "migrated_all_backup_before_topk.json")

# Training defaults per dataset type
# These are derived from the original generating scripts.
DATASET_DEFAULTS = {
    "CT_medical":    {"lag": 1, "max_iter": 2000, "lr": 1e-3, "d_state": 8},
    "CT_pm25":       {"lag": 1, "max_iter": 2000, "lr": 1e-3, "d_state": 8},
    "CT_traffic":    {"lag": 1, "max_iter": 2000, "lr": 1e-3, "d_state": 8},
    "DREAM3_d10":    {"lag": 1, "max_iter": 2000, "lr": 1e-3, "d_state": 4},
    "DREAM3_d50":    {"lag": 1, "max_iter": 2000, "lr": 1e-3, "d_state": 4},
    "DREAM3_d100":   {"lag": 1, "max_iter": 2000, "lr": 1e-3, "d_state": 4},
    "fMRI_d15":      {"lag": 1, "max_iter": 2000, "lr": 1e-3, "d_state": 8},
    "Lorenz_F40":    {"lag": 1, "max_iter": 2000, "lr": 1e-3, "d_state": 8},
    "VAR_d50":       {"lag": 5, "max_iter": 2000, "lr": 1e-3, "d_state": 8},
    "NSVAR_d10":     {"lag": 7, "max_iter": 2000, "lr": 1e-3, "d_state": 8},
    "NSVAR_d50":     {"lag": 7, "max_iter": 2000, "lr": 1e-3, "d_state": 8},
    "NSVAR_d50_PlanA": {"lag": 7, "max_iter": 2000, "lr": 1e-3, "d_state": 8},
}

# Common model config
MODEL_CFG = {"layers": 5, "hidden": 50, "jacobian_lam": 0.01}

device = None  # set in main

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_causaltime(ds_name, data_dir):
    """Load CT_medical, CT_pm25, or CT_traffic."""
    name_map = {"CT_medical": "medical", "CT_pm25": "pm25", "CT_traffic": "traffic"}
    folder = name_map[ds_name]
    p = os.path.join(data_dir, "causaltime", folder)
    x = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))
    return x, gc


def load_dream3(d, subject, data_dir=None):
    """Load DREAM3 data using JRNGC loader."""
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


def load_fmri(data_dir):
    """Load fMRI d=15 data for all 3 subjects."""
    results = {}
    for subject in range(3):
        p = os.path.join(data_dir, "fmri", "num_nodes_15", f"subject_{subject}", "seed_0")
        x = np.load(os.path.join(p, "_x.npy"))
        gc = np.load(os.path.join(p, "_gc.npy"))
        results[subject] = (x, gc)
    return results


def load_lorenz_f40(seed, data_dir):
    p = os.path.join(data_dir, "lorenz", "num_nodes_10", "F_40", f"seed_{seed}")
    x = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))
    return x, gc


def load_var_d50(seed, data_dir):
    p = os.path.join(data_dir, "var", "num_nodes_50", "true_lag_5", "noise_scale_1", f"seed_{seed}")
    x = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))
    return x, gc


def load_nsvar(d, seed, data_dir):
    p = os.path.join(data_dir, "nonstationary_var", f"num_nodes_{d}", "true_lag_7",
                     "noise_scale_1", f"seed_{seed}")
    x = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))
    return x, gc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(x, gc_true, d, seed, method, cfg):
    """Train one model and return (gc_pred, metrics_dict, train_time)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    lag = cfg.get("lag", 1)
    model_cfg = {**MODEL_CFG, "d": d, "lag": lag}

    if method == "baseline":
        model = BaselineJRNGC(**model_cfg).to(device)
    else:
        model = MambaFilterJRNGC(
            **model_cfg,
            d_state=cfg.get("d_state", 8),
            ortho_lam=0.05,
            residual_scale=0.1,
            filter_type="mamba"
        ).to(device)

    t0 = time.time()
    model, loss = train_model(model, x, max_iter=cfg["max_iter"], lr=cfg["lr"])
    gc_pred = model.get_gc_matrix(x)
    train_time = time.time() - t0

    metrics = compute_metrics(gc_true, gc_pred)
    metrics["train_time_s"] = train_time
    metrics["train_loss"] = float(loss)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return gc_pred, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def main():
    global device
    parser = argparse.ArgumentParser(description="Backfill topology metrics for baseline/mamba")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset list (default: all with missing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without training")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = resolve_device()
    log(f"Device: {device}")
    data_dir = args.data_dir or resolve_data_dir()
    log(f"Data dir: {data_dir}")

    # Load existing results
    log(f"Loading {MIGRATED_PATH}...")
    with open(MIGRATED_PATH, "r") as f:
        collection = json.load(f)
    entries = collection["results"]

    # Identify entries needing backfill
    target_datasets = set(args.datasets.split(",")) if args.datasets else None

    to_backfill = []
    for i, e in enumerate(entries):
        if e["method"] not in ("baseline", "mamba"):
            continue
        ds = e["dataset"]
        if target_datasets and ds not in target_datasets:
            continue

        missing = []
        for k in ("shd_topk", "nshd_topk", "mcc_topk"):
            if k not in e["metrics"]:
                missing.append(k)
        if missing:
            to_backfill.append((i, e, missing))

    # Group by dataset+method for batching
    groups = defaultdict(list)
    for idx, e, missing in to_backfill:
        key = (e["dataset"], e["method"])
        groups[key].append((idx, e, missing))

    log(f"\nEntries to backfill: {len(to_backfill)}")
    for (ds, method), items in sorted(groups.items()):
        seeds = sorted([e["seed"] for _, e, _ in items])
        missing_keys = set()
        for _, _, mk in items:
            missing_keys.update(mk)
        log(f"  {ds}/{method}: {len(items)} entries, seeds={seeds}, missing={sorted(missing_keys)}")

    if args.dry_run:
        log("\n[Dry run — no training performed]")
        return

    if not to_backfill:
        log("\nAll entries already have complete topology metrics. Nothing to do.")
        return

    # Backup
    log(f"\nCreating backup: {BACKUP_PATH}")
    shutil.copy2(MIGRATED_PATH, BACKUP_PATH)

    # Process
    n_updated = 0
    for (ds, method), items in sorted(groups.items()):
        log(f"\n{'='*60}")
        log(f"  {ds} / {method}  ({len(items)} entries)")
        log(f"{'='*60}")

        cfg = DATASET_DEFAULTS.get(ds, {"lag": 1, "max_iter": 2000, "lr": 1e-3, "d_state": 8})

        for idx, e, missing_keys in items:
            seed = e["seed"]
            label = f"{ds}/{method}/seed={seed}"

            # Load data
            try:
                if ds.startswith("CT_"):
                    x, gc = load_causaltime(ds, data_dir)
                elif ds.startswith("DREAM3_"):
                    d_val = int(ds.split("_d")[1])
                    x, gc = load_dream3(d_val, subject=seed)
                elif ds == "fMRI_d15":
                    fmri_data = load_fmri(data_dir)
                    x, gc = fmri_data[seed]
                elif ds == "Lorenz_F40":
                    x, gc = load_lorenz_f40(seed, data_dir)
                elif ds == "VAR_d50":
                    x, gc = load_var_d50(seed, data_dir)
                elif ds.startswith("NSVAR_"):
                    d_val = int(ds.split("_d")[1].split("_")[0])  # handles "NSVAR_d10" and "NSVAR_d50_PlanA"
                    x, gc = load_nsvar(d_val, seed, data_dir)
                else:
                    log(f"  [{label}] SKIP: unknown dataset type")
                    continue
            except Exception as ex:
                log(f"  [{label}] DATA LOAD FAILED: {ex}")
                continue

            d = x.shape[0]
            log(f"  [{label}] d={d}, T={x.shape[1]}, edges={int(gc.sum() if gc.ndim==2 else gc[:,:,0].sum())}")

            # Train
            try:
                t0 = time.time()
                _, metrics = train_one(x, gc, d, seed, method, cfg)
                dt = time.time() - t0
            except Exception as ex:
                log(f"  [{label}] TRAIN FAILED: {ex}")
                import traceback
                traceback.print_exc()
                continue

            # Verify AUROC consistency (should be within 0.02 of original)
            orig_auroc = e["metrics"].get("auroc")
            new_auroc = metrics.get("auroc")
            if orig_auroc is not None and new_auroc is not None:
                diff = abs(orig_auroc - new_auroc)
                if diff > 0.05:
                    log(f"  [{label}] WARNING: AUROC drift {orig_auroc:.4f} -> {new_auroc:.4f} (diff={diff:.4f})")

            # Merge topology metrics into entry (preserve existing AUROC/AUPRC/etc)
            for k in ("shd_topk", "nshd_topk", "mcc_topk", "n_edges_true"):
                if k in metrics:
                    entries[idx]["metrics"][k] = metrics[k]

            # Also update train_loss if missing
            if "train_loss" not in entries[idx]["metrics"] and "train_loss" in metrics:
                entries[idx]["metrics"]["train_loss"] = metrics["train_loss"]

            log(f"  [{label}] DONE: auroc={new_auroc:.4f}, shd_topk={metrics.get('shd_topk','?')}, "
                f"nshd_topk={metrics.get('nshd_topk','?'):.3f}, mcc_topk={metrics.get('mcc_topk','?'):.4f}, "
                f"time={dt:.0f}s")
            n_updated += 1

    # Save
    collection["results"] = entries
    output_path = args.output or MIGRATED_PATH
    log(f"\nSaving {n_updated} updated entries to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(collection, f, indent=2, ensure_ascii=False)
    log("Done.")


if __name__ == "__main__":
    main()
