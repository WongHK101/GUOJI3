"""Phase 2 v2 Step 1: Train models and save GC score matrices to disk.

Trains baseline/mamba for all datasets, saves GC scores + ground truth to
results/scores/, and writes a score registry for downstream metric computation.
Metrics are NOT saved to migrated_all.json — that is done separately by
compute_metrics_from_scores.py.

Usage:
    python experiments/backfill_canonical_v2.py
    python experiments/backfill_canonical_v2.py --datasets Lorenz_F40,VAR_d50 --expand-seeds
    python experiments/backfill_canonical_v2.py --deterministic
    python experiments/backfill_canonical_v2.py --dry-run
"""

import random
import torch
import numpy as np
import sys, os, json, time, argparse, shutil, hashlib
from collections import defaultdict

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
_src_dir = os.path.join(_PROJ_ROOT, "src")
if os.path.isdir(_src_dir):
    sys.path.insert(0, _src_dir)
from src.config import resolve_jrngc_path, resolve_data_dir, resolve_device

_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)

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
SCORES_DIR = os.path.join(_PROJ_ROOT, "results", "scores")
REGISTRY_PATH = os.path.join(SCORES_DIR, "score_registry.json")

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
    "NSVAR_d50":     {"lag": 14, "max_iter": 2000, "lr": 1e-3, "d_state": 8},
    "NSVAR_d50_PlanA": {"lag": 14, "max_iter": 2000, "lr": 1e-3, "d_state": 8},
}

MODEL_CFG = {"layers": 5, "hidden": 50, "jacobian_lam": 0.01}
device = None


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data loaders (same as original)
# ---------------------------------------------------------------------------

def load_causaltime(ds_name, data_dir):
    name_map = {"CT_medical": "medical", "CT_pm25": "pm25", "CT_traffic": "traffic"}
    folder = name_map[ds_name]
    p = os.path.join(data_dir, "causaltime", folder)
    return np.load(os.path.join(p, "_x.npy")), np.load(os.path.join(p, "_gc.npy"))


def load_dream3(d, subject, data_dir=None):
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
    results = {}
    for subject in range(3):
        p = os.path.join(data_dir, "fmri", "num_nodes_15", f"subject_{subject}", "seed_0")
        results[subject] = (np.load(os.path.join(p, "_x.npy")),
                            np.load(os.path.join(p, "_gc.npy")))
    return results


def load_lorenz_f40(seed, data_dir):
    p = os.path.join(data_dir, "lorenz", "num_nodes_10", "F_40", f"seed_{seed}")
    return np.load(os.path.join(p, "_x.npy")), np.load(os.path.join(p, "_gc.npy"))


def load_var_d50(seed, data_dir):
    p = os.path.join(data_dir, "var", "num_nodes_50", "true_lag_5", "noise_scale_1", f"seed_{seed}")
    return np.load(os.path.join(p, "_x.npy")), np.load(os.path.join(p, "_gc.npy"))


def load_nsvar(d, seed, data_dir):
    nsvar_dir = os.path.join(_PROJ_ROOT, "data", "nonstationary_var")
    if not os.path.isdir(nsvar_dir):
        nsvar_dir = os.path.join(data_dir, "nonstationary_var")
    lag = 7 if d <= 10 else 14
    p = os.path.join(nsvar_dir, f"num_nodes_{d}", f"true_lag_{lag}",
                     "noise_scale_1", f"seed_{seed}")
    return np.load(os.path.join(p, "_x.npy")), np.load(os.path.join(p, "_gc.npy"))


def load_data(ds, seed, data_dir):
    if ds.startswith("CT_"):
        return load_causaltime(ds, data_dir)
    elif ds.startswith("DREAM3_"):
        return load_dream3(int(ds.split("_d")[1]), subject=seed)
    elif ds == "fMRI_d15":
        return load_fmri(data_dir)[seed]
    elif ds == "Lorenz_F40":
        return load_lorenz_f40(seed, data_dir)
    elif ds == "VAR_d50":
        return load_var_d50(seed, data_dir)
    elif ds.startswith("NSVAR_"):
        return load_nsvar(int(ds.split("_d")[1].split("_")[0]), seed, data_dir)
    raise ValueError(f"Unknown dataset: {ds}")


# ---------------------------------------------------------------------------
# Training (scores only, no metrics stored)
# ---------------------------------------------------------------------------

def train_and_save(x, gc_true, d, seed, method, cfg, deterministic=False):
    """Train model and save GC score matrix + ground truth to disk.
    Returns (gc_pred, metrics_dict) for logging only."""
    set_seed(seed, deterministic=deterministic)

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

    # Compute metrics for logging (NOT saved to entries — compute_metrics_from_scores.py
    # is the canonical source)
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


def compute_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_entry(ds, method, seed, config):
    return {"dataset": ds, "method": method, "seed": seed, "config": config, "metrics": {}}


def main():
    global device
    parser = argparse.ArgumentParser(description="v2 Step 1: Train + save GC scores")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--expand-seeds", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable CUDNN deterministic mode (3-5x slower)")
    args = parser.parse_args()

    device = resolve_device()
    log(f"Device: {device}")
    data_dir = args.data_dir or resolve_data_dir()
    log(f"Data dir: {data_dir}")

    with open(MIGRATED_PATH, "r") as f:
        collection = json.load(f)
    entries = collection["results"]

    target_datasets = set(args.datasets.split(",")) if args.datasets else None

    # Build existing seeds map
    existing_seeds_map = defaultdict(set)
    for e in entries:
        if e["method"] in ("baseline", "mamba"):
            existing_seeds_map[(e["dataset"], e["method"])].add(e["seed"])

    # Build work list
    work_items = []
    for i, e in enumerate(entries):
        if e["method"] not in ("baseline", "mamba"):
            continue
        ds = e["dataset"]
        if target_datasets and ds not in target_datasets:
            continue
        work_items.append(("existing", i, e))

    if args.expand_seeds and target_datasets:
        for ds in target_datasets:
            for method in ("baseline", "mamba"):
                existing = existing_seeds_map.get((ds, method), set())
                cfg = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS["CT_medical"])
                for seed in range(10):
                    if seed not in existing:
                        work_items.append(("new", None, build_entry(ds, method, seed, cfg)))
                        log(f"  [EXPAND] {ds}/{method}/seed={seed} (new)")

    groups = defaultdict(list)
    for item_type, idx, e in work_items:
        groups[(e["dataset"], e["method"])].append((item_type, idx, e))

    n_new = sum(1 for t, _, _ in work_items if t == "new")
    log(f"\nEntries to process: {len(work_items)} ({n_new} new)")
    for (ds, method), items in sorted(groups.items()):
        seeds = sorted([e["seed"] for _, _, e in items])
        log(f"  {ds}/{method}: {len(items)} entries, seeds={seeds}")

    if args.dry_run:
        log("\n[Dry run — no training performed]")
        return

    if not work_items:
        log("\nNo entries to process.")
        return

    # Backup original
    shutil.copy2(MIGRATED_PATH, os.path.join(_PROJ_ROOT, "results", "raw",
                                              "migrated_all_backup_before_v2.json"))

    os.makedirs(SCORES_DIR, exist_ok=True)
    registry = {"scores": [], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "deterministic": args.deterministic}

    n_processed = 0
    for (ds, method), items in sorted(groups.items()):
        log(f"\n{'='*60}")
        log(f"  {ds} / {method}  ({len(items)} entries)")
        log(f"{'='*60}")

        cfg = DATASET_DEFAULTS.get(ds, {"lag": 1, "max_iter": 2000, "lr": 1e-3, "d_state": 8})

        for item_type, idx, e in items:
            seed = e["seed"]
            label = f"{ds}/{method}/seed={seed}"

            try:
                x, gc = load_data(ds, seed, data_dir)
            except Exception as ex:
                log(f"  [{label}] DATA LOAD FAILED: {ex}")
                continue

            d = x.shape[0]
            n_edges = int(gc.sum() if gc.ndim == 2 else gc[:, :, 0].sum())
            log(f"  [{label}] d={d}, T={x.shape[1]}, edges={n_edges}")

            try:
                t0 = time.time()
                gc_pred, metrics = train_and_save(x, gc, d, seed, method, cfg,
                                                  deterministic=args.deterministic)
                dt = time.time() - t0
            except Exception as ex:
                log(f"  [{label}] TRAIN FAILED: {ex}")
                import traceback
                traceback.print_exc()
                continue

            # Save GC score matrix and ground truth
            gc_path = os.path.join(SCORES_DIR, f"{ds}_{method}_seed{seed}_gc.npy")
            gt_path = os.path.join(SCORES_DIR, f"{ds}_{method}_seed{seed}_gt.npy")
            np.save(gc_path, gc_pred)
            np.save(gt_path, gc)

            score_shape = list(gc_pred.shape) if hasattr(gc_pred, 'shape') else None

            # Record in registry
            registry["scores"].append({
                "dataset": ds,
                "method": method,
                "seed": seed,
                "gc_score_path": f"results/scores/{ds}_{method}_seed{seed}_gc.npy",
                "gt_path": f"results/scores/{ds}_{method}_seed{seed}_gt.npy",
                "d": d,
                "shape": score_shape,
                "t": int(x.shape[1]),
                "n_edges": n_edges,
                "item_type": item_type,
            })

            log(f"  [{label}] DONE: auroc={metrics.get('auroc', '?'):.4f}, "
                f"shd_topk={metrics.get('shd_topk', '?')}, "
                f"time={dt:.0f}s  [scores saved]")
            n_processed += 1

    # Save registry
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    log(f"\n{'='*60}")
    log(f"Processed {n_processed} entries. Scores saved to {SCORES_DIR}/")
    log(f"Registry: {REGISTRY_PATH}")
    log(f"Next: run experiments/compute_metrics_from_scores.py")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
