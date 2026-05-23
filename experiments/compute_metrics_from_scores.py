"""Phase 2 v2 Step 2: Compute metrics from saved GC score matrices.

Reads score_registry.json, loads each saved GC score + ground truth pair,
computes all metrics, and builds migrated_all_v2.json and manifest.json.

All metrics in each entry are derived from the SAME saved GC score matrix.
This script can be re-run at any time from saved scores without GPU.

Usage:
    python experiments/compute_metrics_from_scores.py
    python experiments/compute_metrics_from_scores.py --registry results/scores/score_registry.json
"""

import numpy as np
import sys, os, json, time, argparse, hashlib
from collections import defaultdict

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
_src_dir = os.path.join(_PROJ_ROOT, "src")
if os.path.isdir(_src_dir):
    sys.path.insert(0, _src_dir)
from src.config import resolve_jrngc_path

_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)

try:
    from mamba_jrngc_pilot import compute_metrics
except ImportError:
    from src.mamba_jrngc_pilot import compute_metrics

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIGRATED_PATH = os.path.join(_PROJ_ROOT, "results", "raw", "migrated_all.json")
OUTPUT_PATH = os.path.join(_PROJ_ROOT, "results", "raw", "migrated_all_v2.json")
SCORES_DIR = os.path.join(_PROJ_ROOT, "results", "scores")
REGISTRY_PATH = os.path.join(SCORES_DIR, "score_registry.json")
MANIFEST_PATH = os.path.join(SCORES_DIR, "manifest.json")


def log(msg):
    print(msg, flush=True)


def compute_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_self_edge_removed(gc_pred, tolerance=1e-12):
    """Verify diagonal has been zeroed (self-edges removed)."""
    if gc_pred.ndim == 3:
        diag = gc_pred[:, :, 0].diagonal()  # check lag-0 diagonal
    else:
        diag = gc_pred.diagonal()
    return float(np.max(np.abs(diag))) < tolerance


def build_manifest_entry(score_entry, sha256_gc, sha256_gt, self_edge_ok, metrics, gc_pred):
    return {
        "dataset": score_entry["dataset"],
        "method": score_entry["method"],
        "seed": score_entry["seed"],
        "gc_score_path": score_entry["gc_score_path"],
        "gt_path": score_entry["gt_path"],
        "shape": list(gc_pred.shape) if hasattr(gc_pred, 'shape') else score_entry.get("shape"),
        "d": int(gc_pred.shape[0]) if hasattr(gc_pred, 'shape') and gc_pred.ndim >= 1 else score_entry.get("d"),
        "t": score_entry.get("t", None),
        "n_edges_true": score_entry.get("n_edges", None),
        "sha256_gc": sha256_gc,
        "sha256_gt": sha256_gt,
        "self_edge_removed": self_edge_ok,
        "mode": "lag0",  # compute_metrics uses lag-0 ground truth
        "source_script": "experiments/compute_metrics_from_scores.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics_summary": {
            "auroc": metrics.get("auroc"),
            "auprc": metrics.get("auprc"),
            "shd_topk": metrics.get("shd_topk"),
            "nshd_topk": metrics.get("nshd_topk"),
            "mcc_topk": metrics.get("mcc_topk"),
            "f1": metrics.get("f1"),
            "acc": metrics.get("acc"),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="v2 Step 2: Compute metrics from saved scores")
    parser.add_argument("--registry", type=str, default=REGISTRY_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--scores-dir", type=str, default=SCORES_DIR)
    args = parser.parse_args()

    # Load registry
    log(f"Loading registry: {args.registry}")
    with open(args.registry, "r") as f:
        registry = json.load(f)
    log(f"  {len(registry['scores'])} score entries")
    log(f"  deterministic training: {registry.get('deterministic', 'unknown')}")

    # Load original migrated_all.json for non-baseline/mamba entries and metadata
    log(f"Loading original entries: {MIGRATED_PATH}")
    with open(MIGRATED_PATH, "r") as f:
        collection = json.load(f)
    orig_entries = collection["results"]

    # Build lookup: (dataset, method, seed) -> original entry (for config etc.)
    orig_lookup = {}
    for e in orig_entries:
        key = (e["dataset"], e["method"], e["seed"])
        orig_lookup[key] = e

    # Build score registry lookup
    score_lookup = {}
    for s in registry["scores"]:
        key = (s["dataset"], s["method"], s["seed"])
        score_lookup[key] = s

    # Build new entries
    new_entries = []
    manifest_entries = []
    n_baseline_mamba = 0
    n_other = 0
    errors = []

    for e in orig_entries:
        key = (e["dataset"], e["method"], e["seed"])
        if e["method"] in ("baseline", "mamba"):
            # This should have a score file
            score_info = score_lookup.get(key)
            if score_info is None:
                log(f"  WARNING: No score for {key} — keeping original (incomplete)")
                new_entries.append(e)
                continue

            gc_path = os.path.join(args.scores_dir,
                                   os.path.basename(score_info["gc_score_path"]))
            gt_path = os.path.join(args.scores_dir,
                                   os.path.basename(score_info["gt_path"]))

            if not os.path.exists(gc_path):
                errors.append(f"MISSING GC: {gc_path}")
                log(f"  ERROR: {gc_path} not found")
                new_entries.append(e)
                continue
            if not os.path.exists(gt_path):
                errors.append(f"MISSING GT: {gt_path}")
                log(f"  ERROR: {gt_path} not found")
                new_entries.append(e)
                continue

            try:
                gc_pred = np.load(gc_path)
                gc_true = np.load(gt_path)
            except Exception as ex:
                errors.append(f"LOAD ERROR {gc_path}: {ex}")
                log(f"  ERROR loading {gc_path}: {ex}")
                new_entries.append(e)
                continue

            # Compute ALL metrics from saved score matrix
            try:
                metrics = compute_metrics(gc_true, gc_pred)
            except Exception as ex:
                errors.append(f"METRICS ERROR {key}: {ex}")
                log(f"  ERROR computing metrics for {key}: {ex}")
                new_entries.append(e)
                continue

            # Compute checksums
            sha256_gc = compute_sha256(gc_path)
            sha256_gt = compute_sha256(gt_path)
            self_edge_ok = check_self_edge_removed(gc_pred)

            # Build manifest entry
            manifest_entries.append(
                build_manifest_entry(score_info, sha256_gc, sha256_gt, self_edge_ok, metrics, gc_pred)
            )

            # Build result entry — ALL metrics from same GC score matrix
            new_entry = {
                "dataset": e["dataset"],
                "method": e["method"],
                "seed": e["seed"],
                "config": e.get("config", {}),
                "metrics": metrics,
                "artifacts": {
                    "gc_score_path": score_info["gc_score_path"],
                    "gt_path": score_info["gt_path"],
                },
                "provenance": {
                    "source": "canonical_v2",
                    "deterministic_training": registry.get("deterministic", False),
                    "metrics_from_saved_scores": True,
                    "self_edge_removed": self_edge_ok,
                    "sha256_gc": sha256_gc,
                    "sha256_gt": sha256_gt,
                    "score_shape": score_info["shape"],
                },
            }
            new_entries.append(new_entry)
            n_baseline_mamba += 1

            log(f"  {key[0]:20s} {key[1]:10s} seed={key[2]}  "
                f"auroc={metrics.get('auroc', 0):.4f}  "
                f"shd={metrics.get('shd_topk', 0)}  "
                f"self_edge_ok={self_edge_ok}")
        else:
            # Non-baseline/mamba: pass through unchanged
            new_entries.append(e)
            n_other += 1

    # ---- Process orphan scores (in registry but not in original migrated_all.json) ----
    # These come from seed expansion (e.g. Lorenz_F40/VAR_d50 seeds 5-9)
    existing_keys = set(orig_lookup.keys())
    n_orphan = 0
    for (ds, method, seed), score_info in sorted(score_lookup.items()):
        key = (ds, method, seed)
        if key in existing_keys:
            continue  # already processed above

        gc_path = os.path.join(args.scores_dir, os.path.basename(score_info["gc_score_path"]))
        gt_path = os.path.join(args.scores_dir, os.path.basename(score_info["gt_path"]))

        if not os.path.exists(gc_path) or not os.path.exists(gt_path):
            errors.append(f"MISSING ORPHAN: {gc_path}")
            continue

        try:
            gc_pred = np.load(gc_path)
            gc_true = np.load(gt_path)
            metrics = compute_metrics(gc_true, gc_pred)
        except Exception as ex:
            errors.append(f"ORPHAN METRICS ERROR {key}: {ex}")
            continue

        sha256_gc = compute_sha256(gc_path)
        sha256_gt = compute_sha256(gt_path)
        self_edge_ok = check_self_edge_removed(gc_pred)

        manifest_entries.append(
            build_manifest_entry(score_info, sha256_gc, sha256_gt, self_edge_ok, metrics, gc_pred)
        )

        new_entry = {
            "dataset": ds,
            "method": method,
            "seed": seed,
            "config": {},  # config unknown for orphans — use defaults from score
            "metrics": metrics,
            "artifacts": {
                "gc_score_path": score_info["gc_score_path"],
                "gt_path": score_info["gt_path"],
            },
            "provenance": {
                "source": "canonical_v2_seed_expansion",
                "deterministic_training": registry.get("deterministic", False),
                "metrics_from_saved_scores": True,
                "self_edge_removed": self_edge_ok,
                "sha256_gc": sha256_gc,
                "sha256_gt": sha256_gt,
            },
        }
        new_entries.append(new_entry)
        n_orphan += 1

        log(f"  {ds:20s} {method:10s} seed={seed}  "
            f"auroc={metrics.get('auroc', 0):.4f}  "
            f"shd={metrics.get('shd_topk', 0)}  [ORPHAN→NEW]")

    if n_orphan:
        log(f"\n  Added {n_orphan} orphan score entries as new result entries")

    if errors:
        log(f"\n  ERRORS ({len(errors)}):")
        for err in errors:
            log(f"    {err}")

    # Save manifest
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Canonical GC score matrices — all metrics derived from these files",
        "deterministic_training": registry.get("deterministic", False),
        "entries": manifest_entries,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"\nManifest: {MANIFEST_PATH} ({len(manifest_entries)} entries)")

    # Save migrated_all_v2.json
    collection["results"] = new_entries
    with open(args.output, "w") as f:
        json.dump(collection, f, indent=2, ensure_ascii=False)

    log(f"\n{'='*60}")
    log(f"Done: {n_baseline_mamba} baseline/mamba entries (all from saved scores)")
    log(f"      {n_orphan} new entries (seed expansion)")
    log(f"      {n_other} other entries (passthrough)")
    log(f"      {len(errors)} errors")
    log(f"Output: {args.output}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
