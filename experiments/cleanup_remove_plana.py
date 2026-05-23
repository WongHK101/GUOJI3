"""P0 Cleanup: Remove NSVAR_d50_PlanA from all canonical data files.

NSVAR_d50_PlanA was trained on the same data as NSVAR_d50 due to a data-path
bug in backfill_canonical_v2.py (load_nsvar always loads from nonstationary_var/,
never from nonstationary_var_planA/). This caused identical GC scores and SHA256
values. Since PlanA data was never properly generated, we remove it entirely.

Usage:
    python experiments/cleanup_remove_plana.py
"""

import json, os, sys

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

TARGET = "NSVAR_d50_PlanA"
MIGRATED_PATH = os.path.join(_PROJ_ROOT, "results", "raw", "migrated_all_v2.json")
MANIFEST_PATH = os.path.join(_PROJ_ROOT, "results", "scores", "manifest.json")
REGISTRY_PATH = os.path.join(_PROJ_ROOT, "results", "scores", "score_registry.json")
SCORES_DIR = os.path.join(_PROJ_ROOT, "results", "scores")


def fix_utf8_manifest():
    """Re-read and re-write manifest.json as strict UTF-8."""
    # Try multiple encodings
    content = None
    for enc in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']:
        try:
            with open(MANIFEST_PATH, 'rb') as f:
                raw = f.read()
            content = raw.decode(enc)
            # Validate by parsing
            json.loads(content)
            print(f"  manifest.json decoded as {enc}")
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue

    if content is None:
        # Fallback: read with errors='replace'
        with open(MANIFEST_PATH, 'rb') as f:
            raw = f.read()
        content = raw.decode('utf-8', errors='replace')
        print(f"  manifest.json decoded as utf-8 with replace")

    return json.loads(content)


def main():
    print("=== P0 Cleanup: Remove NSVAR_d50_PlanA ===\n")

    # 1. Fix and load manifest
    print("1. Loading manifest.json (fixing UTF-8)...")
    manifest = fix_utf8_manifest()
    n_before = len(manifest["entries"])
    manifest["entries"] = [e for e in manifest["entries"] if e["dataset"] != TARGET]
    n_removed = n_before - len(manifest["entries"])
    print(f"   Removed {n_removed} PlanA entries from manifest ({len(manifest['entries'])} remain)")

    # 2. Fix registry
    print("\n2. Loading score_registry.json...")
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)
    n_before = len(registry["scores"])
    registry["scores"] = [s for s in registry["scores"] if s["dataset"] != TARGET]
    n_removed = n_before - len(registry["scores"])
    print(f"   Removed {n_removed} PlanA entries from registry ({len(registry['scores'])} remain)")

    # 3. Fix migrated_all_v2.json + update audit
    print("\n3. Loading migrated_all_v2.json...")
    with open(MIGRATED_PATH, 'r', encoding='utf-8') as f:
        collection = json.load(f)
    n_before = len(collection["results"])
    collection["results"] = [e for e in collection["results"] if e["dataset"] != TARGET]
    n_removed = n_before - len(collection["results"])
    print(f"   Removed {n_removed} PlanA entries ({len(collection['results'])} remain)")

    # Fix top-level audit
    collection["_audit"] = {
        "script": "experiments/compute_metrics_from_scores.py + experiments/cleanup_remove_plana.py",
        "date": "2026-05-12",
        "n_entries": len(collection["results"]),
        "source": "hybrid_canonical_v2",
        "description": "Canonical v2 results. Baseline/mamba metrics from saved GC score matrices. Other methods from legacy or method-specific results.",
        "baseline_mamba_source": "saved_gc_score_matrices",
        "other_methods_source": "legacy_or_method_specific_results",
        "metrics_from_saved_scores_for": ["baseline", "mamba"],
        "score_manifest": "results/scores/manifest.json",
        "nsvar_d50_planA_removed": "2026-05-12 — removed due to duplicate SHA with NSVAR_d50 (data-path bug in backfill)"
    }
    print(f"   Updated top-level _audit")

    # 4. Remove PlanA .npy files
    print("\n4. Cleaning PlanA .npy files...")
    n_deleted = 0
    for fname in os.listdir(SCORES_DIR):
        if TARGET in fname and fname.endswith('.npy'):
            os.remove(os.path.join(SCORES_DIR, fname))
            n_deleted += 1
    print(f"   Deleted {n_deleted} PlanA .npy files")

    # 5. Save all fixed files
    print("\n5. Saving fixed files...")

    # manifest.json — strict UTF-8
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    # Verify
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        json.load(f)
    print(f"   manifest.json saved + verified UTF-8 ({len(manifest['entries'])} entries)")

    # registry
    with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    print(f"   score_registry.json saved ({len(registry['scores'])} entries)")

    # migrated_all_v2.json
    with open(MIGRATED_PATH, 'w', encoding='utf-8') as f:
        json.dump(collection, f, indent=2, ensure_ascii=False)
    print(f"   migrated_all_v2.json saved ({len(collection['results'])} entries)")

    print(f"\n=== DONE ===")
    print(f"NSVAR_d50_PlanA removed from all canonical files.")
    print(f"Next: re-run generate_appendix_tables.py, run_statistical_tests.py, generate_eligibility_table.py")


if __name__ == "__main__":
    main()
