"""Migrate legacy results/raw/ JSON files to unified schema.

Reads all legacy formats, normalizes method/dataset names, outputs unified entries.
Produces: results/raw/migrated_all.json

Usage:
    python experiments/migrate_legacy_results.py
    python experiments/migrate_legacy_results.py --output results/raw/migrated_all.json
"""

import json, os, sys, argparse
from collections import defaultdict

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from src.schema import (make_result_entry, make_collection, save_collection,
                        make_provenance, expected_results_dir)

RESULTS_DIR = expected_results_dir()

# Normalize method names
METHOD_MAP = {
    "baseline": "baseline",
    "A_baseline": "baseline",
    "mamba": "mamba",
    "mamba_v0": "mamba",
    "F_ds8_fix": "mamba",
    "F_ds8_fixAB": "mamba",
    "F_ds8_OLD": "mamba",
    "tcn": "tcn",
    "cmlp": "cmlp",
    "clstm": "clstm",
    "pcmci": "pcmci",
}

# Normalize dataset names
DATASET_MAP = {
    "VAR_d50": "VAR_d50",
    "Lorenz_F40": "Lorenz_F40",
    "NSVAR_d10": "NSVAR_d10",
    "NSVAR_d50": "NSVAR_d50",
    "NSVAR_d50_PlanA": "NSVAR_d50_PlanA",
    "CT_traffic": "CT_traffic",
    "CT_medical": "CT_medical",
    "CT_pm25": "CT_pm25",
    "fMRI_d15": "fMRI_d15",
    "DREAM3_d10": "DREAM3_d10",
    "DREAM3_d50": "DREAM3_d50",
    "DREAM3_d100": "DREAM3_d100",
    "traffic": "CT_traffic",
    "medical": "CT_medical",
    "pm25": "CT_pm25",
    "causaltime_traffic": "CT_traffic",
    "causaltime_medical": "CT_medical",
    "causaltime_pm25": "CT_pm25",
}


def norm_method(m):
    return METHOD_MAP.get(m, m)


def norm_dataset(d):
    return DATASET_MAP.get(d, d)


def extract_metrics(raw, metric_map=None):
    """Extract known metrics from a legacy raw dict. Normalizes to unified schema names."""
    known = {
        "auroc": "auroc", "auprc": "auprc", "f1": "f1", "acc": "acc",
        "shd": "shd_topk", "nshd": "nshd_topk", "mcc": "mcc_topk",
        "n_edges_true": "n_edges_true", "train_loss": "train_loss",
        "train_time": "train_time_s",
    }
    out = {}
    for old_k, new_k in known.items():
        if old_k in raw:
            v = raw[old_k]
            if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v)):
                out[new_k] = float(v) if isinstance(v, (int, float)) else v
    return out


def migrate_seed_top_level(data, dataset, methods, source="rerun"):
    """Format: {seed_0: {method: {metrics}}, seed_1: ...} or {subj_0: ...}"""
    entries = []
    for key, val in data.items():
        seed = None
        if key.startswith("seed_"):
            seed = int(key.split("_")[1])
        elif key.startswith("subj_"):
            seed = int(key.split("_")[1])
        if seed is None:
            continue
        if isinstance(val, dict):
            for m, met_raw in val.items():
                if not isinstance(met_raw, dict):
                    continue
                m_norm = norm_method(m)
                metrics = extract_metrics(met_raw)
                if "auroc" not in metrics:
                    continue
                entries.append(make_result_entry(
                    dataset=dataset, method=m_norm, seed=seed,
                    metrics=metrics,
                    config={"data": dataset},
                    provenance=make_provenance("experiments/migrate_legacy_results.py",
                                               source=source),
                ))
    return entries


def migrate_dataset_top_level(data, default_dataset="", source="rerun"):
    """Format: {dataset: {seed_N: {method: {metrics}}}}"""
    entries = []
    for ds_key, ds_val in data.items():
        if not isinstance(ds_val, dict):
            continue
        ds = norm_dataset(ds_key) or ds_key
        for seed_key, seed_val in ds_val.items():
            if not isinstance(seed_val, dict):
                continue
            if seed_key.startswith("seed_"):
                seed = int(seed_key.split("_")[1])
            elif seed_key.startswith("subj_"):
                seed = int(seed_key.split("_")[1])
            else:
                continue
            for m, met_raw in seed_val.items():
                if not isinstance(met_raw, dict):
                    continue
                m_norm = norm_method(m)
                metrics = extract_metrics(met_raw)
                if "auroc" not in metrics:
                    continue
                entries.append(make_result_entry(
                    dataset=ds, method=m_norm, seed=seed,
                    metrics=metrics,
                    config={"data": ds},
                    provenance=make_provenance("experiments/migrate_legacy_results.py",
                                               source=source),
                ))
    return entries


def migrate_flat_array_format(data, methods, datasets, source="legacy"):
    """Format: {dataset: [array_of_values_for_each_method], ...}
    Values are raw AUROC arrays (single float per entry).
    """
    entries = []
    for ds, vals in data.items():
        if ds.startswith("_") or not isinstance(vals, list):
            continue
        for i, v in enumerate(vals):
            if i >= len(methods):
                break
            if isinstance(v, (int, float)):
                entries.append(make_result_entry(
                    dataset=norm_dataset(ds), method=norm_method(methods[i]), seed=i,
                    metrics={"auroc": float(v), "auprc": 0.0},
                    config={"data": ds},
                    provenance=make_provenance("experiments/migrate_legacy_results.py",
                                               source=source,
                                               extra={"note": "AUROC only, no per-seed data"}),
                ))
    return entries


def migrate_single_result(data, dataset, method, source="legacy"):
    """Single flat result dict."""
    metrics = extract_metrics(data)
    if "auroc" not in metrics:
        return []
    return [make_result_entry(
        dataset=dataset, method=norm_method(method), seed=0,
        metrics=metrics,
        config={"data": dataset},
        provenance=make_provenance("experiments/migrate_legacy_results.py",
                                   source=source),
    )]


def try_migration(path, filename):
    """Try to migrate a legacy file. Returns list of unified entries."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Skip already-unified files
    if "results" in data and isinstance(data["results"], list):
        return []  # already migrated

    # Skip special files
    if filename in ["factorial_ablation_canonical.json", "factorial_stat_tests.json",
                    "mask_supplement_results.json", "theory_verification_results.json",
                    "interaction_ablation_results.json"]:
        return []

    entries = []

    # consolidated_all.json / final_consolidated.json: flat array format
    if filename in ["consolidated_all.json", "final_consolidated.json"]:
        methods = list(data.get("_methods", []))
        # These files have dataset -> [auroc_for_method_0, auroc_for_method_1, ...]
        # But they also store _methods key. If not present, skip.
        datasets_in_file = [k for k in data if not k.startswith("_")]
        for ds in datasets_in_file:
            if not isinstance(data[ds], list):
                continue
            for i, v in enumerate(data[ds]):
                if i >= len(methods):
                    break
                m = methods[i]
                entries.append(make_result_entry(
                    dataset=norm_dataset(ds), method=norm_method(m), seed=i,
                    metrics={"auroc": float(v), "auprc": 0.0},
                    config={"data": ds},
                    provenance=make_provenance("experiments/migrate_legacy_results.py",
                                               source="legacy",
                                               extra={"note": "AUROC only from consolidated"}),
                ))
        return entries

    # pm25_tcn_standalone_result.json: single flat dict
    if filename == "pm25_tcn_standalone_result.json":
        return migrate_single_result(data, "CT_pm25", "tcn", source="legacy")

    # causaltime_results.json: {dataset: {method: {metrics}}} (no seeds)
    if filename == "causaltime_results.json":
        for ds_key, ds_val in data.items():
            if not isinstance(ds_val, dict):
                continue
            ds = norm_dataset(ds_key)
            for m, met_raw in ds_val.items():
                if not isinstance(met_raw, dict):
                    continue
                metrics = extract_metrics(met_raw)
                if "auroc" not in metrics:
                    continue
                entries.append(make_result_entry(
                    dataset=ds, method=norm_method(m), seed=0,
                    metrics=metrics, config={"data": ds},
                    provenance=make_provenance(
                        "experiments/migrate_legacy_results.py", source="rerun",
                        extra={"note": "single seed; from causaltime_results.json"}),
                ))
        return entries

    # dream3_backfill_results.json
    if filename == "dream3_backfill_results.json":
        return migrate_dataset_top_level(data, source="rerun")

    # ct_medical_3seed_results.json: {seeds: {seed_0: {method: {metrics}}}, config: {...}}
    if filename == "ct_medical_3seed_results.json":
        seeds_data = data.get("seeds", data)
        ds_name = "CT_medical"
        if "config" in data and isinstance(data["config"], dict):
            ds_name = data["config"].get("data", ds_name)
        return migrate_seed_top_level(seeds_data, norm_dataset(ds_name),
                                      ["baseline", "mamba"], source="rerun")

    # var50_3seed_results.json: {seed_0: {baseline: {...}, F_ds8_fix: {...}}}
    if filename == "var50_3seed_results.json":
        return migrate_seed_top_level(data, "VAR_d50",
                                      ["baseline", "mamba"], source="rerun")

    # nsvar50_3seed_results.json
    if filename == "nsvar50_3seed_results.json":
        return migrate_seed_top_level(data, "NSVAR_d50",
                                      ["baseline", "mamba"], source="rerun")

    # fmri_3subj_results.json
    if filename == "fmri_3subj_results.json":
        return migrate_seed_top_level(data, "fMRI_d15",
                                      ["baseline", "mamba"], source="rerun")

    # filter_5seed_fix_results.json
    if filename == "filter_5seed_fix_results.json":
        return migrate_seed_top_level(data, "NSVAR_d10",
                                      ["baseline", "mamba"], source="rerun")

    # priority1_results.json: flat keys like NSVAR_d50_seed0
    if filename == "priority1_results.json":
        for key, val in data.items():
            if not isinstance(val, dict):
                continue
            # Parse dataset_seedN
            for ds_pat in ["NSVAR_d50", "Lorenz_F40"]:
                if key.startswith(ds_pat + "_seed"):
                    seed_str = key[len(ds_pat) + 5:]  # after "_seed"
                    try:
                        seed = int(seed_str)
                    except ValueError:
                        continue
                    for m, met_raw in val.items():
                        if not isinstance(met_raw, dict):
                            continue
                        metrics = extract_metrics(met_raw)
                        if "auroc" not in metrics:
                            continue
                        entries.append(make_result_entry(
                            dataset=ds_pat, method=norm_method(m), seed=seed,
                            metrics=metrics, config={"data": ds_pat},
                            provenance=make_provenance(
                                "experiments/migrate_legacy_results.py",
                                source="rerun"),
                        ))
        return entries

    # multiseed_synthetic_results.json
    if filename == "multiseed_synthetic_results.json":
        return migrate_dataset_top_level(data, source="rerun")

    # var50_tcn_results.json: {seed_0: {tcn: {...}}}
    if filename == "var50_tcn_results.json":
        return migrate_seed_top_level(data, "VAR_d50", ["tcn"], source="rerun")

    # tcn_backfill_results.json
    if filename == "tcn_backfill_results.json":
        return migrate_dataset_top_level(data, source="rerun")

    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir or RESULTS_DIR
    out_path = args.output or os.path.join(results_dir, "migrated_all.json")

    all_entries = []
    stats = {"files": 0, "entries": 0, "skipped": 0}

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(results_dir, fname)

        # Load and try migration
        try:
            entries = try_migration(path, fname)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  SKIP {fname}: {e}")
            stats["skipped"] += 1
            continue

        if entries:
            all_entries.extend(entries)
            print(f"  {fname}: {len(entries)} entries")
            stats["files"] += 1
            stats["entries"] += len(entries)
        else:
            # Check if it's an already-unified file
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if "results" in data and isinstance(data["results"], list):
                    # Normalize dataset/method names even for already-unified files
                    for e in data["results"]:
                        e["dataset"] = norm_dataset(e.get("dataset", ""))
                        e["method"] = norm_method(e.get("method", ""))
                    n = len(data["results"])
                    all_entries.extend(data["results"])
                    print(f"  {fname}: {n} entries (already unified)")
                    stats["files"] += 1
                    stats["entries"] += n
                else:
                    stats["skipped"] += 1
            except Exception:
                stats["skipped"] += 1
                continue

    if not all_entries:
        print("No entries found.")
        return

    # Deduplicate: keep last entry for each (dataset, method, seed)
    seen = {}
    for e in all_entries:
        key = (e["dataset"], e["method"], e["seed"])
        seen[key] = e
    deduped = list(seen.values())
    dupes = len(all_entries) - len(deduped)
    if dupes:
        print(f"\nDeduplicated: {len(all_entries)} → {len(deduped)} entries ({dupes} duplicates)")

    collection = make_collection(
        deduped,
        description="Migrated legacy results in unified schema",
    )
    collection["_audit"] = make_provenance(
        "experiments/migrate_legacy_results.py", source="legacy",
        extra={
            "datasets_covered": sorted(set(e["dataset"] for e in deduped)),
            "methods_covered": sorted(set(e["method"] for e in deduped)),
            "n_entries": len(deduped),
            "source_files": stats["files"],
        }
    )
    save_collection(collection, out_path)
    print(f"\nSaved {len(deduped)} entries from {stats['files']} files to {out_path}")
    print(f"Datasets: {sorted(set(e['dataset'] for e in deduped))}")
    print(f"Methods: {sorted(set(e['method'] for e in deduped))}")


if __name__ == "__main__":
    main()
