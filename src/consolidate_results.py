"""Consolidate all experiment results into an 11-config main table.

Reads all existing result JSON files and produces a unified summary.
Run on cloud: /root/autodl-tmp/GUOJI/mamba_enhanced/
"""
import json, os, glob
import numpy as np


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect_results():
    results = {}

    # ---- NSVAR d=10 (5 seeds) ----
    nsvar_files = [
        "test_filter_5seed_results.json",
        "filter_5seed_fix_results.json",
    ]
    for f in nsvar_files:
        d = load_json(f)
        if d and "seeds" in d:
            for seed_key, seed_data in d["seeds"].items():
                if isinstance(seed_data, dict):
                    for method in seed_data:
                        if "auroc" in seed_data[method]:
                            results.setdefault("NSVAR_d10", {}).setdefault(method, []).append(
                                seed_data[method]["auroc"])

    # ---- TCN ablation results ----
    tcn = load_json("tcn_ablation_results.json")
    if tcn:
        for dataset, data in tcn.items():
            if isinstance(data, dict):
                for method, metrics in data.items():
                    if isinstance(metrics, dict) and "auroc" in metrics:
                        results.setdefault(dataset, {}).setdefault(method, []).append(metrics["auroc"])

    # ---- TCN supplement ----
    tcn_supp = load_json("tcn_supplement_results.json")
    if tcn_supp:
        for dataset, data in tcn_supp.items():
            if isinstance(data, dict):
                for method, metrics in data.items():
                    if isinstance(metrics, dict) and "auroc" in metrics:
                        results.setdefault(dataset, {}).setdefault(method, []).append(metrics["auroc"])

    # ---- Additional baselines ----
    abl = load_json("additional_baselines_results.json")
    if abl:
        for dataset, data in abl.items():
            if isinstance(data, dict):
                for method, metrics in data.items():
                    if isinstance(metrics, dict) and "auroc" in metrics:
                        auroc = metrics["auroc"]
                        if isinstance(auroc, list):
                            results.setdefault(dataset, {}).setdefault(method, []).extend(auroc)
                        else:
                            results.setdefault(dataset, {}).setdefault(method, []).append(auroc)

    # ---- Backfill baselines ----
    bf = load_json("backfill_baselines_results.json")
    if bf:
        for key, metrics in bf.items():
            if isinstance(metrics, dict) and "auroc" in metrics:
                # Parse key like "VAR_d50_pcmci"
                parts = key.split("_")
                if "pcmci" in key.lower():
                    dataset = "VAR_d50"
                    method = "PCMCI+ParCorr"
                elif "lorenz" in key.lower():
                    dataset = "Lorenz_F40"
                    if "gpdc" in key.lower():
                        method = "PCMCI+GPDC"
                    elif "cmlp" in key.lower():
                        method = "cMLP"
                    else:
                        method = key
                else:
                    dataset = key
                    method = "unknown"
                results.setdefault(dataset, {}).setdefault(method, []).append(metrics["auroc"])

    # ---- PCMCI results ----
    pcmci = load_json("pcmci_results.json")
    if pcmci:
        for dataset, data in pcmci.items():
            if isinstance(data, dict) and "auroc" in data:
                results.setdefault(dataset, {}).setdefault("PCMCI+", []).append(data["auroc"])
            elif isinstance(data, list):
                aurocs = [d["auroc"] for d in data if "auroc" in d]
                if aurocs:
                    results.setdefault(dataset, {}).setdefault("PCMCI+", []).extend(aurocs)

    # ---- DREAM3 ----
    dream3 = load_json("dream3_results.json")
    if dream3:
        for key, data in dream3.items():
            if isinstance(data, dict) and "auroc" in data:
                results.setdefault("DREAM3", {}).setdefault(key, []).append(data["auroc"])

    # ---- CausalTime ----
    ct = load_json("causaltime_results.json")
    if ct:
        for dataset, data in ct.items():
            if isinstance(data, dict):
                for method, metrics in data.items():
                    if isinstance(metrics, dict) and "auroc" in metrics:
                        results.setdefault(dataset, {}).setdefault(method, []).append(metrics["auroc"])

    # ---- Priority 1 ----
    p1 = load_json("priority1_results.json")
    if p1:
        for dataset, data in p1.items():
            if isinstance(data, dict):
                for method, metrics in data.items():
                    if isinstance(metrics, dict) and "auroc" in metrics:
                        results.setdefault(dataset, {}).setdefault(method, []).append(metrics["auroc"])

    # ---- Interaction ablation checkpoint ----
    ia = load_json("interaction_ablation_checkpoint.json")
    if ia and "per_seed" in ia:
        for cell, cell_data in ia["per_seed"].items():
            for seed_key, seed_data in cell_data.items():
                if isinstance(seed_data, dict):
                    for method, metrics in seed_data.items():
                        if isinstance(metrics, dict) and "auroc" in metrics:
                            results.setdefault(cell, {}).setdefault(method, []).append(metrics["auroc"])

    # ---- NS Nonlinear ----
    ns_nl = load_json("ns_nonlinear_results.json")
    if ns_nl:
        for seed_key, seed_data in ns_nl.items():
            if isinstance(seed_data, dict):
                for method, metrics in seed_data.items():
                    if isinstance(metrics, dict) and "auroc" in metrics:
                        results.setdefault("NS+Nonlinear", {}).setdefault(method, []).append(metrics["auroc"])

    return results


def print_table(results):
    """Print consolidated table."""
    print("=" * 100)
    print("11-CONFIG MAIN TABLE (preliminary)")
    print("=" * 100)

    # Define expected configurations
    configs = [
        "VAR_d50",
        "Lorenz_F40",
        "NSVAR_d10",
        "NSVAR_d50_PlanA",
        "DREAM3_d10",
        "DREAM3_d50",
        "DREAM3_d100",
        "CT_medical",
        "CT_pm25",
        "CT_traffic",
        "fMRI_d15",
    ]

    methods = ["baseline", "mamba", "tcn", "cMLP", "PCMCI+", "PCMCI+ParCorr", "PCMCI+GPDC", "TCDF"]

    header = f"  {'Dataset':<22}"
    for m in methods:
        header += f" {m:<12}"
    print(header)
    print(f"  {'-'*22}{'-'*13*len(methods)}")

    for cfg in configs:
        row = f"  {cfg:<22}"
        for m in methods:
            if cfg in results and m in results[cfg]:
                vals = results[cfg][m]
                mean = np.mean(vals)
                n = len(vals)
                row += f" {mean:>8.4f}({n}) "
            else:
                row += f" {'N/A':<12}"
        print(row)

    print(f"\n  Raw keys found in results: {sorted(results.keys())}")


if __name__ == "__main__":
    results = collect_results()
    print_table(results)

    with open("consolidated_results.json", "w") as f:
        # Convert to serializable format
        serializable = {}
        for k, v in results.items():
            serializable[k] = {}
            for m, vals in v.items():
                serializable[k][m] = [float(x) for x in vals]
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to consolidated_results.json")
