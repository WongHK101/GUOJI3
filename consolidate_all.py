"""Consolidate all experiment results into 11-config main table — S3 final step."""
import json
import os
import sys
import numpy as np

os.chdir("/root/autodl-tmp/GUOJI/mamba_enhanced")

results = {}

def add(dataset, method, auroc):
    results.setdefault(dataset, {}).setdefault(method, []).append(float(auroc))

# ---- filter_5seed_fix (NSVAR d=10, 5 seeds) ----
d = json.load(open("filter_5seed_fix_results.json"))
for seed_key, seed_data in d.items():
    for method, metrics in seed_data.items():
        if isinstance(metrics, dict) and "auroc" in metrics:
            name = {"A_baseline": "baseline", "F_ds8_OLD": "mamba_v0", "F_ds8_fix": "mamba"}.get(method, method)
            add("NSVAR_d10", name, metrics["auroc"])

# ---- nsvar50_3seed (stationary VAR d=50, 3 seeds) ----
d = json.load(open("nsvar50_3seed_results.json"))
for seed_key, seed_data in d.items():
    for method, metrics in seed_data.items():
        if isinstance(metrics, dict) and "auroc" in metrics:
            add("VAR_d50", method, metrics["auroc"])

# ---- causaltime (3 datasets) ----
d = json.load(open("causaltime_results.json"))
for dataset, methods in d.items():
    ds_name = {"traffic": "CT_traffic", "medical": "CT_medical", "pm25": "CT_pm25"}.get(dataset, dataset)
    for method, metrics in methods.items():
        if isinstance(metrics, dict) and "auroc" in metrics:
            name = {"baseline": "baseline", "F_ds8_fix": "mamba"}.get(method, method)
            add(ds_name, name, metrics["auroc"])

# ---- fmri_3subj (fMRI d=15, 3 subjects) ----
d = json.load(open("fmri_3subj_results.json"))
for subj_key, methods in d.items():
    for method, metrics in methods.items():
        if isinstance(metrics, dict) and "auroc" in metrics:
            name = {"baseline": "baseline", "F_ds8_fix": "mamba"}.get(method, method)
            add("fMRI_d15", name, metrics["auroc"])

# ---- priority1: VAR_d50, Lorenz_F40 baseline+mamba ----
d = json.load(open("priority1_results.json"))
for key, methods in d.items():
    if "NSVAR_d50" in key:
        ds = "VAR_d50"
    elif "Lorenz_F40" in key:
        ds = "Lorenz_F40"
    else:
        continue
    for method, metrics in methods.items():
        if isinstance(metrics, dict) and "auroc" in metrics:
            name = {"baseline": "baseline", "F_ds8_fixAB": "mamba"}.get(method, method)
            add(ds, name, metrics["auroc"])

# ---- tcn_backfill (Lorenz_F40, NSVAR_d50_PlanA, CT_traffic, fMRI_d15) ----
d = json.load(open("tcn_backfill_results.json"))
for config, seeds in d.items():
    for seed_key, methods in seeds.items():
        if isinstance(methods, dict):
            for method, metrics in methods.items():
                if isinstance(metrics, dict) and "auroc" in metrics:
                    add(config, method, metrics["auroc"])

# ---- interaction_ablation: NSVAR_d10 baseline/mamba/tcn (4 cells) ----
d = json.load(open("interaction_ablation_results.json"))
if "per_seed" in d:
    for cell, cell_data in d["per_seed"].items():
        for seed_key, seed_data in cell_data.items():
            if isinstance(seed_data, dict):
                for method, metrics in seed_data.items():
                    if isinstance(metrics, dict) and "auroc" in metrics:
                        add("NSVAR_d10", method, metrics["auroc"])

# ---- Print summary ----
print("=" * 100)
print("AVAILABLE RESULTS PER CONFIG")
print("=" * 100)
configs = [
    "VAR_d50", "Lorenz_F40", "NSVAR_d10", "NSVAR_d50_PlanA",
    "DREAM3_d10", "DREAM3_d50", "DREAM3_d100",
    "CT_medical", "CT_pm25", "CT_traffic", "fMRI_d15",
]
for cfg in configs:
    if cfg in results:
        methods_str = ", ".join([f"{m}(n={len(v)})" for m, v in results[cfg].items()])
        print(f"  {cfg:<22}: {methods_str}")
    else:
        print(f"  {cfg:<22}: NO DATA")

# ---- Main table ----
print()
print("=" * 100)
print("11-CONFIG MAIN TABLE (mean AUROC)")
print("=" * 100)
methods_all = ["baseline", "mamba", "tcn"]
header = f'  {"Dataset":<22}'
for m in methods_all:
    header += f" {m:<14}"
print(header)
print(f'  {"-"*22}{"-"*15*len(methods_all)}')
for cfg in configs:
    row = f"  {cfg:<22}"
    for m in methods_all:
        if cfg in results and m in results[cfg]:
            vals = results[cfg][m]
            row += f" {np.mean(vals):>10.4f}({len(vals)}) "
        else:
            row += f" {'N/A':<14}"
    print(row)

print()
print("All raw keys:", sorted(results.keys()))

# ---- Save consolidated ----
serializable = {}
for k, v in results.items():
    serializable[k] = {}
    for m, vals in v.items():
        serializable[k][m] = [float(x) for x in vals]
with open("consolidated_all.json", "w") as f:
    json.dump(serializable, f, indent=2)
print("\nSaved to consolidated_all.json")
