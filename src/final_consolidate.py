"""Final 11-config main table consolidation — S3 complete (CORRECTED MAPPINGS)."""
import json, os, numpy as np

results = {}

def add(dataset, method, auroc):
    results.setdefault(dataset, {}).setdefault(method, []).append(float(auroc))

base = "E:/GUOJI/mamba_enhanced"

# ---- 1. VAR_d50 (stationary VAR) ----
# baseline + mamba from var50_3seed_results.json
d = json.load(open(f"{base}/var50_3seed_results.json"))
for seed_key, seed_data in d.items():
    for method, metrics in seed_data.items():
        if isinstance(metrics, dict) and "auroc" in metrics:
            name = {"A_baseline": "baseline", "F_ds8_fix": "mamba"}.get(method, method)
            if name in ("baseline", "mamba"):
                add("VAR_d50", name, metrics["auroc"])
# TCN from var50_tcn_results.json
d = json.load(open(f"{base}/var50_tcn_results.json"))
for seed_key, seed_data in d.items():
    if "tcn" in seed_data and isinstance(seed_data["tcn"], dict) and "auroc" in seed_data["tcn"]:
        add("VAR_d50", "tcn", seed_data["tcn"]["auroc"])

# ---- 2. NSVAR_d10 ----
d = json.load(open(f"{base}/filter_5seed_fix_results.json"))
for seed_key, seed_data in d.items():
    for method, metrics in seed_data.items():
        if isinstance(metrics, dict) and "auroc" in metrics:
            name = {"A_baseline": "baseline", "F_ds8_fix": "mamba"}.get(method, method)
            if name in ("baseline", "mamba"):
                add("NSVAR_d10", name, metrics["auroc"])
# TCN from interaction_ablation_results.json
d = json.load(open(f"{base}/interaction_ablation_results.json"))
if "per_seed" in d:
    for cell, cell_data in d["per_seed"].items():
        for seed_key, seed_data in cell_data.items():
            if isinstance(seed_data, dict):
                for method, metrics in seed_data.items():
                    if isinstance(metrics, dict) and "auroc" in metrics and method in ("baseline", "mamba", "tcn"):
                        add("NSVAR_d10", method, metrics["auroc"])

# ---- 3. NSVAR_d50_PlanA ----
d = json.load(open(f"{base}/nsvar50_3seed_results.json"))
for seed_key, seed_data in d.items():
    for method, metrics in seed_data.items():
        if isinstance(metrics, dict) and "auroc" in metrics:
            name = {"A_baseline": "baseline", "F_ds8_fix": "mamba"}.get(method, method)
            if name in ("baseline", "mamba"):
                add("NSVAR_d50_PlanA", name, metrics["auroc"])
# TCN from tcn_backfill_results.json (NSVAR_d50_PlanA key)
d = json.load(open(f"{base}/tcn_backfill_results.json"))
for config, seeds in d.items():
    if config == "NSVAR_d50_PlanA":
        for seed_key, methods in seeds.items():
            if isinstance(methods, dict):
                for method, metrics in methods.items():
                    if isinstance(metrics, dict) and "auroc" in metrics:
                        add("NSVAR_d50_PlanA", "tcn", metrics["auroc"])

# ---- 4. Lorenz_F40 ----
# baseline + mamba from priority1_results.json (Lorenz_F40 entries)
d = json.load(open(f"{base}/priority1_results.json"))
for key, methods in d.items():
    if "Lorenz_F40" not in key:
        continue
    for method, metrics in methods.items():
        if isinstance(metrics, dict) and "auroc" in metrics:
            name = {"baseline": "baseline", "F_ds8_fixAB": "mamba"}.get(method, method)
            if name in ("baseline", "mamba"):
                add("Lorenz_F40", name, metrics["auroc"])
# TCN from tcn_backfill_results.json
d = json.load(open(f"{base}/tcn_backfill_results.json"))
for config, seeds in d.items():
    if config == "Lorenz_F40":
        for seed_key, methods in seeds.items():
            if isinstance(methods, dict):
                for method, metrics in methods.items():
                    if isinstance(metrics, dict) and "auroc" in metrics:
                        add("Lorenz_F40", "tcn", metrics["auroc"])

# ---- 5. DREAM3_d10/50/100 ----
d = json.load(open(f"{base}/dream3_backfill_results.json"))
for cfg_name, subjects in d.items():
    for subj_key, methods in subjects.items():
        if isinstance(methods, dict):
            for method, metrics in methods.items():
                if isinstance(metrics, dict) and "auroc" in metrics:
                    add(cfg_name, method, metrics["auroc"])

# ---- 6. CausalTime: CT_medical, CT_pm25, CT_traffic ----
d = json.load(open(f"{base}/causaltime_results.json"))
for dataset, methods in d.items():
    ds_name = {"traffic": "CT_traffic", "medical": "CT_medical", "pm25": "CT_pm25"}.get(dataset, dataset)
    for method, metrics in methods.items():
        if isinstance(metrics, dict) and "auroc" in metrics:
            name = {"baseline": "baseline", "F_ds8_fix": "mamba"}.get(method, method)
            if name in ("baseline", "mamba"):
                add(ds_name, name, metrics["auroc"])

# CT_medical TCN from log (AUROC=0.4609) — first run completed before crash
add("CT_medical", "tcn", 0.4609)
# CT_pm25 TCN from standalone run
d = json.load(open(f"{base}/pm25_tcn_standalone_result.json"))
if "auroc" in d:
    add("CT_pm25", "tcn", d["auroc"])
# CT_traffic TCN from tcn_backfill_results.json (structure: {config: {method: metrics}})
d = json.load(open(f"{base}/tcn_backfill_results.json"))
ct = d.get("CT_traffic", {})
if isinstance(ct, dict):
    # Direct method->metrics (no seed wrapper)
    if "auroc" in ct:
        add("CT_traffic", "tcn", ct["auroc"])
    elif "tcn" in ct and isinstance(ct["tcn"], dict) and "auroc" in ct["tcn"]:
        add("CT_traffic", "tcn", ct["tcn"]["auroc"])
    else:
        # Seed-wrapped structure
        for seed_key, methods in ct.items():
            if isinstance(methods, dict):
                for method, metrics in methods.items():
                    if isinstance(metrics, dict) and "auroc" in metrics:
                        add("CT_traffic", method, metrics["auroc"])

# ---- 7. fMRI_d15 ----
d = json.load(open(f"{base}/fmri_3subj_results.json"))
for subj_key, methods in d.items():
    for method, metrics in methods.items():
        if isinstance(metrics, dict) and "auroc" in metrics:
            name = {"baseline": "baseline", "F_ds8_fix": "mamba"}.get(method, method)
            if name in ("baseline", "mamba"):
                add("fMRI_d15", name, metrics["auroc"])
# TCN from tcn_backfill_results.json
d = json.load(open(f"{base}/tcn_backfill_results.json"))
for config, seeds in d.items():
    if config == "fMRI_d15":
        for seed_key, methods in seeds.items():
            if isinstance(methods, dict):
                for method, metrics in methods.items():
                    if isinstance(metrics, dict) and "auroc" in metrics:
                        add("fMRI_d15", "tcn", metrics["auroc"])

# ============================================================
# PRINT MAIN TABLE
# ============================================================
print("=" * 100)
print("11-CONFIG MAIN TABLE (AUROC: mean ± std, n seeds)")
print("=" * 100)
configs = [
    "VAR_d50", "Lorenz_F40", "NSVAR_d10", "NSVAR_d50_PlanA",
    "DREAM3_d10", "DREAM3_d50", "DREAM3_d100",
    "CT_medical", "CT_pm25", "CT_traffic", "fMRI_d15",
]
methods_all = ["baseline", "mamba", "tcn"]
header = f'  {"Dataset":<22}'
for m in methods_all:
    header += f" {m:<20}"
print(header)
print(f'  {"-"*22}{"-"*21*len(methods_all)}')
for cfg in configs:
    row = f"  {cfg:<22}"
    for m in methods_all:
        if cfg in results and m in results[cfg]:
            vals = results[cfg][m]
            row += f" {np.mean(vals):>8.4f}±{np.std(vals):.2f} (n={len(vals)})"
        else:
            row += f" {'N/A':<20}"
    print(row)

# ============================================================
# RAW DATA
# ============================================================
print()
print("=" * 100)
print("RAW AUROC VALUES BY CONFIG")
print("=" * 100)
for cfg in configs:
    if cfg in results:
        print(f"{cfg}:")
        for m, vals in sorted(results[cfg].items()):
            print(f"  {m}: {[round(v, 4) for v in vals]}")
    else:
        print(f"{cfg}: NO DATA")

# Save
serializable = {}
for k, v in results.items():
    serializable[k] = {}
    for m, vals in v.items():
        serializable[k][m] = [float(x) for x in vals]
with open(f"{base}/final_consolidated.json", "w") as f:
    json.dump(serializable, f, indent=2)
print(f"\nSaved to final_consolidated.json")
