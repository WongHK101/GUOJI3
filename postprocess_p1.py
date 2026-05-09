"""P1-4: Post-process all experiment results into summary tables.

Reads all JSON result files and produces:
  1. Complete AUROC/AUPRC/nSHD/MCC table
  2. Per-dataset summary statistics (mean ± std across seeds)
  3. Comparison table: baseline vs ISTF-Mamba vs TCN (where available)
"""
import json
import os
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(name):
    path = os.path.join(OUTPUT_DIR, name)
    with open(path) as f:
        return json.load(f)


def fmt_mean_std(values, decimals=4):
    """Format mean ± std from list of values."""
    v = np.array(values)
    return f"{np.mean(v):.{decimals}f} ± {np.std(v):.{decimals}f}"


def extract_metrics_per_seed(data, metric="auroc"):
    """Extract a metric across seeds from a {seed_N: {model: {metric: val}}} dict."""
    seeds = {}
    models = set()
    for seed_key, seed_data in data.items():
        if not seed_key.startswith("seed_"):
            continue
        for model_name in seed_data:
            models.add(model_name)
    for model in sorted(models):
        vals = []
        for seed_key in sorted(data.keys()):
            if not seed_key.startswith("seed_"):
                continue
            if model in data[seed_key]:
                vals.append(data[seed_key][model].get(metric, float("nan")))
        seeds[model] = vals
    return seeds


def extract_seed_results(data):
    """Handle nested 'seeds' structure used in some files."""
    if "seeds" in data:
        return extract_metrics_per_seed(data["seeds"])
    return extract_metrics_per_seed(data)


def main():
    print("=" * 70)
    print("P1-4: COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 70)

    # ---- P1-3: Multi-seed synthetic ----
    ms = load_json("multiseed_synthetic_results.json")
    print("\n--- P1-3: Multi-seed Synthetic ---")
    for dataset in ["VAR_d50", "Lorenz_F40", "NSVAR_d10", "NSVAR_d50_PlanA"]:
        if dataset not in ms:
            continue
        d = ms[dataset]
        print(f"\n  {dataset}:")
        for metric in ["auroc", "auprc", "nshd", "mcc"]:
            seeds_data = extract_metrics_per_seed(d, metric)
            for model, vals in seeds_data.items():
                print(f"    {model:12s} {metric:6s}: {fmt_mean_std(vals)}")

    # ---- P1-1: CT_medical 3-seed (d_state=8) ----
    ct = load_json("ct_medical_3seed_results.json")
    print(f"\n\n--- P1-1: CT_medical 3-seed (d_state=8) ---")
    ct_seeds = extract_metrics_per_seed(ct["seeds"])
    for metric in ["auroc", "auprc", "nshd", "mcc", "shd"]:
        for model, vals in ct_seeds.items():
            if metric in ct["seeds"]["seed_0"][model]:
                print(f"  {model:12s} {metric:6s}: {fmt_mean_std(vals)}  [{', '.join(f'{v:.4f}' for v in vals)}]")

    # ---- DREAM3 ----
    try:
        d3 = load_json("dream3_backfill_results.json")
        print(f"\n\n--- DREAM3 Backfill ---")
        for dataset in sorted(d3.keys()):
            if not isinstance(d3[dataset], dict):
                continue
            for metric in ["auroc", "auprc", "nshd"]:
                seeds_data = extract_seed_results(d3[dataset])
                for model, vals in seeds_data.items():
                    if vals:
                        print(f"  {dataset:20s} {model:12s} {metric:6s}: {fmt_mean_std(vals)}")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # ---- fMRI ----
    try:
        fmri = load_json("fmri_3subj_results.json")
        print(f"\n\n--- fMRI 3-subject ---")
        for dataset in sorted(fmri.keys()):
            if not isinstance(fmri[dataset], dict):
                continue
            for metric in ["auroc", "auprc", "nshd"]:
                seeds_data = extract_seed_results(fmri[dataset])
                for model, vals in seeds_data.items():
                    if vals:
                        print(f"  {dataset:20s} {model:12s} {metric:6s}: {fmt_mean_std(vals)}")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # ---- Theory Verification ----
    try:
        tv = load_json("theory_verification_results.json")
        print(f"\n\n--- Theory Verification ---")
        if isinstance(tv, dict):
            for k, v in tv.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
                elif isinstance(v, dict):
                    print(f"  {k}: {json.dumps(v, indent=4)[:200]}")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # ---- Interaction Ablation (2×2 factorial) ----
    try:
        ia = load_json("interaction_ablation_results.json")
        print(f"\n\n--- Interaction Ablation (2×2 factorial) ---")
        if isinstance(ia, dict):
            for k, v in sorted(ia.items()):
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for kk, vv in v.items():
                        if isinstance(vv, float):
                            print(f"    {kk}: {vv:.4f}")
                        elif isinstance(vv, dict):
                            auroc = vv.get("auroc", "N/A")
                            auprc = vv.get("auprc", "N/A")
                            print(f"    {kk}: AUROC={auroc}, AUPRC={auprc}")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # ---- P1-5: Mask Supplement ----
    try:
        mask = load_json("mask_supplement_results.json")
        print(f"\n\n--- P1-5: Mask Supplement ---")
        for k, v in mask.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # ---- CausalTime consolidated ----
    try:
        ct_all = load_json("causaltime_results.json")
        print(f"\n\n--- CausalTime (all) ---")
        if isinstance(ct_all, dict):
            for k, v in ct_all.items():
                print(f"  {k}: {v}")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # ---- Priority1 summary ----
    try:
        p1 = load_json("priority1_results.json")
        print(f"\n\n--- Priority1 Summary ---")
        if isinstance(p1, dict):
            for k, v in p1.items():
                print(f"  {k}: {v}")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # ============================================================
    # MASTER TABLE: All configs, AUROC only
    # ============================================================
    print(f"\n\n{'='*70}")
    print("MASTER AUROC TABLE")
    print(f"{'='*70}")
    print(f"{'Dataset':<25s} {'Baseline':>18s} {'ISTF-Mamba':>18s} {'TCN':>18s} {'Δ Mamba':>10s}")
    print("-" * 90)

    configs = []

    # VAR_d50 (from P1-3, 5 seeds)
    if "VAR_d50" in ms:
        d = ms["VAR_d50"]
        b_au = extract_metrics_per_seed(d, "auroc").get("baseline", [])
        m_au = extract_metrics_per_seed(d, "auroc").get("mamba", [])
        configs.append(("VAR_d50 (5 seeds)", b_au, m_au, []))

    # Lorenz_F40
    if "Lorenz_F40" in ms:
        d = ms["Lorenz_F40"]
        b_au = extract_metrics_per_seed(d, "auroc").get("baseline", [])
        m_au = extract_metrics_per_seed(d, "auroc").get("mamba", [])
        configs.append(("Lorenz_F40 (5 seeds)", b_au, m_au, []))

    # NSVAR_d10
    if "NSVAR_d10" in ms:
        d = ms["NSVAR_d10"]
        b_au = extract_metrics_per_seed(d, "auroc").get("baseline", [])
        m_au = extract_metrics_per_seed(d, "auroc").get("mamba", [])
        configs.append(("NSVAR_d10 (5 seeds)", b_au, m_au, []))

    # NSVAR_d50_PlanA
    if "NSVAR_d50_PlanA" in ms:
        d = ms["NSVAR_d50_PlanA"]
        b_au = extract_metrics_per_seed(d, "auroc").get("baseline", [])
        m_au = extract_metrics_per_seed(d, "auroc").get("mamba", [])
        configs.append(("NSVAR_d50_PlanA (3s)", b_au, m_au, []))

    # CT_medical 3-seed
    ct_b = [ct["seeds"][f"seed_{s}"]["baseline"]["auroc"] for s in range(3)]
    ct_m = [ct["seeds"][f"seed_{s}"]["mamba"]["auroc"] for s in range(3)]
    configs.append(("CT_medical (3 seeds)", ct_b, ct_m, []))

    # CausalTime traffic/pm25 (1 seed from consolidated)
    ca = load_json("consolidated_all.json")
    for ct_name in ["CT_traffic", "CT_pm25"]:
        if ct_name in ca:
            b_val = ca[ct_name].get("baseline", [])
            m_val = ca[ct_name].get("mamba", [])
            if b_val:
                configs.append((f"{ct_name} (1 seed)", b_val, m_val, []))

    # DREAM3 (5 subjects, 1 seed each)
    try:
        d3 = load_json("dream3_backfill_results.json")
        for subj_name in sorted(d3.keys()):
            if not isinstance(d3[subj_name], dict):
                continue
            b_vals = []
            m_vals = []
            for sk in sorted(d3[subj_name].keys()):
                if not sk.startswith("seed_"):
                    continue
                sd = d3[subj_name][sk]
                if "baseline" in sd:
                    b_vals.append(sd["baseline"].get("auroc", float("nan")))
                if "mamba" in sd:
                    m_vals.append(sd["mamba"].get("auroc", float("nan")))
            if b_vals:
                configs.append((f"DREAM3 {subj_name}", b_vals, m_vals, []))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # fMRI
    try:
        fmri = load_json("fmri_3subj_results.json")
        for subj_name in sorted(fmri.keys()):
            if not isinstance(fmri[subj_name], dict):
                continue
            b_vals = []
            m_vals = []
            t_vals = []
            for sk in sorted(fmri[subj_name].keys()):
                if not sk.startswith("seed_"):
                    continue
                sd = fmri[subj_name][sk]
                if "baseline" in sd:
                    b_vals.append(sd["baseline"].get("auroc", float("nan")))
                if "mamba" in sd:
                    m_vals.append(sd["mamba"].get("auroc", float("nan")))
                if "tcn" in sd:
                    t_vals.append(sd["tcn"].get("auroc", float("nan")))
            if b_vals:
                configs.append((f"fMRI {subj_name}", b_vals, m_vals, t_vals))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Print table
    total_b, total_m = [], []
    for name, b_au, m_au, t_au in configs:
        b_str = fmt_mean_std(b_au) if b_au else "N/A"
        m_str = fmt_mean_std(m_au) if m_au else "N/A"
        t_str = fmt_mean_std(t_au) if t_au else "N/A"
        if b_au and m_au:
            delta = np.mean(m_au) - np.mean(b_au)
            d_str = f"{delta:+.4f}"
            total_b.extend(b_au)
            total_m.extend(m_au)
        else:
            d_str = "N/A"
        print(f"  {name:<25s} {b_str:>18s} {m_str:>18s} {t_str:>18s} {d_str:>10s}")

    if total_b:
        print("-" * 90)
        print(f"  {'OVERALL MEAN':<25s} {fmt_mean_std(total_b):>18s} {fmt_mean_std(total_m):>18s} {'':>18s} {np.mean(total_m)-np.mean(total_b):+10.4f}")

    print(f"\nResults saved to individual JSON files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
