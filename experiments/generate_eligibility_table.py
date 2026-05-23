"""Generate datasets eligibility table: inferential vs descriptive only.

Classifies each dataset by whether it supports paired statistical inference
(baseline + mamba both >= 5 seeds) or is descriptive-only.

Usage:
    python experiments/generate_eligibility_table.py
"""

import json, os, sys
from collections import defaultdict

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

MIGRATED_PATH = os.path.join(_PROJ_ROOT, "results", "raw", "migrated_all_v2.json")
TABLES_DIR = os.path.join(_PROJ_ROOT, "results", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

MIN_SEEDS_FOR_INFERENCE = 5

DATASET_INFO = {
    "CT_medical":    ("Real", "Medical time series (PM2.5-related ER visits)"),
    "CT_pm25":       ("Real", "Environmental: PM2.5 air quality"),
    "CT_traffic":    ("Real", "Transportation: traffic flow"),
    "DREAM3_d10":    ("Synthetic", "Gene regulatory network (d=10), small sample"),
    "DREAM3_d50":    ("Synthetic", "Gene regulatory network (d=50), medium"),
    "DREAM3_d100":   ("Synthetic", "Gene regulatory network (d=100), large"),
    "fMRI_d15":      ("Real", "Resting-state fMRI (d=15)"),
    "Lorenz_F40":    ("Synthetic", "Chaotic Lorenz-96 (d=10, F=40)"),
    "VAR_d50":       ("Synthetic", "Stationary linear VAR (d=50, lag=5)"),
    "NSVAR_d10":     ("Synthetic", "Nonstationary VAR (d=10, lag=7)"),
    "NSVAR_d50":     ("Synthetic", "Nonstationary VAR (d=50, lag=14)"),
    "NSVAR_d50_PlanA": ("Synthetic", "Nonstationary VAR PlanA (d=50, lag=14)"),
}


def main():
    with open(MIGRATED_PATH, encoding='utf-8') as f:
        data = json.load(f)

    # Count baseline + mamba seeds per dataset
    ds_seeds = defaultdict(lambda: {"baseline": set(), "mamba": set()})
    for e in data["results"]:
        if e["method"] in ("baseline", "mamba"):
            ds_seeds[e["dataset"]][e["method"]].add(e["seed"])

    rows = []
    for ds in sorted(ds_seeds):
        info = DATASET_INFO.get(ds, ("?", "Unknown"))
        n_base = len(ds_seeds[ds]["baseline"])
        n_mamba = len(ds_seeds[ds]["mamba"])
        n_min = min(n_base, n_mamba)
        eligible = n_min >= MIN_SEEDS_FOR_INFERENCE

        if eligible:
            reason = "Paired Wilcoxon (Holm-Bonferroni corrected)"
        elif n_min == 0:
            reason = "Missing baseline or mamba"
        elif ds in ("DREAM3_d10", "DREAM3_d50", "DREAM3_d100"):
            reason = "Only 3 seeds (DREAM3 fixed design); severely sample-limited"
        elif ds in ("CT_pm25", "CT_traffic"):
            reason = "Only 1 seed; real-data series without replicates"
        elif ds == "fMRI_d15":
            reason = "Only 3 seeds (3 subjects); insufficient for paired test"
        elif ds in ("NSVAR_d50", "NSVAR_d50_PlanA"):
            reason = "Only 3 seeds; nonstationary VAR medium-scale"
        else:
            reason = f"Only {n_min} seeds (< {MIN_SEEDS_FOR_INFERENCE})"

        rows.append({
            "dataset": ds,
            "type": info[0],
            "description": info[1],
            "n_baseline": n_base,
            "n_mamba": n_mamba,
            "n_min": n_min,
            "eligible": eligible,
            "reason": reason,
        })

    # Print summary
    n_infer = sum(1 for r in rows if r["eligible"])
    n_desc = sum(1 for r in rows if not r["eligible"])
    print(f"Datasets: {len(rows)} total, {n_infer} inferential, {n_desc} descriptive-only")
    print()
    for r in rows:
        status = "INFERENTIAL" if r["eligible"] else "descriptive"
        print(f"  {r['dataset']:20s} {r['type']:10s} base={r['n_baseline']} mam={r['n_mamba']} "
              f"→ {status:12s} | {r['reason']}")

    # Save CSV
    csv_path = os.path.join(TABLES_DIR, "dataset_eligibility.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "type", "description", "n_baseline", "n_mamba",
            "n_min", "eligible", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {csv_path}")

    # Save LaTeX
    tex_path = os.path.join(TABLES_DIR, "dataset_eligibility.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(to_latex(rows))
    print(f"LaTeX saved to {tex_path}")


def to_latex(rows):
    parts = []
    parts.append(r"% Dataset eligibility: inferential vs descriptive-only")
    parts.append(r"\begin{table}[ht]")
    parts.append(r"  \centering")
    parts.append(r"  \caption{Datasets: eligibility for statistical inference}")
    parts.append(r"  \label{tab:dataset_eligibility}")
    parts.append(r"  \begin{tabular}{l l c c c l}")
    parts.append(r"    \toprule")
    parts.append(r"    Dataset & Type & $n_{\mathrm{base}}$ & $n_{\mathrm{mamba}}$ & Eligible & Reason \\")
    parts.append(r"    \midrule")

    # Inferential first, then descriptive
    inferential = [r for r in rows if r["eligible"]]
    descriptive = [r for r in rows if not r["eligible"]]

    for r in inferential:
        parts.append(
            f"    {r['dataset']} & {r['type']} & {r['n_baseline']} & {r['n_mamba']} & "
            f"Yes & {r['reason']} \\\\"
        )
    if inferential and descriptive:
        parts.append(r"    \midrule")
    for r in descriptive:
        parts.append(
            f"    {r['dataset']} & {r['type']} & {r['n_baseline']} & {r['n_mamba']} & "
            f"No & {r['reason']} \\\\"
        )

    parts.append(r"    \bottomrule")
    parts.append(r"  \end{tabular}")
    parts.append(r"  \vspace{4pt}")
    parts.append(r"  \textit{Note:} Eligibility requires $\ge "
                 + str(MIN_SEEDS_FOR_INFERENCE)
                 + r"$ paired seeds for both baseline and ISTF-Mamba. "
                 r"Only eligible datasets enter Wilcoxon signed-rank tests with "
                 r"Holm-Bonferroni correction.")
    parts.append(r"\end{table}")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
