"""Task 2: Statistical tests on 10-seed factorial results.

Produces paper-ready paired-delta table with:
- Paired t-test (two-sided)
- Wilcoxon signed-rank
- Cohen's d
- 95% CI (t-distribution)
- Holm-Bonferroni corrected p-values
"""

import json, os
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "raw")
os.makedirs(OUT_DIR, exist_ok=True)

# Load canonical
with open(os.path.join(OUT_DIR, "factorial_ablation_canonical.json")) as f:
    canon = json.load(f)

primary = canon["primary"]["per_seed"]
cells = ["Stat+Linear", "Stat+Nonlinear", "NS+Linear", "NS+Nonlinear"]

# Extract per-cell per-model AUROC values
def get_vals(seeds_dict, cell, model):
    vals = []
    for sk in sorted(seeds_dict.keys()):
        if cell in seeds_dict[sk] and model in seeds_dict[sk][cell]:
            vals.append(seeds_dict[sk][cell][model]["summary_max"]["auroc"])
    return np.array(vals)

# Paired comparisons
comparisons = [
    ("Mamba", "Baseline", "mamba", "baseline"),
    ("TCN", "Baseline", "tcn", "baseline"),
    ("Mamba", "TCN", "mamba", "tcn"),
]

rows = []
all_p_values = []

for comp_label_a, comp_label_b, model_a, model_b in comparisons:
    for cell in cells:
        va = get_vals(primary, cell, model_a)
        vb = get_vals(primary, cell, model_b)
        if len(va) == 0 or len(vb) == 0:
            continue
        deltas = va - vb
        n = len(deltas)
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas, ddof=1)
        se_delta = std_delta / np.sqrt(n)

        # Paired t-test (two-sided)
        t_stat, p_t = stats.ttest_rel(va, vb)

        # Wilcoxon signed-rank
        w_stat, p_w = stats.wilcoxon(va, vb, zero_method="zsplit")

        # Cohen's d
        pooled_std = np.sqrt((np.std(va, ddof=1)**2 + np.std(vb, ddof=1)**2) / 2)
        cohens_d = mean_delta / pooled_std if pooled_std > 0 else 0.0

        # 95% CI (t-distribution)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_lo = mean_delta - t_crit * se_delta
        ci_hi = mean_delta + t_crit * se_delta

        row = {
            "cell": cell,
            "comparison": f"{comp_label_a} - {comp_label_b}",
            "mean_a": round(float(np.mean(va)), 4),
            "mean_b": round(float(np.mean(vb)), 4),
            "delta_mean": round(float(mean_delta), 4),
            "ci_95": f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            "ci_lo": round(float(ci_lo), 4),
            "ci_hi": round(float(ci_hi), 4),
            "t_stat": round(float(t_stat), 4),
            "p_t": float(p_t),
            "p_wilcoxon": float(p_w),
            "cohens_d": round(float(cohens_d), 4),
            "n": n,
        }
        rows.append(row)
        all_p_values.append((p_t, row))

# Holm-Bonferroni correction on all p-values
all_p_values.sort(key=lambda x: x[0])
m = len(all_p_values)
for rank, (p_val, row) in enumerate(all_p_values):
    holm_threshold = 0.05 / (m - rank)
    row["p_corr_holm"] = min(float(p_val * (m - rank)), 1.0)
    row["significant_05"] = bool(row["p_corr_holm"] < 0.05)

# Build output
output = {
    "_audit": {
        "script": "experiments/factorial_stat_tests.py",
        "source": "results/raw/factorial_ablation_canonical.json",
        "methods": "Paired t-test (two-sided), Wilcoxon signed-rank, Cohen's d, 95% CI (t-dist), Holm-Bonferroni"
    },
    "n_comparisons_total": len(rows),
    "rows": rows,
}

# Pretty-print table for verification
header = f"{'Cell':<18} {'Comparison':<20} {'Δ mean':>8} {'95% CI':<22} {'t':>7} {'p_t':>8} {'p_corr':>8} {'d':>7} {'sig'}"
print(header)
print("-" * len(header))
for row in rows:
    sig = "**" if row["significant_05"] else "  "
    print(f"{row['cell']:<18} {row['comparison']:<20} {row['delta_mean']:>8.4f} {row['ci_95']:<22} {row['t_stat']:>7.3f} {row['p_t']:>8.4f} {row['p_corr_holm']:>8.4f} {row['cohens_d']:>7.3f} {sig}")

out_path = os.path.join(OUT_DIR, "factorial_stat_tests.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nWritten to {out_path}")

# Print key findings
print("\nKey findings (p_corr < 0.05):")
sig_rows = [r for r in rows if r["significant_05"]]
if sig_rows:
    for r in sig_rows:
        direction = ">" if r["delta_mean"] > 0 else "<"
        print(f"  {r['cell']} {r['comparison']}: Δ={r['delta_mean']:.4f}, d={r['cohens_d']:.3f}, t={r['t_stat']:.3f}, p_corr={r['p_corr_holm']:.4f}")
else:
    print("  (none)")
