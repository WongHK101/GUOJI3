"""Unified paired statistical tests across all methods and datasets.

Performs Wilcoxon signed-rank test (paired by seed) between ISTF-Mamba and each
other method, per dataset, per metric. Only runs when both methods have ≥5 seeds.
Outputs CSV + LaTeX to results/tables/.

Usage:
    python experiments/run_statistical_tests.py
    python experiments/run_statistical_tests.py --results-dir results/raw/
"""

import json, os, sys, argparse
from collections import defaultdict
import numpy as np

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from src.schema import load_collection, expected_results_dir

RESULTS_DIR = expected_results_dir()
TABLES_DIR = os.path.join(_PROJ_ROOT, "results", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

METRICS = ["auroc", "auprc", "shd_topk", "nshd_topk", "mcc_topk", "f1", "acc"]
METRIC_LABELS = {
    "auroc": "AUROC", "auprc": "AUPRC", "shd_topk": "SHD$_{topk}$",
    "nshd_topk": "nSHD$_{topk}$", "mcc_topk": "MCC$_{topk}$",
    "f1": "F1", "acc": "Acc",
}
MIN_SEEDS = 5  # minimum paired observations for Wilcoxon


def load_all_entries(results_dir):
    """Load from migrated_all.json first (canonical), fall back to scanning."""
    # Prefer v2 (canonical, all metrics from same GC score) over legacy
    for fname in ["migrated_all_v2.json", "migrated_all.json"]:
        migrated_path = os.path.join(results_dir, fname)
        if os.path.exists(migrated_path):
            try:
                coll = load_collection(migrated_path)
                entries = coll.get("results", [])
                if entries:
                    return entries
            except Exception:
                pass
    entries = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(results_dir, fname)
        try:
            coll = load_collection(path)
        except Exception:
            continue
        for e in coll.get("results", []):
            key = (e["dataset"], e["method"], e["seed"])
            if key in entries:
                if len(e.get("metrics", {})) > len(entries[key].get("metrics", {})):
                    entries[key] = e
            else:
                entries[key] = e
    return list(entries.values())


def wilcoxon_signed_rank_paired(x, y):
    """Two-sided Wilcoxon signed-rank test p-value. x, y are 1D arrays paired by index.
    Returns (p_value, n_effective) where n_effective is count of non-zero differences."""
    from scipy.stats import wilcoxon
    x, y = np.asarray(x), np.asarray(y)
    d = x - y
    # Remove zero differences (with tolerance for floating point)
    nonzero = np.abs(d) > 1e-15
    d = d[nonzero]
    n = len(d)
    if n < MIN_SEEDS:
        return float("nan"), n
    if n == 0:
        return 1.0, 0
    try:
        res = wilcoxon(d, alternative='two-sided', method='exact' if n <= 25 else 'approx')
        return float(res.pvalue), n
    except Exception:
        return float("nan"), n


def compute_tests(entries):
    """For each dataset and metric, test mamba vs each other method."""
    # Group by (dataset, seed) -> {method: metrics_dict}
    ds_seed_methods = defaultdict(lambda: defaultdict(dict))
    for e in entries:
        ds_seed_methods[e["dataset"]][e["seed"]][e["method"]] = e.get("metrics", {})

    rows = []
    for ds in sorted(ds_seed_methods):
        seeds = list(ds_seed_methods[ds].keys())
        methods = set()
        for s in seeds:
            methods.update(ds_seed_methods[ds][s].keys())

        if "mamba" not in methods:
            continue

        for metric in METRICS:
            # Extract mamba values
            mamba_vals = []
            for s in sorted(seeds):
                m = ds_seed_methods[ds][s].get("mamba", {}).get(metric)
                if m is not None:
                    mamba_vals.append(m)
                else:
                    mamba_vals.append(float("nan"))

            for method in sorted(methods):
                if method == "mamba":
                    continue
                base_vals = []
                for s in sorted(seeds):
                    v = ds_seed_methods[ds][s].get(method, {}).get(metric)
                    if v is not None:
                        base_vals.append(v)
                    else:
                        base_vals.append(float("nan"))

                # Only use seeds where both have values
                paired_mamba = []
                paired_base = []
                for mv, bv in zip(mamba_vals, base_vals):
                    if not (np.isnan(mv) or np.isnan(bv)):
                        paired_mamba.append(mv)
                        paired_base.append(bv)

                if len(paired_mamba) < MIN_SEEDS:
                    continue

                p, n_eff = wilcoxon_signed_rank_paired(paired_mamba, paired_base)
                mean_diff = np.mean(np.array(paired_mamba) - np.array(paired_base))

                rows.append({
                    "dataset": ds,
                    "metric": metric,
                    "method_a": "mamba",
                    "method_b": method,
                    "n_pairs": len(paired_mamba),
                    "n_effective": n_eff,
                    "mean_diff": float(mean_diff),
                    "p_value": float(p),
                    "significant_05": not np.isnan(p) and bool(p < 0.05),
                    "significant_01": not np.isnan(p) and bool(p < 0.01),
                    # Holm-Bonferroni columns (filled below)
                    "holm_adj_p": float("nan"),
                    "holm_family_size": 0,
                    "holm_rank": None,
                    "holm_sig_05": False,
                    "holm_sig_01": False,
                })

    # ---- Holm-Bonferroni correction ----
    # Correction family: all comparisons within each metric (across datasets),
    # INCLUDING NaN p-values (degenerate tests count toward m).
    for metric in METRICS:
        metric_rows = [(i, r) for i, r in enumerate(rows) if r["metric"] == metric]
        # Pass ALL rows (including NaN) — family size = total comparisons attempted
        p_indexed = [(r["p_value"], i) for i, r in metric_rows]
        holm_result = holm_bonferroni(p_indexed, alpha=0.05)
        for i, r in metric_rows:
            if i in holm_result:
                h = holm_result[i]
                rows[i]["holm_adj_p"] = h["holm_adj_p"]
                rows[i]["holm_family_size"] = h["family_size"]
                rows[i]["holm_rank"] = h["rank"]
                rows[i]["holm_sig_05"] = h["rejected_05"]
                rows[i]["holm_sig_01"] = h["rejected_01"]

    # Summary per metric
    for metric in METRICS:
        m_count = sum(1 for r in rows if r["metric"] == metric)
        if m_count > 0:
            n_sig_raw = sum(1 for r in rows if r["metric"] == metric and r["significant_05"])
            n_sig_holm = sum(1 for r in rows if r["metric"] == metric and r["holm_sig_05"])
            family_size = rows[[i for i, r in enumerate(rows) if r["metric"] == metric][0]]["holm_family_size"]
            print(f"  {metric}: m={family_size}, {m_count} comparisons, raw p<0.05: {n_sig_raw}, Holm p<0.05: {n_sig_holm}")

    return rows


def holm_bonferroni(p_values, alpha=0.05):
    """Holm-Bonferroni correction with adjusted p-values, family size, and rank.

    NaN p-values count toward family size m but are treated as p=1 (cannot reject).
    This is conservative and appropriate when a test was attempted but returned
    degenerate (e.g., too few non-zero differences for Wilcoxon).

    Args:
        p_values: list of (p_value, row_index) tuples (NaN allowed, counted in m).
        alpha: family-wise error rate (default 0.05).

    Returns:
        dict mapping row_index -> {
            'holm_adj_p': float (adjusted p-value, NaN→NaN),
            'family_size': int (total m),
            'rank': int or None (1-based, None for NaN),
            'rejected_05': bool, 'rejected_01': bool,
        }
    """
    m = len(p_values)
    if m == 0:
        return {}

    # Sort: valid p-values first (ascending), NaN treated as 1.0 at end
    indexed = []
    nan_indices = []
    for p, i in p_values:
        if np.isnan(p):
            nan_indices.append(i)
        else:
            indexed.append((p, i))
    indexed.sort(key=lambda x: x[0])

    result = {}
    # Monotonicity: adjusted p-value can't decrease as rank increases
    prev_adj_p = -1.0

    for rank_zero_based, (p, idx) in enumerate(indexed):
        rank = rank_zero_based + 1  # 1-based rank
        # Holm step-down adjusted p-value
        adj_p = min(1.0, p * (m - rank_zero_based))
        adj_p = max(adj_p, prev_adj_p)  # enforce monotonicity
        prev_adj_p = adj_p

        rejected_05 = adj_p < 0.05
        rejected_01 = adj_p < 0.01

        result[idx] = {
            'holm_adj_p': round(float(adj_p), 6),
            'family_size': m,
            'rank': rank,
            'rejected_05': rejected_05,
            'rejected_01': rejected_01,
        }

    # NaN p-values: cannot reject, but count toward family
    for idx in nan_indices:
        result[idx] = {
            'holm_adj_p': float('nan'),
            'family_size': m,
            'rank': None,
            'rejected_05': False,
            'rejected_01': False,
        }

    return result


def format_pvalue(p):
    if np.isnan(p):
        return "---"
    if p < 0.001:
        return "$<$0.001"
    if p < 0.01:
        return f"{p:.4f}"
    if p < 0.05:
        return f"{p:.4f}"
    return f"{p:.4f}"


def format_adj_p(p):
    """Format adjusted p-value with significance marker."""
    if np.isnan(p):
        return "---"
    s = f"{p:.4f}"
    if p < 0.01:
        s += r"^{\dagger\dagger}"
    elif p < 0.05:
        s += r"^{\dagger}"
    return s


def to_latex(rows):
    """Generate a multi-panel LaTeX table: one sub-table per metric.
    Includes raw p-value, Holm adjusted p-value, family size, and rank."""
    parts = []
    parts.append(r"% Auto-generated paired Wilcoxon tests: ISTF-Mamba vs baselines")
    parts.append(r"% Columns: dataset, method, n, mean Δ, raw p, Holm adj p (rank/m)")
    parts.append(r"\begin{table}[ht]")
    parts.append(r"  \centering")
    parts.append(r"  \caption{Paired Wilcoxon signed-rank tests: ISTF-Mamba vs baselines}")
    parts.append(r"  \label{tab:stat_tests}")

    for metric in METRICS:
        metric_rows = [r for r in rows if r["metric"] == metric]
        if not metric_rows:
            continue
        label = METRIC_LABELS.get(metric, metric)

        # Get family size from first row
        m = metric_rows[0]["holm_family_size"]

        parts.append(r"  \subtable{" + label + r" ($m=" + str(m) + r"$)}{")
        parts.append(r"    \begin{tabular}{l l c c c c c}")
        parts.append(r"      \toprule")
        parts.append(r"      Dataset & Baseline & $n$ & $\Delta$ mean & $p_{\mathrm{raw}}$ & $p_{\mathrm{Holm}}$ & Rank \\")
        parts.append(r"      \midrule")

        for r in sorted(metric_rows, key=lambda x: (x["dataset"], x["method_b"])):
            rank_str = f"{r['holm_rank']}/{m}" if r['holm_rank'] is not None else "---"
            parts.append(
                f"      {r['dataset']} & {r['method_b']} & {r['n_pairs']} & "
                f"{r['mean_diff']:+.4f} & {format_pvalue(r['p_value'])} & "
                f"{format_adj_p(r['holm_adj_p'])} & {rank_str} \\\\"
            )

        parts.append(r"      \bottomrule")
        parts.append(r"    \end{tabular}")
        parts.append(r"  }")
        parts.append(r"")

    parts.append(r"  \vspace{4pt}")
    parts.append(r"  \textit{Note:} $^{\dagger}\,p_{\mathrm{Holm}}<0.05$, "
                  r"$^{\dagger\dagger}\,p_{\mathrm{Holm}}<0.01$, "
                  r"Holm-Bonferroni step-down within each metric. "
                  r"Rank is 1-based among valid $p$-values; $m$ = total comparisons "
                  r"(incl.\ degenerate tests with $n_{\mathrm{eff}}<5$). "
                  r"Two-sided Wilcoxon, paired by seed.")
    parts.append(r"\end{table}")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir or RESULTS_DIR
    output_dir = args.output_dir or TABLES_DIR
    os.makedirs(output_dir, exist_ok=True)

    entries = load_all_entries(results_dir)
    print(f"Loaded {len(entries)} result entries from {results_dir}")

    rows = compute_tests(entries)
    print(f"Computed {len(rows)} test rows")

    if not rows:
        print("No rows with n ≥ {} seeds for mamba vs baselines.".format(MIN_SEEDS))
        return

    # Summary
    sig_05 = sum(1 for r in rows if r["significant_05"])
    sig_01 = sum(1 for r in rows if r["significant_01"])
    holm_05 = sum(1 for r in rows if r["holm_sig_05"])
    holm_01 = sum(1 for r in rows if r["holm_sig_01"])
    print(f"  Raw    p < 0.05: {sig_05}, p < 0.01: {sig_01}")
    print(f"  Holm   p < 0.05: {holm_05}, p < 0.01: {holm_01}")
    for r in rows:
        raw_sig = " **" if r["significant_01"] else (" *" if r["significant_05"] else "")
        holm_sig = " H01" if r["holm_sig_01"] else (" H05" if r["holm_sig_05"] else "")
        p_str = f"{r['p_value']:.4f}" if not np.isnan(r['p_value']) else "nan"
        adj_str = f"{r['holm_adj_p']:.4f}" if not np.isnan(r['holm_adj_p']) else "nan"
        rank_str = f"rank={r['holm_rank']}/{r['holm_family_size']}" if r['holm_rank'] is not None else "rank=nan"
        print(f"  {r['dataset']:20s} {r['metric']:12s} mamba vs {r['method_b']:10s} "
              f"n={r['n_pairs']}(eff={r['n_effective']}) Δ={r['mean_diff']:+.4f} "
              f"p_raw={p_str}{raw_sig} p_holm={adj_str}{holm_sig} {rank_str}")

    # Save CSV
    csv_path = os.path.join(output_dir, "statistical_tests.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "metric", "method_a", "method_b", "n_pairs",
            "n_effective", "mean_diff", "p_value",
            "significant_05", "significant_01",
            "holm_adj_p", "holm_family_size", "holm_rank",
            "holm_sig_05", "holm_sig_01"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {csv_path}")

    # Save LaTeX
    tex_path = os.path.join(output_dir, "statistical_tests.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(to_latex(rows))
    print(f"LaTeX saved to {tex_path}")


if __name__ == "__main__":
    main()
