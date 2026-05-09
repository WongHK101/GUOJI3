"""Post-process existing result JSONs to add nSHD, seed-based CIs, and expanded metrics.

Reads per-seed result JSONs, computes:
  - nSHD = SHD / n_edges_true
  - 95% t-distribution confidence intervals
  - Summary tables with expanded metrics

Usage: python recompute_metrics.py
Output: enriched_results.json + printed summary tables
"""
import json, os, sys, numpy as np
from collections import defaultdict

# Known n_edges_true for each dataset (from ground truth)
# Values determined from experiment logs
DATASET_EDGES = {
    # NSVAR d=10, P=7: per-seed gc shape reveals edge count
    "nsvar_d10_p7": 22,        # approximate, varies by seed
    "nsvar_d50_p14_plana": 72,  # Plan A, stable coeff + exogenous RW
    "var_d50_p5_stat": 98,      # stationary VAR d=50
    "lorenz_d10_f40": 20,       # Lorenz-96, F=40, d=10
    # DREAM3: varies by subject (we handle per-subject)
    "dream3_d10": None,  # per-subject
    "dream3_d50": None,
    "dream3_d100": None,
    "fMRI_d15": 45,             # approximate
    "causaltime_medical": 153,
    "causaltime_pm25": 354,
    "causaltime_traffic": 160,  # approximate
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def enrich_results(data, known_edges=None):
    """Add nSHD to each metric dict. If known_edges provided, use it;
    otherwise estimate from per-seed SHD values."""
    enriched = {}
    for seed_key, seed_data in data.items():
        if seed_key.startswith("_"):
            continue  # skip summary keys
        enriched[seed_key] = {}
        for model_key, metrics in seed_data.items():
            m = dict(metrics)
            # nSHD
            n_edges = known_edges if known_edges else metrics.get("n_edges_true", 1)
            if n_edges is None:
                # Try from the metrics themselves (new format)
                n_edges = metrics.get("n_edges_true", 1)
            m["nshd"] = m.get("nshd", m["shd"] / max(n_edges, 1))
            enriched[seed_key][model_key] = m
    return enriched


def compute_summary(data):
    """Compute mean ± std and 95% CI across seeds for each model."""
    # Collect per-model metrics across seeds
    model_metrics = defaultdict(lambda: defaultdict(list))
    for seed_key, seed_data in data.items():
        if seed_key.startswith("_"):
            continue
        for model_key, metrics in seed_data.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    model_metrics[model_key][k].append(float(v))

    # Compute statistics
    from scipy import stats as scipy_stats
    summary = {}
    for model_key, metric_dict in model_metrics.items():
        summary[model_key] = {}
        for metric, vals in metric_dict.items():
            vals_arr = np.array(vals)
            mean = np.mean(vals_arr)
            std = np.std(vals_arr, ddof=1) if len(vals) > 1 else 0.0
            se = std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            if len(vals) > 1:
                t_crit = scipy_stats.t.ppf(0.975, df=len(vals) - 1)
                ci_lower = mean - t_crit * se
                ci_upper = mean + t_crit * se
            else:
                ci_lower, ci_upper = mean, mean
            summary[model_key][metric] = {
                "mean": float(mean),
                "std": float(std),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
            }
    return summary


def print_summary_table(summary, models=None):
    """Pretty-print summary table."""
    if models is None:
        models = list(summary.keys())

    # Key metrics to display
    display_metrics = ["auroc", "auprc", "f1", "shd", "nshd", "mcc"]

    # Header
    header = f"{'Model':<30}"
    for m in display_metrics:
        header += f" {m:>20}"
    print(header)
    print("-" * len(header))

    for model in models:
        if model not in summary:
            continue
        row = f"{model:<30}"
        for m in display_metrics:
            if m in summary[model]:
                s = summary[model][m]
                if m in ("shd",):
                    row += f" {s['mean']:>7.0f}±{s['std']:<5.0f}     "
                elif m in ("nshd",):
                    row += f" {s['mean']:>7.3f}±{s['std']:<7.3f}  "
                else:
                    row += f" {s['mean']:>7.4f}±{s['std']:<7.4f}  "
            else:
                row += f" {'N/A':>20}"
        print(row)

    # Print 95% CI for AUROC specifically
    print(f"\n95% Confidence Intervals (AUROC):")
    for model in models:
        if model in summary and "auroc" in summary[model]:
            s = summary[model]["auroc"]
            print(f"  {model:<30}: {s['mean']:.4f} [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]")


def process_file(filepath, known_edges=None, label=""):
    """Process a single result JSON file."""
    print(f"\n{'='*70}")
    print(f"  {label or filepath}")
    print(f"{'='*70}")

    data = load_json(filepath)

    # Check if this has per-seed structure
    has_seeds = any(k.startswith("seed_") or k.startswith("subject_") for k in data)
    if not has_seeds:
        print("  (flat structure, skipping)")
        return None, None

    enriched = enrich_results(data, known_edges)
    summary = compute_summary(enriched)

    models = list(summary.keys())
    print_summary_table(summary, models)

    return enriched, summary


def main():
    base = "/root/autodl-tmp/GUOJI/mamba_enhanced"
    os.chdir(base)

    all_summaries = {}

    # Process each result file with known edge counts
    files_to_process = [
        ("filter_5seed_fix_results.json", 22, "NSVAR d=10 P=7 (5 seeds)"),
        ("var50_3seed_results.json", 98, "VAR d=50 stationary (3 seeds)"),
        ("nsvar50_3seed_results.json", 72, "NSVAR d=50 PlanA (3 seeds)"),
        ("causaltime_results.json", None, "CausalTime (medical/pm25/traffic)"),
        ("fmri_3subj_results.json", 45, "fMRI d=15 (3 subjects)"),
    ]

    for fname, edges, label in files_to_process:
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            try:
                enriched, summary = process_file(fpath, edges, label)
                if summary:
                    all_summaries[label] = summary
            except Exception as e:
                print(f"  ERROR: {e}")
        else:
            print(f"\n  SKIP {fname} (not found)")

    # Save enriched summary
    output = {}
    for label, summary in all_summaries.items():
        output[label] = summary

    with open("enriched_metrics_summary.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved enriched summary to enriched_metrics_summary.json")


if __name__ == "__main__":
    main()
