"""Auto-generate appendix tables (AUPRC, nSHD, MCC) from results/raw/ JSON files.

Reads all collections in results/raw/, extracts metrics, produces:
  - appendix_metrics.csv — machine-readable
  - appendix_metrics.tex — LaTeX table body

Usage:
    python experiments/generate_appendix_tables.py
    python experiments/generate_appendix_tables.py --output-dir results/tables/
"""

import json, os, sys, argparse
from collections import defaultdict
import numpy as np

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from src.schema import load_collection, validate_collection, expected_results_dir


def load_all_results(results_dir):
    """Load from migrated_all_v2.json (canonical v2). Falls back to v1, then directory scan."""
    # Prefer v2 (canonical, PlanA removed, 176 entries, 11 datasets)
    for preferred in ["migrated_all_v2.json", "migrated_all.json"]:
        migrated_path = os.path.join(results_dir, preferred)
        if os.path.exists(migrated_path):
            try:
                data = load_collection(migrated_path)
                entries = data.get("results", [])
                if entries:
                    print(f"  Loaded {len(entries)} entries from {preferred}")
                    return entries
            except Exception as e:
                print(f"  Warning: failed to load {preferred}: {e}")
    # Fallback: scan all files
    all_entries = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(results_dir, fname)
        if fname == "factorial_ablation_canonical.json":
            continue
        try:
            data = load_collection(path)
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            continue
        if "results" in data and isinstance(data["results"], list):
            for e in data["results"]:
                key = (e.get("dataset", "?"), e.get("method", "?"), e.get("seed", "?"))
                if key not in all_entries:
                    all_entries[key] = e
                else:
                    existing = all_entries[key]
                    if len(e.get("metrics", {})) > len(existing.get("metrics", {})):
                        all_entries[key] = e
                    elif e.get("provenance", {}).get("source") == "rerun" \
                            and existing.get("provenance", {}).get("source") != "rerun":
                        all_entries[key] = e
        return list(all_entries.values())


def group_entries(entries, metric):
    """Group entries by (dataset, method), compute per-group mean±std.

    Returns dict: dataset -> method -> {mean, std, n, values}
    """
    groups = defaultdict(lambda: defaultdict(list))
    for e in entries:
        ds = e.get("dataset", "?")
        m = e.get("method", "?")
        if metric in e.get("metrics", {}):
            groups[ds][m].append(e["metrics"][metric])

    summary = {}
    for ds in sorted(groups):
        summary[ds] = {}
        for m in sorted(groups[ds]):
            vals = groups[ds][m]
            summary[ds][m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
    return summary


def format_mean_std(mean, std, n_digits=4):
    """Format as 'mean ± std' (plain text / CSV)"""
    return f"{mean:.{n_digits}f} ± {std:.{n_digits}f}"


def format_mean_std_latex(mean, std, n_digits=4):
    r"""Format as 'mean $\pm$ std' (LaTeX math mode)"""
    return f"{mean:.{n_digits}f} $\\pm$ {std:.{n_digits}f}"


def generate_latex_table(summary, metric_name, datasets, methods):
    """Generate LaTeX table body for one metric."""
    lines = []
    header = "Dataset & " + " & ".join(methods) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for ds in datasets:
        if ds not in summary:
            continue
        cells = [ds.replace("_", "\\_")]
        for m in methods:
            if m in summary[ds]:
                s = summary[ds][m]
                cells.append(format_mean_std_latex(s["mean"], s["std"]))
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines)


def generate_csv(summary, datasets, methods):
    """Generate CSV for all metrics."""
    lines = ["dataset,method,mean,std,n"]
    for ds in datasets:
        if ds not in summary:
            continue
        for m in methods:
            if m in summary[ds]:
                s = summary[ds][m]
                lines.append(f"{ds},{m},{s['mean']:.4f},{s['std']:.4f},{s['n']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir or expected_results_dir()
    out_dir = args.output_dir or os.path.join(_PROJ_ROOT, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)

    entries = load_all_results(results_dir)
    if not entries:
        print(f"No entries found in {results_dir}")
        return

    print(f"Loaded {len(entries)} result entries from {results_dir}")

    # Determine available datasets and methods
    datasets = sorted(set(e["dataset"] for e in entries))
    methods = sorted(set(e["method"] for e in entries))
    print(f"Datasets: {datasets}")
    print(f"Methods: {methods}")

    metrics = ["auroc", "auprc", "nshd_topk", "mcc_topk", "f1", "acc", "shd_topk"]
    all_csv_lines = []
    all_tex = {}

    for metric in metrics:
        summary = group_entries(entries, metric)
        if not summary:
            continue

        # CSV
        csv_path = os.path.join(out_dir, f"appendix_{metric}.csv")
        csv_content = generate_csv(summary, datasets, methods)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)
        print(f"  {csv_path}")

        # LaTeX
        tex_body = generate_latex_table(summary, metric, datasets, methods)
        tex_path = os.path.join(out_dir, f"appendix_{metric}.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_body)
        print(f"  {tex_path}")
        all_tex[metric] = tex_body

    # Combined LaTeX
    combined_path = os.path.join(out_dir, "appendix_all_metrics.tex")
    with open(combined_path, "w", encoding="utf-8") as f:
        for metric in metrics:
            if metric in all_tex:
                f.write(f"\n% ==== {metric.upper()} ====\n")
                f.write(all_tex[metric])
                f.write("\n")
    print(f"\nCombined: {combined_path}")
    print(f"\n{len(entries)} entries, {len(datasets)} datasets, {len(methods)} methods")


if __name__ == "__main__":
    main()
