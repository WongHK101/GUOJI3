"""Merge factorial JSONs into canonical file with full protocol/audit metadata.

Per expert requirement, the output includes:
- Full D2 params and design protocol
- Audit trail (source files, git info, date)
- Selection rule documentation
- Primary (10-seed, max_iter=2000) and Appendix (3-seed, max_iter=5000) sections
"""

import json, os, subprocess
from datetime import date

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Get git info if available
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(RESULTS_DIR),
        stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    git_commit = "N/A (not a git repo)"

# Load source files
with open(os.path.join(RESULTS_DIR, "factorial_D2_10seed_iter2000.json")) as f:
    d10 = json.load(f)

with open(os.path.join(RESULTS_DIR, "factorial_D2_10seed_tcn.json")) as f:
    dtcn = json.load(f)

with open(os.path.join(RESULTS_DIR, "factorial_D2_3seed_iter5000.json")) as f:
    d5k = json.load(f)

# --- Build per-seed primary data (merge baseline/mamba from d10 with tcn from dtcn) ---
primary_seeds = {}
for seed_str in d10["D2"]["seeds"]:
    seed_data = d10["D2"]["seeds"][seed_str]
    tcn_seed_data = dtcn["D2"]["seeds"].get(seed_str, {})
    primary_seeds[f"seed_{seed_str}"] = {}
    for cell_name in ["Stat+Linear", "Stat+Nonlinear", "NS+Linear", "NS+Nonlinear"]:
        cell = {}
        if cell_name in seed_data:
            if "baseline" in seed_data[cell_name]:
                cell["baseline"] = seed_data[cell_name]["baseline"]
            if "mamba" in seed_data[cell_name]:
                cell["mamba"] = seed_data[cell_name]["mamba"]
        if cell_name in tcn_seed_data and "tcn" in tcn_seed_data[cell_name]:
            cell["tcn"] = tcn_seed_data[cell_name]["tcn"]
        primary_seeds[f"seed_{seed_str}"][cell_name] = cell

# --- Build per-seed appendix data (3-seed iter=5000) ---
appendix_seeds = {}
for seed_str in d5k["D2"]["seeds"]:
    seed_data = d5k["D2"]["seeds"][seed_str]
    appendix_seeds[f"seed_{seed_str}"] = {}
    for cell_name in ["Stat+Linear", "Stat+Nonlinear", "NS+Linear", "NS+Nonlinear"]:
        if cell_name in seed_data:
            appendix_seeds[f"seed_{seed_str}"][cell_name] = seed_data[cell_name]

# --- Compute summary statistics ---
import numpy as np

def compute_summary(seeds_dict, models):
    cells = ["Stat+Linear", "Stat+Nonlinear", "NS+Linear", "NS+Nonlinear"]
    summary = {}
    for cell in cells:
        summary[cell] = {}
        for m in models:
            vals = []
            for sk in seeds_dict:
                if cell in seeds_dict[sk] and m in seeds_dict[sk][cell]:
                    vals.append(seeds_dict[sk][cell][m]["summary_max"]["auroc"])
            if vals:
                summary[cell][m] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "n": len(vals),
                    "values": [round(float(v), 4) for v in vals]
                }
        # Compute deltas
        if "baseline" in summary[cell]:
            bm = summary[cell]["baseline"]["mean"]
            for m in models:
                if m != "baseline" and m in summary[cell]:
                    mm = summary[cell][m]["mean"]
                    # Paired deltas
                    bv = np.array([seeds_dict[sk][cell]["baseline"]["summary_max"]["auroc"]
                                   for sk in seeds_dict
                                   if cell in seeds_dict[sk] and "baseline" in seeds_dict[sk][cell]
                                   and m in seeds_dict[sk][cell]])
                    mv = np.array([seeds_dict[sk][cell][m]["summary_max"]["auroc"]
                                   for sk in seeds_dict
                                   if cell in seeds_dict[sk] and "baseline" in seeds_dict[sk][cell]
                                   and m in seeds_dict[sk][cell]])
                    deltas = mv - bv
                    summary[cell][f"delta_{m}_minus_baseline"] = {
                        "mean": round(float(np.mean(deltas)), 4),
                        "std": round(float(np.std(deltas)), 4),
                        "values": [round(float(d), 4) for d in deltas]
                    }
    return summary

primary_models = ["baseline", "mamba", "tcn"]
appendix_models = ["baseline", "mamba"]

canonical = {
    "_audit": {
        "created": str(date.today()),
        "git_commit": git_commit,
        "script": "experiments/merge_factorial_canonical.py",
        "source_files": [
            "results/factorial_D2_10seed_iter2000.json",
            "results/factorial_D2_10seed_tcn.json",
            "results/factorial_D2_3seed_iter5000.json"
        ]
    },
    "protocol": {
        "setting": "D2",
        "params": {
            "coeff_scale": 0.40,
            "noise_scale": 0.15,
            "regime_shift_strength": 0.20,
            "nonlinear_strength": 0.50
        },
        "design": {
            "d": 10, "T": 600, "lag": 3,
            "sparsity": 0.2,
            "cells": ["Stat+Linear", "Stat+Nonlinear", "NS+Linear", "NS+Nonlinear"],
            "shared_graph_per_seed": True,
            "graph_generation": "VAR sparsity=0.2, random lag k~U(0,lag-1) per edge",
            "nonlinearity": "pred = (1-α)pred + α·s·tanh(pred/s), s=std(pred)",
            "nonstationarity": "A_k(t) = A_k(0) + drift_scale * smoothed_random_walk(t)"
        },
        "selection_rule": "Chosen by baseline Stat+Linear summary_max AUROC in [0.75, 0.90]. Not selected based on Mamba delta.",
        "metric": "summary_max (GT = any edge at any lag, PR = max(|gc_pred|, axis=2))",
        "calibration_history": "A/B/C (too hard) → D/E/F/D2 → D2 selected"
    },
    "primary": {
        "max_iter": 2000,
        "n_seeds": 10,
        "models": primary_models,
        "summary": compute_summary(primary_seeds, primary_models),
        "per_seed": primary_seeds
    },
    "appendix": {
        "convergence_diagnostic": {
            "max_iter": 5000,
            "n_seeds": 3,
            "models": appendix_models,
            "note": "Extended training degrades Mamba in all cells, especially NS. Included as over-training diagnostic, not as primary evidence.",
            "summary": compute_summary(appendix_seeds, appendix_models),
            "per_seed": appendix_seeds
        }
    }
}

out_path = os.path.join(RAW_DIR, "factorial_ablation_canonical.json")
with open(out_path, "w") as f:
    json.dump(canonical, f, indent=2)
print(f"Written to {out_path}")
print(f"Size: {os.path.getsize(out_path)} bytes")

# Print quick summary for verification
print("\nPrimary summary (10-seed):")
for cell in ["Stat+Linear", "Stat+Nonlinear", "NS+Linear", "NS+Nonlinear"]:
    parts = []
    for m in primary_models:
        if m in canonical["primary"]["summary"][cell]:
            s = canonical["primary"]["summary"][cell][m]
            parts.append(f"{m}={s['mean']:.4f}±{s['std']:.3f}")
    print(f"  {cell:<18} {'  '.join(parts)}")
