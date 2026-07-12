"""Generate the final Phase 8 manuscript figures from frozen artifacts only.

This script performs no model training. It reads the frozen Track A replication,
P0 semantic-audit, and final bounded-lambda aggregate artifacts, exports source
data tables, and renders editable SVG/PDF plus 600-dpi PNG files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


COLORS = {
    "ink": "#202124",
    "gray": "#6B6F73",
    "grid": "#D7DADC",
    "neutral": "#E8EAED",
    "raw": "#356AA0",
    "raw_soft": "#DCE8F4",
    "aux": "#B84A4A",
    "aux_soft": "#F3DCDC",
    "audit": "#287D78",
    "audit_soft": "#D9ECEA",
    "gold": "#B58218",
    "gold_soft": "#F4E7C1",
    "pass": "#3E7C59",
    "pass_soft": "#DDEBDD",
    "fail": "#A33F3F",
    "fail_soft": "#F2D7D7",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.06,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
    )


def save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, dpi in (("svg", None), ("pdf", None), ("png", 600)):
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.03}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
        outputs.append(path)
    plt.close(fig)
    return outputs


def box(ax, xy, width, height, text, *, face, edge, fontsize=7.0, weight="normal"):
    patch = Rectangle(xy, width, height, facecolor=face, edgecolor=edge, linewidth=1.0)
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["ink"],
        linespacing=1.15,
    )
    return patch


def arrow(ax, start, end, *, color, dashed=False, width=1.5):
    sx, sy = start
    ex, ey = end
    if abs(sx - ex) > 1e-8 and abs(sy - ey) > 1e-8:
        raise ValueError("Figure 1 connectors must be horizontal or vertical")
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            linestyle=(0, (3, 2)) if dashed else "-",
            color=color,
        )
    )


def frame_panel(ax, label: str, title: str) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (0.12, 0.12),
            9.76,
            9.68,
            facecolor="none",
            edgecolor=COLORS["gray"],
            linewidth=0.9,
            linestyle=(0, (3, 2)),
        )
    )
    ax.text(0.45, 9.38, label, fontsize=10, fontweight="bold", va="center")
    ax.text(1.15, 9.38, title, fontsize=8.5, fontweight="bold", va="center")


def draw_figure1(output_dir: Path) -> list[Path]:
    fig = plt.figure(figsize=(7.25, 3.15))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.10, 1.12, 1.08], wspace=0.05)

    ax = fig.add_subplot(grid[0, 0])
    frame_panel(ax, "a", "Declared predictive routes")
    box(ax, (0.55, 6.55), 2.35, 1.25, "raw history\n$X_{t-K:t-1}$", face=COLORS["raw_soft"], edge=COLORS["raw"], fontsize=5.8, weight="bold")
    box(ax, (0.55, 4.35), 2.35, 1.25, "auxiliary input\n$c_t$", face=COLORS["aux_soft"], edge=COLORS["aux"], fontsize=5.8, weight="bold")
    box(ax, (4.05, 4.35), 2.65, 3.45, "conditioned\npredictor\n$f_\\theta(X,c)$", face="#F7F7F7", edge=COLORS["ink"], fontsize=5.8, weight="bold")
    box(ax, (7.75, 5.45), 1.65, 1.25, "prediction\n$\\hat{x}_t$", face=COLORS["audit_soft"], edge=COLORS["audit"], fontsize=5.5, weight="bold")
    arrow(ax, (2.90, 7.17), (4.05, 7.17), color=COLORS["raw"])
    arrow(ax, (2.90, 4.97), (4.05, 4.97), color=COLORS["aux"])
    arrow(ax, (6.70, 6.07), (7.75, 6.07), color=COLORS["ink"])
    box(
        ax,
        (0.55, 1.05),
        8.85,
        2.15,
        "reported partial score\n$S^x \\leftarrow \\partial \\hat{x}/\\partial X$ with $c$ fixed\nauxiliary predictive route omitted",
        face=COLORS["neutral"],
        edge=COLORS["gray"],
        fontsize=6.4,
        weight="bold",
    )

    ax = fig.add_subplot(grid[0, 1])
    frame_panel(ax, "b", "Coverage declaration")
    box(
        ax,
        (0.45, 7.75),
        9.10,
        1.05,
        "$C=(V_{score},V_{penalty},P_{pred},M_{coord},H_{attr})$",
        face="#F7F7F7",
        edge=COLORS["ink"],
        fontsize=6.4,
        weight="bold",
    )
    headers = ["route class", "score", "penalty", "exempt"]
    widths = [3.55, 1.65, 1.90, 1.65]
    x0 = [0.45, 4.10, 5.85, 7.85]
    for x, w, text in zip(x0, widths, headers):
        ax.text(x + w / 2, 7.18, text, ha="center", va="center", fontsize=6.1, fontweight="bold")
    rows = [
        (5.72, "raw history $X$", "yes", "yes", "no", COLORS["raw_soft"], COLORS["raw"]),
        (4.17, "auxiliary $c$", "no", "no", "no", COLORS["aux_soft"], COLORS["aux"]),
    ]
    for y, route, scored, penalized, exempt, face, edge in rows:
        for x, w, text, cell_face in zip(x0, widths, [route, scored, penalized, exempt], [face, "#FAFAFA", "#FAFAFA", "#FAFAFA"]):
            box(ax, (x, y), w, 1.05, text, face=cell_face, edge=edge, fontsize=6.3, weight="bold")
    box(
        ax,
        (0.45, 1.00),
        9.10,
        2.15,
        "partial: $J^x=\\partial f(X,c)/\\partial X$\n"
        "total (if $c=g(X)$): $d f/dX=J^x+J^c\\,dg/dX$\n"
        "score and penalty coverage are audited separately",
        face=COLORS["audit_soft"],
        edge=COLORS["audit"],
        fontsize=5.9,
        weight="bold",
    )

    ax = fig.add_subplot(grid[0, 2])
    frame_panel(ax, "c", "Claim-specific audit")
    dimensions = [
        "A  score-route completeness",
        "B  penalty-route completeness",
        "C  score--penalty alignment",
        "D  coordinate identity",
        "E  attribution horizon",
    ]
    for idx, text in enumerate(dimensions):
        box(
            ax,
            (0.55, 7.75 - idx * 1.28),
            8.90,
            0.90,
            text,
            face=COLORS["audit_soft"] if idx < 3 else COLORS["neutral"],
            edge=COLORS["audit"] if idx < 3 else COLORS["gray"],
            fontsize=6.1,
            weight="bold",
        )
    ax.text(0.70, 2.00, "diagnostic outputs", fontsize=6.0, fontweight="bold", va="center")
    labels = ["COVERED", "PARTIAL", "COORD-AMBIG.", "HORIZON-TRUNC.", "UNASSESSED"]
    fills = [COLORS["pass_soft"], COLORS["gold_soft"], COLORS["aux_soft"], COLORS["neutral"], "#FAFAFA"]
    edges = [COLORS["pass"], COLORS["gold"], COLORS["aux"], COLORS["gray"], COLORS["gray"]]
    for idx, (text, face, edge) in enumerate(zip(labels, fills, edges)):
        x = 0.55 + (idx % 3) * 3.0
        y = 1.08 if idx < 3 else 0.30
        if idx >= 3:
            x = 2.05 + (idx - 3) * 3.35
        box(ax, (x, y), 2.65 if idx < 3 else 3.0, 0.62, text, face=face, edge=edge, fontsize=4.5, weight="bold")

    return save_figure(fig, output_dir, "fig1_jacobian_coverage_audit_phase8_final")


def _capacity_arrays(track: Mapping[str, object]):
    dconds = np.asarray([0, 1, 2, 4, 8, 16], dtype=int)
    mse_rows, auroc_rows = [], []
    for pair in track["capacity"]["pairs"]:  # type: ignore[index]
        lookup = {int(row["d_cond"]): row for row in pair["concat_by_capacity"]}
        mse_rows.append([pair["baseline"]["fixed_target_prediction_mse"]] + [lookup[x]["fixed_target_prediction_mse"] for x in dconds[1:]])
        auroc_rows.append([pair["baseline"]["partial_nominal_auroc"]] + [lookup[x]["partial_nominal_auroc"] for x in dconds[1:]])
    return dconds, np.asarray(mse_rows), np.asarray(auroc_rows)


def _coefficient_rows(track_root: Path) -> list[dict]:
    rows = []
    for seed in range(11201, 11206):
        model_seed = seed + 10000
        base = load_json(track_root / "runs" / f"P8-COEF-D{seed}-M{model_seed}-baseline_jrngc" / "metrics.json")
        concat = load_json(track_root / "runs" / f"P8-COEF-D{seed}-M{model_seed}-concat_x_only" / "metrics.json")
        rows.append(
            {
                "data_seed": seed,
                "baseline_auroc": base["graph_metrics"]["partial_nominal"]["auroc"],
                "concat_partial_auroc": concat["graph_metrics"]["partial_nominal"]["auroc"],
                "concat_total_auroc": concat["graph_metrics"]["total_nominal"]["auroc"],
                "baseline_coefficient_r": base["coefficient_r_partial_lag1"],
                "concat_partial_coefficient_r": concat["coefficient_r_partial_lag1"],
                "concat_total_coefficient_r": concat["coefficient_r_total_lag1"],
            }
        )
    return rows


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def seed_lines(ax, x, values, *, color, mean_color, ylabel, title):
    values = np.asarray(values, dtype=float)
    for row in values:
        ax.plot(x, row, color=color, alpha=0.28, linewidth=0.8, marker="o", markersize=2.5)
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=0)
    ax.errorbar(x, mean, yerr=sd, color=mean_color, linewidth=1.8, marker="o", markersize=4.0, capsize=2.2, label="mean +/- SD")
    ax.set_xlabel(r"auxiliary dimension $d_{cond}$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_title(title, loc="left", fontsize=8.3, fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.7)


def draw_figure2(track_root: Path, output_dir: Path, source_dir: Path) -> list[Path]:
    track = load_json(track_root / "replication_aggregate_and_gates.json")
    dconds, mse, auroc = _capacity_arrays(track)
    coeff = _coefficient_rows(track_root)
    interventions = track["fixed_target_interventions"]["pairs"]

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15))
    ax = axes[0, 0]
    seed_lines(ax, dconds, mse, color=COLORS["gray"], mean_color=COLORS["aux"], ylabel="fixed-target pure MSE", title="Prediction improves with auxiliary capacity")
    panel_label(ax, "a")

    ax = axes[0, 1]
    seed_lines(ax, dconds, auroc, color=COLORS["gray"], mean_color=COLORS["raw"], ylabel="partial nominal AUROC", title="Partial graph fidelity degrades")
    ax.set_ylim(0.15, 1.03)
    panel_label(ax, "b")

    ax = axes[1, 0]
    metric_labels = ["AUROC", "coefficient r"]
    methods = ["Baseline", "Concat partial", "Concat total"]
    method_colors = [COLORS["gray"], COLORS["aux"], COLORS["audit"]]
    for metric_idx, metric in enumerate(("auroc", "coefficient_r")):
        for method_idx, method in enumerate(("baseline", "concat_partial", "concat_total")):
            values = [row[f"{method}_{metric}"] for row in coeff]
            xpos = metric_idx * 4 + method_idx
            jitter = np.linspace(-0.10, 0.10, len(values))
            ax.scatter(xpos + jitter, values, s=14, color=method_colors[method_idx], alpha=0.58, zorder=3)
            mean, sd = mean_sd(values)
            ax.errorbar(xpos, mean, yerr=sd, color=method_colors[method_idx], marker="s", markersize=5, capsize=2.2, linewidth=1.2, zorder=4)
    ax.set_xticks([1, 5])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("direct graph metric")
    ax.set_title("Graph and coefficient degradation replicate", loc="left", fontsize=8.3, fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    handles = [plt.Line2D([], [], color=color, marker="s", linestyle="", label=label) for color, label in zip(method_colors, methods)]
    ax.legend(handles=handles, loc="lower left", fontsize=6.2, ncol=1)
    panel_label(ax, "c")

    ax = axes[1, 1]
    conditions = ["Mask", "Shuffle"]
    raw_values = [
        [pair["concat_fixed_target_prediction_mse_delta"]["mask_x"] for pair in interventions],
        [pair["concat_fixed_target_prediction_mse_delta"]["shuffle_x_only"] for pair in interventions],
    ]
    aux_values = [
        [pair["concat_fixed_target_prediction_mse_delta"]["mask_c"] for pair in interventions],
        [pair["concat_fixed_target_prediction_mse_delta"]["shuffle_c_only"] for pair in interventions],
    ]
    for idx, condition in enumerate(conditions):
        for offset, values, color, label in ((-0.18, raw_values[idx], COLORS["raw"], "raw history X"), (0.18, aux_values[idx], COLORS["aux"], "auxiliary c")):
            x = idx + offset
            jitter = np.linspace(-0.04, 0.04, len(values))
            ax.scatter(x + jitter, values, color=color, alpha=0.55, s=15, zorder=3)
            mean, sd = mean_sd(values)
            ax.errorbar(x, mean, yerr=sd, color=color, marker="s", markersize=5, capsize=2.2, linewidth=1.2, zorder=4)
        for raw, aux in zip(raw_values[idx], aux_values[idx]):
            ax.plot([idx - 0.18, idx + 0.18], [raw, aux], color=COLORS["grid"], linewidth=0.7, zorder=1)
    ax.set_xticks(range(2))
    ax.set_xticklabels(conditions)
    ax.set_ylabel(r"$\Delta$ fixed-target pure MSE")
    ax.set_title("Corrected interventions favor raw history", loc="left", fontsize=8.3, fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    handles = [plt.Line2D([], [], color=COLORS["raw"], marker="s", linestyle="", label="raw history X"), plt.Line2D([], [], color=COLORS["aux"], marker="s", linestyle="", label="auxiliary c")]
    ax.legend(handles=handles, loc="upper right", fontsize=6.2)
    panel_label(ax, "d")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.95, bottom=0.09, wspace=0.31, hspace=0.36)

    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "figure2_capacity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["paired_seed_index", "d_cond", "fixed_target_prediction_mse", "partial_nominal_auroc"])
        for seed_idx in range(mse.shape[0]):
            for idx, d_cond in enumerate(dconds):
                writer.writerow([seed_idx + 1, int(d_cond), mse[seed_idx, idx], auroc[seed_idx, idx]])
    with (source_dir / "figure2_coefficient.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coeff[0]))
        writer.writeheader()
        writer.writerows(coeff)
    with (source_dir / "figure2_interventions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["data_seed", "condition", "raw_history_delta", "auxiliary_delta"])
        for idx, pair in enumerate(interventions):
            writer.writerow([pair["data_seed"], "mask", raw_values[0][idx], aux_values[0][idx]])
            writer.writerow([pair["data_seed"], "shuffle", raw_values[1][idx], aux_values[1][idx]])
    return save_figure(fig, output_dir, "fig2_replicated_decoupling_phase8_final")


def draw_figure3(track_root: Path, p0_paths: Sequence[Path], output_dir: Path, source_dir: Path) -> list[Path]:
    audits = [load_json(path) for path in p0_paths]
    coeff = _coefficient_rows(track_root)

    fig = plt.figure(figsize=(7.25, 3.15))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.02, 0.95, 1.15], wspace=0.42)

    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    panel_label(ax, "a")
    ax.text(0.0, 9.45, "Four derivative objects", fontsize=8.3, fontweight="bold", ha="left")
    entries = [
        ("partial raw", "$J^x=\\partial f/\\partial X$", COLORS["raw_soft"], COLORS["raw"]),
        ("auxiliary", "$J^c=\\partial f/\\partial c$", COLORS["aux_soft"], COLORS["aux"]),
        ("filtered", "$J^{x'}=\\partial f/\\partial X'$", COLORS["gold_soft"], COLORS["gold"]),
        ("raw chain", "$J^{raw}=d f(F(X))/dX$", COLORS["audit_soft"], COLORS["audit"]),
    ]
    for idx, (name, formula, face, edge) in enumerate(entries):
        box(ax, (0.10, 7.35 - idx * 1.85), 9.55, 1.25, f"{name}\n{formula}", face=face, edge=edge, fontsize=6.3, weight="bold")
    ax.text(4.85, 0.45, "State coordinates and horizon before graph interpretation.", ha="center", fontsize=5.8, color=COLORS["gray"])

    ax = fig.add_subplot(grid[0, 1])
    values = {
        "Concat\npartial/total r": [row["concat_partial_vs_total_derivative"]["offdiag_score_correlation"] for row in audits],
        "Legacy\nfiltered/raw r": [row["istf_mamba_filtered_vs_raw_chain"]["offdiag_score_correlation"] for row in audits],
        "Legacy\ntop-k Jaccard": [row["istf_mamba_filtered_vs_raw_chain"]["topk_jaccard"] for row in audits],
    }
    colors = [COLORS["aux"], COLORS["gold"], COLORS["gray"]]
    for idx, ((label, vals), color) in enumerate(zip(values.items(), colors)):
        jitter = np.linspace(-0.10, 0.10, len(vals))
        ax.scatter(idx + jitter, vals, color=color, alpha=0.55, s=16)
        mean, sd = mean_sd(vals)
        ax.errorbar(idx, mean, yerr=sd, marker="s", markersize=5, color=color, capsize=2.3, linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score agreement")
    ax.set_xticks(range(3))
    ax.set_xticklabels(values.keys(), fontsize=5.8)
    ax.set_title("Five-seed semantic audit", loc="left", x=0.10, fontsize=8.3, fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    panel_label(ax, "b")

    ax = fig.add_subplot(grid[0, 2])
    metrics = ["AUROC", "coefficient r"]
    methods = ["Baseline", "Concat partial", "Concat total"]
    keys = [
        ("baseline_auroc", "concat_partial_auroc", "concat_total_auroc"),
        ("baseline_coefficient_r", "concat_partial_coefficient_r", "concat_total_coefficient_r"),
    ]
    colors = [COLORS["gray"], COLORS["aux"], COLORS["audit"]]
    for metric_idx, key_set in enumerate(keys):
        for method_idx, key in enumerate(key_set):
            vals = [row[key] for row in coeff]
            x = metric_idx * 4 + method_idx
            ax.scatter(x + np.linspace(-0.08, 0.08, len(vals)), vals, color=colors[method_idx], alpha=0.48, s=14)
            mean, sd = mean_sd(vals)
            ax.errorbar(x, mean, yerr=sd, marker="s", markersize=5, color=colors[method_idx], capsize=2.2, linewidth=1.2)
    ax.set_xticks([1, 5])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("direct graph metric")
    ax.set_title("Total-score-only evaluation does not repair", loc="left", x=0.10, fontsize=7.9, fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    handles = [plt.Line2D([], [], color=color, marker="s", linestyle="", label=label) for color, label in zip(colors, methods)]
    ax.legend(handles=handles, loc="lower left", fontsize=5.9)
    panel_label(ax, "c")

    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.14)
    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "figure3_semantic_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "concat_partial_total_pearson", "legacy_filtered_raw_pearson", "legacy_filtered_raw_topk_jaccard"])
        for idx, audit in enumerate(audits):
            writer.writerow(
                [
                    idx,
                    audit["concat_partial_vs_total_derivative"]["offdiag_score_correlation"],
                    audit["istf_mamba_filtered_vs_raw_chain"]["offdiag_score_correlation"],
                    audit["istf_mamba_filtered_vs_raw_chain"]["topk_jaccard"],
                ]
            )
    return save_figure(fig, output_dir, "fig3_score_semantics_phase8_final")


def _tradeoff_rows(final: Mapping[str, object]) -> list[dict]:
    output = []
    for seed_row in final["data_seed_level_values"]:  # type: ignore[index]
        seed = int(seed_row["data_seed"])
        concat = seed_row["comparators"]["concat_x_only"]
        methods = {
            "Concat x-only": concat,
            "Baseline JRNGC": seed_row["comparators"]["baseline_jrngc"],
            "Full auxiliary lc10": seed_row["comparators"]["full_aux_lc10"],
            r"$\lambda=3\times10^{-4}$": seed_row["repairs"]["lambda_0.0003"],
            r"$\lambda=10^{-3}$": seed_row["repairs"]["lambda_0.001"],
            r"$\lambda=3\times10^{-3}$": seed_row["repairs"]["lambda_0.003"],
            r"$\lambda=10^{-2}$": seed_row["repairs"]["lambda_0.01"],
        }
        for method, metrics in methods.items():
            output.append(
                {
                    "data_seed": seed,
                    "method": method,
                    "relative_mse_percent": 100.0 * (metrics["fixed_target_prediction_mse"] - concat["fixed_target_prediction_mse"]) / concat["fixed_target_prediction_mse"],
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "coefficient_r": metrics["coefficient_r"],
                    "fixed_target_prediction_mse": metrics["fixed_target_prediction_mse"],
                }
            )
    return output


def draw_figure4(final_path: Path, output_dir: Path, source_dir: Path) -> list[Path]:
    final = load_json(final_path)
    rows = _tradeoff_rows(final)
    labels = ["Concat x-only", "Baseline JRNGC", "Full auxiliary lc10", r"$\lambda=3\times10^{-4}$", r"$\lambda=10^{-3}$", r"$\lambda=3\times10^{-3}$", r"$\lambda=10^{-2}$"]
    colors = [COLORS["aux"], COLORS["gray"], COLORS["gold"], "#6BA7A1", "#398D87", "#20706C", "#124F4C"]
    markers = ["X", "s", "D", "o", "o", "o", "o"]

    fig = plt.figure(figsize=(7.25, 5.25))
    grid = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.34)
    for panel_idx, (metric, ylabel, title) in enumerate(
        [
            ("auroc", "total nominal AUROC", "Direct graph ranking"),
            ("auprc", "total nominal AUPRC", "Class-imbalance-aware recovery"),
            ("coefficient_r", "lag-1 coefficient r", "Coefficient fidelity"),
        ]
    ):
        ax = fig.add_subplot(grid[panel_idx // 2, panel_idx % 2])
        for label, color, marker in zip(labels, colors, markers):
            group = [row for row in rows if row["method"] == label]
            xvals = [row["relative_mse_percent"] for row in group]
            yvals = [row[metric] for row in group]
            ax.scatter(xvals, yvals, color=color, alpha=0.40, s=18, marker=marker, zorder=2)
            xmean, xsd = mean_sd(xvals)
            ymean, ysd = mean_sd(yvals)
            ax.errorbar(xmean, ymean, xerr=xsd, yerr=ysd, color=color, marker=marker, markersize=5.5, capsize=2.2, linewidth=1.1, label=label, zorder=3)
        ax.axvline(10.0, color=COLORS["fail"], linestyle=(0, (3, 2)), linewidth=0.9)
        ax.text(11.5, 0.04, "mean MSE gate", transform=ax.get_xaxis_transform(), color=COLORS["fail"], fontsize=5.7, va="bottom")
        ax.set_xlabel("pure-MSE degradation vs concat (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=8.3, fontweight="bold")
        ax.grid(color=COLORS["grid"], linewidth=0.55, alpha=0.65)
        panel_label(ax, chr(ord("a") + panel_idx))

    ax = fig.add_subplot(grid[1, 1])
    lambda_reports = final["lambda_reports"]
    rows_labels = ["0.0003", "0.001", "0.003", "0.01"]
    columns = ["effect", "pure MSE", "safety", "semantic", "eligible"]
    matrix = []
    for report in lambda_reports:
        matrix.append(
            [
                report["repair_minus_concat"]["effect_gate_passed"],
                report["repair_minus_concat"]["pure_mse_gate_passed"],
                report["comparator_safety"]["passed"],
                report["semantic_compute_gate_passed"],
                report["eligible"],
            ]
        )
    ax.set_xlim(0, len(columns))
    ax.set_ylim(0, len(rows_labels))
    for ridx, row in enumerate(matrix):
        y = len(rows_labels) - ridx - 1
        for cidx, passed in enumerate(row):
            face = COLORS["pass_soft"] if passed else COLORS["fail_soft"]
            edge = COLORS["pass"] if passed else COLORS["fail"]
            ax.add_patch(Rectangle((cidx + 0.05, y + 0.08), 0.9, 0.84, facecolor=face, edgecolor=edge, linewidth=0.8))
            ax.text(cidx + 0.5, y + 0.5, "PASS" if passed else "FAIL", ha="center", va="center", fontsize=5.5, fontweight="bold", color=edge)
    ax.set_xticks(np.arange(len(columns)) + 0.5)
    ax.set_xticklabels(columns, rotation=25, ha="right", fontsize=6.1)
    ax.set_yticks(np.arange(len(rows_labels)) + 0.5)
    ax.set_yticklabels(rows_labels[::-1])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Pre-specified pilot-go decision", loc="left", fontsize=8.3, fontweight="bold")
    ax.text(0.02, -0.20, "No lambda eligible; confirmation not executed", transform=ax.transAxes, fontsize=6.5, color=COLORS["fail"], fontweight="bold")
    panel_label(ax, "d")

    legend_handles = [
        plt.Line2D([], [], color=color, marker=marker, linestyle="", label=label)
        for label, color, marker in zip(labels, colors, markers)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=4,
        fontsize=5.7,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.09)
    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "figure4_repair_tradeoff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (source_dir / "figure4_gate_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lambda", *columns])
        for label, row in zip(rows_labels, matrix):
            writer.writerow([label, *row])
    return save_figure(fig, output_dir, "fig4_coverage_repair_tradeoff_phase8_final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-a-root", type=Path, required=True)
    parser.add_argument("--p0-audit-dir", type=Path, required=True)
    parser.add_argument("--final-aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    p0_paths = sorted(args.p0_audit_dir.glob("p0_jacobian_semantics_d6_iter120_refactor_seed*.json"))
    if len(p0_paths) != 5:
        raise FileNotFoundError(f"Expected five P0 audit files, found {len(p0_paths)}")
    inputs = [
        args.track_a_root / "replication_aggregate_and_gates.json",
        args.final_aggregate,
        *p0_paths,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    generated: list[Path] = []
    generated += draw_figure1(args.output_dir)
    generated += draw_figure2(args.track_a_root, args.output_dir, args.source_data_dir)
    generated += draw_figure3(args.track_a_root, p0_paths, args.output_dir, args.source_data_dir)
    generated += draw_figure4(args.final_aggregate, args.output_dir, args.source_data_dir)
    manifest = {
        "purpose": "Final Phase 8 manuscript figures generated exclusively from frozen artifacts.",
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs],
        "generated": [{"path": str(path), "sha256": sha256(path)} for path in generated],
        "source_data": [
            {"path": str(path), "sha256": sha256(path)}
            for path in sorted(args.source_data_dir.glob("*.csv"))
        ],
        "scientific_boundary": {
            "fixed_target_intervention": "Raw history was more prediction-critical; auxiliary-dominance claim withdrawn.",
            "repair": "No lambda passed complete pilot-go; confirmation was not eligible or executed.",
            "legacy_mamba": "Score-semantics diagnostic only; no graph-recovery or performance claim.",
        },
    }
    manifest_path = args.output_dir / "phase8_final_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(generated)} exports and {len(manifest['source_data'])} source-data tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
