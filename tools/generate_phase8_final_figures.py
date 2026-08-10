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
        "font.size": 7.0,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "phase8-kbs-visual-rework-v2",
    }
)


COLORS = {
    "ink": "#272B2F",
    "gray": "#6F757B",
    "gray_light": "#BBC0C4",
    "grid": "#E5E7E9",
    "neutral": "#F5F6F7",
    "raw": "#5D778E",
    "raw_mid": "#8194A5",
    "raw_soft": "#EFF3F6",
    "aux": "#98635F",
    "aux_soft": "#F5EEEE",
    "audit": "#596A77",
    "audit_soft": "#F0F2F3",
    "pass": "#64766C",
    "pass_soft": "#F1F4F2",
    "fail": "#98635F",
    "fail_soft": "#F5EEEE",
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
        -0.065,
        1.025,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["ink"],
    )


def save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, dpi in (("svg", None), ("pdf", None), ("png", 600)):
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.03}
        if suffix == "svg":
            kwargs["metadata"] = {"Date": None, "Creator": "Matplotlib"}
        elif suffix == "pdf":
            kwargs["metadata"] = {
                "CreationDate": None,
                "ModDate": None,
                "Creator": "Matplotlib",
                "Producer": "Matplotlib",
            }
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


def box(ax, xy, width, height, text, *, face, edge, fontsize=7.0, weight="normal", linestyle="-"):
    patch = Rectangle(
        xy,
        width,
        height,
        facecolor=face,
        edgecolor=edge,
        linewidth=0.9,
        linestyle=linestyle,
    )
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
            linewidth=0.75,
            linestyle=(0, (2.2, 2.2)),
        )
    )
    ax.text(0.42, 9.40, label, fontsize=9.2, fontweight="bold", va="center")
    ax.text(1.08, 9.40, title, fontsize=8.0, fontweight="bold", va="center")


def draw_figure1(output_dir: Path) -> list[Path]:
    fig = plt.figure(figsize=(7.25, 3.45))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.48, 1.08, 1.18], wspace=0.045)

    # The composition follows the full-width KBS framework diagrams: one
    # dominant predictive flow, followed by compact derivative and audit views.
    ax = fig.add_subplot(grid[0, 0])
    frame_panel(ax, "a", "Declared predictive routes")
    box(
        ax,
        (0.48, 6.70),
        2.15,
        1.25,
        "raw history\n$X_{t-K:t-1}$",
        face=COLORS["raw_soft"],
        edge=COLORS["raw"],
        fontsize=5.9,
        weight="bold",
    )
    box(
        ax,
        (0.48, 4.48),
        2.15,
        1.25,
        "auxiliary\n$c_t$",
        face=COLORS["aux_soft"],
        edge=COLORS["aux"],
        fontsize=5.9,
        weight="bold",
    )
    ax.text(1.55, 4.18, "declare route origin", ha="center", va="top", fontsize=4.7, color=COLORS["gray"])
    box(
        ax,
        (3.72, 4.58),
        2.65,
        3.28,
        "concat\n$[X;c]$\n\npredictor\n$f_\\theta(X,c)$",
        face="#FAFAFA",
        edge=COLORS["ink"],
        fontsize=5.8,
        weight="bold",
    )
    box(
        ax,
        (7.60, 5.55),
        1.72,
        1.28,
        "prediction\n$\\hat{x}_t$",
        face=COLORS["audit_soft"],
        edge=COLORS["audit"],
        fontsize=5.8,
        weight="bold",
    )
    arrow(ax, (2.63, 7.32), (3.72, 7.32), color=COLORS["raw"], width=1.35)
    arrow(ax, (2.63, 5.10), (3.72, 5.10), color=COLORS["aux"], width=1.35)
    arrow(ax, (6.37, 6.19), (7.60, 6.19), color=COLORS["ink"], width=1.25)

    ax.text(0.55, 3.22, "reported knowledge object", fontsize=5.5, fontweight="bold", color=COLORS["gray"])
    box(
        ax,
        (0.48, 1.62),
        4.05,
        1.22,
        "scored + penalized\n$J^x=\\partial\\hat{x}/\\partial X\\;|_{c\\;fixed}$",
        face=COLORS["raw_soft"],
        edge=COLORS["raw"],
        fontsize=5.7,
        weight="bold",
    )
    box(
        ax,
        (4.82, 1.62),
        4.50,
        1.22,
        "omitted route\n$J^c=\\partial\\hat{x}/\\partial c$",
        face="#FFFFFF",
        edge=COLORS["aux"],
        fontsize=5.7,
        weight="bold",
        linestyle=(0, (2.5, 2.0)),
    )
    ax.text(7.07, 1.18, "x-only graph is route-incomplete", ha="center", fontsize=5.2, color=COLORS["aux"], fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    frame_panel(ax, "b", "Coverage declaration")
    box(
        ax,
        (0.48, 7.88),
        9.04,
        0.86,
        "$C=(V_{score},V_{penalty},P_{pred},$\n$M_{coord},H_{attr})$",
        face="#FAFAFA",
        edge=COLORS["ink"],
        fontsize=5.2,
        weight="bold",
    )
    ax.text(0.60, 7.18, "route ledger", fontsize=5.8, fontweight="bold", color=COLORS["gray"])
    headers = ["route", "score", "penalty", "exempt"]
    widths = [3.20, 1.80, 2.10, 1.55]
    x0 = [0.48, 3.82, 5.76, 8.00]
    for x, w, text in zip(x0, widths, headers):
        ax.text(x + w / 2, 6.70, text, ha="center", va="center", fontsize=5.4, fontweight="bold")
    rows = [
        (5.38, "raw $X$", "yes", "yes", "no", COLORS["raw_soft"], COLORS["raw"]),
        (4.05, "aux $c$", "no", "no", "no", COLORS["aux_soft"], COLORS["aux"]),
    ]
    for y, route, scored, penalized, exempt, face, edge in rows:
        for x, w, text, cell_face in zip(
            x0,
            widths,
            [route, scored, penalized, exempt],
            [face, "#FFFFFF", "#FFFFFF", "#FFFFFF"],
        ):
            box(ax, (x, y), w, 0.96, text, face=cell_face, edge=edge, fontsize=5.6, weight="bold")
    box(
        ax,
        (0.48, 1.12),
        9.04,
        2.05,
        "if $c=g(X)$, total raw attribution is\n"
        "$\\frac{df}{dX}=J^x+J^c\\,\\frac{dg}{dX}$\n"
        "score and penalty status remain separate",
        face=COLORS["audit_soft"],
        edge=COLORS["audit"],
        fontsize=5.1,
        weight="bold",
    )

    ax = fig.add_subplot(grid[0, 2])
    frame_panel(ax, "c", "Distinct audit dimensions")
    dimensions = [
        (0.58, 6.78, "A", "score-route\ncompleteness"),
        (5.10, 6.78, "B", "penalty-route\ncompleteness"),
        (0.58, 4.76, "C", "score--penalty\nalignment"),
        (5.10, 4.76, "D--E", "coordinate +\nhorizon validity"),
    ]
    for x, y, code, text in dimensions:
        box(ax, (x, y), 4.02, 1.34, text, face="#FFFFFF", edge=COLORS["raw_mid"], fontsize=5.35, weight="bold")
        ax.text(x + 0.23, y + 1.10, code, ha="left", va="center", fontsize=4.6, color=COLORS["raw"], fontweight="bold")

    ax.text(0.65, 3.86, "claim-specific profile", fontsize=5.3, fontweight="bold", color=COLORS["gray"])
    labels = ["COVERED", "PARTIAL", "COORD-AMBIG.", "HORIZON-TRUNC.", "UNASSESSED"]
    edges = [COLORS["pass"], COLORS["aux"], COLORS["aux"], COLORS["gray"], COLORS["gray"]]
    positions = [(0.58, 2.78, 2.52), (3.25, 2.78, 2.08), (5.48, 2.78, 3.94), (1.32, 1.70, 3.42), (5.02, 1.70, 3.22)]
    for (x, y, w), text, edge in zip(positions, labels, edges):
        box(ax, (x, y), w, 0.68, text, face="#FFFFFF", edge=edge, fontsize=4.45, weight="bold")
    ax.text(4.95, 0.88, "coexisting diagnostic flags, not guarantees", ha="center", fontsize=4.8, color=COLORS["gray"])

    return save_figure(fig, output_dir, "fig1_jacobian_coverage_audit_phase8_final")


def draw_figure2_architecture(output_dir: Path) -> list[Path]:
    """Render the source-verified controlled concat data and derivative paths."""
    fig = plt.figure(figsize=(7.25, 2.48))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.18, 1.08], wspace=0.20)

    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    panel_label(ax, "a")
    ax.text(0.0, 9.55, "Causal prefix transform", fontsize=7.5, fontweight="bold", ha="left")
    box(ax, (0.25, 6.65), 2.65, 1.15, "raw prefix\n$X_{0:t-1}$", face=COLORS["raw_soft"], edge=COLORS["raw"], fontsize=5.6, weight="bold")
    box(ax, (3.75, 6.65), 2.50, 1.15, "$g_\\phi$\nstate-space block", face="#FFFFFF", edge=COLORS["gray"], fontsize=5.05, weight="bold")
    box(ax, (7.10, 6.65), 2.65, 1.15, "auxiliary prefix\n$c_{0:t-1}$", face=COLORS["aux_soft"], edge=COLORS["aux"], fontsize=5.4, weight="bold")
    arrow(ax, (2.90, 7.22), (3.75, 7.22), color=COLORS["raw"], width=1.0)
    arrow(ax, (6.25, 7.22), (7.10, 7.22), color=COLORS["aux"], width=1.0)

    box(ax, (0.25, 3.65), 2.65, 1.10, "raw lag window\n$X_{t-K:t-1}$", face="#FFFFFF", edge=COLORS["raw"], fontsize=5.35, weight="bold")
    box(ax, (7.10, 3.65), 2.65, 1.10, "auxiliary lag window\n$c_{t-K:t-1}$", face="#FFFFFF", edge=COLORS["aux"], fontsize=5.1, weight="bold")
    arrow(ax, (1.58, 6.65), (1.58, 4.75), color=COLORS["raw"], width=0.9)
    arrow(ax, (8.42, 6.65), (8.42, 4.75), color=COLORS["aux"], width=0.9)

    box(ax, (3.58, 1.38), 2.84, 1.04, "raw target $x_t$\nexcluded from history", face="#FFFFFF", edge=COLORS["aux"], fontsize=5.0, weight="bold", linestyle=(0, (2.2, 1.8)))
    ax.text(5.0, 0.66, "prefix-stateful; no reset at the lag boundary", ha="center", fontsize=4.7, color=COLORS["gray"])

    ax = fig.add_subplot(grid[0, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    panel_label(ax, "b")
    ax.text(0.0, 9.55, "Prediction path and raw target", fontsize=7.5, fontweight="bold", ha="left")
    box(ax, (0.18, 6.80), 2.30, 1.10, "$X_{t-K:t-1}$", face=COLORS["raw_soft"], edge=COLORS["raw"], fontsize=5.8, weight="bold")
    box(ax, (0.18, 4.50), 2.30, 1.10, "$c_{t-K:t-1}$", face=COLORS["aux_soft"], edge=COLORS["aux"], fontsize=5.8, weight="bold")
    box(ax, (3.45, 4.48), 2.85, 3.44, "concat + flatten\n$[X;c]$\n\nJRNGC predictor\n$f_\\theta$", face="#FFFFFF", edge=COLORS["ink"], fontsize=5.45, weight="bold")
    arrow(ax, (2.48, 7.35), (3.45, 7.35), color=COLORS["raw"], width=1.05)
    arrow(ax, (2.48, 5.05), (3.45, 5.05), color=COLORS["aux"], width=1.05)
    box(ax, (7.22, 6.28), 2.30, 1.10, "prediction\n$\\hat{x}_t$", face=COLORS["audit_soft"], edge=COLORS["audit"], fontsize=5.6, weight="bold")
    arrow(ax, (6.30, 6.83), (7.22, 6.83), color=COLORS["ink"], width=1.0)
    box(ax, (7.22, 3.96), 2.30, 1.10, "pure MSE\n$\\|\\hat{x}_t-x_t\\|^2$", face="#FFFFFF", edge=COLORS["gray"], fontsize=5.15, weight="bold")
    arrow(ax, (8.37, 6.28), (8.37, 5.06), color=COLORS["gray"], width=0.85)
    box(ax, (7.22, 1.64), 2.30, 1.10, "fixed raw target\n$x_t$", face="#FFFFFF", edge=COLORS["aux"], fontsize=5.25, weight="bold")
    arrow(ax, (8.37, 2.74), (8.37, 3.96), color=COLORS["aux"], width=0.85)
    ax.text(3.25, 8.55, "prediction gradients pass through both routes", fontsize=4.65, color=COLORS["gray"], ha="center")

    ax = fig.add_subplot(grid[0, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    panel_label(ax, "c")
    ax.text(0.0, 9.55, "Three derivative objects", fontsize=7.5, fontweight="bold", ha="left")
    derivative_rows = [
        (7.28, "$J^x=\\partial\\hat{x}/\\partial X\\;|_{c\\;fixed}$", "x-only score + penalty", COLORS["raw"]),
        (4.78, "$[J^x,J^c]$", "full auxiliary penalty", COLORS["aux"]),
        (2.28, "$d\\hat{x}/dX=J^x+J^c\\,dg/dX$", "total raw-chain score", COLORS["audit"]),
    ]
    for idx, (y, formula, label, edge) in enumerate(derivative_rows, start=1):
        ax.text(0.20, y + 0.68, str(idx), ha="center", va="center", fontsize=5.2, color=edge, fontweight="bold")
        box(ax, (0.72, y), 8.98, 1.36, formula, face="#FFFFFF", edge=edge, fontsize=5.25, weight="bold")
        ax.text(5.20, y - 0.36, label, ha="center", fontsize=4.8, color=COLORS["gray"])
    ax.text(5.20, 0.62, "same predictor; distinct score and regularizer semantics", ha="center", fontsize=4.7, color=COLORS["gray"])

    fig.subplots_adjust(left=0.045, right=0.992, top=0.91, bottom=0.08)
    return save_figure(fig, output_dir, "fig2_controlled_concat_architecture_phase8_final")


def draw_figure3_workflow(output_dir: Path) -> list[Path]:
    """Render the reusable audit sequence that replaces the text-only algorithm box."""
    fig = plt.figure(figsize=(7.25, 2.18))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.18)

    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    panel_label(ax, "a")
    ax.text(0.0, 9.48, "Claim-specific audit sequence", fontsize=7.5, fontweight="bold", ha="left")
    phases = [
        (0.15, "1  DECLARE", "graph object\nroute classes\nexemptions"),
        (2.63, "2  TRACE", "score derivative\npenalty derivative\nwindow aggregation"),
        (5.11, "3  VALIDATE", "source map\nattribution horizon\nprovenance"),
        (7.59, "4  PROFILE", "retain every\napplicable flag\n+ unresolved item"),
    ]
    for idx, (x, heading, body) in enumerate(phases):
        box(ax, (x, 4.65), 2.08, 3.20, f"{heading}\n\n{body}", face="#FFFFFF", edge=COLORS["raw_mid"], fontsize=5.15, weight="bold")
        if idx < len(phases) - 1:
            arrow(ax, (x + 2.08, 6.25), (x + 2.45, 6.25), color=COLORS["gray"], width=0.85)

    checks = [
        (0.20, "A", "score routes"),
        (2.18, "B", "penalty routes"),
        (4.18, "C", "alignment"),
        (6.18, "D", "coordinates"),
        (8.18, "E", "horizon"),
    ]
    for x, code, label in checks:
        ax.add_patch(Rectangle((x, 1.62), 0.55, 0.55, facecolor="#FFFFFF", edgecolor=COLORS["audit"], linewidth=0.9))
        ax.text(x + 0.275, 1.895, code, ha="center", va="center", fontsize=4.8, color=COLORS["audit"], fontweight="bold")
        ax.text(x + 0.72, 1.895, label, ha="left", va="center", fontsize=4.55, color=COLORS["ink"])
    ax.text(4.95, 0.62, "score and penalty coverage are audited separately", ha="center", fontsize=4.75, color=COLORS["gray"])

    ax = fig.add_subplot(grid[0, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    panel_label(ax, "b")
    ax.text(0.0, 9.48, "Audit-profile output", fontsize=7.5, fontweight="bold", ha="left")
    ax.add_patch(Rectangle((0.25, 3.05), 9.45, 4.95, facecolor="#FFFFFF", edgecolor=COLORS["gray"], linewidth=0.85))
    ax.text(0.62, 7.48, "declared graph claim", fontsize=5.1, fontweight="bold", color=COLORS["ink"])
    ax.plot([0.62, 9.28], [7.12, 7.12], color=COLORS["grid"], linewidth=0.8)
    labels = ["COVERED", "PARTIAL", "COORD-AMBIG.", "HORIZON-TRUNC.", "UNASSESSED"]
    edges = [COLORS["pass"], COLORS["aux"], COLORS["aux"], COLORS["gray"], COLORS["gray"]]
    y_positions = [6.30, 5.50, 4.70, 3.90, 3.10]
    for label, edge, y in zip(labels, edges, y_positions):
        ax.add_patch(Rectangle((0.70, y), 0.42, 0.42, facecolor="#FFFFFF", edgecolor=edge, linewidth=0.8))
        ax.text(1.42, y + 0.21, label, ha="left", va="center", fontsize=4.65, color=edge, fontweight="bold")
    ax.text(4.98, 2.12, "multiple flags may coexist", ha="center", fontsize=4.8, color=COLORS["gray"])
    ax.text(4.98, 1.24, "diagnostic record, not a causal certificate", ha="center", fontsize=4.7, color=COLORS["gray"])

    fig.subplots_adjust(left=0.045, right=0.992, top=0.90, bottom=0.08)
    return save_figure(fig, output_dir, "fig3_claim_specific_audit_workflow_phase8_final")


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
        ax.plot(x, row, color=color, alpha=0.48, linewidth=0.65, marker="o", markersize=2.2, markerfacecolor="white")
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=0)
    ax.errorbar(
        x,
        mean,
        yerr=sd,
        color=mean_color,
        linewidth=1.45,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.2,
        markersize=3.8,
        capsize=2.0,
        label="mean +/- SD",
    )
    ax.set_xlabel(r"auxiliary dimension $d_{cond}$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_title(title, loc="left", fontsize=7.6, fontweight="bold", pad=5)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.72)


def draw_figure4(
    track_root: Path,
    output_dir: Path,
    source_dir: Path,
    *,
    stem: str = "fig4_replicated_decoupling_phase8_final",
) -> list[Path]:
    track = load_json(track_root / "replication_aggregate_and_gates.json")
    dconds, mse, auroc = _capacity_arrays(track)
    coeff = _coefficient_rows(track_root)
    interventions = track["fixed_target_interventions"]["pairs"]

    fig = plt.figure(figsize=(7.25, 4.65))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.04, 0.96], wspace=0.30, hspace=0.42)
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )
    ax = axes[0, 0]
    seed_lines(
        ax,
        dconds,
        mse,
        color=COLORS["gray_light"],
        mean_color=COLORS["raw"],
        ylabel="fixed-target pure MSE",
        title="Pure prediction error",
    )
    panel_label(ax, "a")

    ax = axes[0, 1]
    seed_lines(
        ax,
        dconds,
        auroc,
        color=COLORS["gray_light"],
        mean_color=COLORS["raw"],
        ylabel="partial nominal AUROC",
        title="Partial graph ranking",
    )
    ax.set_ylim(0.15, 1.03)
    panel_label(ax, "b")

    ax = axes[1, 0]
    metric_labels = ["AUROC", "coefficient r"]
    methods = ["Baseline", "Concat partial", "Concat total"]
    method_colors = [COLORS["gray"], COLORS["aux"], COLORS["raw"]]
    for metric_idx, metric in enumerate(("auroc", "coefficient_r")):
        for method_idx, method in enumerate(("baseline", "concat_partial", "concat_total")):
            values = [row[f"{method}_{metric}"] for row in coeff]
            xpos = metric_idx * 4 + method_idx
            jitter = np.linspace(-0.10, 0.10, len(values))
            ax.scatter(xpos + jitter, values, s=12, facecolor="white", edgecolor=method_colors[method_idx], linewidth=0.8, alpha=0.82, zorder=3)
            mean, sd = mean_sd(values)
            ax.errorbar(xpos, mean, yerr=sd, color=method_colors[method_idx], marker="s", markersize=4.5, capsize=2.0, linewidth=1.05, zorder=4)
    ax.set_xticks([1, 5])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("direct graph metric")
    ax.set_title("Direct graph fidelity", loc="left", fontsize=7.6, fontweight="bold", pad=5)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.72)
    handles = [plt.Line2D([], [], color=color, marker="s", linestyle="", label=label) for color, label in zip(method_colors, methods)]
    ax.legend(handles=handles, loc="lower left", fontsize=5.7, ncol=1, handletextpad=0.45, borderaxespad=0.2)
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
            ax.scatter(x + jitter, values, facecolor="white", edgecolor=color, linewidth=0.8, alpha=0.84, s=13, zorder=3)
            mean, sd = mean_sd(values)
            ax.errorbar(x, mean, yerr=sd, color=color, marker="s", markersize=4.5, capsize=2.0, linewidth=1.05, zorder=4)
        for raw, aux in zip(raw_values[idx], aux_values[idx]):
            ax.plot([idx - 0.18, idx + 0.18], [raw, aux], color=COLORS["gray_light"], linewidth=0.55, alpha=0.60, zorder=1)
    ax.set_xticks(range(2))
    ax.set_xticklabels(conditions)
    ax.set_ylabel(r"$\Delta$ fixed-target pure MSE")
    ax.set_title("Fixed-target route interventions", loc="left", fontsize=7.6, fontweight="bold", pad=5)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.72)
    handles = [plt.Line2D([], [], color=COLORS["raw"], marker="s", linestyle="", label="raw history X"), plt.Line2D([], [], color=COLORS["aux"], marker="s", linestyle="", label="auxiliary c")]
    ax.legend(handles=handles, loc="upper right", fontsize=5.7, handletextpad=0.45, borderaxespad=0.2)
    panel_label(ax, "d")

    fig.subplots_adjust(left=0.082, right=0.988, top=0.96, bottom=0.095)

    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "figure4_capacity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["paired_seed_index", "d_cond", "fixed_target_prediction_mse", "partial_nominal_auroc"])
        for seed_idx in range(mse.shape[0]):
            for idx, d_cond in enumerate(dconds):
                writer.writerow([seed_idx + 1, int(d_cond), mse[seed_idx, idx], auroc[seed_idx, idx]])
    with (source_dir / "figure4_coefficient.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coeff[0]))
        writer.writeheader()
        writer.writerows(coeff)
    with (source_dir / "figure4_interventions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["data_seed", "condition", "raw_history_delta", "auxiliary_delta"])
        for idx, pair in enumerate(interventions):
            writer.writerow([pair["data_seed"], "mask", raw_values[0][idx], aux_values[0][idx]])
            writer.writerow([pair["data_seed"], "shuffle", raw_values[1][idx], aux_values[1][idx]])
    return save_figure(fig, output_dir, stem)


def draw_figure5(
    track_root: Path,
    p0_paths: Sequence[Path],
    output_dir: Path,
    source_dir: Path,
    *,
    stem: str = "fig5_score_semantics_phase8_final",
) -> list[Path]:
    audits = [load_json(path) for path in p0_paths]
    coeff = _coefficient_rows(track_root)

    fig = plt.figure(figsize=(7.25, 3.20))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.33, 0.86, 1.08], wspace=0.38)

    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    panel_label(ax, "a")
    ax.text(0.0, 9.50, "Derivative coordinates and support", fontsize=7.6, fontweight="bold", ha="left")

    # Conditioned path: partial derivatives are distinct graph objects.
    box(ax, (0.25, 7.28), 1.60, 0.82, "$X$", face=COLORS["raw_soft"], edge=COLORS["raw"], fontsize=6.2, weight="bold")
    box(ax, (0.25, 5.95), 1.60, 0.82, "$c$", face=COLORS["aux_soft"], edge=COLORS["aux"], fontsize=6.2, weight="bold")
    box(ax, (3.00, 6.08), 2.10, 1.88, "$f_\\theta(X,c)$", face="#FAFAFA", edge=COLORS["ink"], fontsize=6.0, weight="bold")
    box(ax, (6.25, 6.60), 1.42, 0.86, "$\\hat{x}$", face=COLORS["audit_soft"], edge=COLORS["audit"], fontsize=6.2, weight="bold")
    arrow(ax, (1.85, 7.69), (3.00, 7.69), color=COLORS["raw"], width=1.05)
    arrow(ax, (1.85, 6.36), (3.00, 6.36), color=COLORS["aux"], width=1.05)
    arrow(ax, (5.10, 7.03), (6.25, 7.03), color=COLORS["ink"], width=1.0)
    box(ax, (8.05, 7.22), 1.67, 0.72, "$J^x=\\partial f/\\partial X$", face=COLORS["raw_soft"], edge=COLORS["raw"], fontsize=4.9, weight="bold")
    box(ax, (8.05, 6.08), 1.67, 0.72, "$J^c=\\partial f/\\partial c$", face=COLORS["aux_soft"], edge=COLORS["aux"], fontsize=4.9, weight="bold")

    # Transformed path: filtered coordinates differ from original-input chain attribution.
    y = 3.82
    blocks = [
        (0.25, 1.35, "$X$", COLORS["raw_soft"], COLORS["raw"]),
        (2.30, 1.45, "$F_\\phi$", COLORS["neutral"], COLORS["raw_mid"]),
        (4.45, 1.45, "$X'$", COLORS["neutral"], COLORS["raw_mid"]),
        (6.60, 1.42, "$f_\\theta$", "#FAFAFA", COLORS["ink"]),
        (8.72, 1.00, "$\\hat{x}$", COLORS["audit_soft"], COLORS["audit"]),
    ]
    for x, width, text, face, edge in blocks:
        box(ax, (x, y), width, 0.88, text, face=face, edge=edge, fontsize=5.8, weight="bold")
    for start, end, color in [
        ((1.60, y + 0.44), (2.30, y + 0.44), COLORS["raw"]),
        ((3.75, y + 0.44), (4.45, y + 0.44), COLORS["raw_mid"]),
        ((5.90, y + 0.44), (6.60, y + 0.44), COLORS["raw_mid"]),
        ((8.02, y + 0.44), (8.72, y + 0.44), COLORS["ink"]),
    ]:
        arrow(ax, start, end, color=color, width=0.9)
    ax.text(5.18, 3.34, "$J^{x'}=\\partial f/\\partial X'$", ha="center", fontsize=5.1, color=COLORS["raw_mid"], fontweight="bold")
    ax.text(4.98, 2.72, "$J^{raw}=d f(F(X))/dX$", ha="center", fontsize=5.1, color=COLORS["audit"], fontweight="bold")

    # The horizon strip makes the nominal-versus-extended distinction visible.
    ax.add_patch(Rectangle((0.25, 1.34), 7.45, 0.42, facecolor=COLORS["neutral"], edgecolor=COLORS["gray"], linewidth=0.75))
    ax.add_patch(Rectangle((6.06, 1.34), 1.64, 0.42, facecolor=COLORS["raw_soft"], edgecolor=COLORS["raw"], linewidth=0.75))
    ax.text(3.02, 1.55, "earlier transformation support", ha="center", va="center", fontsize=4.6, color=COLORS["gray"])
    ax.text(6.88, 1.55, "nominal $K$", ha="center", va="center", fontsize=4.6, color=COLORS["raw"], fontweight="bold")
    ax.text(8.05, 1.55, "$H_{attr}$", ha="left", va="center", fontsize=5.3, color=COLORS["ink"], fontweight="bold")
    ax.text(4.85, 0.62, "Declare coordinates and horizon before graph interpretation.", ha="center", fontsize=4.8, color=COLORS["gray"])

    ax = fig.add_subplot(grid[0, 1])
    values = {
        "Concat\npartial/total r": [row["concat_partial_vs_total_derivative"]["offdiag_score_correlation"] for row in audits],
        "Legacy\nfiltered/raw r": [row["istf_mamba_filtered_vs_raw_chain"]["offdiag_score_correlation"] for row in audits],
        "Legacy\ntop-k Jaccard": [row["istf_mamba_filtered_vs_raw_chain"]["topk_jaccard"] for row in audits],
    }
    colors = [COLORS["aux"], COLORS["raw_mid"], COLORS["gray"]]
    for idx, ((label, vals), color) in enumerate(zip(values.items(), colors)):
        jitter = np.linspace(-0.10, 0.10, len(vals))
        ax.scatter(idx + jitter, vals, facecolor="white", edgecolor=color, linewidth=0.8, alpha=0.88, s=13)
        mean, sd = mean_sd(vals)
        ax.errorbar(idx, mean, yerr=sd, marker="s", markersize=4.4, color=color, capsize=2.0, linewidth=1.05)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score agreement")
    ax.set_xticks(range(3))
    ax.set_xticklabels(values.keys(), fontsize=5.8)
    ax.set_title("Score agreement", loc="left", x=0.08, fontsize=7.6, fontweight="bold", pad=5)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.72)
    panel_label(ax, "b")

    ax = fig.add_subplot(grid[0, 2])
    metrics = ["AUROC", "coefficient r"]
    methods = ["Baseline", "Concat partial", "Concat total"]
    keys = [
        ("baseline_auroc", "concat_partial_auroc", "concat_total_auroc"),
        ("baseline_coefficient_r", "concat_partial_coefficient_r", "concat_total_coefficient_r"),
    ]
    colors = [COLORS["gray"], COLORS["aux"], COLORS["raw"]]
    for metric_idx, key_set in enumerate(keys):
        for method_idx, key in enumerate(key_set):
            vals = [row[key] for row in coeff]
            x = metric_idx * 4 + method_idx
            ax.scatter(x + np.linspace(-0.08, 0.08, len(vals)), vals, facecolor="white", edgecolor=colors[method_idx], linewidth=0.8, alpha=0.82, s=12)
            mean, sd = mean_sd(vals)
            ax.errorbar(x, mean, yerr=sd, marker="s", markersize=4.4, color=colors[method_idx], capsize=2.0, linewidth=1.05)
    ax.set_xticks([1, 5])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("direct graph metric")
    ax.set_title("Post-hoc total scoring", loc="left", x=0.08, fontsize=7.6, fontweight="bold", pad=5)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.72)
    handles = [plt.Line2D([], [], color=color, marker="s", linestyle="", label=label) for color, label in zip(colors, methods)]
    ax.legend(handles=handles, loc="lower left", fontsize=5.5, handletextpad=0.45, borderaxespad=0.2)
    panel_label(ax, "c")

    fig.subplots_adjust(left=0.055, right=0.988, top=0.93, bottom=0.16)
    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "figure5_semantic_audit.csv").open("w", newline="", encoding="utf-8") as handle:
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
    return save_figure(fig, output_dir, stem)


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


def draw_figure6(
    final_path: Path,
    output_dir: Path,
    source_dir: Path,
    *,
    stem: str = "fig6_coverage_repair_tradeoff_phase8_final",
) -> list[Path]:
    final = load_json(final_path)
    rows = _tradeoff_rows(final)
    labels = ["Concat x-only", "Baseline JRNGC", "Full auxiliary lc10", r"$\lambda=3\times10^{-4}$", r"$\lambda=10^{-3}$", r"$\lambda=3\times10^{-3}$", r"$\lambda=10^{-2}$"]
    # Keep method families distinct without assigning a new hue to every
    # regularization strength. Lambda values use one muted blue and shape.
    colors = [
        COLORS["aux"],
        COLORS["gray"],
        COLORS["raw_mid"],
        COLORS["raw"],
        COLORS["raw"],
        COLORS["raw"],
        COLORS["raw"],
    ]
    markers = ["X", "s", "D", "o", "^", "v", "P"]

    fig = plt.figure(figsize=(7.25, 4.17))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.70], wspace=0.42, hspace=0.54)
    for panel_idx, (metric, ylabel, title) in enumerate(
        [
            ("auroc", "total nominal AUROC", "Direct graph AUROC"),
            ("auprc", "total nominal AUPRC", "Direct graph AUPRC"),
            ("coefficient_r", "lag-1 coefficient r", "Coefficient fidelity"),
        ]
    ):
        ax = fig.add_subplot(grid[0, panel_idx])
        for label, color, marker in zip(labels, colors, markers):
            group = [row for row in rows if row["method"] == label]
            xvals = [row["relative_mse_percent"] for row in group]
            yvals = [row[metric] for row in group]
            ax.scatter(
                xvals,
                yvals,
                facecolor="white",
                edgecolor=color,
                linewidth=0.85,
                alpha=0.76,
                s=15,
                marker=marker,
                zorder=2,
            )
            xmean, xsd = mean_sd(xvals)
            ymean, ysd = mean_sd(yvals)
            ax.errorbar(
                xmean,
                ymean,
                xerr=xsd,
                yerr=ysd,
                color=color,
                marker=marker,
                markersize=4.8,
                capsize=1.8,
                linewidth=0.95,
                label=label,
                zorder=3,
            )
        ax.axvline(10.0, color=COLORS["fail"], linestyle=(0, (3, 2)), linewidth=0.9)
        ax.text(12.0, 0.035, "10% gate", transform=ax.get_xaxis_transform(), color=COLORS["fail"], fontsize=5.0, va="bottom")
        ax.set_xlabel("pure-MSE change vs concat (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold", pad=5)
        ax.grid(color=COLORS["grid"], linewidth=0.48, alpha=0.72)
        panel_label(ax, chr(ord("a") + panel_idx))

    ax = fig.add_subplot(grid[1, :])
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
    ax.set_xlim(-0.10, 5.10)
    ax.set_ylim(-0.12, len(rows_labels) + 0.10)
    for line_x in range(len(columns) + 1):
        ax.plot([line_x, line_x], [0.12, 3.88], color=COLORS["grid"], linewidth=0.55, zorder=0)
    for line_y in range(len(rows_labels) + 1):
        ax.plot([0.0, 5.0], [line_y, line_y], color=COLORS["grid"], linewidth=0.55, zorder=0)
    for ridx, row in enumerate(matrix):
        y = len(rows_labels) - ridx - 1
        for cidx, passed in enumerate(row):
            edge = COLORS["pass"] if passed else COLORS["fail"]
            marker = "o" if passed else "x"
            if passed:
                ax.scatter(cidx + 0.5, y + 0.5, s=24, marker=marker, facecolors="none", edgecolors=edge, linewidth=1.0, zorder=2)
            else:
                ax.scatter(cidx + 0.5, y + 0.5, s=24, marker=marker, color=edge, linewidth=1.0, zorder=2)
            ax.text(cidx + 0.5, y + 0.18, "pass" if passed else "fail", ha="center", va="center", fontsize=4.2, color=edge)
    ax.set_xticks(np.arange(len(columns)) + 0.5)
    ax.set_xticklabels(columns, fontsize=5.8)
    ax.set_yticks(np.arange(len(rows_labels)) + 0.5)
    ax.set_yticklabels(rows_labels[::-1], fontsize=5.8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Pre-specified effect, fit, safety, and semantic gates", loc="left", fontsize=7.5, fontweight="bold", pad=5)
    panel_label(ax, "d")

    legend_handles = [
        plt.Line2D([], [], color=color, marker=marker, linestyle="", label=label)
        for label, color, marker in zip(labels, colors, markers)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=4,
        fontsize=5.4,
        columnspacing=1.05,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.835, bottom=0.075)
    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "figure6_repair_tradeoff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (source_dir / "figure6_gate_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lambda", *columns])
        for label, row in zip(rows_labels, matrix):
            writer.writerow([label, *row])
    return save_figure(fig, output_dir, stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-a-root", type=Path, required=True)
    parser.add_argument("--p0-audit-dir", type=Path, required=True)
    parser.add_argument("--final-aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument(
        "--figures",
        choices=("all", "results"),
        default="all",
        help="Generate all legacy-layout figures or result figures 4--6 only.",
    )
    parser.add_argument(
        "--submission-v4-names",
        action="store_true",
        help="Name result figures by their submission-v4 main-text numbers (2--4).",
    )
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
    if args.figures == "all":
        generated += draw_figure1(args.output_dir)
        generated += draw_figure2_architecture(args.output_dir)
        generated += draw_figure3_workflow(args.output_dir)
    stems = {
        "figure4": (
            "fig2_prediction_knowledge_decoupling_v4"
            if args.submission_v4_names
            else "fig4_replicated_decoupling_phase8_final"
        ),
        "figure5": (
            "fig3_derivative_semantics_v4"
            if args.submission_v4_names
            else "fig5_score_semantics_phase8_final"
        ),
        "figure6": (
            "fig4_graph_prediction_frontier_v4"
            if args.submission_v4_names
            else "fig6_coverage_repair_tradeoff_phase8_final"
        ),
    }
    generated += draw_figure4(
        args.track_a_root,
        args.output_dir,
        args.source_data_dir,
        stem=stems["figure4"],
    )
    generated += draw_figure5(
        args.track_a_root,
        p0_paths,
        args.output_dir,
        args.source_data_dir,
        stem=stems["figure5"],
    )
    generated += draw_figure6(
        args.final_aggregate,
        args.output_dir,
        args.source_data_dir,
        stem=stems["figure6"],
    )
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
