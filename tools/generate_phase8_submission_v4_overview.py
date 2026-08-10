"""Render the integrated Phase 8 route-resolved Jacobian audit overview.

This deterministic schematic contains no experimental values. It consolidates
the predictor architecture, derivative semantics, and audit outputs that were
spread across Figures 1--3 of the v3 manuscript.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from generate_phase8_submission_v3_concept_figures import (
    COLORS,
    arrow,
    box,
    elbow_arrow,
    matrix_glyph,
    network_glyph,
    panel,
    save_figure,
    signal_strip,
    status_marker,
    tensor_glyph,
)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "phase8-kbs-top-journal-narrative-v4",
    }
)


def route_tag(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    width: float,
) -> None:
    box(ax, x, y, width, 0.48, face=COLORS["white"], edge=color, linewidth=0.70)
    ax.text(
        x + width / 2,
        y + 0.24,
        text,
        ha="center",
        va="center",
        fontsize=4.25,
        color=color,
        fontweight="bold",
    )


def draw_panel_a(ax: plt.Axes) -> None:
    """Architecture: raw prefix, causal auxiliary route, predictor, graph."""
    panel(ax, "a", "Route-resolved predictor and graph object")

    # Raw prefix and nominal history window.
    signal_strip(ax, 0.25, 5.65, 2.25, 2.85, color=COLORS["raw"], phase=0.25, n=4)
    ax.axvspan(1.38, 2.15, ymin=0.565, ymax=0.85, color=COLORS["raw_soft"], zorder=-1)
    ax.axvline(2.18, ymin=0.56, ymax=0.86, color=COLORS["ink"], linewidth=0.75, linestyle=(0, (2, 2)))
    ax.text(0.25, 8.84, r"observed prefix  $X_{0:t-1}$", fontsize=5.0, color=COLORS["ink"], fontweight="bold")
    ax.text(1.76, 8.48, r"$X_{t-K:t-1}$", ha="center", fontsize=4.25, color=COLORS["raw"], fontweight="bold")
    ax.text(2.23, 5.27, r"target $x_t$", ha="center", fontsize=4.2, color=COLORS["ink"], fontweight="bold")

    # Causal prefix-stateful transformation and auxiliary sequence.
    box(ax, 0.47, 3.60, 1.82, 0.72, face=COLORS["white"], edge=COLORS["gray_mid"], linewidth=0.78)
    ax.text(1.38, 3.97, r"causal $g_\phi$", ha="center", va="center", fontsize=4.7, color=COLORS["ink"], fontweight="bold")
    arrow(ax, (1.38, 5.65), (1.38, 4.32), color=COLORS["gray"], linewidth=0.90)
    signal_strip(ax, 0.25, 1.05, 2.25, 1.82, color=COLORS["aux"], phase=1.15, n=3)
    arrow(ax, (1.38, 3.60), (1.38, 2.87), color=COLORS["aux"], linewidth=1.00)
    ax.text(0.25, 0.62, r"auxiliary prefix  $c_{0:t-1}=g_\phi(X_{0:t-1})$", fontsize=4.55, color=COLORS["aux"], fontweight="bold")

    # Shared predictor; orthogonal route connectors retain the two-lane logic.
    route_tag(ax, 2.86, 7.42, "raw route", color=COLORS["raw"], width=1.10)
    route_tag(ax, 2.86, 2.08, "aux route", color=COLORS["aux"], width=1.10)
    arrow(ax, (2.50, 7.09), (4.08, 7.09), color=COLORS["raw"], linewidth=1.25)
    elbow_arrow(ax, (2.50, 1.96), (3.53, 1.96), (3.53, 4.35), color=COLORS["aux"], linewidth=1.25)
    arrow(ax, (3.53, 4.35), (4.08, 4.35), color=COLORS["aux"], linewidth=1.25)

    network_glyph(ax, 4.08, 3.17, 1.65, 4.72, edge=COLORS["ink"])
    ax.text(4.90, 8.35, "shared JRNGC predictor", ha="center", fontsize=5.1, color=COLORS["ink"], fontweight="bold")
    ax.text(4.90, 2.72, r"$\widehat{x}_t=f_\theta(X,c)$", ha="center", fontsize=4.7, color=COLORS["ink"])
    route_tag(ax, 4.22, 1.70, "score + penalty", color=COLORS["raw"], width=1.36)
    route_tag(ax, 4.22, 1.02, "prediction route", color=COLORS["aux"], width=1.36)

    # Fixed target and prediction objective.
    arrow(ax, (5.73, 5.53), (6.32, 5.53), color=COLORS["ink"], linewidth=1.05)
    box(ax, 6.32, 4.78, 1.25, 1.50, face=COLORS["white"], edge=COLORS["ink"], linewidth=0.82)
    ax.text(6.95, 5.77, r"prediction", ha="center", fontsize=4.25, color=COLORS["gray"])
    ax.text(6.95, 5.25, r"$\widehat{x}_t$", ha="center", fontsize=6.6, color=COLORS["ink"], fontweight="bold")
    box(ax, 6.32, 2.66, 1.25, 1.05, face=COLORS["white"], edge=COLORS["gray_mid"], linewidth=0.72)
    ax.text(6.95, 3.18, r"fixed raw $x_t$", ha="center", va="center", fontsize=4.45, color=COLORS["ink"], fontweight="bold")
    arrow(ax, (6.95, 4.78), (6.95, 3.71), color=COLORS["gray"], linewidth=0.85)
    ax.text(7.76, 4.23, r"$\mathcal{L}_{\rm pred}$", fontsize=5.0, color=COLORS["ink"], fontweight="bold")

    # Reported Jacobian tensor and directed graph.
    elbow_arrow(ax, (5.73, 6.72), (8.03, 6.72), (8.03, 7.52), color=COLORS["raw"], linewidth=1.0)
    tensor_glyph(ax, 8.03, 6.66, 0.82, 0.82, color=COLORS["raw"], slices=3)
    ax.text(8.52, 8.08, r"$\bar{J}^{x}_{ijh}$", ha="center", fontsize=4.75, color=COLORS["raw"], fontweight="bold")
    arrow(ax, (8.92, 7.11), (9.18, 7.11), color=COLORS["gray"], linewidth=0.85)
    score = np.asarray([[0.06, 0.78, 0.18], [0.43, 0.05, 0.64], [0.12, 0.52, 0.08]])
    matrix_glyph(ax, 9.18, 6.54, 0.62, 1.10, score, color=COLORS["ink"], label="graph", fontsize=3.85)

    # Route discrepancy is shown as a precise derivative relation, not a prose card.
    ax.plot([7.92, 7.92], [0.85, 6.55], color=COLORS["grid"], linewidth=0.70)
    ax.text(8.15, 5.48, r"reported", fontsize=4.0, color=COLORS["gray"], fontweight="bold")
    ax.text(8.15, 4.96, r"$J^x=\partial\widehat{x}/\partial X\,|_{c}$", fontsize=4.75, color=COLORS["raw"])
    ax.text(8.15, 3.92, r"predictive chain", fontsize=4.0, color=COLORS["gray"], fontweight="bold")
    ax.text(8.15, 3.38, r"$d\widehat{x}/dX=J^x+J^c\,dg_\phi/dX$", fontsize=4.55, color=COLORS["ink"])
    ax.plot([8.15, 9.72], [2.82, 2.82], color=COLORS["grid"], linewidth=0.65)
    ax.text(8.15, 2.28, "same prediction", fontsize=4.15, color=COLORS["text"], fontweight="bold")
    ax.text(8.15, 1.70, "different graph object", fontsize=4.15, color=COLORS["aux"], fontweight="bold")
    ax.text(8.15, 1.12, "when route coverage differs", fontsize=4.0, color=COLORS["gray"])


def draw_panel_b(ax: plt.Axes) -> None:
    """Three Jacobian objects on the same computation graph."""
    panel(ax, "b", "Derivative objects and coordinate systems")
    labels = [
        ("partial score", r"$J^x=\partial\widehat{x}/\partial X\,|_{c}$", COLORS["raw"]),
        ("full-coordinate penalty", r"$[J^x,J^c]$", COLORS["aux"]),
        ("total raw-chain score", r"$d\widehat{x}/dX$", COLORS["ink"]),
    ]
    y_positions = [7.12, 4.18, 1.24]
    partial = np.asarray([[0.05, 0.72, 0.15], [0.22, 0.08, 0.61], [0.12, 0.42, 0.06]])
    indirect = np.asarray([[0.00, 0.18, 0.40], [0.38, 0.00, 0.10], [0.08, 0.27, 0.00]])
    total = np.clip(partial + indirect, 0.0, 1.0)
    matrices = [partial, indirect, total]
    axis_labels = [
        ("target", "source", "raw lag"),
        ("target", "coordinate", "nominal lag"),
        ("target", "source", "raw lag"),
    ]
    for (title, formula, color), ypos, values, axes in zip(
        labels, y_positions, matrices, axis_labels
    ):
        ax.text(0.32, ypos + 1.82, title, fontsize=4.8, color=COLORS["ink"], fontweight="bold")
        ax.text(0.32, ypos + 1.17, formula, fontsize=5.0, color=color)
        matrix_glyph(ax, 4.10, ypos + 0.52, 1.40, 1.28, values, color=color)
        arrow(ax, (5.62, ypos + 1.16), (6.55, ypos + 1.16), color=color, linewidth=0.90)
        tensor_glyph(ax, 6.55, ypos + 0.58, 1.18, 1.18, color=color, slices=3)
        ax.text(8.15, ypos + 1.50, axes[0], fontsize=3.8, color=COLORS["gray"])
        ax.text(8.15, ypos + 1.03, axes[1], fontsize=3.8, color=COLORS["gray"])
        ax.text(8.15, ypos + 0.56, axes[2], fontsize=3.8, color=COLORS["gray"])
        if ypos > 1.5:
            ax.plot([0.32, 9.62], [ypos - 0.12, ypos - 0.12], color=COLORS["grid"], linewidth=0.55)
    ax.text(4.95, 0.30, "one predictor; route set and differentiation coordinate define the graph object", ha="center", fontsize=4.3, color=COLORS["text"], fontweight="bold")


def draw_panel_c(ax: plt.Axes) -> None:
    """Claim-specific audit output as route, coordinate, and horizon glyphs."""
    panel(ax, "c", "Coverage profile for a graph claim")

    # Route ledger.
    ax.text(0.35, 8.92, "route ledger", fontsize=4.8, color=COLORS["ink"], fontweight="bold")
    cols = ["predict", "score", "penalty"]
    for idx, label in enumerate(cols):
        ax.text(3.45 + 1.55 * idx, 8.92, label, ha="center", fontsize=3.9, color=COLORS["gray"], fontweight="bold")
    rows = [("raw $X$", COLORS["raw"], ["yes", "yes", "yes"]), ("aux $c$", COLORS["aux"], ["yes", "no", "no"])]
    for ridx, (label, color, states) in enumerate(rows):
        ypos = 7.92 - ridx * 1.15
        ax.text(0.55, ypos, label, va="center", fontsize=4.55, color=color, fontweight="bold")
        ax.plot([1.90, 7.85], [ypos - 0.50, ypos - 0.50], color=COLORS["grid"], linewidth=0.45)
        for cidx, state in enumerate(states):
            status_marker(ax, 3.45 + 1.55 * cidx, ypos, status=state, color=color, radius=0.15)

    ax.plot([0.35, 9.62], [5.92, 5.92], color=COLORS["grid"], linewidth=0.65)

    # Coordinate map and horizon support carry more visual weight than status text.
    ax.text(0.35, 5.38, "source-coordinate map", fontsize=4.65, color=COLORS["ink"], fontweight="bold")
    identity = np.eye(4)
    matrix_glyph(ax, 0.55, 3.18, 2.20, 1.72, identity, color=COLORS["raw"], label=r"$M_{\rm coord}$", fontsize=4.0)
    arrow(ax, (2.92, 4.04), (3.52, 4.04), color=COLORS["gray"], linewidth=0.85)
    graph = np.asarray([[0.02, 0.81, 0.10, 0.23], [0.30, 0.03, 0.64, 0.18], [0.12, 0.48, 0.02, 0.73], [0.55, 0.16, 0.36, 0.04]])
    matrix_glyph(ax, 3.52, 3.18, 2.20, 1.72, graph, color=COLORS["ink"], label=r"$S_{GC}$", fontsize=4.0)

    ax.text(6.25, 5.38, "attribution support", fontsize=4.65, color=COLORS["ink"], fontweight="bold")
    lags = np.arange(1, 13)
    mass = np.asarray([0.95, 0.66, 0.44, 0.31, 0.23, 0.18, 0.14, 0.11, 0.09, 0.07, 0.06, 0.05])
    xpos = 6.40 + 0.27 * (lags - 1)
    ypos = 3.20 + 1.42 * mass
    ax.fill_between(xpos, 3.20, ypos, where=lags > 3, color=COLORS["aux_soft"], alpha=1.0)
    ax.plot(xpos, ypos, color=COLORS["raw"], linewidth=1.0)
    ax.axvline(xpos[2], ymin=0.31, ymax=0.50, color=COLORS["aux"], linewidth=0.75, linestyle=(0, (2, 2)))
    ax.text(xpos[2], 4.79, "$K$", ha="center", fontsize=4.0, color=COLORS["aux"], fontweight="bold")
    ax.text(7.82, 2.84, "raw lag $h$", ha="center", fontsize=3.9, color=COLORS["gray"])

    ax.plot([0.35, 9.62], [2.28, 2.28], color=COLORS["grid"], linewidth=0.65)
    dimensions = [
        (0.45, "score route", "partial", COLORS["aux"]),
        (2.40, "penalty route", "partial", COLORS["aux"]),
        (4.35, "coordinate", "raw aligned", COLORS["raw"]),
        (6.30, "horizon", "declared", COLORS["raw"]),
        (8.25, "graph", "auditable", COLORS["ink"]),
    ]
    for xpos0, title, value, color in dimensions:
        ax.text(xpos0 + 0.62, 1.78, title, ha="center", fontsize=3.75, color=COLORS["gray"])
        box(ax, xpos0, 0.70, 1.24, 0.72, face=COLORS["white"], edge=color, linewidth=0.75)
        ax.text(xpos0 + 0.62, 1.06, value, ha="center", va="center", fontsize=3.95, color=color, fontweight="bold")
    ax.text(5.00, 0.24, "declare the graph object before comparing or reusing it", ha="center", fontsize=4.3, color=COLORS["text"], fontweight="bold")


def draw_overview(output_dir: Path) -> list[Path]:
    """Build the asymmetric schematic-led composite."""
    fig = plt.figure(figsize=(7.25, 4.72))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.08, 0.92],
        width_ratios=[1.03, 0.97],
        hspace=0.22,
        wspace=0.12,
    )
    draw_panel_a(fig.add_subplot(grid[0, :]))
    draw_panel_b(fig.add_subplot(grid[1, 0]))
    draw_panel_c(fig.add_subplot(grid[1, 1]))
    fig.subplots_adjust(left=0.025, right=0.995, top=0.945, bottom=0.045)
    return save_figure(fig, output_dir, "fig1_route_resolved_jacobian_audit_v4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in draw_overview(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
