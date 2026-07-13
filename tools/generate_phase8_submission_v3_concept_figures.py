"""Generate the Phase 8 submission-v3 conceptual figures.

The figures are schematic and source-verified; this script does not read model
results or run experiments.  It exports editable SVG/PDF and 600-dpi PNG files
for the manuscript's coverage mechanism, controlled concat architecture, and
claim-specific audit workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "phase8-kbs-visual-narrative-rework-v3",
    }
)


COLORS = {
    "ink": "#252A2E",
    "text": "#394047",
    "gray": "#737B82",
    "gray_mid": "#A8AFB5",
    "grid": "#DDE1E4",
    "panel": "#F6F7F8",
    "raw": "#55758E",
    "raw_mid": "#7F96A8",
    "raw_soft": "#EAF0F4",
    "aux": "#95635E",
    "aux_mid": "#B58A85",
    "aux_soft": "#F3EDEC",
    "ok": "#6F757A",
    "ok_soft": "#F1F2F2",
    "white": "#FFFFFF",
}


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, dpi in (("svg", None), ("pdf", None), ("png", 600)):
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {
            "bbox_inches": "tight",
            "facecolor": "white",
            "pad_inches": 0.025,
        }
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
                "\n".join(
                    line.rstrip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
        outputs.append(path)
    plt.close(fig)
    return outputs


def panel(ax: plt.Axes, label: str, title: str, *, face: str = "white") -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_facecolor(face)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(
        Rectangle(
            (0.02, 0.02),
            9.96,
            9.94,
            facecolor="none",
            edgecolor=COLORS["grid"],
            linewidth=0.55,
            linestyle=(0, (3.0, 2.2)),
            clip_on=False,
            zorder=-5,
        )
    )
    ax.text(
        0.0,
        10.16,
        label,
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="bottom",
    )
    ax.text(
        0.72,
        10.16,
        title,
        fontsize=7.8,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="bottom",
    )


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str = "white",
    edge: str = COLORS["gray_mid"],
    linewidth: float = 0.8,
    linestyle: str | tuple = "-",
    radius: float = 0.0,
) -> Rectangle:
    del radius  # Rectangular modules keep the diagrams compact and technical.
    patch = Rectangle(
        (x, y),
        w,
        h,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["gray"],
    linewidth: float = 1.0,
    dashed: bool = False,
    head: bool = True,
) -> None:
    if abs(start[0] - end[0]) > 1e-9 and abs(start[1] - end[1]) > 1e-9:
        raise ValueError("Process connectors must be horizontal or vertical")
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>" if head else "-",
            mutation_scale=8.5,
            linewidth=linewidth,
            linestyle=(0, (3.0, 2.0)) if dashed else "-",
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def elbow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    corner: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.0,
    dashed: bool = False,
) -> None:
    arrow(
        ax,
        start,
        corner,
        color=color,
        linewidth=linewidth,
        dashed=dashed,
        head=False,
    )
    arrow(
        ax,
        corner,
        end,
        color=color,
        linewidth=linewidth,
        dashed=dashed,
        head=True,
    )


def signal_strip(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    color: str,
    phase: float,
    n: int = 3,
    linewidth: float = 0.72,
) -> None:
    box(ax, x, y, w, h, face=COLORS["white"], edge=COLORS["grid"], linewidth=0.65)
    t = np.linspace(0.0, 1.0, 120)
    for idx in range(n):
        baseline = y + h * (idx + 0.5) / n
        wave = (
            0.22 * np.sin(2 * np.pi * (1.0 + 0.35 * idx) * t + phase + idx)
            + 0.08 * np.cos(2 * np.pi * (4.0 + idx) * t + 0.4 * phase)
        )
        ax.plot(
            x + 0.06 * w + 0.88 * w * t,
            baseline + wave * h / n,
            color=color,
            linewidth=linewidth,
            clip_on=True,
        )
        if idx < n - 1:
            yline = y + h * (idx + 1) / n
            ax.plot([x, x + w], [yline, yline], color=COLORS["grid"], linewidth=0.45)


def matrix_glyph(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    values: np.ndarray,
    *,
    color: str,
    frame: str = COLORS["gray_mid"],
    label: str | None = None,
    fontsize: float = 4.5,
) -> None:
    values = np.asarray(values, dtype=float)
    rows, cols = values.shape
    for row in range(rows):
        for col in range(cols):
            alpha = 0.08 + 0.78 * float(np.clip(values[row, col], 0.0, 1.0))
            cell = Rectangle(
                (x + col * w / cols, y + (rows - 1 - row) * h / rows),
                w / cols,
                h / rows,
                facecolor=matplotlib.colors.to_rgba(color, alpha),
                edgecolor=COLORS["white"],
                linewidth=0.35,
            )
            ax.add_patch(cell)
    box(ax, x, y, w, h, face="none", edge=frame, linewidth=0.65)
    if label:
        ax.text(
            x + w / 2,
            y - 0.27,
            label,
            ha="center",
            va="top",
            fontsize=fontsize,
            color=COLORS["text"],
        )


def network_glyph(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    edge: str = COLORS["ink"],
) -> None:
    box(ax, x, y, w, h, face=COLORS["white"], edge=edge, linewidth=0.85)
    layers = [3, 4, 3]
    xs = [x + 0.18 * w, x + 0.50 * w, x + 0.82 * w]
    coords: list[list[tuple[float, float]]] = []
    for xpos, count in zip(xs, layers):
        ys = np.linspace(y + 0.20 * h, y + 0.80 * h, count)
        coords.append([(xpos, float(ypos)) for ypos in ys])
    for left, right in zip(coords[:-1], coords[1:]):
        for source in left:
            for target in right:
                ax.plot(
                    [source[0], target[0]],
                    [source[1], target[1]],
                    color=COLORS["grid"],
                    linewidth=0.42,
                    zorder=1,
                )
    for layer_index, layer in enumerate(coords):
        for xpos, ypos in layer:
            face = COLORS["raw_soft"] if layer_index == 0 else COLORS["panel"]
            ax.add_patch(
                Circle(
                    (xpos, ypos),
                    radius=0.065 * min(w, h),
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.55,
                    zorder=2,
                )
            )


def tensor_glyph(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    color: str,
    slices: int = 3,
) -> None:
    values = np.asarray(
        [
            [0.10, 0.70, 0.15, 0.32],
            [0.58, 0.08, 0.84, 0.20],
            [0.16, 0.48, 0.10, 0.66],
            [0.74, 0.18, 0.40, 0.08],
        ]
    )
    for depth in reversed(range(slices)):
        dx = depth * 0.12 * w
        dy = depth * 0.12 * h
        matrix_glyph(
            ax,
            x + dx,
            y + dy,
            w,
            h,
            np.roll(values, depth, axis=1),
            color=color,
            frame=COLORS["gray_mid"],
        )


def status_marker(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    status: str,
    color: str,
    radius: float = 0.16,
) -> None:
    ax.add_patch(
        Circle((x, y), radius, facecolor=COLORS["white"], edgecolor=color, linewidth=0.85)
    )
    if status == "yes":
        ax.plot(
            [x - 0.07, x - 0.01, x + 0.09],
            [y, y - 0.07, y + 0.08],
            color=color,
            linewidth=0.9,
            solid_capstyle="round",
        )
    elif status == "no":
        ax.plot([x - 0.07, x + 0.07], [y - 0.07, y + 0.07], color=color, linewidth=0.85)
        ax.plot([x - 0.07, x + 0.07], [y + 0.07, y - 0.07], color=color, linewidth=0.85)
    else:
        ax.text(x, y - 0.01, "?", ha="center", va="center", fontsize=4.5, color=color, fontweight="bold")


def draw_figure1(output_dir: Path) -> list[Path]:
    """Schematic-led overview of route coverage and Jacobian interpretation."""
    fig = plt.figure(figsize=(7.25, 4.12))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.55, 1.0],
        height_ratios=[1.04, 0.96],
        wspace=0.13,
        hspace=0.27,
    )

    ax = fig.add_subplot(grid[:, 0])
    panel(ax, "a", "Prediction routes and the reported graph")
    signal_strip(ax, 0.35, 7.55, 2.25, 1.38, color=COLORS["raw"], phase=0.2)
    ax.text(0.38, 9.18, "raw history  $X$", fontsize=5.6, fontweight="bold", color=COLORS["raw"])
    signal_strip(ax, 0.35, 4.90, 2.25, 1.38, color=COLORS["aux"], phase=1.0)
    ax.text(0.38, 6.53, r"auxiliary route  $c=g_\phi(X)$", fontsize=5.6, fontweight="bold", color=COLORS["aux"])

    box(ax, 3.10, 4.78, 1.25, 4.25, face=COLORS["white"], edge=COLORS["gray_mid"])
    ax.text(3.72, 8.62, "concat", ha="center", va="center", fontsize=5.5, fontweight="bold")
    ax.text(3.72, 6.93, "$[X;c]$", ha="center", va="center", fontsize=7.0)
    ax.text(3.72, 5.35, "two route\nclasses", ha="center", va="center", fontsize=4.7, color=COLORS["gray"])
    arrow(ax, (2.60, 8.24), (3.10, 8.24), color=COLORS["raw"], linewidth=1.25)
    arrow(ax, (2.60, 5.59), (3.10, 5.59), color=COLORS["aux"], linewidth=1.25)

    network_glyph(ax, 4.95, 5.35, 2.35, 3.15)
    ax.text(6.13, 8.74, r"shared predictor  $f_\theta$", ha="center", fontsize=5.4, fontweight="bold")
    arrow(ax, (4.35, 6.90), (4.95, 6.90), color=COLORS["ink"], linewidth=1.0)
    box(ax, 7.95, 6.05, 1.62, 1.38, face=COLORS["white"], edge=COLORS["ink"])
    ax.text(8.76, 6.74, "prediction\n" + r"$\widehat{x}_t$", ha="center", va="center", fontsize=5.4, fontweight="bold")
    arrow(ax, (7.30, 6.74), (7.95, 6.74), color=COLORS["ink"], linewidth=1.0)

    box(ax, 7.78, 3.74, 1.96, 1.20, face=COLORS["white"], edge=COLORS["gray_mid"])
    ax.text(8.76, 4.34, "fixed-target MSE", ha="center", va="center", fontsize=4.7, fontweight="bold")
    ax.text(8.76, 3.93, r"$\|\widehat{x}_t-x_t\|_2^2$", ha="center", va="center", fontsize=5.4)
    arrow(ax, (8.76, 6.05), (8.76, 4.94), color=COLORS["gray"], linewidth=0.9)

    ax.plot([0.35, 9.60], [3.20, 3.20], color=COLORS["grid"], linewidth=0.75)
    ax.text(0.35, 2.86, "reported graph object", fontsize=5.3, fontweight="bold", color=COLORS["text"])
    ax.text(0.35, 2.35, r"$J^{x}=\partial\widehat{x}/\partial X\;|_{c\,\mathrm{fixed}}$", fontsize=5.3, color=COLORS["raw"])
    partial = np.asarray(
        [
            [0.08, 0.74, 0.12, 0.18],
            [0.20, 0.10, 0.62, 0.08],
            [0.42, 0.16, 0.08, 0.68],
            [0.12, 0.34, 0.22, 0.08],
        ]
    )
    matrix_glyph(ax, 3.55, 1.12, 1.45, 1.45, partial, color=COLORS["raw"], label="partial Jacobian")
    arrow(ax, (2.78, 1.84), (3.55, 1.84), color=COLORS["raw"], linewidth=1.0)
    ax.text(2.70, 1.84, "score", ha="right", va="center", fontsize=4.7, color=COLORS["raw"])
    ax.text(5.42, 1.84, r"$\Longrightarrow$", ha="center", va="center", fontsize=8.0, color=COLORS["gray"])
    matrix_glyph(ax, 6.05, 1.12, 1.45, 1.45, (partial > 0.35).astype(float), color=COLORS["ink"], label="directed graph")
    elbow_arrow(
        ax,
        (2.60, 5.25),
        (2.88, 5.25),
        (2.88, 2.32),
        color=COLORS["aux"],
        linewidth=0.95,
        dashed=True,
    )
    ax.text(1.22, 1.52, "auxiliary sensitivity\nchanges training but is absent\nfrom the x-only score", ha="center", va="center", fontsize=4.45, color=COLORS["aux"])
    ax.text(8.62, 1.75, "prediction and graph\ncan move independently", ha="center", va="center", fontsize=5.0, color=COLORS["text"], fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "b", "Raw-chain anatomy")
    ax.text(0.25, 8.88, r"$\frac{d\widehat{x}}{dX}$", fontsize=8.0, color=COLORS["ink"], va="center")
    ax.text(2.20, 8.88, "$=$", fontsize=7.5, color=COLORS["gray"], va="center")
    ax.text(2.92, 8.88, "$J^{x}$", fontsize=8.0, color=COLORS["raw"], va="center", fontweight="bold")
    ax.text(4.08, 8.88, "$+$", fontsize=7.5, color=COLORS["gray"], va="center")
    ax.text(4.82, 8.88, "$J^{c}$", fontsize=8.0, color=COLORS["aux"], va="center", fontweight="bold")
    ax.text(6.12, 8.88, r"$\frac{dg_\phi}{dX}$", fontsize=7.2, color=COLORS["aux"], va="center")
    ax.text(0.25, 7.86, "total raw attribution", fontsize=4.8, color=COLORS["gray"])

    indirect = np.asarray(
        [
            [0.04, 0.12, 0.38, 0.10],
            [0.24, 0.06, 0.14, 0.44],
            [0.18, 0.32, 0.06, 0.20],
            [0.42, 0.10, 0.24, 0.04],
        ]
    )
    total = np.clip(partial + 0.75 * indirect, 0.0, 1.0)
    matrix_glyph(ax, 0.45, 4.62, 2.12, 2.12, partial, color=COLORS["raw"], label="$J^{x}$  direct")
    ax.text(2.95, 5.68, "+", ha="center", va="center", fontsize=9.0, color=COLORS["gray"])
    matrix_glyph(ax, 3.38, 4.62, 2.12, 2.12, indirect, color=COLORS["aux"], label=r"$J^{c}dg_\phi/dX$  indirect")
    ax.text(5.88, 5.68, "=", ha="center", va="center", fontsize=9.0, color=COLORS["gray"])
    matrix_glyph(ax, 6.32, 4.62, 2.12, 2.12, total, color=COLORS["ink"], label="total raw-chain")
    ax.text(0.45, 2.18, "Absolute Jacobians are averaged over eligible windows,", fontsize=4.45, color=COLORS["gray"])
    ax.text(0.45, 1.47, "then aggregated over the declared lag support.", fontsize=4.45, color=COLORS["gray"])
    ax.plot([0.45, 8.48], [0.78, 0.78], color=COLORS["grid"], linewidth=0.65)
    ax.text(4.46, 0.28, "route decomposition precedes graph interpretation", ha="center", fontsize=4.55, color=COLORS["text"], fontweight="bold")

    ax = fig.add_subplot(grid[1, 1])
    panel(ax, "c", "Coverage declaration and profile")
    ax.text(0.25, 8.82, "$C=(V_{score},V_{penalty},P_{pred},M_{coord},H_{attr})$", fontsize=6.2, color=COLORS["ink"], fontweight="bold")
    columns = ["route", "score", "penalty", "source", "horizon"]
    xs = [0.25, 2.82, 4.40, 6.04, 7.74]
    widths = [2.35, 1.32, 1.38, 1.42, 1.62]
    for xpos, width, label in zip(xs, widths, columns):
        ax.text(xpos + width / 2, 7.62, label, ha="center", fontsize=4.5, color=COLORS["gray"], fontweight="bold")
    rows = [
        (5.88, "raw $X$", ["yes", "yes", "yes", "yes"], COLORS["raw"], COLORS["raw_soft"]),
        (4.43, "aux $c$", ["no", "no", "?", "?"], COLORS["aux"], COLORS["aux_soft"]),
    ]
    for ypos, label, statuses, edge, face in rows:
        box(ax, xs[0], ypos, widths[0], 1.02, face=face, edge=edge, linewidth=0.75)
        ax.text(xs[0] + widths[0] / 2, ypos + 0.51, label, ha="center", va="center", fontsize=5.0, fontweight="bold", color=edge)
        for xpos, width, status in zip(xs[1:], widths[1:], statuses):
            box(ax, xpos, ypos, width, 1.02, face=COLORS["white"], edge=COLORS["grid"], linewidth=0.6)
            status_marker(ax, xpos + width / 2, ypos + 0.51, status=status, color=edge, radius=0.14)
    labels = [
        (0.35, "score routes", "PARTIAL", COLORS["aux"]),
        (2.35, "penalty routes", "PARTIAL", COLORS["aux"]),
        (4.52, "alignment", "PARTIAL", COLORS["aux"]),
        (6.15, "coordinates", "CHECK", COLORS["gray"]),
        (8.00, "horizon", "CHECK", COLORS["gray"]),
    ]
    for xpos, title, status, edge in labels:
        ax.text(xpos + 0.65, 2.90, title, ha="center", fontsize=3.95, color=COLORS["gray"])
        box(ax, xpos, 1.60, 1.30, 0.92, face=COLORS["white"], edge=edge, linewidth=0.75)
        ax.text(xpos + 0.65, 2.06, status, ha="center", va="center", fontsize=4.25, color=edge, fontweight="bold")
    ax.text(4.75, 0.56, "one claim-specific profile; multiple dimensions remain visible", ha="center", fontsize=4.45, color=COLORS["text"], fontweight="bold")

    fig.subplots_adjust(left=0.025, right=0.995, top=0.935, bottom=0.055)
    return save_figure(fig, output_dir, "fig1_jacobian_coverage_system_v3")


def draw_figure2(output_dir: Path) -> list[Path]:
    """Source-verified controlled concat architecture and derivative objects."""
    fig = plt.figure(figsize=(7.25, 3.42))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.16, 1.12, 1.05], wspace=0.16)

    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "a", "Causal prefix and lag-window sampling")
    t = np.linspace(0.0, 1.0, 160)
    y_offsets = [7.65, 6.55, 5.45]
    for idx, baseline in enumerate(y_offsets):
        wave = 0.28 * np.sin(2 * np.pi * (1.25 + 0.4 * idx) * t + 0.7 * idx) + 0.09 * np.cos(2 * np.pi * 5 * t)
        ax.plot(0.55 + 8.75 * t, baseline + wave, color=COLORS["raw"], linewidth=0.68)
        ax.text(0.20, baseline, f"$x_{idx+1}$", ha="left", va="center", fontsize=4.5, color=COLORS["raw"])
    ax.axvspan(5.50, 8.30, ymin=0.47, ymax=0.89, facecolor=COLORS["raw_soft"], edgecolor="none")
    ax.axvline(8.30, ymin=0.45, ymax=0.92, color=COLORS["gray"], linewidth=0.75, linestyle=(0, (2, 2)))
    ax.text(6.90, 9.08, "predictor history  $t-K{:}t-1$", ha="center", fontsize=4.65, color=COLORS["raw"], fontweight="bold")
    ax.text(8.42, 4.72, "target  $x_t$", ha="left", va="center", fontsize=4.6, color=COLORS["ink"], fontweight="bold")
    ax.text(0.55, 4.55, "raw prefix  $X_{0:t-1}$", fontsize=4.8, color=COLORS["text"], fontweight="bold")

    box(ax, 0.55, 2.70, 2.08, 1.02, face=COLORS["white"], edge=COLORS["gray_mid"])
    ax.text(1.59, 3.21, r"$g_\phi$  causal state-space", ha="center", va="center", fontsize=4.55, fontweight="bold")
    arrow(ax, (1.59, 4.32), (1.59, 3.72), color=COLORS["raw"], linewidth=0.9)
    signal_strip(ax, 3.35, 2.32, 4.95, 1.75, color=COLORS["aux"], phase=1.25, n=2, linewidth=0.67)
    arrow(ax, (2.63, 3.21), (3.35, 3.21), color=COLORS["aux"], linewidth=0.95)
    ax.axvspan(6.75, 8.30, ymin=0.22, ymax=0.41, facecolor=COLORS["aux_soft"], edgecolor="none")
    ax.text(5.82, 1.86, "auxiliary prefix  $c_{0:t-1}$", ha="center", fontsize=4.6, color=COLORS["aux"], fontweight="bold")
    ax.text(5.00, 0.72, "prefix-stateful; the raw target and future samples are excluded", ha="center", fontsize=4.35, color=COLORS["gray"])

    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "b", "Two-lane predictor and fixed raw target")
    signal_strip(ax, 0.25, 7.15, 2.50, 1.28, color=COLORS["raw"], phase=0.3, n=2)
    signal_strip(ax, 0.25, 4.70, 2.50, 1.28, color=COLORS["aux"], phase=1.1, n=2)
    ax.text(1.50, 8.73, "$X_{t-K:t-1}$", ha="center", fontsize=5.0, color=COLORS["raw"], fontweight="bold")
    ax.text(1.50, 6.28, "$c_{t-K:t-1}$", ha="center", fontsize=5.0, color=COLORS["aux"], fontweight="bold")
    network_glyph(ax, 3.60, 4.78, 2.42, 3.60)
    ax.text(4.81, 8.72, "concat + JRNGC", ha="center", fontsize=5.1, color=COLORS["ink"], fontweight="bold")
    arrow(ax, (2.75, 7.79), (3.60, 7.79), color=COLORS["raw"], linewidth=1.05)
    arrow(ax, (2.75, 5.34), (3.60, 5.34), color=COLORS["aux"], linewidth=1.05)
    box(ax, 6.80, 6.10, 1.90, 1.36, face=COLORS["white"], edge=COLORS["ink"])
    ax.text(7.75, 6.78, r"$\widehat{x}_t=f_\theta([X;c])$", ha="center", va="center", fontsize=5.1, fontweight="bold")
    arrow(ax, (6.02, 6.78), (6.80, 6.78), color=COLORS["ink"], linewidth=0.95)
    box(ax, 6.45, 2.92, 2.60, 1.46, face=COLORS["panel"], edge=COLORS["gray_mid"])
    ax.text(7.75, 3.78, r"$\mathcal{L}_{pred}=\|\widehat{x}_t-x_t\|_2^2$", ha="center", va="center", fontsize=5.0, fontweight="bold")
    ax.text(7.75, 3.28, "pure fixed-target MSE", ha="center", va="center", fontsize=4.25, color=COLORS["gray"])
    arrow(ax, (7.75, 6.10), (7.75, 4.38), color=COLORS["gray"], linewidth=0.85)
    box(ax, 0.45, 1.03, 2.20, 1.08, face=COLORS["white"], edge=COLORS["ink"])
    ax.text(1.55, 1.57, "fixed raw target  $x_t$", ha="center", va="center", fontsize=4.8, fontweight="bold")
    elbow_arrow(ax, (2.65, 1.57), (7.75, 1.57), (7.75, 2.92), color=COLORS["ink"], linewidth=0.85)
    ax.text(4.65, 0.42, "prediction gradients traverse both lanes", ha="center", fontsize=4.35, color=COLORS["text"], fontweight="bold")

    ax = fig.add_subplot(grid[0, 2])
    panel(ax, "c", "Three derivative objects on one computation graph")
    rows = [
        (7.26, "partial score", r"$J^{x}=\partial\widehat{x}/\partial X\;|_{c\,fixed}$", "raw", "off"),
        (4.47, "full coordinate penalty", "$[J^{x},J^{c}]$", "raw", "aux"),
        (1.68, "total raw-chain score", r"$d\widehat{x}/dX=J^{x}+J^{c}dg_\phi/dX$", "total", "total"),
    ]
    for ypos, title, formula, raw_state, aux_state in rows:
        box(ax, 0.25, ypos, 9.25, 2.06, face=COLORS["white"], edge=COLORS["grid"], linewidth=0.65)
        ax.text(0.55, ypos + 1.62, title, ha="left", fontsize=4.75, color=COLORS["ink"], fontweight="bold")
        ax.text(9.18, ypos + 1.62, formula, ha="right", fontsize=4.45, color=COLORS["text"])
        raw_color = COLORS["raw"]
        aux_color = COLORS["aux"]
        ax.add_patch(Circle((1.10, ypos + 0.66), 0.18, facecolor=COLORS["raw_soft"], edgecolor=raw_color, linewidth=0.75))
        ax.add_patch(Circle((1.10, ypos + 1.14), 0.18, facecolor=COLORS["aux_soft"], edgecolor=aux_color, linewidth=0.75))
        network_glyph(ax, 3.00, ypos + 0.42, 1.42, 0.98)
        ax.add_patch(Circle((6.10, ypos + 0.91), 0.20, facecolor=COLORS["white"], edgecolor=COLORS["ink"], linewidth=0.75))
        if raw_state in {"raw", "total"}:
            arrow(ax, (1.28, ypos + 0.66), (3.00, ypos + 0.66), color=raw_color, linewidth=1.0)
        if aux_state in {"aux", "total"}:
            arrow(ax, (1.28, ypos + 1.14), (3.00, ypos + 1.14), color=aux_color, linewidth=1.0)
        else:
            arrow(ax, (1.28, ypos + 1.14), (2.78, ypos + 1.14), color=aux_color, linewidth=0.8, dashed=True)
            ax.text(2.07, ypos + 1.38, "held fixed", ha="center", fontsize=3.75, color=aux_color)
        arrow(ax, (4.42, ypos + 0.91), (5.90, ypos + 0.91), color=COLORS["ink"], linewidth=0.85)
        ax.text(6.10, ypos + 0.91, r"$\hat{x}$", ha="center", va="center", fontsize=4.2, fontweight="bold")
        if raw_state == "total":
            ax.text(7.02, ypos + 0.68, r"differentiate through $g_\phi$", ha="left", fontsize=3.9, color=aux_color)
    ax.text(4.90, 0.52, "same predictor; different derivative, coordinate, and route coverage", ha="center", fontsize=4.3, color=COLORS["gray"])

    fig.subplots_adjust(left=0.025, right=0.995, top=0.925, bottom=0.065)
    return save_figure(fig, output_dir, "fig2_controlled_concat_computation_v3")


def draw_figure3(output_dir: Path) -> list[Path]:
    """Operational audit pipeline and diagnostic failure signatures."""
    fig = plt.figure(figsize=(7.25, 3.20))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.16)

    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "a", "From a graph claim to an auditable profile")
    stage_x = [0.25, 2.75, 5.25, 7.75]
    stage_titles = ["DECLARE", "TRACE", "VALIDATE", "PROFILE"]
    stage_subtitles = ["graph object", "exact derivative", "map + horizon", "retain flags"]
    for idx, (xpos, title, subtitle) in enumerate(zip(stage_x, stage_titles, stage_subtitles), start=1):
        ax.text(xpos, 9.15, str(idx), fontsize=4.5, color=COLORS["gray"], fontweight="bold")
        ax.text(xpos + 0.40, 9.15, title, fontsize=5.2, color=COLORS["ink"], fontweight="bold")
        ax.text(xpos, 8.58, subtitle, fontsize=4.25, color=COLORS["gray"])
        if idx < 4:
            arrow(ax, (xpos + 1.88, 5.80), (xpos + 2.35, 5.80), color=COLORS["gray"], linewidth=0.85)

    # Stage 1: direct graph object.
    adjacency = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    matrix_glyph(ax, 0.35, 4.55, 1.62, 1.62, adjacency, color=COLORS["ink"], label=r"$S_{GC}$: target $\times$ source")
    ax.text(1.16, 3.58, "nominal lag $K$", ha="center", fontsize=4.1, color=COLORS["text"])

    # Stage 2: route ledger plus computation graph.
    box(ax, 2.78, 4.36, 1.82, 2.16, face=COLORS["white"], edge=COLORS["grid"], linewidth=0.7)
    ax.text(3.12, 6.12, "route", fontsize=3.9, color=COLORS["gray"], fontweight="bold")
    ax.text(3.94, 6.12, "score", fontsize=3.9, color=COLORS["gray"], fontweight="bold")
    ax.text(3.08, 5.48, "$X$", fontsize=4.4, color=COLORS["raw"], fontweight="bold")
    status_marker(ax, 4.10, 5.50, status="yes", color=COLORS["raw"], radius=0.13)
    ax.text(3.08, 4.75, "$c$", fontsize=4.4, color=COLORS["aux"], fontweight="bold")
    status_marker(ax, 4.10, 4.77, status="no", color=COLORS["aux"], radius=0.13)
    ax.text(3.69, 3.58, r"$\partial\hat{x}/\partial(\cdot)$", ha="center", fontsize=4.0, color=COLORS["text"])

    # Stage 3: lag-indexed Jacobian tensor and support map.
    tensor_glyph(ax, 5.32, 4.55, 1.42, 1.42, color=COLORS["raw"], slices=3)
    ax.text(6.05, 3.58, r"$\bar{J}_{ijh}$ over eligible windows", ha="center", fontsize=4.0, color=COLORS["text"])
    ax.plot([5.35, 7.18], [3.07, 3.07], color=COLORS["grid"], linewidth=0.6)
    for idx, xpos in enumerate(np.linspace(5.45, 7.05, 6), start=1):
        height = [0.72, 0.54, 0.36, 0.22, 0.12, 0.07][idx - 1]
        ax.add_patch(Rectangle((xpos, 2.05), 0.18, height, facecolor=COLORS["raw_mid"], edgecolor="none"))
    ax.text(6.28, 1.70, "lag support", ha="center", fontsize=3.95, color=COLORS["gray"])

    # Stage 4: compact five-dimension profile.
    labels = ["score", "penalty", "align", "coord", "horizon"]
    statuses = ["no", "no", "no", "yes", "?"]
    colors = [COLORS["aux"], COLORS["aux"], COLORS["aux"], COLORS["ok"], COLORS["gray"]]
    for row, (label, status, color) in enumerate(zip(labels, statuses, colors)):
        ypos = 6.25 - row * 0.78
        ax.text(7.88, ypos, label, ha="left", va="center", fontsize=4.1, color=COLORS["text"])
        status_marker(ax, 9.34, ypos, status=status, color=color, radius=0.13)
    ax.text(8.68, 1.70, "claim-specific profile", ha="center", fontsize=4.1, color=COLORS["gray"])
    ax.plot([0.35, 9.65], [0.95, 0.95], color=COLORS["grid"], linewidth=0.65)
    ax.text(5.00, 0.35, r"architecture declaration $\rightarrow$ derivative tensor $\rightarrow$ graph interpretation", ha="center", fontsize=4.45, color=COLORS["text"], fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "b", "Three coverage signatures")
    rows = [7.02, 4.14, 1.26]
    titles = ["route discrepancy", "coordinate mixing", "attribution beyond nominal lag"]
    for ypos, title in zip(rows, titles):
        box(ax, 0.18, ypos, 9.60, 2.30, face=COLORS["white"], edge=COLORS["grid"], linewidth=0.65)
        ax.text(0.45, ypos + 1.88, title, fontsize=4.85, fontweight="bold", color=COLORS["ink"])

    partial = np.asarray([[0.05, 0.70, 0.15], [0.25, 0.08, 0.55], [0.10, 0.38, 0.06]])
    total = np.asarray([[0.05, 0.72, 0.44], [0.52, 0.08, 0.57], [0.18, 0.64, 0.06]])
    matrix_glyph(ax, 0.55, 7.35, 1.55, 1.25, partial, color=COLORS["raw"], label="partial")
    ax.text(2.48, 7.98, "vs", ha="center", va="center", fontsize=4.3, color=COLORS["gray"])
    matrix_glyph(ax, 2.87, 7.35, 1.55, 1.25, total, color=COLORS["ink"], label="total")
    ax.text(5.10, 8.22, "$r_{partial,total}$", fontsize=4.65, color=COLORS["text"], fontweight="bold")
    ax.text(5.10, 7.58, "top-$k$ Jaccard", fontsize=4.35, color=COLORS["text"])
    ax.text(5.10, 7.05, "$M_{missing}$", fontsize=4.35, color=COLORS["aux"])

    raw_map = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mixed_map = np.asarray([[0.45, 0.55, 0.20], [0.35, 0.25, 0.60], [0.20, 0.50, 0.40]])
    matrix_glyph(ax, 0.55, 4.47, 1.55, 1.25, raw_map, color=COLORS["raw"], label="raw sources")
    ax.text(2.48, 5.10, r"$\rightarrow$", ha="center", va="center", fontsize=6.0, color=COLORS["gray"])
    matrix_glyph(ax, 2.87, 4.47, 1.55, 1.25, mixed_map, color=COLORS["aux"], label="mixed coordinates")
    ax.text(5.10, 5.28, r"$M_{coord}: z\mapsto X$", fontsize=4.65, color=COLORS["text"], fontweight="bold")
    ax.text(5.10, 4.58, "cross-variable leakage", fontsize=4.35, color=COLORS["aux"])
    ax.text(5.10, 4.06, "source identity audit", fontsize=4.35, color=COLORS["text"])

    lags = np.arange(1, 13)
    mass = np.asarray([0.86, 0.58, 0.40, 0.29, 0.22, 0.17, 0.13, 0.10, 0.08, 0.06, 0.05, 0.04])
    ax.plot(0.65 + 0.31 * (lags - 1), 1.67 + 0.85 * mass, color=COLORS["raw"], linewidth=1.0)
    ax.axvline(0.65 + 0.31 * 2, ymin=0.14, ymax=0.29, color=COLORS["aux"], linewidth=0.8, linestyle=(0, (2, 2)))
    ax.fill_between(0.65 + 0.31 * (lags - 1), 1.67, 1.67 + 0.85 * mass, where=lags > 3, color=COLORS["aux_soft"], alpha=1.0)
    ax.text(1.27, 2.52, "nominal $K$", ha="center", fontsize=3.8, color=COLORS["aux"])
    ax.text(2.45, 1.35, "raw lag $h$", ha="center", fontsize=3.8, color=COLORS["gray"])
    ax.text(5.10, 2.48, "tail mass", fontsize=4.65, color=COLORS["text"], fontweight="bold")
    ax.text(5.10, 1.83, "$S_{nominal}$ vs $S_{full-H}$", fontsize=4.35, color=COLORS["text"])
    ax.text(5.10, 1.30, "omitted-support audit", fontsize=4.35, color=COLORS["aux"])
    ax.text(4.95, 0.38, "each signature has a distinct remediation and claim boundary", ha="center", fontsize=4.35, color=COLORS["gray"])

    fig.subplots_adjust(left=0.025, right=0.995, top=0.925, bottom=0.065)
    return save_figure(fig, output_dir, "fig3_claim_specific_audit_signatures_v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = []
    outputs.extend(draw_figure1(args.output_dir))
    outputs.extend(draw_figure2(args.output_dir))
    outputs.extend(draw_figure3(args.output_dir))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
