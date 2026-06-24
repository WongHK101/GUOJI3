"""
Figure 1 v6 -- compact orthogonal-panel candidate.

Design constraints requested by the PI:
- no diagonal or curved arrows;
- three panels separated by dashed frames;
- restrained palette and typographic hierarchy;
- no text/arrow/shape overlap;
- standalone candidate only, not automatically inserted into the manuscript.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)

DARK = "#2B2B2B"
MID = "#6F6F6F"
LIGHT = "#F0F0F0"
GRID = "#D8D8D8"
BLUE = "#1B5A9A"
BLUE_FILL = "#EAF1F8"
RED = "#B64A4A"
RED_FILL = "#F6EAEA"

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "paper-data",
    "figures",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def init_panel(ax, label, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    frame = FancyBboxPatch(
        (0.010, 0.030),
        0.970,
        0.930,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor="white",
        edgecolor=MID,
        linewidth=0.75,
        linestyle=(0, (4, 3)),
        transform=ax.transAxes,
        zorder=0,
        clip_on=False,
    )
    ax.add_patch(frame)
    ax.text(
        0.045,
        0.910,
        label,
        ha="left",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=DARK,
        transform=ax.transAxes,
    )
    ax.text(
        0.120,
        0.910,
        title,
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=DARK,
        transform=ax.transAxes,
    )


def add_box(ax, x, y, w, h, text, edge=DARK, fill="white", lw=0.9, fs=6.2, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=3,
        clip_on=False,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight=weight,
            color=edge if edge != MID else DARK,
            transform=ax.transAxes,
            zorder=4,
            linespacing=1.0,
        )
    return patch


def arrow_segment(ax, start, end, color=DARK, lw=1.0, arrow=True, ls="-", z=2):
    if arrow:
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            linestyle=ls,
            color=color,
            transform=ax.transAxes,
            zorder=z,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
        )
        ax.add_patch(patch)
    else:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=lw,
            linestyle=ls,
            transform=ax.transAxes,
            zorder=z,
            clip_on=False,
        )


def ortho_arrow(ax, pts, color=DARK, lw=1.0, ls="-"):
    """Draw an orthogonal polyline; final segment carries the arrow head."""
    for p0, p1 in zip(pts[:-2], pts[1:-1]):
        arrow_segment(ax, p0, p1, color=color, lw=lw, arrow=False, ls=ls)
    arrow_segment(ax, pts[-2], pts[-1], color=color, lw=lw, arrow=True, ls=ls)


def draw_matrix(ax, x, y, w, h, color=BLUE, label=r"$X_{t-L:t}$"):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="white",
            edgecolor=color,
            linewidth=0.95,
            transform=ax.transAxes,
            zorder=3,
        )
    )
    rng = np.random.default_rng(8)
    vals = 0.20 + 0.70 * rng.random((4, 6))
    vals[1, 2:4] = [0.85, 0.70]
    vals[2, 0:2] = [0.72, 0.86]
    for i in range(4):
        for j in range(6):
            xx = x + 0.010 + j * (w - 0.020) / 6
            yy = y + 0.012 + (3 - i) * (h - 0.024) / 4
            ww = (w - 0.026) / 6
            hh = (h - 0.030) / 4
            ax.add_patch(
                Rectangle(
                    (xx, yy),
                    ww,
                    hh,
                    facecolor=color,
                    alpha=0.18 + 0.55 * vals[i, j],
                    edgecolor="white",
                    linewidth=0.25,
                    transform=ax.transAxes,
                    zorder=4,
                )
            )
    ax.text(
        x + w / 2,
        y - 0.045,
        label,
        ha="center",
        va="top",
        fontsize=6.0,
        color=color,
        transform=ax.transAxes,
        zorder=5,
    )


def draw_vector(ax, x, y, w, h, color=RED, label=r"$c_t$"):
    add_box(ax, x, y, w, h, "", edge=color, fill="white", lw=0.95)
    for j, alpha in enumerate([0.25, 0.50, 0.68, 0.38]):
        ax.add_patch(
            Rectangle(
                (x + 0.020 + j * (w - 0.050) / 4, y + 0.024),
                (w - 0.065) / 4,
                h - 0.048,
                facecolor=color,
                alpha=alpha,
                edgecolor="white",
                linewidth=0.25,
                transform=ax.transAxes,
                zorder=4,
            )
        )
    ax.text(
        x + w / 2,
        y - 0.042,
        label,
        ha="center",
        va="top",
        fontsize=6.0,
        color=color,
        transform=ax.transAxes,
        zorder=5,
    )


def draw_heatmap(ax, x, y, w, h, mat, edge, label):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="white",
            edgecolor=edge,
            linewidth=0.85,
            transform=ax.transAxes,
            zorder=3,
        )
    )
    rows, cols = mat.shape
    for i in range(rows):
        for j in range(cols):
            val = mat[i, j]
            xx = x + j * w / cols
            yy = y + (rows - 1 - i) * h / rows
            ax.add_patch(
                Rectangle(
                    (xx + 0.002, yy + 0.002),
                    w / cols - 0.004,
                    h / rows - 0.004,
                    facecolor=edge,
                    alpha=0.08 + 0.78 * val,
                    edgecolor="white",
                    linewidth=0.18,
                    transform=ax.transAxes,
                    zorder=4,
                )
            )
    ax.text(
        x + w / 2,
        y + h + 0.035,
        label,
        ha="center",
        va="bottom",
        fontsize=6.0,
        fontweight="bold",
        color=edge,
        transform=ax.transAxes,
        zorder=5,
    )


def panel_a(ax):
    init_panel(ax, "a", "concat shortcut")
    draw_matrix(ax, 0.070, 0.635, 0.205, 0.180, BLUE, r"$X_{t-L:t}$")
    draw_vector(ax, 0.070, 0.280, 0.205, 0.130, RED, r"$c_t$")
    add_box(ax, 0.405, 0.510, 0.185, 0.160, "concat", fill=LIGHT, fs=6.6)
    add_box(ax, 0.685, 0.510, 0.205, 0.160, "predictor", fill="white", fs=6.6)
    add_box(ax, 0.735, 0.205, 0.115, 0.115, r"$\hat y$", fill=LIGHT, fs=7.0)

    # Orthogonal routes into concat.
    ortho_arrow(ax, [(0.275, 0.725), (0.335, 0.725), (0.335, 0.590), (0.405, 0.590)], BLUE, 1.25)
    ortho_arrow(ax, [(0.275, 0.345), (0.335, 0.345), (0.335, 0.555), (0.405, 0.555)], RED, 1.45)
    ortho_arrow(ax, [(0.590, 0.590), (0.685, 0.590)], DARK, 1.0)
    ortho_arrow(ax, [(0.787, 0.510), (0.787, 0.320)], DARK, 0.95)

    add_box(ax, 0.390, 0.205, 0.210, 0.130, r"$J_x$", edge=BLUE, fill=BLUE_FILL, lw=0.85, fs=6.4)
    arrow_segment(ax, (0.498, 0.510), (0.498, 0.335), color=BLUE, lw=0.9, arrow=True, ls=(0, (3, 2)))
    ax.text(
        0.055,
        0.090,
        r"score: $S_{ij}=\|\partial f_i/\partial x_j\|$",
        ha="left",
        va="center",
        fontsize=5.7,
        color=BLUE,
        transform=ax.transAxes,
    )

def panel_b(ax):
    init_panel(ax, "b", "input-space repair")
    draw_matrix(ax, 0.060, 0.575, 0.175, 0.165, BLUE, r"$X$")
    add_box(ax, 0.315, 0.535, 0.275, 0.195, r"$F_\theta$" + "\nMamba | TCN", edge=BLUE, fill=BLUE_FILL, lw=1.05, fs=6.6, weight="bold")
    add_box(ax, 0.660, 0.565, 0.135, 0.135, r"$X'$" + "\nsame $d$", edge=BLUE, fill="white", lw=0.95, fs=6.2)
    add_box(ax, 0.835, 0.565, 0.120, 0.135, r"MLP" + "\n" + r"$\hat y$", edge=DARK, fill=LIGHT, lw=0.9, fs=6.1)

    ortho_arrow(ax, [(0.235, 0.658), (0.315, 0.658)], BLUE, 1.2)
    ortho_arrow(ax, [(0.590, 0.632), (0.660, 0.632)], BLUE, 1.2)
    ortho_arrow(ax, [(0.795, 0.632), (0.835, 0.632)], DARK, 1.0)

    add_box(
        ax,
        0.070,
        0.130,
        0.860,
        0.145,
        r"$X'=F_\theta(X)\in\mathbb{R}^{T\times d}$"
        + "     "
        + r"$\mathcal{L}=\mathcal{L}_{pred}+\lambda_J\|J_x\|_1+\lambda_O\|X'-X\|_2^2$",
        edge=MID,
        fill="white",
        lw=0.75,
        fs=6.0,
    )
    ax.text(
        0.335,
        0.385,
        r"$J_x=(\partial\hat y/\partial X')(\partial X'/\partial X)$",
        ha="left",
        va="center",
        fontsize=5.7,
        color=BLUE,
        transform=ax.transAxes,
    )

def panel_c(ax):
    init_panel(ax, "c", "score semantics")

    true = np.array(
        [
            [0.0, 0.85, 0.0, 0.0, 0.55],
            [0.0, 0.0, 0.75, 0.0, 0.0],
            [0.65, 0.0, 0.0, 0.70, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.80],
            [0.45, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    concat = np.array(
        [
            [0.0, 0.20, 0.42, 0.10, 0.14],
            [0.28, 0.0, 0.18, 0.35, 0.11],
            [0.08, 0.37, 0.0, 0.24, 0.34],
            [0.32, 0.12, 0.20, 0.0, 0.16],
            [0.14, 0.31, 0.09, 0.27, 0.0],
        ]
    )
    istf = np.clip(true + 0.06 * np.array(
        [
            [0, 1, 0, 0, -1],
            [0, 0, -1, 1, 0],
            [-1, 0, 0, 1, 0],
            [1, 0, 0, 0, -1],
            [1, 0, 0, 0, 0],
        ]
    ), 0, 1)

    draw_heatmap(ax, 0.070, 0.300, 0.150, 0.365, true, BLUE, "True A")
    draw_heatmap(ax, 0.290, 0.300, 0.150, 0.365, concat, RED, "Concat S")
    draw_heatmap(ax, 0.510, 0.300, 0.150, 0.365, istf, BLUE, "ISTF S")
    ortho_arrow(ax, [(0.220, 0.482), (0.290, 0.482)], MID, 0.8)
    ortho_arrow(ax, [(0.440, 0.482), (0.510, 0.482)], MID, 0.8)

    # Small coefficient diagnostic, neutral two-tone bars.
    x0, y0, w, h = 0.725, 0.250, 0.225, 0.440
    ax.plot([x0, x0], [y0, y0 + h], color=DARK, lw=0.75, transform=ax.transAxes)
    ax.plot([x0, x0 + w], [y0, y0], color=DARK, lw=0.75, transform=ax.transAxes)
    ax.plot([x0, x0 + w], [y0 + h * (1.0 / 1.15), y0 + h * (1.0 / 1.15)],
            color=MID, lw=0.6, ls=(0, (3, 2)), transform=ax.transAxes)
    methods = ["Base", "Concat", "ISTF"]
    pearson = [0.9544, 0.2719, 0.9682]
    norm = [1.0867, 0.6289, 1.0433]
    group_w = w / 3.5
    bar_w = group_w * 0.28
    for i, (p, n) in enumerate(zip(pearson, norm)):
        gx = x0 + 0.025 + i * group_w
        hp = h * p / 1.15
        hn = h * n / 1.15
        ax.add_patch(Rectangle((gx, y0), bar_w, hp, facecolor="#5C6F85", edgecolor="none",
                               transform=ax.transAxes, zorder=4, alpha=0.92))
        ax.add_patch(Rectangle((gx + bar_w * 1.25, y0), bar_w, hn, facecolor="#A39178", edgecolor="none",
                               transform=ax.transAxes, zorder=4, alpha=0.92))
        ax.text(gx + bar_w, y0 - 0.040, methods[i], ha="center", va="top", fontsize=5.3,
                color=DARK, transform=ax.transAxes)
    ax.text(x0 + w / 2, y0 + h + 0.040, "coefficient fidelity", ha="center", va="bottom",
            fontsize=6.0, fontweight="bold", color=DARK, transform=ax.transAxes)
    ax.text(x0 + w + 0.015, y0 + h * 0.86, "1", ha="left", va="center", fontsize=5.2,
            color=MID, transform=ax.transAxes)
    ax.text(x0 + 0.055, y0 + h + 0.003, "r", ha="center", va="bottom", fontsize=5.3,
            color="#5C6F85", transform=ax.transAxes)
    ax.text(x0 + 0.095, y0 + h + 0.003, "norm", ha="left", va="bottom", fontsize=5.3,
            color="#A39178", transform=ax.transAxes)


def save_figure(fig, name):
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 600)]:
        path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        kwargs = {} if dpi is None else {"dpi": dpi}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"  Saved {path}")


def main():
    fig = plt.figure(figsize=(7.15, 3.85), facecolor="white")
    ax_a = fig.add_axes([0.030, 0.535, 0.460, 0.420])
    ax_b = fig.add_axes([0.510, 0.535, 0.460, 0.420])
    ax_c = fig.add_axes([0.030, 0.065, 0.940, 0.420])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    save_figure(fig, "fig1_istf_architecture_v6")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
