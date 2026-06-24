"""
Figure 1 v7 -- dense orthogonal layout candidate.

Compared with v6, this candidate targets actual area usage rather than height
compression: shorter orthogonal arrows, tighter dashed panel frames, larger
content blocks, and square score-map panels in c.
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
        "legend.frameon": False,
    }
)

DARK = "#2B2B2B"
MID = "#666666"
PANEL = "#F5F5F5"
LIGHT = "#EFEFEF"
BLUE = "#1B5A9A"
BLUE_FILL = "#E8F0F8"
RED = "#B64A4A"
RED_FILL = "#F6E8E8"
BROWN = "#A39178"
STEEL = "#66798F"

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "paper-data",
    "figures",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def panel(ax, label, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    frame = FancyBboxPatch(
        (0.008, 0.020),
        0.984,
        0.960,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        facecolor=PANEL,
        edgecolor=MID,
        linewidth=0.75,
        linestyle=(0, (4, 3)),
        transform=ax.transAxes,
        zorder=0,
        clip_on=False,
    )
    ax.add_patch(frame)
    ax.text(0.045, 0.925, label, ha="left", va="center",
            fontsize=8.3, fontweight="bold", color=DARK, transform=ax.transAxes)
    ax.text(0.115, 0.925, title, ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=DARK, transform=ax.transAxes)


def box(ax, x, y, w, h, text="", edge=DARK, fill="white", lw=0.9, fs=6.3, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=3,
        clip_on=False,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, color=edge if edge != MID else DARK,
                linespacing=0.95, transform=ax.transAxes, zorder=4)
    return patch


def segment(ax, p0, p1, color=DARK, lw=1.0, arrow=False, ls="-"):
    if arrow:
        patch = FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=8, linewidth=lw,
            color=color, linestyle=ls, shrinkA=0, shrinkB=0,
            transform=ax.transAxes, zorder=2, clip_on=False,
        )
        ax.add_patch(patch)
    else:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw,
                ls=ls, transform=ax.transAxes, zorder=2, clip_on=False)


def ortho(ax, pts, color=DARK, lw=1.0, ls="-"):
    for p0, p1 in zip(pts[:-2], pts[1:-1]):
        segment(ax, p0, p1, color=color, lw=lw, arrow=False, ls=ls)
    segment(ax, pts[-2], pts[-1], color=color, lw=lw, arrow=True, ls=ls)


def matrix(ax, x, y, w, h, color=BLUE, label=r"$X$"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color,
                           linewidth=0.95, transform=ax.transAxes, zorder=3))
    vals = np.array(
        [
            [0.65, 0.42, 0.70, 0.50, 0.58, 0.33],
            [0.36, 0.72, 0.52, 0.80, 0.45, 0.62],
            [0.76, 0.58, 0.44, 0.53, 0.69, 0.38],
            [0.48, 0.83, 0.62, 0.41, 0.51, 0.73],
        ]
    )
    for i in range(4):
        for j in range(6):
            xx = x + 0.008 + j * (w - 0.016) / 6
            yy = y + 0.010 + (3 - i) * (h - 0.020) / 4
            ww = (w - 0.024) / 6
            hh = (h - 0.026) / 4
            ax.add_patch(Rectangle((xx, yy), ww, hh, facecolor=color,
                                   alpha=0.18 + 0.55 * vals[i, j],
                                   edgecolor="white", linewidth=0.22,
                                   transform=ax.transAxes, zorder=4))
    ax.text(x + w / 2, y - 0.040, label, ha="center", va="top",
            fontsize=6.2, color=color, transform=ax.transAxes)


def vector(ax, x, y, w, h, color=RED, label=r"$c_t$"):
    box(ax, x, y, w, h, "", edge=color, fill="white", lw=0.9)
    for j, alpha in enumerate([0.25, 0.47, 0.65, 0.35]):
        ax.add_patch(Rectangle((x + 0.018 + j * (w - 0.050) / 4, y + 0.025),
                               (w - 0.064) / 4, h - 0.050, facecolor=color,
                               alpha=alpha, edgecolor="white", linewidth=0.2,
                               transform=ax.transAxes, zorder=4))
    ax.text(x + w / 2, y - 0.038, label, ha="center", va="top",
            fontsize=6.1, color=color, transform=ax.transAxes)


def square_width(ax, height_frac):
    fig = ax.figure
    bbox = ax.get_position()
    return height_frac * (bbox.height * fig.get_figheight()) / (bbox.width * fig.get_figwidth())


def heat_square(ax, x, y, h, mat, edge, label):
    w = square_width(ax, h)
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=edge,
                           linewidth=0.95, transform=ax.transAxes, zorder=3))
    rows, cols = mat.shape
    for i in range(rows):
        for j in range(cols):
            val = mat[i, j]
            ax.add_patch(Rectangle((x + j * w / cols + 0.002,
                                    y + (rows - 1 - i) * h / rows + 0.002),
                                   w / cols - 0.004, h / rows - 0.004,
                                   facecolor=edge, alpha=0.08 + 0.78 * val,
                                   edgecolor="white", linewidth=0.18,
                                   transform=ax.transAxes, zorder=4))
    ax.text(x + w / 2, y + h + 0.030, label, ha="center", va="bottom",
            fontsize=6.0, fontweight="bold", color=edge, transform=ax.transAxes)
    return w


def draw_a(ax):
    panel(ax, "a", "concat shortcut")
    matrix(ax, 0.060, 0.660, 0.215, 0.190, BLUE, r"$X_{t-L:t}$")
    vector(ax, 0.060, 0.260, 0.215, 0.140, RED, r"$c_t$")
    box(ax, 0.390, 0.505, 0.190, 0.175, "concat", fill=LIGHT, fs=6.7)
    box(ax, 0.675, 0.505, 0.210, 0.175, "predictor", fill="white", fs=6.7)
    box(ax, 0.738, 0.195, 0.120, 0.120, r"$\hat y$", fill=LIGHT, fs=7.0)
    box(ax, 0.382, 0.190, 0.215, 0.135, r"$J_x$", edge=BLUE, fill=BLUE_FILL, lw=0.9, fs=6.6)

    ortho(ax, [(0.275, 0.755), (0.330, 0.755), (0.330, 0.595), (0.390, 0.595)], BLUE, 1.25)
    ortho(ax, [(0.275, 0.330), (0.330, 0.330), (0.330, 0.555), (0.390, 0.555)], RED, 1.45)
    ortho(ax, [(0.580, 0.595), (0.675, 0.595)], DARK, 1.0)
    ortho(ax, [(0.780, 0.505), (0.780, 0.315)], DARK, 0.95)
    segment(ax, (0.490, 0.505), (0.490, 0.325), BLUE, 0.9, True, (0, (3, 2)))
    ax.text(0.055, 0.095, r"score: $S_{ij}=\|\partial f_i/\partial x_j\|$",
            ha="left", va="center", fontsize=5.8, color=BLUE, transform=ax.transAxes)


def draw_b(ax):
    panel(ax, "b", "input-space repair")
    matrix(ax, 0.060, 0.630, 0.185, 0.175, BLUE, r"$X$")
    box(ax, 0.315, 0.585, 0.285, 0.210, r"$F_\theta$" + "\nMamba | TCN",
        edge=BLUE, fill=BLUE_FILL, lw=1.0, fs=6.8, weight="bold")
    box(ax, 0.665, 0.620, 0.140, 0.145, r"$X'$" + "\nsame $d$",
        edge=BLUE, fill="white", lw=0.95, fs=6.3)
    box(ax, 0.845, 0.620, 0.115, 0.145, r"MLP" + "\n" + r"$\hat y$",
        edge=DARK, fill=LIGHT, lw=0.9, fs=6.1)
    ortho(ax, [(0.245, 0.718), (0.315, 0.718)], BLUE, 1.2)
    ortho(ax, [(0.600, 0.692), (0.665, 0.692)], BLUE, 1.2)
    ortho(ax, [(0.805, 0.692), (0.845, 0.692)], DARK, 1.0)
    ax.text(0.335, 0.425, r"$J_x=(\partial\hat y/\partial X')(\partial X'/\partial X)$",
            ha="left", va="center", fontsize=5.9, color=BLUE, transform=ax.transAxes)
    box(
        ax,
        0.070,
        0.145,
        0.860,
        0.155,
        r"$X'=F_\theta(X)\in\mathbb{R}^{T\times d}$"
        + "     "
        + r"$\mathcal{L}=\mathcal{L}_{pred}+\lambda_J\|J_x\|_1+\lambda_O\|X'-X\|_2^2$",
        edge=MID,
        fill="white",
        lw=0.75,
        fs=5.9,
    )


def draw_c(ax):
    panel(ax, "c", "score semantics")
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
        [[0, 1, 0, 0, -1], [0, 0, -1, 1, 0], [-1, 0, 0, 1, 0],
         [1, 0, 0, 0, -1], [1, 0, 0, 0, 0]]
    ), 0, 1)

    h = 0.620
    y = 0.145
    w = heat_square(ax, 0.075, y, h, true, BLUE, "True A")
    w2 = heat_square(ax, 0.270, y, h, concat, RED, "Concat S")
    w3 = heat_square(ax, 0.465, y, h, istf, BLUE, "ISTF S")
    ortho(ax, [(0.075 + w, y + h / 2), (0.270, y + h / 2)], MID, 0.8)
    ortho(ax, [(0.270 + w2, y + h / 2), (0.465, y + h / 2)], MID, 0.8)

    x0, y0, bw, bh = 0.680, 0.165, 0.220, 0.600
    ax.plot([x0, x0], [y0, y0 + bh], color=DARK, lw=0.75, transform=ax.transAxes)
    ax.plot([x0, x0 + bw], [y0, y0], color=DARK, lw=0.75, transform=ax.transAxes)
    ax.plot([x0, x0 + bw], [y0 + bh * (1.0 / 1.15), y0 + bh * (1.0 / 1.15)],
            color=MID, lw=0.6, ls=(0, (3, 2)), transform=ax.transAxes)
    methods = ["Base", "Concat", "ISTF"]
    pearson = [0.9544, 0.2719, 0.9682]
    norm = [1.0867, 0.6289, 1.0433]
    group = bw / 3.45
    bar = group * 0.30
    for i, (p, n) in enumerate(zip(pearson, norm)):
        gx = x0 + 0.018 + i * group
        ax.add_patch(Rectangle((gx, y0), bar, bh * p / 1.15, facecolor=STEEL,
                               edgecolor="none", alpha=0.94, transform=ax.transAxes, zorder=4))
        ax.add_patch(Rectangle((gx + bar * 1.25, y0), bar, bh * n / 1.15, facecolor=BROWN,
                               edgecolor="none", alpha=0.94, transform=ax.transAxes, zorder=4))
        ax.text(gx + bar, y0 - 0.045, methods[i], ha="center", va="top",
                fontsize=5.3, color=DARK, transform=ax.transAxes)
    ax.text(x0 + bw / 2, y0 + bh + 0.035, "coefficient fidelity",
            ha="center", va="bottom", fontsize=6.0, fontweight="bold",
            color=DARK, transform=ax.transAxes)
    ax.text(x0 + 0.052, y0 + bh + 0.002, "r", ha="center", va="bottom",
            fontsize=5.3, color=STEEL, transform=ax.transAxes)
    ax.text(x0 + 0.090, y0 + bh + 0.002, "norm", ha="left", va="bottom",
            fontsize=5.3, color=BROWN, transform=ax.transAxes)
    ax.text(x0 + bw + 0.010, y0 + bh * (1.0 / 1.15), "1", ha="left",
            va="center", fontsize=5.3, color=MID, transform=ax.transAxes)


def save(fig, name):
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 600)]:
        path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", **({} if dpi is None else {"dpi": dpi}))
        print(f"  Saved {path}")


def main():
    fig = plt.figure(figsize=(7.15, 3.18), facecolor="white")
    ax_a = fig.add_axes([0.018, 0.515, 0.480, 0.465])
    ax_b = fig.add_axes([0.502, 0.515, 0.480, 0.465])
    ax_c = fig.add_axes([0.018, 0.040, 0.964, 0.450])
    draw_a(ax_a)
    draw_b(ax_b)
    draw_c(ax_c)
    save(fig, "fig1_istf_architecture_v7")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
