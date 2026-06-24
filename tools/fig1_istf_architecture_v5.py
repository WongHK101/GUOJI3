"""
Figure 1 v5 -- mechanism-rich ISTF architecture figure for the KBS paper.

This version replaces the text-heavy v4 flowchart with a composite method
figure: shortcut mechanism, ISTF repair, score-map semantics, and compact
diagnostic values already reported in the manuscript.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["xtick.labelsize"] = 6.4
plt.rcParams["ytick.labelsize"] = 6.4
plt.rcParams["legend.frameon"] = False

BLUE = "#0F4D92"
BLUE_LIGHT = "#E9F1FA"
RED = "#B64342"
RED_LIGHT = "#F8E8E6"
GREEN = "#2E9E44"
GREEN_LIGHT = "#EAF6ED"
GOLD = "#B68A2A"
GOLD_LIGHT = "#FBF4DD"
PEARSON_C = "#4C627F"
NORM_C = "#8B6F47"
GRAY = "#777777"
LIGHT_GRAY = "#ECECEC"
DARK = "#262626"

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "paper-data",
    "figures",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def panel(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.text(
        0.0,
        1.035,
        title,
        ha="left",
        va="bottom",
        fontsize=8.1,
        fontweight="bold",
        color=DARK,
        transform=ax.transAxes,
    )


def box(ax, x, y, w, h, text="", ec=DARK, fc="white", lw=1.0, fs=6.4,
        fw="normal", tc=None, ls="-", pad=0.012, z=3):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size=0.018",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=z,
        clip_on=False,
        transform=ax.transAxes,
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
            fontweight=fw,
            color=tc if tc is not None else ec,
            transform=ax.transAxes,
            zorder=z + 1,
            linespacing=1.05,
        )
    return patch


def arrow(ax, p1, p2, color=DARK, lw=1.0, rad=0.0, style="-|>",
          mutation=9, ls="-", z=2, alpha=1.0):
    patch = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        linestyle=ls,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        transform=ax.transAxes,
        zorder=z,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def mini_lag_stack(ax, x, y, w, h, color=BLUE, label=r"$X_{t-L:t}$"):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="white",
            edgecolor=color,
            linewidth=1.1,
            transform=ax.transAxes,
            zorder=3,
        )
    )
    rng = np.random.default_rng(4)
    rows, cols = 5, 7
    vals = 0.15 + 0.75 * rng.random((rows, cols))
    vals[1, 2:5] = [0.85, 0.92, 0.72]
    vals[3, 1:4] = [0.70, 0.78, 0.88]
    for i in range(rows):
        for j in range(cols):
            xx = x + 0.012 + j * (w - 0.024) / cols
            yy = y + 0.016 + i * (h - 0.041) / rows
            ww = (w - 0.032) / cols
            hh = (h - 0.052) / rows
            ax.add_patch(
                Rectangle(
                    (xx, yy),
                    ww,
                    hh,
                    facecolor=color,
                    alpha=0.18 + 0.58 * vals[i, j],
                    edgecolor="white",
                    linewidth=0.25,
                    transform=ax.transAxes,
                    zorder=4,
                )
            )
    ax.text(
        x + w / 2,
        y - 0.035,
        label,
        ha="center",
        va="top",
        fontsize=6.4,
        color=color,
        transform=ax.transAxes,
        zorder=5,
    )


def mini_aux_vector(ax, x, y, w, h):
    box(ax, x, y, w, h, "", ec=RED, fc="white", lw=1.1, pad=0.006)
    for j, alpha in enumerate([0.28, 0.60, 0.36, 0.75]):
        ax.add_patch(
            Rectangle(
                (x + 0.022 + j * (w - 0.055) / 4, y + 0.025),
                (w - 0.070) / 4,
                h - 0.050,
                facecolor=RED,
                alpha=alpha,
                edgecolor="white",
                linewidth=0.35,
                transform=ax.transAxes,
                zorder=4,
            )
        )
    ax.text(
        x + w / 2,
        y - 0.032,
        r"$c_t$",
        ha="center",
        va="top",
        fontsize=6.4,
        color=RED,
        transform=ax.transAxes,
    )


def draw_jacobian_card(ax, x, y, w, h, edge_color, caption, cmap_color):
    box(ax, x, y, w, h, "", ec=edge_color, fc="white", lw=0.9, pad=0.006)
    data = np.array(
        [
            [0.03, 0.86, 0.10, 0.05],
            [0.06, 0.04, 0.72, 0.12],
            [0.62, 0.06, 0.05, 0.18],
            [0.09, 0.14, 0.57, 0.03],
        ]
    )
    if cmap_color == "shortcut":
        data = np.array(
            [
                [0.04, 0.24, 0.13, 0.44],
                [0.36, 0.05, 0.18, 0.12],
                [0.12, 0.26, 0.06, 0.30],
                [0.41, 0.16, 0.21, 0.05],
            ]
        )
    for i in range(4):
        for j in range(4):
            val = data[i, j]
            color = RED if cmap_color == "shortcut" else edge_color
            ax.add_patch(
                Rectangle(
                    (x + 0.025 + j * (w - 0.05) / 4, y + 0.042 + (3 - i) * (h - 0.085) / 4),
                    (w - 0.058) / 4,
                    (h - 0.096) / 4,
                    facecolor=color,
                    alpha=0.12 + 0.76 * val,
                    edgecolor="white",
                    linewidth=0.3,
                    transform=ax.transAxes,
                    zorder=5,
                )
            )
    ax.text(
        x + w / 2,
        y + h - 0.020,
        caption,
        ha="center",
        va="top",
        fontsize=5.9,
        color=edge_color,
        transform=ax.transAxes,
        zorder=6,
    )


def draw_shortcut_panel(ax):
    panel(ax, "a  Shortcut: prediction and score decouple")

    # Penalty scope around the original input branch.
    box(
        ax,
        0.025,
        0.115,
        0.445,
        0.775,
        "",
        ec=BLUE,
        fc="none",
        lw=0.9,
        ls=(0, (3, 2)),
        pad=0.006,
        z=1,
    )
    ax.text(
        0.045,
        0.165,
        r"$S_{ij}=\left\|\partial f_i/\partial x_j\right\|$",
        ha="left",
        va="center",
        fontsize=6.0,
        color=BLUE,
        transform=ax.transAxes,
    )

    mini_lag_stack(ax, 0.06, 0.62, 0.19, 0.20, BLUE, r"$X_{t-L:t}$")
    mini_aux_vector(ax, 0.065, 0.27, 0.18, 0.125)

    box(ax, 0.355, 0.54, 0.195, 0.16, "concat", ec=DARK, fc=LIGHT_GRAY, lw=0.9, fs=6.7)
    box(ax, 0.650, 0.52, 0.235, 0.19, "prediction\nnetwork", ec=DARK, fc="white", lw=1.0, fs=6.4)
    box(ax, 0.700, 0.24, 0.145, 0.115, r"$\hat{y}_{t+1}$", ec=DARK, fc=LIGHT_GRAY, lw=0.8, fs=7.0)
    draw_jacobian_card(ax, 0.315, 0.18, 0.19, 0.20, BLUE, r"$J_x$", "normal")

    arrow(ax, (0.25, 0.71), (0.355, 0.63), color=BLUE, lw=1.45)
    arrow(ax, (0.245, 0.335), (0.355, 0.585), color=RED, lw=2.2, rad=-0.10)
    arrow(ax, (0.55, 0.62), (0.65, 0.62), color=DARK, lw=1.0)
    arrow(ax, (0.767, 0.52), (0.767, 0.355), color=DARK, lw=0.95)
    arrow(ax, (0.438, 0.54), (0.410, 0.38), color=BLUE, lw=0.95, ls="--", mutation=7)

    ax.text(
        0.055,
        0.840,
        "penalty scope: original input only",
        ha="left",
        va="center",
        fontsize=5.8,
        color=BLUE,
        transform=ax.transAxes,
    )


def draw_repair_panel(ax):
    panel(ax, "b  ISTF: filtering stays in input space")

    mini_lag_stack(ax, 0.035, 0.61, 0.18, 0.20, BLUE, r"$X_{t-L:t}$")
    box(ax, 0.300, 0.555, 0.315, 0.235, "", ec=GREEN, fc=GREEN_LIGHT, lw=1.3)
    ax.text(
        0.457,
        0.735,
        r"$F_\theta$",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=GREEN,
        transform=ax.transAxes,
    )
    box(ax, 0.330, 0.595, 0.105, 0.075, "Mamba", ec=GREEN, fc="white", lw=0.8, fs=5.7, pad=0.006)
    ax.text(0.457, 0.632, "OR", ha="center", va="center", fontsize=5.8, fontweight="bold",
            color=DARK, transform=ax.transAxes)
    box(ax, 0.482, 0.595, 0.105, 0.075, "TCN", ec=GREEN, fc="white", lw=0.8, fs=5.7, pad=0.006)
    box(ax, 0.715, 0.61, 0.18, 0.19, r"$X'_{t-L:t}$" + "\n(same d)", ec=GREEN, fc="white", lw=1.1, fs=6.4)
    box(ax, 0.385, 0.245, 0.30, 0.14, "prediction\nnetwork", ec=DARK, fc="white", lw=1.0, fs=6.4)
    box(ax, 0.745, 0.257, 0.13, 0.105, r"$\hat{y}_{t+1}$", ec=DARK, fc=LIGHT_GRAY, lw=0.8, fs=7.0)

    arrow(ax, (0.215, 0.705), (0.300, 0.675), color=GREEN, lw=1.45)
    arrow(ax, (0.615, 0.675), (0.715, 0.700), color=GREEN, lw=1.45)
    arrow(ax, (0.805, 0.610), (0.620, 0.385), color=GREEN, lw=1.05, rad=-0.08)
    arrow(ax, (0.685, 0.315), (0.745, 0.312), color=DARK, lw=0.95)

    ax.text(
        0.175,
        0.305,
        r"$J_x = (\partial \hat{y}/\partial X')(\partial X'/\partial X)$",
        ha="left",
        va="center",
        fontsize=5.8,
        color=BLUE,
        transform=ax.transAxes,
    )

    box(
        ax,
        0.090,
        0.045,
        0.795,
        0.120,
        r"$X' = F_\theta(X)\in\mathbb{R}^{T\times d}$     "
        r"$\mathcal{L}=\mathcal{L}_{pred}+\lambda_J\|J_x\|_1+\lambda_O\|X'-X\|_2^2$",
        ec=GREEN,
        fc="white",
        lw=0.8,
        fs=6.0,
        tc=DARK,
        pad=0.010,
    )


def draw_score_panel(fig):
    ax = fig.add_axes([0.045, 0.075, 0.915, 0.315])
    panel(ax, "c  Score-map semantics and diagnostic consistency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

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
    istf = np.clip(
        true
        + np.array(
            [
                [0, 0.06, 0.03, 0.02, -0.04],
                [0.02, 0, -0.03, 0.04, 0.02],
                [-0.05, 0.02, 0, 0.05, 0.01],
                [0.04, 0.01, 0.03, 0, -0.02],
                [0.03, -0.01, 0.02, 0.01, 0],
            ]
        ),
        0,
        1,
    )

    heat_specs = [
        ([0.085, 0.115, 0.132, 0.210], true, "True A", BLUE),
        ([0.285, 0.115, 0.132, 0.210], concat, "Concat score", RED),
        ([0.485, 0.115, 0.132, 0.210], istf, "ISTF score", GREEN),
    ]
    for pos, mat, title, color in heat_specs:
        hax = fig.add_axes(pos)
        hax.imshow(mat, vmin=0, vmax=1, cmap="YlGnBu", interpolation="nearest")
        hax.set_xticks([])
        hax.set_yticks([])
        for spine in hax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color(color)
        hax.set_title(title, fontsize=6.4, color=color, pad=2.5, fontweight="bold")

    # Arrows between score maps are drawn in the parent panel coordinate system.
    arrow(ax, (0.212, 0.520), (0.275, 0.520), color=GRAY, lw=0.7, mutation=7)
    arrow(ax, (0.412, 0.520), (0.475, 0.520), color=GRAY, lw=0.7, mutation=7)
    bax = fig.add_axes([0.705, 0.135, 0.220, 0.185])
    methods = ["Baseline", "Concat", "ISTF"]
    pearson = np.array([0.9544, 0.2719, 0.9682])
    norm_ratio = np.array([1.0867, 0.6289, 1.0433])
    x = np.arange(len(methods))
    width = 0.30
    bax.bar(x - width / 2, pearson, width, color=PEARSON_C, alpha=0.90, label="Pearson r", zorder=3)
    bax.bar(x + width / 2, norm_ratio, width, color=NORM_C, alpha=0.78, label="Norm ratio", zorder=3)
    bax.axhline(1.0, color=GRAY, lw=0.65, ls="--", zorder=1)
    bax.set_ylim(0, 1.15)
    bax.set_xticks(x)
    bax.set_xticklabels(methods, fontsize=5.8)
    bax.set_yticks([0, 0.5, 1.0])
    bax.tick_params(axis="both", length=2.5, width=0.6, pad=1.5)
    bax.set_ylabel("coefficient\nmetric", fontsize=5.8, labelpad=1)
    bax.set_title("Controlled VAR diagnostic", fontsize=6.4, pad=2.5, fontweight="bold")
    bax.legend(loc="upper center", bbox_to_anchor=(0.52, -0.30), ncol=2, fontsize=5.4,
               handlelength=1.0, columnspacing=0.65)
    bax.grid(axis="y", color="#D9D9D9", linewidth=0.45, zorder=0)
    for spine in ["left", "bottom"]:
        bax.spines[spine].set_linewidth(0.7)


def save_figure(fig, name):
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 600)]:
        path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        kwargs = {} if dpi is None else {"dpi": dpi}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"  Saved {path}")


def main():
    fig = plt.figure(figsize=(7.15, 4.15), facecolor="white")
    ax_a = fig.add_axes([0.045, 0.515, 0.435, 0.365])
    ax_b = fig.add_axes([0.535, 0.515, 0.425, 0.365])

    draw_shortcut_panel(ax_a)
    draw_repair_panel(ax_b)
    draw_score_panel(fig)

    save_figure(fig, "fig1_istf_architecture_v5")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
