"""
Figure 1 v4 -- ISTF Architecture (KBS figure*).

Panel a explicitly separates the original input x_t from the auxiliary input
c_t, so the side channel is not implied to be generated from x_t. The Jacobian
penalty scope is shown as a dashed blue frame around the x-only path.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

BLUE = "#0F4D92"
RED = "#B64342"
GREEN = "#2E9E44"
GRAY = "#767676"
DARK = "#272727"
LGRAY = "#E8E8E8"
GREEN_FILL = "#F2F9F2"

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "paper-data",
    "figures",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def box(
    ax,
    x,
    y,
    w,
    h,
    text="",
    color=DARK,
    fill="white",
    lw=1.0,
    fs=6.5,
    fw="normal",
    tc=None,
    pad=0.025,
    ls="-",
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad}",
        facecolor=fill,
        edgecolor=color,
        linewidth=lw,
        linestyle=ls,
        zorder=3,
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
            color=tc or color,
            zorder=4,
            transform=ax.transAxes,
        )


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.0, rad=0.0, zorder=2, style="->"):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=8,
        color=color,
        linewidth=lw,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
        transform=ax.transAxes,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)


def txt(ax, x, y, s, color=DARK, fs=6.5, fw="normal", ha="center", va="center"):
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=fs,
        fontweight=fw,
        color=color,
        zorder=5,
        transform=ax.transAxes,
    )


def setup_ax(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_title(title, fontsize=7.5, fontweight="bold", color=DARK, loc="left", pad=4)


def draw_panel_a(ax):
    # Two independent inputs.
    box(ax, 0.05, 0.78, 0.36, 0.10, r"$\mathbf{x}_t$" + "\noriginal input", color=BLUE, fill=LGRAY, fs=6.2)
    box(ax, 0.59, 0.78, 0.36, 0.10, r"$\mathbf{c}_t$" + "\nauxiliary input", color=RED, fill="white", fs=6.2)

    # Dashed Jacobian scope around the x-only branch.
    box(
        ax,
        0.015,
        0.43,
        0.44,
        0.49,
        "",
        color=BLUE,
        fill="none",
        lw=0.8,
        pad=0.010,
        ls=(0, (3, 2)),
    )
    txt(ax, 0.045, 0.462, "penalized\nw.r.t. x only", color=BLUE, fs=5.2, ha="left")

    box(ax, 0.08, 0.56, 0.30, 0.10, "Original\npath", color=BLUE, fs=6.0, lw=1.0)
    box(ax, 0.62, 0.56, 0.30, 0.10, "Shortcut\npath", color=RED, fs=6.0, lw=1.0)
    box(ax, 0.18, 0.28, 0.64, 0.12, "Concat + prediction block", color=DARK, fill=LGRAY, fs=6.4, lw=1.0)
    txt(ax, 0.50, 0.07, r"$\hat{\mathbf{y}}_{t+1}$", color=DARK, fs=6.8, fw="bold")

    # Flow: blue x path and red auxiliary shortcut enter the same block.
    arrow(ax, 0.23, 0.78, 0.23, 0.66, color=BLUE, lw=1.2)
    arrow(ax, 0.23, 0.56, 0.23, 0.40, color=BLUE, lw=1.15)
    arrow(ax, 0.77, 0.78, 0.77, 0.66, color=RED, lw=1.2)
    arrow(ax, 0.77, 0.56, 0.62, 0.40, color=RED, lw=1.35)
    arrow(ax, 0.50, 0.28, 0.50, 0.12, color=DARK, lw=1.0)

    txt(ax, 0.83, 0.47, "shortcut", color=RED, fs=5.6, fw="bold", ha="left")


def draw_panel_b(ax):
    box(ax, 0.22, 0.78, 0.56, 0.10, r"$\mathbf{x}_t$" + "\n(d-dim input)", color=DARK, fill=LGRAY, fs=6.5)
    box(ax, 0.10, 0.44, 0.80, 0.16, "", color=GREEN, fill="white", lw=1.6)
    txt(ax, 0.50, 0.56, "Input-Space Temporal Filter", color=GREEN, fs=7, fw="bold")
    txt(ax, 0.50, 0.49, r"$\mathbf{x}_t \rightarrow \mathbf{x}_t^\prime$   (d-dim confined)", color=DARK, fs=6)
    ax.plot([0.04, 0.96], [0.40, 0.40], color=GREEN, lw=0.8, ls=":", clip_on=False, zorder=2, transform=ax.transAxes)
    txt(ax, 0.50, 0.365, "no side channel", color=GREEN, fs=5.5)
    arrow(ax, 0.50, 0.78, 0.50, 0.60, color=GREEN, lw=1.2)
    box(ax, 0.22, 0.13, 0.56, 0.09, "Prediction MLP", color=DARK, fill=LGRAY, fs=6.5)
    arrow(ax, 0.50, 0.44, 0.50, 0.22, color=GREEN, lw=1.1)
    txt(ax, 0.60, 0.31, r"$\mathbf{x}_t^\prime$", color=GREEN, fs=6, fw="bold")
    txt(ax, 0.50, 0.05, r"$\hat{\mathbf{y}}_{t+1}$", color=DARK, fs=6.5, fw="bold")
    arrow(ax, 0.50, 0.13, 0.50, 0.07, color=DARK, lw=0.9)
    txt(ax, 0.08, 0.26, "all MLP inputs\npenalized", color=GREEN, fs=5.4, ha="left")


def draw_panel_c(ax):
    box(ax, 0.22, 0.78, 0.56, 0.10, r"$\mathbf{x}_t$" + "\n(d-dim input)", color=DARK, fill=LGRAY, fs=6.5)
    box(ax, 0.02, 0.48, 0.44, 0.14, "", color=GREEN, fill="white", lw=1.3)
    txt(ax, 0.24, 0.58, "Mamba", color=GREEN, fs=7, fw="bold")
    txt(ax, 0.24, 0.51, "selective SSM", color=DARK, fs=6)
    box(ax, 0.54, 0.48, 0.44, 0.14, "", color=GREEN, fill="white", lw=1.3)
    txt(ax, 0.76, 0.58, "TCN", color=GREEN, fs=7, fw="bold")
    txt(ax, 0.76, 0.51, "temporal conv", color=DARK, fs=6)
    arrow(ax, 0.36, 0.78, 0.24, 0.62, color=GREEN, lw=1.0)
    arrow(ax, 0.64, 0.78, 0.76, 0.62, color=GREEN, lw=1.0)
    box(ax, 0.22, 0.31, 0.56, 0.08, r"$\mathbf{x}_t^\prime$", color=GREEN, fill=LGRAY, fs=6.5)
    arrow(ax, 0.24, 0.48, 0.40, 0.39, color=GREEN, lw=0.9)
    arrow(ax, 0.76, 0.48, 0.60, 0.39, color=GREEN, lw=0.9)
    box(ax, 0.06, 0.04, 0.88, 0.14, "", color=GREEN, fill=GREEN_FILL, lw=0.9)
    txt(ax, 0.50, 0.145, "shared ISTF constraint", color=GREEN, fs=5.5, fw="bold")
    txt(ax, 0.50, 0.095, "input-space confinement + orthogonality", color=DARK, fs=6)


def main():
    fig = plt.figure(figsize=(7.0, 2.55), facecolor="white")
    margins = {"left": 0.04, "right": 0.98, "bot": 0.08, "top": 0.88}
    panel_w = 0.29
    gap = 0.03

    axes = [
        fig.add_axes([margins["left"], margins["bot"], panel_w, margins["top"] - margins["bot"]]),
        fig.add_axes([margins["left"] + panel_w + gap, margins["bot"], panel_w, margins["top"] - margins["bot"]]),
        fig.add_axes([margins["left"] + 2 * (panel_w + gap), margins["bot"], panel_w, margins["top"] - margins["bot"]]),
    ]

    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c(axes[2])

    setup_ax(axes[0], "a  Auxiliary-channel shortcut")
    setup_ax(axes[1], "b  ISTF repair")
    setup_ax(axes[2], "c  Filter instantiations")

    for ext in ["svg", "pdf", "png"]:
        path = os.path.join(OUTPUT_DIR, f"fig1_istf_architecture_v4.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=(600 if ext == "png" else None))
        print(f"  Saved {path}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
