"""
Figure 2 — Mechanistic Diagnostics (KBS figure*, nature-figure conventions).
v2.2: 3-panel (a,b,c) layout with panel-a split into stacked AUROC/loss.
Clean KBS style: unified font/color, no in-figure annotations, no dual-y squeeze.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 7
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['legend.frameon'] = False
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['axes.labelsize'] = 7.5

BLUE      = '#0F4D92'
RED       = '#B64342'
GREEN     = '#2E9E44'
GRAY      = '#767676'
DARK      = '#272727'
CONCAT_C  = '#D98C7A'
ISTF_C    = '#5B9E6F'

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "paper-data", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_panel_label(ax, label, x=-0.10, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='bottom', ha='left', color=DARK)


def save_figure(fig, name):
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 600)]:
        p = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        kw = {} if dpi is None else {"dpi": dpi}
        fig.savefig(p, bbox_inches="tight", **kw)
    print(f"  {name}.{{svg,pdf,png}}")


def main():
    fig = plt.figure(figsize=(7.0, 3.45), facecolor='white')

    # Grid: left column panel (a) is taller, right column split for (b) top, (c) bottom
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.97, bottom=0.10, top=0.79,
                          wspace=0.32, hspace=0.56,
                          width_ratios=[1.1, 0.9])

    ax_a1 = fig.add_subplot(gs[0, 0])  # d_cond sweep — AUROC
    ax_a2 = fig.add_subplot(gs[1, 0])  # d_cond sweep — Loss
    ax_b  = fig.add_subplot(gs[0, 1])  # Intervention sensitivity
    ax_c  = fig.add_subplot(gs[1, 1])  # Coefficient fidelity

    draw_sweep_auroc(ax_a1)
    draw_sweep_loss(ax_a2)
    draw_intervention(ax_b)
    draw_coefficient_fidelity(ax_c)

    add_panel_label(ax_a1, 'a')
    add_panel_label(ax_a2, '')
    add_panel_label(ax_b, 'b')
    add_panel_label(ax_c, 'c')

    # Sub-label for the two sweep panels
    ax_a1.text(0.50, 1.12, r'Auxiliary-channel dimension sweep ($d_{\mathrm{cond}}$)',
               transform=ax_a1.transAxes, fontsize=7.2, fontweight='bold',
               color=DARK, ha='center', va='bottom')

    legend_handles = [
        Patch(facecolor=CONCAT_C, alpha=0.9, label='Concat-JRNGC'),
        Patch(facecolor=ISTF_C, alpha=0.9, label='ISTF-Mamba'),
        Patch(facecolor=BLUE, alpha=0.85, label='Pearson r'),
        Patch(facecolor=GREEN, alpha=0.55, label='Norm shrinkage'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.61, 0.975),
               ncol=4, fontsize=6.5, handlelength=1.0, handleheight=0.8,
               columnspacing=1.1, borderpad=0.2, labelspacing=0.3)

    save_figure(fig, "fig2_diagnostics_v2")
    plt.close(fig)
    print('Done.')


# ═══════════════════════════════════════════════════════════════════════

def draw_sweep_auroc(ax):
    d_cond = [0, 1, 2, 4, 8, 16]
    auroc  = [0.9013, 0.8372, 0.7192, 0.4962, 0.3500, 0.5218]
    x = np.arange(len(d_cond))
    bars = ax.bar(x, auroc, 0.50, color=BLUE, alpha=0.85, edgecolor='white',
                  linewidth=0.5, zorder=3)
    for bar, val in zip(bars, auroc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=5.2, color=BLUE)
    ax.axhline(y=0.5, color=GRAY, linewidth=0.6, linestyle='--', zorder=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in d_cond], fontsize=6.5)
    ax.set_ylabel('AUROC', fontsize=7.5)
    ax.set_ylim(0, 1.06)
    ax.tick_params(axis='x', length=3)


def draw_sweep_loss(ax):
    d_cond = [0, 1, 2, 4, 8, 16]
    loss   = [0.00895, 0.00558, 0.00472, 0.00321, 0.00254, 0.00211]
    x = np.arange(len(d_cond))
    ax.plot(x, loss, 'o-', color=RED, linewidth=1.5, markersize=5,
            markerfacecolor='white', markeredgewidth=1.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in d_cond], fontsize=6.5)
    ax.set_xlabel(r'$d_{\mathrm{cond}}$', fontsize=7.5)
    ax.set_ylabel('Train loss', fontsize=7.5)
    ax.tick_params(axis='x', length=3)


def draw_intervention(ax):
    conditions = ['Mask x', 'Shuffle x']
    concat_delta = [1.683741, 3.035019]
    istf_delta   = [29.153231, 2.879076]
    x = np.arange(len(conditions))
    width = 0.30
    ax.bar(x - width/2, concat_delta, width, color=CONCAT_C, alpha=0.9,
           edgecolor='white', linewidth=0.5, label='Concat-JRNGC', zorder=3)
    ax.bar(x + width/2, istf_delta, width, color=ISTF_C, alpha=0.9,
           edgecolor='white', linewidth=0.5, label='ISTF-Mamba', zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=7)
    ax.set_ylabel(r'$\Delta$ pred loss (log)', fontsize=7.5)
    ax.set_yscale('log')
    ax.set_ylim(0.9, 42)
    ax.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30]))
    ax.yaxis.set_major_formatter(FixedFormatter(['1', '3', '10', '30']))
    ax.yaxis.set_minor_locator(FixedLocator([2, 5, 20, 40]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='x', length=3)
    ax.tick_params(axis='y', which='minor', length=2, color=GRAY)
    for xpos, val in zip(x - width/2, concat_delta):
        ax.text(xpos, val * 1.16, f'{val:.2f}', ha='center', va='bottom',
                fontsize=5.8, color=DARK, zorder=5)
    for xpos, val in zip(x + width/2, istf_delta):
        ax.text(xpos, val * 1.12, f'{val:.2f}', ha='center', va='bottom',
                fontsize=5.8, color=DARK, zorder=5)
    ax.set_title('Intervention sensitivity', fontsize=8, fontweight='bold',
                 color=DARK, loc='center', pad=4)


def draw_coefficient_fidelity(ax):
    methods = ['Baseline\nJRNGC', 'Concat\nJRNGC', 'ISTF-\nMamba']
    pearson_r = [0.9544, 0.2719, 0.9682]
    shrinkage  = [1.0867, 0.6289, 1.0433]
    x = np.arange(len(methods))
    width = 0.30

    ax.bar(x - width/2, pearson_r, width, color=BLUE, alpha=0.85,
           edgecolor='white', linewidth=0.5, zorder=3)
    ax.bar(x + width/2, shrinkage, width, color=GREEN, alpha=0.55,
           edgecolor='white', linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=6.5)
    ax.tick_params(axis='x', length=3)
    ax.set_ylabel("Fidelity score", fontsize=7.5)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color=GRAY, linewidth=0.6, linestyle='--', zorder=1, alpha=0.7)
    ax.set_title('Coefficient fidelity', fontsize=8, fontweight='bold', color=DARK,
                 loc='center', pad=4)


if __name__ == '__main__':
    main()
