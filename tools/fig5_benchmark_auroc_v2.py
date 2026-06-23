"""
Figure 5 — Inferential Benchmark AUROC (KBS figure* format).
v4: figure* layout, publication-friendly x labels, no in-plot significance key.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 8
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['legend.frameon'] = False
plt.rcParams['xtick.labelsize'] = 7.5
plt.rcParams['ytick.labelsize'] = 7.5
plt.rcParams['axes.labelsize'] = 8.5

BASELINE_C  = '#7884B4'
MAMBA_C     = '#F0C0CC'
PCMCI_C     = '#A0A0A0'
DARK        = '#272727'
GRAY        = '#767676'
SIG_STAR    = '#2E9E44'
NS_COLOR    = '#767676'

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "paper-data", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(fig, name):
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 600)]:
        p = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        kw = {} if dpi is None else {"dpi": dpi}
        fig.savefig(p, bbox_inches="tight", **kw)
    print(f"  {name}.{{svg,pdf,png}}")


def main():
    fig, ax = plt.subplots(figsize=(7.0, 3.0), facecolor='white')  # figure* width

    datasets = ['CT-medical', 'Lorenz-F40', 'VAR (d=50)', 'NSVAR (d=10)']
    baseline_means = [0.458, 0.938, 0.715, 0.930]
    baseline_stds  = [0.024, 0.012, 0.034, 0.024]
    mamba_means    = [0.500, 0.939, 0.678, 0.946]
    mamba_stds     = [0.022, 0.014, 0.034, 0.028]
    pcmci_vals     = [0.481, 0.689, 0.514, 0.505]

    # Significance categories
    sig_labels  = ['*', 'n.s.', 'n.s.', 'deg.']
    sig_colors  = [SIG_STAR, NS_COLOR, NS_COLOR, NS_COLOR]

    x = np.arange(len(datasets))
    width = 0.22
    n = len(datasets)

    # Bars — wider spacing for figure*
    ax.bar(x - width, baseline_means, width, color=BASELINE_C, alpha=0.9,
           edgecolor='white', linewidth=0.4, label='JRNGC', zorder=3)
    ax.bar(x, mamba_means, width, color=MAMBA_C, alpha=0.9,
           edgecolor='white', linewidth=0.4, label='ISTF-Mamba', zorder=3)
    ax.bar(x + width, pcmci_vals, width, color=PCMCI_C, alpha=0.8,
           edgecolor='white', linewidth=0.4, label='PCMCI+', zorder=3)

    # Error bars (baseline and mamba only)
    for i in range(n):
        ax.errorbar(i - width, baseline_means[i], yerr=baseline_stds[i],
                    fmt='none', ecolor=DARK, capsize=3, linewidth=0.7, zorder=4)
        ax.errorbar(i, mamba_means[i], yerr=mamba_stds[i],
                    fmt='none', ecolor=DARK, capsize=3, linewidth=0.7, zorder=4)

    # Significance annotations above bars
    max_vals = [max(baseline_means[i] + baseline_stds[i],
                    mamba_means[i] + mamba_stds[i]) for i in range(n)]
    for i in range(n):
        y_pos = max_vals[i] + 0.035
        ax.text(i, y_pos, sig_labels[i], ha='center', va='bottom',
                fontsize=7.5, fontweight='bold', color=sig_colors[i], zorder=5)

    # Chance line
    ax.axhline(y=0.5, color=GRAY, linewidth=0.6, linestyle='--', zorder=1, alpha=0.7)
    ax.text(-0.45, 0.505, 'chance 0.5', fontsize=6, color=GRAY, va='bottom')

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=7.5)
    ax.set_ylabel('AUROC', fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis='x', length=3)

    # Legend — horizontal above plot area, no overlap
    ax.legend(fontsize=7.5, loc='lower left', ncol=3,
              handlelength=1.0, handleheight=0.8, borderpad=0.3, labelspacing=0.4,
              bbox_to_anchor=(0.0, 1.02))

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.86)

    save_figure(fig, "fig5_benchmark_auroc_v2")
    plt.close(fig)
    print('Done.')


if __name__ == '__main__':
    main()
