#!/usr/bin/env python
r"""
Generate Figure 2: Mechanistic diagnostics of auxiliary-channel shortcut learning.
Three panels: (a) d_cond sweep, (b) intervention sensitivity, (c) Jacobian norm ratio.
Source data: real GPU experiments (2026-05-13, frozen v2).
Output: PDF + SVG vector figures.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

# ── paths ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "diagnostic_results")
FACTORIAL_DIR = os.path.join(ROOT, "paper-data", "factorial")
OUT_DIR = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "text.usetex": False,
    "mathtext.fontset": "stix",
})

# ── colour palette ─────────────────────────────────────────────────────────
C_BLUE   = "#2166ac"
C_ORANGE = "#d6604d"
C_GREEN  = "#4daf4a"
C_GREY   = "#777777"
C_RED    = "#b2182b"
C_TEAL   = "#008080"

# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════

# Panel (a): d_cond sweep
with open(os.path.join(DATA_DIR, "exp2_dcond_sweep.json")) as f:
    dcond_data = json.load(f)
keys_ordered = [
    "Baseline (d_cond=0)", "Concat d_cond=1", "Concat d_cond=2",
    "Concat d_cond=4", "Concat d_cond=8", "Concat d_cond=16",
]
d_conds = np.array([0, 1, 2, 4, 8, 16])
aurocs = np.array([dcond_data[k]["auroc"] for k in keys_ordered])
losses = np.array([dcond_data[k]["train_loss"] for k in keys_ordered])

# Panel (b): intervention sensitivity
with open(os.path.join(DATA_DIR, "mask_shuffle_results.json")) as f:
    mask_data = json.load(f)

# Panel (c): Jacobian norm ratio
with open(os.path.join(FACTORIAL_DIR, "diagnostics_nslinear_5seed_D2.json")) as f:
    jac_data = json.load(f)
jac_ratios = [jac_data["per_seed"][f"seed_{i}"]["jacobian_ratio"] for i in range(5)]
jac_mean = jac_data["summary"]["jacobian_ratio_mean"]

# ══════════════════════════════════════════════════════════════════════════════
# Build figure
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(7.2, 2.75))

# ── Panel (a): d_cond sweep (dual y-axis) ──────────────────────────────────
ax1 = fig.add_axes([0.050, 0.18, 0.290, 0.76])
ax2 = ax1.twinx()

line1, = ax1.plot(d_conds, aurocs, "o-", color=C_BLUE, ms=5.5, lw=1.5,
                  markerfacecolor="white", markeredgewidth=1.2)
line2, = ax2.plot(d_conds, losses, "s-", color=C_ORANGE, ms=5.5, lw=1.5,
                  markerfacecolor="white", markeredgewidth=1.2)

# Baseline reference line
ax1.axhline(y=0.90, color=C_GREY, ls="--", lw=0.7, alpha=0.6)
ax1.text(14.5, 0.903, "baseline\nAUROC", color=C_GREY, fontsize=6.5, ha="left", va="bottom")

ax1.set_xlabel(r"$d_{\mathrm{cond}}$")
ax1.set_ylabel("AUROC", color=C_BLUE)
ax2.set_ylabel("Training loss", color=C_ORANGE)
ax1.set_xticks(d_conds)
ax1.set_xlim(-1, 18)
ax1.set_ylim(0.20, 1.05)
ax2.set_ylim(0.001, 0.011)
ax1.tick_params(axis="y", colors=C_BLUE)
ax2.tick_params(axis="y", colors=C_ORANGE)
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.ticklabel_format(axis="y", style="plain")

# Decoupling annotation
ax1.annotate("", xy=(9.5, 0.35), xytext=(9.5, 0.88),
             arrowprops=dict(arrowstyle="<->", color=C_GREY, lw=0.7))
ax1.text(11.8, 0.55, r"loss--causal$\;$decoupling",
         color=C_GREY, fontsize=6, ha="left", va="center")

leg1 = ax1.legend([line1, line2], ["AUROC", "Training loss"],
                  loc="lower left", frameon=True, fancybox=False,
                  edgecolor="0.7", fontsize=7)
leg1.get_frame().set_linewidth(0.5)

ax1.text(-0.28, 1.07, "(a)", transform=ax1.transAxes, fontsize=10, fontweight="bold")

# ── Panel (b): Intervention sensitivity (broken axis) ──────────────────────
# Use full y-range; tall ISTF mask bar (30.17) visually dominates
# and the dramatic ratio is the key finding.
ax3 = fig.add_axes([0.390, 0.18, 0.250, 0.76])

interventions = [r"$\mathbf{x}$-masking", r"$\mathbf{x}$-shuffling"]
concat_vals = [mask_data["concat_mask_delta"], mask_data["concat_shuffle_delta"]]
istf_vals   = [mask_data["istf_mask_delta"],   mask_data["istf_shuffle_delta"]]

x = np.arange(len(interventions))
width = 0.30

bars1 = ax3.bar(x - width/2, concat_vals, width, color=C_ORANGE, edgecolor="white",
                lw=0.3, label="Concat JRNGC")
bars2 = ax3.bar(x + width/2, istf_vals,   width, color=C_BLUE,   edgecolor="white",
                lw=0.3, label="ISTF-Mamba")

# Value labels atop each bar
for bar, val in zip(bars1, concat_vals):
    va = "bottom" if val < 2 else "top"
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f"{val:.2f}", ha="center", va=va, fontsize=7, color=C_ORANGE)
for bar, val in zip(bars2, istf_vals):
    va = "bottom" if val < 2 else "top"
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f"{val:.2f}", ha="center", va=va, fontsize=7, color=C_BLUE)

ax3.set_xticks(x)
ax3.set_xticklabels(interventions)
ax3.set_ylabel(r"Prediction loss increase $\Delta\mathcal{L}_{\mathrm{pred}}$")
ax3.set_ylim(0, 34)

leg3 = ax3.legend(fontsize=6.5, frameon=True, fancybox=False, edgecolor="0.7",
                  loc="upper left")
leg3.get_frame().set_linewidth(0.5)

# Ratio annotation
ax3.text(0.98, 0.40, r"Concat / ISTF sensitivity: $5.4\%$",
         transform=ax3.transAxes, fontsize=6.5, ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.25", fc="0.97", ec="0.7", lw=0.4))

ax3.text(-0.32, 1.07, "(b)", transform=ax3.transAxes, fontsize=10, fontweight="bold")

# ── Panel (c): Jacobian norm ratio ─────────────────────────────────────────
ax4 = fig.add_axes([0.690, 0.18, 0.280, 0.76])

seeds = ["1", "2", "3", "4", "5"]
bar_colors = [C_ORANGE if r >= 1.0 else C_GREEN for r in jac_ratios]
bars = ax4.bar(seeds, jac_ratios, width=0.55, color=bar_colors, edgecolor="white", lw=0.3)

ax4.axhline(y=1.0, color=C_GREY, ls="--", lw=0.8, alpha=0.7)
ax4.axhline(y=jac_mean, color=C_RED, ls=":", lw=1.1, alpha=0.8)
ax4.text(4.4, jac_mean + 0.004, f"mean = {jac_mean:.3f}", color=C_RED,
         fontsize=6.5, ha="right", va="bottom")

# "baseline = 1.0" label
ax4.text(4.4, 1.004, "baseline = 1.0", color=C_GREY, fontsize=6.5,
         ha="right", va="bottom")

# Value labels atop bars
for bar, val in zip(bars, jac_ratios):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f"{val:.3f}", ha="center", va="bottom", fontsize=7)

ax4.set_xlabel("Paired seed")
ax4.set_ylabel(r"Jacobian norm ratio$\;\;$" + "\n" + "(ISTF-Mamba / Baseline)")
ax4.set_ylim(0.88, 1.15)

# Annotation
ax4.text(0.98, 0.07, r"$4/5$ seeds $>$ 1.0", transform=ax4.transAxes,
         fontsize=6.5, ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.25", fc="0.97", ec="0.7", lw=0.4))

ax4.text(-0.36, 1.07, "(c)", transform=ax4.transAxes, fontsize=10, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
for fmt in ["pdf", "svg", "png"]:
    outpath = os.path.join(OUT_DIR, f"figure2_diagnostics.{fmt}")
    dpi_kw = {"dpi": 300} if fmt == "png" else {}
    fig.savefig(outpath, format=fmt, **dpi_kw)
    print(f"Saved ({fmt.upper()}): {outpath}")

plt.close(fig)
print("Done. Figure 2 generated successfully.")
