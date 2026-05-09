"""Generate paper figures for Mamba-Enhanced JRNGC (ICLR 2027 target)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})
OUT = "figures/"
import os; os.makedirs(OUT, exist_ok=True)

# ====== Figure 1: DREAM3 d=50 SHD reduction (HERO) ======
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# SHD bar plot
methods = ['JRNGC\n(Baseline)', '+Mamba\nFilter']
shd_vals = [260.7, 100.7]
colors = ['#4472C4', '#ED7D31']
bars = ax1.bar(methods, shd_vals, color=colors, width=0.5, edgecolor='white', linewidth=0.8)
ax1.bar_label(bars, fmt='%.1f', fontsize=13, fontweight='bold')
ax1.set_ylabel('SHD')
ax1.set_title('DREAM3 d=50: SHD Reduced 61%')
# Add reduction arrow
ax1.annotate('-61.4%', xy=(0.5, 140), fontsize=14, fontweight='bold', color='#C00000',
             ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE0E0', alpha=0.8))

# 3-subject breakdown
subjects = ['S0', 'S1', 'S2']
bl_shd = [234, 296, 252]
mb_shd = [73, 109, 120]
x = np.arange(len(subjects))
w = 0.35
ax2.bar(x - w/2, bl_shd, w, label='JRNGC Baseline', color='#4472C4', edgecolor='white')
ax2.bar(x + w/2, mb_shd, w, label='+Mamba Filter', color='#ED7D31', edgecolor='white')
ax2.set_xticks(x)
ax2.set_xticklabels(subjects)
ax2.set_ylabel('SHD')
ax2.set_title('Per-Subject Breakdown')
ax2.legend(frameon=False)

fig.suptitle('Figure 1: DREAM3 d=50 — Core Evidence for Mamba Filter', fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(OUT + 'fig1_dream3_shd.pdf')
fig.savefig(OUT + 'fig1_dream3_shd.png')
plt.close()
print("  Fig 1: DREAM3 SHD saved")

# ====== Figure 2: Cross-Dataset AUROC Comparison ======
fig, ax = plt.subplots(figsize=(12, 5))

datasets = ['NSVAR\nd=10', 'Lorenz-96\nF=40', 'VAR\nd=50 stat', 'NSVAR\nd=50\nPlan A',
            'fMRI\nd=15', 'DREAM3\nd=10', 'DREAM3\nd=50', 'DREAM3\nd=100',
            'CT\ntraffic', 'CT\nmedical', 'CT\npm25']
bl_auroc = [0.9296, 0.9350, 0.7145, 0.6497, 0.5255, 0.5113, 0.4956, 0.5305,
            0.4084, 0.4766, 0.4288]
mb_auroc = [0.9457, 0.9374, 0.6963, 0.6358, 0.4439, 0.5442, 0.5273, 0.5233,
            0.3889, 0.5596, 0.4668]
tcn_auroc = [0.9465, None, 0.7064, None, None, None, 0.4771, None, None, None, None]
pcmci_auroc = [0.578, 0.689, 0.514, 0.512, 0.694, 0.557, 0.528, 0.525, 0.494, 0.481, 0.501]

x = np.arange(len(datasets))
w = 0.2

ax.bar(x - 1.5*w, bl_auroc, w, label='JRNGC Baseline', color='#4472C4', edgecolor='white', linewidth=0.5)
ax.bar(x - 0.5*w, mb_auroc, w, label='+Mamba Filter (ours)', color='#ED7D31', edgecolor='white', linewidth=0.5)

# TCN only where available
tcn_x, tcn_y = [], []
for i, v in enumerate(tcn_auroc):
    if v is not None:
        tcn_x.append(i + 0.5*w)
        tcn_y.append(v)
ax.bar(tcn_x, tcn_y, w, label='+TCN Filter (ablation)', color='#A5A5A5', edgecolor='white', linewidth=0.5)

# PCMCI+
ax.bar(x + 1.5*w, pcmci_auroc, w, label='PCMCI+ (statistical)', color='#70AD47', edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=9)
ax.set_ylabel('AUROC')
ax.set_title('Cross-Dataset AUROC Comparison', fontweight='bold')
ax.legend(frameon=False, ncol=2)
ax.set_ylim(0, 1.05)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
fig.savefig(OUT + 'fig2_auroc_comparison.pdf')
fig.savefig(OUT + 'fig2_auroc_comparison.png')
plt.close()
print("  Fig 2: AUROC comparison saved")

# ====== Figure 3: Variance Stability ======
fig, ax = plt.subplots(figsize=(6, 4.5))

datasets_var = ['NSVAR\nd=10', 'DREAM3\nd=10', 'DREAM3\nd=50']
bl_std = [0.0236, 0.0466, 0.0319]
mb_std = [0.0280, 0.0079, 0.0269]

x = np.arange(len(datasets_var))
w = 0.35
ax.bar(x - w/2, bl_std, w, label='JRNGC Baseline', color='#4472C4', edgecolor='white')
ax.bar(x + w/2, mb_std, w, label='+Mamba Filter', color='#ED7D31', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(datasets_var)
ax.set_ylabel('AUROC Standard Deviation')
ax.set_title('Cross-Seed Variance Stability', fontweight='bold')
ax.legend(frameon=False)

# Highlight DREAM3 d=10 improvement
ax.annotate('-83%', xy=(1, 0.025), fontsize=12, fontweight='bold', color='#C00000',
            ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='#D5F5D5', alpha=0.8))

plt.tight_layout()
fig.savefig(OUT + 'fig3_variance.pdf')
fig.savefig(OUT + 'fig3_variance.png')
plt.close()
print("  Fig 3: Variance stability saved")

# ====== Figure 4: TCN Ablation ======
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

abl_configs = [
    ("NSVAR d=10", [0.9296, 0.9457, 0.9465], [0.0236, 0.0280, 0.0268]),
    ("DREAM3 d=50", [0.4165, 0.4475, 0.4771], [0.0467, 0.0352, 0.0374]),
    ("VAR d=50 stat", [0.7145, 0.6963, 0.7064], [0.0113, 0.0354, 0.0215]),
]
abl_methods = ['Baseline', '+Mamba', '+TCN']
abl_colors = ['#4472C4', '#ED7D31', '#A5A5A5']

for ax_i, (name, vals, errs) in enumerate(abl_configs):
    ax = axes[ax_i]
    bars = ax.bar(abl_methods, vals, color=abl_colors, width=0.5, edgecolor='white')
    # Add value labels
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title(name, fontweight='bold')
    if ax_i == 0:
        ax.set_ylabel('AUROC')

fig.suptitle('Figure 4: TCN Filter Ablation — Architecture vs. Mechanism', fontweight='bold')
plt.tight_layout()
fig.savefig(OUT + 'fig4_tcn_ablation.pdf')
fig.savefig(OUT + 'fig4_tcn_ablation.png')
plt.close()
print("  Fig 4: TCN ablation saved")

# ====== Figure 5: PCMCI+ vs Ours (gap plot) ======
fig, ax = plt.subplots(figsize=(8, 5))

ds_labels = ['NSVAR', 'Lorenz', 'VAR50', 'NSVAR\nPlanA', 'fMRI',
             'DREAM3\n10', 'DREAM3\n50', 'DREAM3\n100', 'CT\ntraffic', 'CT\nmedical', 'CT\npm25']
gaps = [mb - pc for mb, pc in zip(mb_auroc, pcmci_auroc)]
gap_colors = ['#2E7D32' if g > 0 else '#C62828' for g in gaps]
bars = ax.bar(range(len(ds_labels)), gaps, color=gap_colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(range(len(ds_labels)))
ax.set_xticklabels(ds_labels, fontsize=8, rotation=45, ha='right')
ax.set_ylabel('Δ AUROC (Mamba - PCMCI+)')
ax.set_title('Mamba Filter vs PCMCI+: Performance Gap', fontweight='bold')

# Add value labels
for bar, g in zip(bars, gaps):
    y_pos = bar.get_height() + (0.02 if g > 0 else -0.04)
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{g:+.3f}',
            ha='center', fontsize=8, fontweight='bold',
            color='#2E7D32' if g > 0 else '#C62828')

plt.tight_layout()
fig.savefig(OUT + 'fig5_pcmci_gap.pdf')
fig.savefig(OUT + 'fig5_pcmci_gap.png')
plt.close()
print("  Fig 5: PCMCI+ gap saved")

print(f"\nAll figures saved to {OUT}")

# ====== Figure 6: AUPRC Comparison (supplementary) ======
fig, ax = plt.subplots(figsize=(12, 5))

# AUPRC data (from enriched metrics where available, else from original)
bl_auprc = [0.2014, None, 0.1113, 0.2186, None, None, None, None, None, None, None]
mb_auprc = [0.2298, None, 0.0982, 0.1853, None, None, None, None, None, None, None]
tcn_auprc = [None, None, None, None, None, None, None, None, None, None, None]

# Only plot where we have data
has_auprc = [i for i, v in enumerate(bl_auprc) if v is not None]
ds_labels_auprc = [datasets[i] for i in has_auprc]

x_ap = np.arange(len(has_auprc))
w_ap = 0.35
ax.bar(x_ap - w_ap/2, [bl_auprc[i] for i in has_auprc], w_ap,
       label='JRNGC Baseline', color='#4472C4', edgecolor='white')
ax.bar(x_ap + w_ap/2, [mb_auprc[i] for i in has_auprc], w_ap,
       label='+Mamba Filter', color='#ED7D31', edgecolor='white')
ax.set_xticks(x_ap)
ax.set_xticklabels(ds_labels_auprc)
ax.set_ylabel('AUPRC')
ax.set_title('AUPRC Comparison (Available Datasets)', fontweight='bold')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
fig.savefig(OUT + 'fig6_auprc_comparison.pdf')
fig.savefig(OUT + 'fig6_auprc_comparison.png')
plt.close()
print("  Fig 6: AUPRC comparison saved")

# ====== Figure 7: nSHD Comparison ======
fig, ax = plt.subplots(figsize=(10, 4.5))

nshd_configs = [
    ("NSVAR\nd=10", 0.155, 0.127),
    ("VAR\nd=50 stat", 1.116, 1.116),
    ("NSVAR\nd=50\nPlanA", 1.759, 1.755),
    ("DREAM3\nd=50", 2.060, 0.787),
]
nshd_labels = [c[0] for c in nshd_configs]
bl_nshd = [c[1] for c in nshd_configs]
mb_nshd = [c[2] for c in nshd_configs]

x_ns = np.arange(len(nshd_configs))
w_ns = 0.35
bars1 = ax.bar(x_ns - w_ns/2, bl_nshd, w_ns, label='JRNGC Baseline', color='#4472C4', edgecolor='white')
bars2 = ax.bar(x_ns + w_ns/2, mb_nshd, w_ns, label='+Mamba Filter', color='#ED7D31', edgecolor='white')
ax.set_xticks(x_ns)
ax.set_xticklabels(nshd_labels)
ax.set_ylabel('nSHD (SHD / |E_true|)')
ax.set_title('Normalized SHD Comparison', fontweight='bold')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.2)

# Highlight DREAM3 improvement
ax.annotate('-61.8%', xy=(3, 0.9), fontsize=12, fontweight='bold', color='#C00000',
            ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='#D5F5D5', alpha=0.8))

plt.tight_layout()
fig.savefig(OUT + 'fig7_nshd_comparison.pdf')
fig.savefig(OUT + 'fig7_nshd_comparison.png')
plt.close()
print("  Fig 7: nSHD comparison saved")

# ====== Figure 8: Expanded AUROC with 95% CI error bars ======
fig, ax = plt.subplots(figsize=(12, 5))

ds_ci = ['NSVAR\nd=10', 'VAR\nd=50 stat', 'NSVAR\nd=50\nPlanA']
bl_mean_ci = [0.9296, 0.7145, 0.6497]
bl_ci_half = [0.0327, 0.0344, 0.0215]  # half-width of 95% CI
mb_mean_ci = [0.9457, 0.6963, 0.6358]
mb_ci_half = [0.0388, 0.1076, 0.0643]

x_ci = np.arange(len(ds_ci))
w_ci = 0.35
ax.bar(x_ci - w_ci/2, bl_mean_ci, w_ci, label='JRNGC Baseline', color='#4472C4', edgecolor='white')
ax.bar(x_ci + w_ci/2, mb_mean_ci, w_ci, label='+Mamba Filter', color='#ED7D31', edgecolor='white')
ax.errorbar(x_ci - w_ci/2, bl_mean_ci, yerr=bl_ci_half, fmt='none',
            ecolor='black', capsize=4, linewidth=0.8)
ax.errorbar(x_ci + w_ci/2, mb_mean_ci, yerr=mb_ci_half, fmt='none',
            ecolor='black', capsize=4, linewidth=0.8)
ax.set_xticks(x_ci)
ax.set_xticklabels(ds_ci)
ax.set_ylabel('AUROC')
ax.set_title('AUROC with 95% Confidence Intervals', fontweight='bold')
ax.legend(frameon=False)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
fig.savefig(OUT + 'fig8_auroc_ci.pdf')
fig.savefig(OUT + 'fig8_auroc_ci.png')
plt.close()
print("  Fig 8: AUROC with CI saved")

print(f"\nAll {8} figures saved to {OUT}")
