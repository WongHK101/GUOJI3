"""Generate main paper figures for ISTF-Mamba (TNNLS).

Figure 1: ISTF architecture / shortcut repair overview (3 panels)
Figure 2: Shortcut diagnostic figure (d_cond sweep + mask/shuffle + Jacobian ratio)

These are CONCEPTUAL SKETCHES to lock in the paper narrative before writing Section 1.
Final versions will use GPU-computed data where needed.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np
import json
import os

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_PROJ_ROOT, "figures") + os.sep
os.makedirs(OUT, exist_ok=True)

# ─── Color palette ───
C_BASELINE = '#4472C4'     # blue
C_MAMBA = '#ED7D31'        # orange
C_TCN = '#70AD47'          # green
C_PCMCI = '#A5A5A5'        # grey
C_SHORTCUT = '#C00000'     # red (for problem)
C_REPAIR = '#2E7D32'        # green (for fix)
C_INPUT = '#5B9BD5'         # light blue
C_AUX = '#FF6B6B'           # coral
C_GRAD = '#FFD700'          # gold

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: ISTF Architecture / Shortcut Repair Overview (conceptual diagram)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_architecture_figure():
    fig = plt.figure(figsize=(18, 6.5))

    # ── Panel A: JRNGC + Auxiliary-Channel Shortcut ──
    ax_a = fig.add_subplot(1, 3, 1)
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis('off')
    ax_a.set_title('(a) JRNGC with Auxiliary Channel:\nShortcut Learning',
                   fontweight='bold', fontsize=11, color=C_SHORTCUT, pad=10)

    # Input box
    rect_x = FancyBboxPatch((0.5, 6.5), 2.0, 1.2, boxstyle="round,pad=0.1",
                             facecolor=C_INPUT, edgecolor='#2F5496', linewidth=1.5, alpha=0.8)
    ax_a.add_patch(rect_x)
    ax_a.text(1.5, 7.1, r'$\mathbf{x}_t$', ha='center', va='center', fontsize=12, fontweight='bold')
    ax_a.text(1.5, 6.75, '(original input)', ha='center', va='center', fontsize=8, color='#333')

    # Auxiliary box
    rect_c = FancyBboxPatch((4.5, 8.0), 2.0, 1.2, boxstyle="round,pad=0.1",
                             facecolor=C_AUX, edgecolor='#CC4444', linewidth=1.5, alpha=0.7)
    ax_a.add_patch(rect_c)
    ax_a.text(5.5, 8.6, r'$\mathbf{c}_t$', ha='center', va='center', fontsize=12, fontweight='bold')
    ax_a.text(5.5, 8.25, '(auxiliary channel)', ha='center', va='center', fontsize=8, color='#333')

    # Concat/FiLM block
    rect_concat = FancyBboxPatch((3.8, 4.8), 4.0, 1.0, boxstyle="round,pad=0.1",
                                  facecolor='#FFE0E0', edgecolor=C_SHORTCUT, linewidth=1.5, alpha=0.9)
    ax_a.add_patch(rect_concat)
    ax_a.text(5.8, 5.3, 'Concat / FiLM', ha='center', va='center', fontsize=10, fontweight='bold', color=C_SHORTCUT)
    ax_a.text(5.8, 5.0, '(unpenalized side channel)', ha='center', va='center', fontsize=7.5, color='#666')

    # Prediction MLP
    rect_mlp = FancyBboxPatch((3.8, 2.5), 4.0, 1.2, boxstyle="round,pad=0.1",
                               facecolor='#D6E4F0', edgecolor='#2F5496', linewidth=1.5)
    ax_a.add_patch(rect_mlp)
    ax_a.text(5.8, 3.3, 'JRNGC Prediction MLP', ha='center', va='center', fontsize=10, fontweight='bold')
    ax_a.text(5.8, 2.9, 'Jacobian penalty on x-path only', ha='center', va='center', fontsize=8, color=C_SHORTCUT)

    # Loss output
    rect_loss = FancyBboxPatch((4.5, 0.8), 2.5, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#FFCCCC', edgecolor=C_SHORTCUT, linewidth=1.5, alpha=0.8)
    ax_a.add_patch(rect_loss)
    ax_a.text(5.75, 1.2, 'Low Loss', ha='center', va='center', fontsize=10, fontweight='bold', color=C_SHORTCUT)
    ax_a.text(5.75, 0.95, 'Invalid Causal Jacobian', ha='center', va='center', fontsize=7.5, color=C_SHORTCUT)

    # Arrows Panel A
    ax_a.annotate('', xy=(3.8, 5.3), xytext=(2.5, 6.8),
                  arrowprops=dict(arrowstyle='->', color='#2F5496', lw=1.5))
    ax_a.annotate('', xy=(3.8, 5.3), xytext=(5.5, 8.0),
                  arrowprops=dict(arrowstyle='->', color=C_SHORTCUT, lw=1.5, linestyle='dashed'))
    ax_a.annotate('', xy=(5.8, 3.7), xytext=(5.8, 4.8),
                  arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    ax_a.annotate('', xy=(5.75, 1.6), xytext=(5.8, 2.5),
                  arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

    # Annotations
    ax_a.text(2.3, 5.8, 'gradient\nblocked by\nJacobian penalty', ha='center', fontsize=7, color='#666',
              bbox=dict(boxstyle='round', facecolor='#FFFACD', alpha=0.7))
    ax_a.text(7.3, 6.5, 'gradient\nunblocked,\nlow-loss path', ha='center', fontsize=7, color=C_SHORTCUT,
              bbox=dict(boxstyle='round', facecolor='#FFE0E0', alpha=0.7))

    # Red X mark on Jacobian
    ax_a.text(5.8, 2.7, 'X', fontsize=16, color=C_SHORTCUT, ha='center', va='center', fontweight='bold',
              bbox=dict(boxstyle='circle,pad=0.1', facecolor='white', edgecolor=C_SHORTCUT, lw=1.5, alpha=0.9))

    # ── Panel B: ISTF Repair ──
    ax_b = fig.add_subplot(1, 3, 2)
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.axis('off')
    ax_b.set_title('(b) Input-Space Temporal Filtering (ISTF):\nStructural Repair',
                   fontweight='bold', fontsize=11, color=C_REPAIR, pad=10)

    # Input
    rect_x2 = FancyBboxPatch((0.5, 5.5), 2.0, 1.2, boxstyle="round,pad=0.1",
                              facecolor=C_INPUT, edgecolor='#2F5496', linewidth=1.5, alpha=0.8)
    ax_b.add_patch(rect_x2)
    ax_b.text(1.5, 6.1, r'$\mathbf{x}_t$', ha='center', va='center', fontsize=12, fontweight='bold')

    # ISTF Filter block
    rect_filter = FancyBboxPatch((3.5, 5.0), 3.5, 2.2, boxstyle="round,pad=0.1",
                                  facecolor='#D4EDDA', edgecolor=C_REPAIR, linewidth=2.0)
    ax_b.add_patch(rect_filter)
    ax_b.text(5.25, 6.7, 'ISTF Filter', ha='center', va='center', fontsize=11, fontweight='bold', color=C_REPAIR)
    ax_b.text(5.25, 6.25, '(Mamba SSM / TCN)', ha='center', va='center', fontsize=9, color='#333')
    ax_b.text(5.25, 5.85, 'Input-space confined', ha='center', va='center', fontsize=8, color=C_REPAIR)
    ax_b.text(5.25, 5.45, r'$\mathbf{x}_t \rightarrow \mathbf{x}_t^{\prime}$  same dim $d$',
             ha='center', va='center', fontsize=7.5, color='#555')

    # Filtered output
    rect_xp = FancyBboxPatch((3.8, 2.2), 3.0, 1.0, boxstyle="round,pad=0.1",
                              facecolor='#BBDEFB', edgecolor='#2F5496', linewidth=1.5, alpha=0.8)
    ax_b.add_patch(rect_xp)
    ax_b.text(5.3, 2.7, r"$\mathbf{x}_t^{\prime}$", ha='center', va='center', fontsize=11, fontweight='bold')

    # Prediction MLP
    rect_mlp2 = FancyBboxPatch((3.8, 0.5), 3.0, 1.0, boxstyle="round,pad=0.1",
                                facecolor='#D6E4F0', edgecolor='#2F5496', linewidth=1.5)
    ax_b.add_patch(rect_mlp2)
    ax_b.text(5.3, 1.0, 'JRNGC MLP', ha='center', va='center', fontsize=10, fontweight='bold')

    # GC output
    ax_b.text(5.3, 0.15, 'Valid Causal Jacobian → GC Graph', ha='center', va='center',
              fontsize=8, fontweight='bold', color=C_REPAIR)

    # Arrows Panel B
    ax_b.annotate('', xy=(3.5, 6.1), xytext=(2.5, 6.1),
                  arrowprops=dict(arrowstyle='->', color='#2F5496', lw=1.5))
    ax_b.annotate('', xy=(5.3, 3.2), xytext=(5.3, 5.0),
                  arrowprops=dict(arrowstyle='->', color=C_REPAIR, lw=2.0))
    ax_b.annotate('', xy=(5.3, 1.5), xytext=(5.3, 2.2),
                  arrowprops=dict(arrowstyle='->', color='#2F5496', lw=1.5))

    # Design principles box
    rect_principles = FancyBboxPatch((0.3, 0.8), 2.8, 2.5, boxstyle="round,pad=0.1",
                                      facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=1.0, alpha=0.6)
    ax_b.add_patch(rect_principles)
    ax_b.text(1.7, 3.0, 'Design Principles:', fontsize=7.5, fontweight='bold', ha='center')
    ax_b.text(1.7, 2.6, '1. Input-space\n    confinement', fontsize=7, ha='center')
    ax_b.text(1.7, 2.1, '2. Near-identity\n    initialization', fontsize=7, ha='center')
    ax_b.text(1.7, 1.6, '3. Orthogonality\n    regularization', fontsize=7, ha='center')

    # Checkmark
    ax_b.text(7.2, 5.8, 'OK', fontsize=14, color=C_REPAIR, ha='center', va='center', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#D4EDDA', edgecolor=C_REPAIR, lw=1.0, alpha=0.8))
    ax_b.text(8.0, 5.8, 'No side\nchannel', fontsize=7, color=C_REPAIR, ha='center')

    # ── Panel C: Instantiations & Operating Regime ──
    ax_c = fig.add_subplot(1, 3, 3)
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)
    ax_c.axis('off')
    ax_c.set_title('(c) Filter Instantiations &\nOperating Boundary',
                   fontweight='bold', fontsize=11, pad=10)

    # Mamba block
    rect_mamba = FancyBboxPatch((0.5, 7.0), 3.8, 2.5, boxstyle="round,pad=0.1",
                                 facecolor='#FFF3E0', edgecolor=C_MAMBA, linewidth=1.5)
    ax_c.add_patch(rect_mamba)
    ax_c.text(2.4, 9.0, 'ISTF-Mamba', fontsize=10, fontweight='bold', color=C_MAMBA, ha='center')
    ax_c.text(2.4, 8.5, 'Selective SSM', fontsize=8, ha='center')
    ax_c.text(2.4, 8.1, '• content-dependent gating', fontsize=7, ha='center')
    ax_c.text(2.4, 7.75, '• long-range memory', fontsize=7, ha='center')
    ax_c.text(2.4, 7.4, '• suited for long T/d', fontsize=7, ha='center')

    # TCN block
    rect_tcn = FancyBboxPatch((5.7, 7.0), 3.8, 2.5, boxstyle="round,pad=0.1",
                               facecolor='#E8F5E9', edgecolor=C_TCN, linewidth=1.5)
    ax_c.add_patch(rect_tcn)
    ax_c.text(7.6, 9.0, 'ISTF-TCN', fontsize=10, fontweight='bold', color=C_TCN, ha='center')
    ax_c.text(7.6, 8.5, 'Temporal Convolution', fontsize=8, ha='center')
    ax_c.text(7.6, 8.1, '• fixed receptive field', fontsize=7, ha='center')
    ax_c.text(7.6, 7.75, '• efficient local patterns', fontsize=7, ha='center')
    ax_c.text(7.6, 7.4, '• suited for moderate T/d', fontsize=7, ha='center')

    # Operating regime summary
    rect_regime = FancyBboxPatch((0.5, 1.0), 9.0, 5.2, boxstyle="round,pad=0.1",
                                  facecolor='#F5F5F5', edgecolor='#999', linewidth=1.0)
    ax_c.add_patch(rect_regime)
    ax_c.text(5.0, 5.8, 'Empirical Operating Boundary (frozen v2 data)', fontsize=9, fontweight='bold', ha='center')

    # Regime table
    regimes = [
        ('Non-stationary\nReal-world (CT_medical)', 'Statistically reliable\nmodest improvement', '#D4EDDA'),
        ('Chaotic Nonlinear\n(Lorenz_F40)', 'Near-neutral\n(ceiling effect)', '#FFF9C4'),
        ('Stationary Linear\n(VAR_d50)', 'Negative boundary\n(AUPRC, F1 worse)', '#FFCCCC'),
        ('Non-stationary\nLow-T (NSVAR_d10)', 'Degenerate/Ceiling\n(insufficient pairs)', '#E0E0E0'),
    ]
    for i, (regime, result, color) in enumerate(regimes):
        y = 4.5 - i * 0.95
        rect = FancyBboxPatch((1.0, y - 0.35), 3.5, 0.8, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='#999', linewidth=0.8, alpha=0.7)
        ax_c.add_patch(rect)
        ax_c.text(2.75, y + 0.05, regime, fontsize=7, ha='center', va='center')
        ax_c.text(6.5, y + 0.05, result, fontsize=7.5, ha='center', va='center',
                  fontweight='bold')

    ax_c.text(5.0, 0.25, 'ISTF is a structural safeguard, not a universal booster',
              fontsize=9, fontweight='bold', ha='center', style='italic', color='#555')

    plt.tight_layout(pad=2)
    fig.savefig(OUT + 'fig1_istf_architecture.pdf', facecolor='white')
    fig.savefig(OUT + 'fig1_istf_architecture.png', facecolor='white', dpi=300)
    plt.close()
    print("  Fig 1: ISTF Architecture saved")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Shortcut Diagnostic Figure (conceptual + available data)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_diagnostic_figure():
    fig = plt.figure(figsize=(18, 5.5))

    # ── Panel A: d_cond Sweep (conceptual sketch) ──
    ax_a = fig.add_subplot(1, 3, 1)

    # Conceptual curves showing the shortcut phenomenon
    d_cond_vals = np.array([0, 2, 4, 8, 16, 32, 64])

    # AUROC curve: drops as auxiliary dimension increases
    auroc_concat = np.array([0.82, 0.78, 0.72, 0.64, 0.58, 0.53, 0.51])
    auroc_istf = np.array([0.82, 0.81, 0.80, 0.79, 0.78, 0.76, 0.74])

    # Loss curve
    loss_concat = np.array([0.45, 0.38, 0.32, 0.27, 0.24, 0.22, 0.21])
    loss_istf = np.array([0.45, 0.43, 0.41, 0.40, 0.38, 0.37, 0.36])

    ax_a.plot(d_cond_vals, auroc_concat, 'o-', color=C_SHORTCUT, lw=2, markersize=7, label='Concat: AUROC')
    ax_a.plot(d_cond_vals, auroc_istf, 's--', color=C_MAMBA, lw=2, markersize=7, label='ISTF-Mamba: AUROC')

    ax_a.set_xlabel('Auxiliary Dimension d_cond', fontweight='bold')
    ax_a.set_ylabel('AUROC', fontweight='bold', color='#333')
    ax_a.set_title('(a) d_cond Sweep: Shortcut via\nAuxiliary Channel', fontweight='bold', fontsize=11)
    ax_a.legend(loc='lower left', frameon=False, fontsize=8)
    ax_a.set_ylim(0.4, 0.9)
    ax_a.grid(True, alpha=0.3)
    ax_a.annotate('Shortcut:\nAUROC collapses\nas d_cond grows', xy=(40, 0.57), fontsize=8.5,
                  color=C_SHORTCUT, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='#FFE0E0', alpha=0.8))
    ax_a.annotate('ISTF:\nno side-channel,\nAUROC stable', xy=(35, 0.78), fontsize=8.5,
                  color=C_MAMBA, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8))
    # Watermark
    ax_a.text(32, 0.42, '[DATA PENDING: run test_shortcut_diagnostics.py]',
              fontsize=7, color='#999', style='italic', ha='center')

    # ── Panel B: Mask/Shuffle Dependency (conceptual) ──
    ax_b = fig.add_subplot(1, 3, 2)

    conditions = ['Full\nModel', 'Mask\nx only', 'Mask\nc only', 'Mask\nboth', 'Shuffle\nx']
    bar_colors = [C_BASELINE, C_SHORTCUT, C_AUX, '#999', '#FFD54F']

    # Conceptual values showing c-channel dependency
    concat_vals = [0.75, 0.70, 0.51, 0.50, 0.68]
    istf_vals = [0.76, 0.26, 0.74, 0.25, 0.24]

    x = np.arange(len(conditions))
    w = 0.35
    bars1 = ax_b.bar(x - w/2, concat_vals, w, label='JRNGC + Concat', color='#FFB3B3', edgecolor=C_SHORTCUT, linewidth=1.2)
    bars2 = ax_b.bar(x + w/2, istf_vals, w, label='ISTF-Mamba', color=C_MAMBA, edgecolor='#BF5700', linewidth=1.2, alpha=0.8)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(conditions, fontsize=8)
    ax_b.set_ylabel('AUROC', fontweight='bold')
    ax_b.set_title('(b) Mask/Shuffle Intervention:\nAuxiliary Channel Dominance', fontweight='bold', fontsize=11)
    ax_b.legend(loc='upper right', frameon=False, fontsize=8)
    ax_b.set_ylim(0.1, 0.9)
    ax_b.grid(True, alpha=0.2, axis='y')

    # Annotations
    ax_b.annotate('c-channel\ncarries prediction', xy=(1.5, 0.55), fontsize=7.5,
                  color=C_SHORTCUT, ha='center',
                  bbox=dict(boxstyle='round', facecolor='#FFE0E0', alpha=0.7))
    ax_b.annotate('ISTF: no fallback\n→ Jacobian active', xy=(3.5, 0.30), fontsize=7.5,
                  color=C_MAMBA, ha='center',
                  bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.7))
    ax_b.text(2.0, 0.15, '[DATA PENDING: run test_mask_shuffle.py]',
              fontsize=7, color='#999', style='italic', ha='center')

    # ── Panel C: Jacobian Ratio from Factorial Diagnostics ──
    ax_c = fig.add_subplot(1, 3, 3)

    # Load real factorial diagnostic data
    diag_path = os.path.join(_PROJ_ROOT, 'paper-data', 'factorial', 'diagnostics_nslinear_5seed_D2.json')
    try:
        with open(diag_path, 'r') as f:
            diag_data = json.load(f)
        per_seed = diag_data.get('per_seed', {})
        seeds = sorted(per_seed.keys())
        jac_ratios = [per_seed[s]['jacobian_ratio'] for s in seeds]
        selectivity = [per_seed[s].get('selectivity_index', 0) for s in seeds]

        ax_c.bar(range(len(seeds)), jac_ratios, color=['#FF9800' if r > 1.0 else '#4CAF50' for r in jac_ratios],
                 edgecolor='#333', linewidth=0.8)
        ax_c.axhline(y=1.0, color='#333', linestyle='--', lw=1.5, alpha=0.5)
        ax_c.set_xticks(range(len(seeds)))
        ax_c.set_xticklabels([s.replace('seed_', 'S') for s in seeds])
        ax_c.set_ylabel('Jacobian Norm Ratio (Mamba / Baseline)', fontweight='bold')
        ax_c.set_title('(c) Jacobian Fidelity Preservation\n(NS+Linear, 5 seeds, real v2 data)',
                       fontweight='bold', fontsize=11)
        ax_c.set_ylim(0.9, 1.15)
        ax_c.grid(True, alpha=0.2, axis='y')

        mean_jr = np.mean(jac_ratios)
        ax_c.axhline(y=mean_jr, color=C_MAMBA, linestyle='-', lw=1.0, alpha=0.7)
        ax_c.text(len(seeds) - 0.5, mean_jr + 0.01, f'Mean={mean_jr:.3f}',
                  fontsize=8, color=C_MAMBA, ha='right')
        ax_c.annotate(r'Ratio $\geq$ 1: Jacobian\npreserved or enhanced',
                      xy=(2, 1.07), fontsize=8, color='#FF9800',
                      bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.7))
    except FileNotFoundError:
        ax_c.text(2, 1.0, '[No diagnostic data]', fontsize=12, ha='center', color='#999')
        ax_c.set_title('(c) Jacobian Fidelity Preservation\n(conceptual)', fontweight='bold', fontsize=11)

    plt.tight_layout(pad=2)
    fig.savefig(OUT + 'fig2_shortcut_diagnostics.pdf', facecolor='white')
    fig.savefig(OUT + 'fig2_shortcut_diagnostics.png', facecolor='white', dpi=300)
    plt.close()
    print("  Fig 2: Shortcut Diagnostics saved")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Benchmark AUROC Bar Chart (4 inferential datasets, frozen v2 data)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_benchmark_auroc():
    """Load from migrated_all_v2.json (frozen v2 data)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    results_path = os.path.join(_PROJ_ROOT, 'results', 'raw', 'migrated_all_v2.json')
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 4 inferential datasets
    datasets = ['CT_medical', 'Lorenz_F40', 'VAR_d50', 'NSVAR_d10']
    methods = ['baseline', 'mamba', 'pcmci']
    labels = ['JRNGC\nBaseline', 'ISTF-\nMamba', 'PCMCI+']
    colors = [C_BASELINE, C_MAMBA, C_PCMCI]

    x = np.arange(len(datasets))
    w = 0.25

    for mi, (method, label, color) in enumerate(zip(methods, labels, colors)):
        means, stds = [], []
        for ds in datasets:
            vals = [e['metrics']['auroc'] for e in data['results']
                    if e['dataset'] == ds and e['method'] == method]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(x + (mi - 1) * w, means, w, label=label, color=color,
                      edgecolor='white', linewidth=0.8, yerr=stds, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', '\n') for d in datasets])
    ax.set_ylabel('AUROC', fontweight='bold')
    ax.set_title('AUROC on 4 Inferential Datasets (frozen v2)', fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # Significance annotations
    sig_annotations = {
        'CT_medical': ('*', 0.52, 'Holm-sig.\n(5/7 metrics)'),
        'Lorenz_F40': ('n.s.', 0.96, 'near-neutral'),
        'VAR_d50': ('n.s.', 0.74, 'directional only\n(adj p=0.059)'),
        'NSVAR_d10': ('N/A', 0.96, 'degenerate\n(insuff. pairs)'),
    }
    for i, ds in enumerate(datasets):
        sym, y, note = sig_annotations[ds]
        ax.text(i, y, sym, ha='center', fontsize=9, fontweight='bold')
        ax.text(i + 0.35, y - 0.04, note, fontsize=7, color='#666')

    plt.tight_layout()
    fig.savefig(OUT + 'fig3_benchmark_auroc.pdf', facecolor='white')
    fig.savefig(OUT + 'fig3_benchmark_auroc.png', facecolor='white', dpi=300)
    plt.close()
    print("  Fig 3: Benchmark AUROC saved")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating paper main figures (TNNLS)...")
    print(f"Output: {os.path.abspath(OUT)}")
    draw_architecture_figure()
    draw_diagnostic_figure()
    draw_benchmark_auroc()
    print("Done.")
