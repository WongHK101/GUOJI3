# ISTF-Mamba: Input-Space Temporal Filtering for JRNGC

Structural shortcut diagnosis and repair for Jacobian-regularized neural Granger causality.

## Overview

Jacobian-regularized neural Granger causality (JRNGC) discovers causal graphs by penalizing the input-output Jacobian of a prediction MLP. We identify a critical architectural vulnerability: when auxiliary temporal channels (concat, FiLM) are introduced to handle non-stationarity, gradient descent routes predictions through **unpenalized side channels**, suppressing the Jacobian entries that constitute the GC measurement signal.

**Input-Space Temporal Filtering (ISTF)** repairs this by confining all temporal processing to the original *d*-dimensional input space, eliminating unpenalized pathways. The architecture uses two core structural constraints:

1. **Input-space confinement** — all temporal processing stays in the original variable space
2. **Orthogonality-based drift control** — explicit penalty on filter drift from identity

Near-identity initialization (zero output projection, residual scale ε=0.1) is used as a conservative implementation choice, not a standalone constraint.

## Key Results

Final inferential benchmark (4 datasets, ≥5 paired seeds, Holm-Bonferroni corrected):

| Dataset | Seeds | Baseline AUROC | ISTF-Mamba AUROC | Δ | Interpretation |
|---------|-------|---------------|-----------------|---|----------------|
| CT_medical | 10 | 0.458 +/- 0.024 | 0.500 +/- 0.022 | +0.042 | 5/7 metrics Holm-significant |
| Lorenz_F40 | 10 | 0.938 +/- 0.012 | 0.939 +/- 0.014 | +0.001 | 0/7 metrics significant (neutral) |
| VAR_d50 | 10 | 0.715 +/- 0.034 | 0.678 +/- 0.034 | -0.036 | AUROC directional only; AUPRC/F1 significantly worse |
| NSVAR_d10 | 5 | 0.930 +/- 0.024 | 0.946 +/- 0.028 | +0.016 | Degenerate on most metrics |

ISTF-Mamba is not a universal booster — it is a structural safeguard whose benefit depends on whether the dataset regime creates incentives for auxiliary-channel shortcut learning.

## Repository Structure

```
mamba_enhanced/
├── src/
│   ├── mamba_jrngc_pilot.py          # Core: all model definitions + training + metrics
│   │   ├── MambaPreprocessor         #   Aux channel (used by Concat/FiLM)
│   │   ├── MambaJRNGC                #   Concat architecture (shortcut-vulnerable)
│   │   ├── FiLMJRNGC                 #   FiLM architecture (shortcut-vulnerable)
│   │   ├── MambaFilterJRNGC          #   ISTF architecture (ours, filter_type="mamba")
│   │   ├── TCNFilterJRNGC            #   ISTF architecture (ours, filter_type="tcn")
│   │   ├── BaselineJRNGC             #   Original JRNGC (no temporal context)
│   │   ├── train_model()             #   Unified training loop
│   │   └── compute_metrics()         #   AUROC/AUPRC/SHD/nSHD/MCC evaluation
│   ├── minimal_mamba.py              # MambaBlock + SelectiveSSM (pure PyTorch)
│   ├── nonstationary_var.py          # Non-stationary VAR data generator
│   ├── config.py                     # Path resolution, argument parser, device selection
│   ├── schema.py                     # Result schema definitions
│   ├── factorial_data.py             # Factorial experiment data
│   └── __init__.py
│
├── experiments/                       # All experiment scripts (run from repo root)
│   ├── test_mask_shuffle.py          # Shortcut diagnostic: mask/shuffle intervention
│   ├── test_shortcut_diagnostics.py  # Shortcut diagnostic suite
│   ├── test_mask_supplement.py       # Three-way mask intervention
│   ├── run_statistical_tests.py      # Holm-Bonferroni paired tests (7 metrics × 4 datasets)
│   ├── generate_appendix_tables.py   # LaTeX appendix tables from canonical JSON
│   ├── generate_eligibility_table.py # Dataset eligibility classification
│   ├── gen_paper_main_figures.py     # Paper figure generation
│   ├── backfill_canonical_v2.py      # Canonical rerun with deterministic seeding + GC scores
│   ├── backfill_topology_metrics.py  # Topology metric backfill (legacy)
│   ├── compute_metrics_from_scores.py # Recompute metrics from saved GC score matrices
│   ├── consolidate_all.py            # Aggregate all JSON results
│   ├── run_ct_medical_10seed.py      # CT_medical 10-seed full run
│   ├── run_neural_gc_baseline.py     # Neural GC baseline (cMLP, cLSTM)
│   ├── run_pcmci_baseline.py         # PCMCI+ baseline comparison
│   ├── test_causaltime.py            # CausalTime benchmark (medical/pm25/traffic)
│   ├── test_dream3_backfill.py       # DREAM3 gene regulation benchmark
│   ├── test_fmri_3subj.py            # fMRI neuroimaging benchmark
│   ├── test_tcn_backfill.py          # TCN filter ablation across configs
│   ├── test_interaction_ablation.py  # 2×2 factorial ablation
│   ├── test_theory_verification.py   # Theorem numerical verification
│   ├── test_concat_full_penalty.py   # Auxiliary-channel penalty control
│   ├── factorial_diagnostics.py      # Factorial experiment diagnostics
│   ├── factorial_stat_tests.py       # Factorial statistical tests
│   ├── reproduce_factorial_full.py   # Full factorial reproduction
│   ├── risk_mitigation_20260515/     # Risk-mitigation experiments (5 scripts)
│   └── ...                           # Additional experiment scripts
│
├── tests/
│   ├── test_core.py                  # Unit tests for core modules
│   └── __init__.py
│
├── paper-data-prefreeze-20260512/    # Frozen paper data assets
├── results/                           # Experiment outputs (raw/ tracked, others ignored)
├── README.md                         # This file
└── .gitignore
```

## Getting Started

### Environment

```bash
conda create -n jrngc_bw python=3.10
conda activate jrngc_bw
pip install torch numpy scipy scikit-learn
```

### Data

Datasets must be placed under `data/` (not tracked in git). Required structure:
- `data/var/` — Stationary VAR (from JRNGC repo)
- `data/lorenz/` — Lorenz-96 (from JRNGC repo)
- `data/nonstationary_var/` — Non-stationary VAR (generated by `nonstationary_var.py`)
- `data/causaltime/` — CausalTime medical/traffic/pm25
- JRNGC datasets expected at `../JRNGC/data/`

### Running Experiments

All experiment scripts live in `experiments/` and should be run from the repo root:

```bash
cd mamba_enhanced

# Shortcut diagnostic (~20 min)
python experiments/test_mask_shuffle.py

# CT_medical 10-seed run
python experiments/run_ct_medical_10seed.py

# Full canonical backfill (all 90 baseline+mamba entries, ~1.5h GPU)
python experiments/backfill_canonical_v2.py

# Statistical tests on results
python experiments/run_statistical_tests.py

# Generate appendix tables
python experiments/generate_appendix_tables.py
```

### Model Training API

```python
from src.mamba_jrngc_pilot import MambaFilterJRNGC, train_model, compute_metrics

model = MambaFilterJRNGC(
    d=10, lag=7, layers=5, hidden=50,
    jacobian_lam=0.01, d_state=4, ortho_lam=0.05,
    residual_scale=0.1, filter_type="mamba"  # or "tcn"
)
model, loss = train_model(model, x, max_iter=5000, lr=1e-3)
gc_matrix = model.get_gc_matrix(x)
metrics = compute_metrics(gc_true, gc_matrix)
```

## Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_state` | 4 | Mamba SSM state dimension (CT_medical uses 8 as pre-specified exception) |
| `d_cond` | 4 | Auxiliary channel dimension (Concat/FiLM only) |
| `jacobian_lam` | 0.01 | Jacobian L1 penalty weight |
| `ortho_lam` | 0.05 | Orthogonality regularization weight |
| `residual_scale` | 0.1 | Filter residual scale ε |
| `layers` | 5 | MLP depth |
| `hidden` | 50 | MLP hidden dimension |
| `max_iter` | 2000–5000 | Training iterations |
| `lr` | 1e-3 | Learning rate (Adam) |

## Maintenance Notes

- This README should be updated whenever new experiments are added or key results change.
- Result JSON files in `results/raw/` are committed for traceability; regenerate after re-running experiments.
- Paper LaTeX source is maintained separately in `IEEE-Transactions-LaTeX2e-templates-and-instructions/istf_jrngc.tex`.
- Paper data asset index: `paper-data-prefreeze-20260512/` (frozen canonical results).
- Cloud runner scripts are on AutoDL server at `/root/autodl-tmp/GUOJI/mamba_enhanced/`.
