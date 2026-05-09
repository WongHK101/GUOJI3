# ISTF-Mamba: Input-Space Temporal Filtering for JRNGC

Structural shortcut diagnosis and repair for Jacobian-regularized neural Granger causality.

## Overview

Jacobian-regularized neural Granger causality (JRNGC, ICML 2024) discovers causal graphs by penalizing the input-output Jacobian of a prediction MLP. We identify a critical architectural vulnerability: when auxiliary temporal channels (concat, FiLM) are introduced to handle non-stationarity, gradient descent routes predictions through **unpenalized side channels**, suppressing the Jacobian entries that constitute the GC measurement signal.

**Input-Space Temporal Filtering (ISTF)** repairs this by processing all temporal information within the original *d*-dimensional input space, eliminating unpenalized pathways. Three design principles:

1. **Input-space confinement** — all temporal processing stays in the original variable space
2. **Near-identity initialization** — starts from JRNGC baseline (zero output projection, residual scale ε=0.1)
3. **Orthogonality regularization** — explicit penalty on filter drift from identity

## Key Results (2026-05-10)

### Shortcut Diagnosis (P1-5): Three-Way Mask Intervention

NSVAR d=10, seed 0. Trained Concat JRNGC + ISTF, then measured Δloss under interventions:

| Intervention | Concat Δloss | ISTF Δloss | Interpretation |
|-------------|-------------|-----------|----------------|
| mask_x_only (zero x, keep c) | **+1.68** | +29.15 | Concat ~17× less sensitive → c provides backup |
| mask_c_only (keep x, zero c) | +0.44 | — | c not dominant when x is available |
| mask_both (zero both) | +1.66 | +29.15 | Full input dependency baseline |

> **Finding:** Auxiliary channel c is a **learned backup pathway** — not the primary prediction route, but exploited when input is degraded.

### CT_medical 3-Seed Validation (P1-1, d_state=8)

d=40, T=1200, lag=1. 3 seeds (0–2).

| | Baseline | ISTF-Mamba | Δ |
|--|----------|------------|----|
| Mean ± Std | 0.4741 ± 0.0236 | 0.5136 ± 0.0329 | **+3.95pp (+8.3% relative)** |
| Best seed (0) | 0.4726 | 0.5596 | +8.7pp (+18.4% relative) |
| Worst seed (2) | 0.5036 | 0.4844 | −1.9pp (−3.8% relative) |

Seed variance (±3.3pp std) is substantial. With T=1200 (T/d=30), this is unlikely due to sample insufficiency; more likely reflects seed-dependent optimization paths and clinical non-stationarity heterogeneity.

### Multi-Seed Synthetic (P1-3)

ISTF-Mamba is approximately neutral on standard benchmarks (Δ within ±1%) — expected: ISTF is a structural safeguard, not an unconditional booster.

| Dataset | Seeds | Baseline AUROC | ISTF-Mamba AUROC | Δ |
|---------|-------|---------------|-----------------|---|
| VAR_d50 | 5 | 0.7178 ± 0.0331 | 0.7237 ± 0.0263 | +0.0058 |
| Lorenz_F40 | 5 | 0.9417 ± 0.0119 | 0.9462 ± 0.0087 | +0.0046 |
| NSVAR_d10 | 5 | 0.9279 ± 0.0293 | 0.9269 ± 0.0336 | −0.0010 |
| NSVAR_d50_PlanA | 3 | 0.6476 ± 0.0053 | 0.6406 ± 0.0246 | −0.0070 |

## Repository Structure

```
mamba_enhanced/
├── mamba_jrngc_pilot.py          # Core: all model definitions
│   ├── MambaPreprocessor         #   Aux channel (used by Concat/FiLM)
│   ├── MambaJRNGC                #   Concat architecture (shortcut-vulnerable)
│   ├── FiLMJRNGC                 #   FiLM architecture (shortcut-vulnerable)
│   ├── MambaFilterJRNGC          #   ISTF architecture (ours, filter_type="mamba")
│   ├── TCNFilterJRNGC            #   ISTF architecture (ours, filter_type="tcn")
│   ├── BaselineJRNGC             #   Original JRNGC (no temporal context)
│   ├── train_model()             #   Unified training loop
│   └── compute_metrics()         #   AUROC/AUPRC/SHD/nSHD/MCC evaluation
│
├── minimal_mamba.py              # MambaBlock + SelectiveSSM (pure PyTorch)
├── nonstationary_var.py          # Non-stationary VAR data generator
│
├── test_mask_supplement.py       # P1-5: Three-way mask intervention
├── test_ct_medical_3seed.py      # P1-1: CT_medical 3-seed validation
├── test_multiseed_synthetic.py   # P1-3: Multi-seed synthetic (4 configs)
├── test_shortcut_diagnostics.py  # P1: Shortcut diagnostic suite
├── test_interaction_ablation.py  # P1: 2×2 factorial ablation
├── test_theory_verification.py   # P1: Theorem numerical verification
├── test_concat_full_penalty.py   # P1: Auxiliary-channel penalty control
├── test_causaltime.py            # CausalTime benchmark (medical/pm25/traffic)
├── test_dream3_backfill.py       # DREAM3 gene regulation benchmark
├── test_fmri_3subj.py            # fMRI neuroimaging benchmark
├── test_tcn_backfill.py          # TCN filter ablation across configs
├── test_pcmci.py                 # PCMCI+ baseline comparison
│
├── *test_filter_5seed*.py        # Filter 5-seed validation
├── *test_var50*.py               # VAR_d50 experiments
├── *test_nsvar50*.py             # NSVAR_d50 experiments
├── *test_ns_nonlinear*.py        # Non-stationary nonlinear experiments
│
├── postprocess_p1.py             # P1-4: Comprehensive results post-processing
├── consolidate_all.py            # Aggregate all JSON results
├── gen_paper_figures.py          # Paper figure generation
├── filter_selection_criterion.py # Deployment heuristic
│
├── *_results.json                # Result files (AUROC/AUPRC/SHD/nSHD/MCC)
├── causality_preserving_design.md # Design rationale document
├── theory_jacobian_bound.md      # Theoretical analysis notes
└── README.md                     # This file
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

### Running a Single Experiment

```bash
# Three-way mask diagnostic (~20 min)
python test_mask_supplement.py

# CT_medical 3-seed validation (~20 min)
python test_ct_medical_3seed.py

# Full multi-seed synthetic (~3.5 hours, 36 training runs)
python test_multiseed_synthetic.py

# Post-processing all results
python postprocess_p1.py
```

### Model Training API

```python
from mamba_jrngc_pilot import MambaFilterJRNGC, train_model, compute_metrics

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
| `d_state` | 4 | Mamba SSM state dimension (CT_medical uses 8) |
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
- Result JSON files are committed for traceability; regenerate after re-running experiments.
- Paper LaTeX source is maintained separately in `IEEE-Transactions-LaTeX2e-templates-and-instructions/istf_jrngc.tex`.
- Cloud runner scripts (`run_p1.sh`, etc.) are on AutoDL server at `/root/autodl-tmp/GUOJI/mamba_enhanced/`.
