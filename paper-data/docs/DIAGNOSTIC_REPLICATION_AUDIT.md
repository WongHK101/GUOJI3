# Diagnostic Replication Audit

**Audit date:** 2026-07-10  
**Purpose:** Disclose the aggregation unit, seed count, configuration, and score semantics for every controlled diagnostic used in the Route-B v2 manuscript. This is a provenance audit of frozen files, not a rerun.

## Result summary

| Diagnostic | Reported aggregation | Classification in v2 | Decision |
| --- | --- | --- | --- |
| d_cond sweep | One generator seed and one model seed across the capacity values. | Single-run controlled diagnostic. | Keep, explicitly label in text and Fig. 2 caption. |
| Mask/shuffle intervention | NSVAR data seed 0 and model seed 0. | Single-run controlled diagnostic. | Keep, explicitly label in text and Fig. 2 caption. |
| Coefficient recovery | One controlled VAR generator/model seed. | Single-run controlled diagnostic. | Keep, explicitly label in text and Fig. 2 caption. |
| Full auxiliary-Jacobian penalty | Five saved seeds, summarized as mean +/- saved population SD. | Five-seed controlled diagnostic. | Keep, report mean +/- SD and draw error bars. |
| P0 score-semantics audit | Five saved CPU smoke seeds (0-4), each with the same d/T/lag/iteration setting. | Five-seed semantic diagnostic. | Keep, redraw Fig. 3 with the five-seed aggregate. |

## Audit details

### d_cond sweep

- Artifact: `E:\GUOJI\mamba_enhanced\diagnostic_results\exp2_dcond_sweep.json`
- Script: `E:\GUOJI\mamba_enhanced\experiments\test_shortcut_diagnostics.py`, function `exp2_dcond_sweep`.
- Data/model setting: VAR(1), d=8, T=300, lag=1, generator seed 42. Each capacity sets both Torch and NumPy seed 42 before model construction.
- Model: baseline JRNGC at d_cond=0 or concat MambaJRNGC otherwise; 3 layers, hidden=32, d_state=4, jacobian_lam=0.01, max_iter=1500, lr=1e-3.
- Score: x-only Jacobian graph score.
- Frozen observation: loss falls from 0.0089488 to 0.0021074; AUROC is non-monotonic, reaches 0.3500 at d_cond=8, and is 0.5218 at d_cond=16.
- Manuscript status: single-run controlled diagnostic. It must not be described as a seed mean, a monotonic law, or a benchmark result.

### Mask/shuffle intervention

- Artifact: `E:\GUOJI\mamba_enhanced\results\raw\mask_supplement_results.json`
- Script: `E:\GUOJI\mamba_enhanced\experiments\test_mask_supplement.py`.
- Data/model setting: NSVAR d=10, true lag=7, data seed 0. Concat and no-aux models are each initialized with Torch/NumPy seed 0; both run for 2,000 iterations.
- Model: concat MambaJRNGC (5 layers, hidden=50, d_cond=4, d_state=4, jacobian_lam=0.01) plus an input-space no-aux control.
- Diagnostic: prediction-loss changes after masking x, masking c, masking both, or shuffling x. It is not itself a graph score.
- Manuscript status: single-run controlled diagnostic. It supports declared auxiliary-route usage only.

### Coefficient recovery

- Artifact: `E:\GUOJI\mamba_enhanced\diagnostic_results\exp4_coefficient_recovery.json`
- Script: `E:\GUOJI\mamba_enhanced\experiments\test_shortcut_diagnostics.py`, function `exp4_coefficient_recovery`.
- Data/model setting: VAR(1), d=8, T=500, lag=1, generator seed 42. Each baseline/concat/filter model resets Torch and NumPy seed 42, then trains for 2,000 iterations at lr=1e-3.
- Score: x-only Jacobian-derived coefficient fidelity compared with known off-diagonal transition coefficients.
- Manuscript status: single-run controlled diagnostic. Only the baseline/concat comparison is used in the Route-B main text.

### Full auxiliary-Jacobian penalty

- Artifacts: `E:\GUOJI\mamba_enhanced\risk_mitigation_results\full_aux_jacobian_penalty.json` and `.csv`.
- Script: `E:\GUOJI\mamba_enhanced\experiments\risk_mitigation_20260515\run_full_aux_penalty.py`.
- Data/model setting: VAR(1), d=8, T=500, lag=1, d_cond=4, five seeds, max_iter=2,000, lr=1e-3; data are generated with seed `100*seed+42`.
- Comparison: baseline, concat x-only, and several full-penalty variants. The v2 manuscript uses `full_lc_10`, where lambda_c/lambda_x=10.
- Score: x-only graph recovery and coefficient fidelity. The regularizer, not the graph score, is expanded to include the auxiliary coordinates.
- Frozen summary: concat x-only AUROC 0.5420 +/- 0.0854; full_lc_10 AUROC 0.8962 +/- 0.0409. Coefficient correlations are 0.2646 +/- 0.2539 and 0.9394 +/- 0.0341, respectively.
- Manuscript status: five-seed controlled diagnostic. It shows controlled mitigation through expanded penalty-route coverage; it does not establish score-route completeness.

### P0 score-semantics audit

- Artifacts: `E:\GUOJI\mamba_enhanced\results\p0_audit\p0_jacobian_semantics_d6_iter120_refactor_seed0.json` through `...seed4.json`.
- Script: `E:\GUOJI\mamba_enhanced\experiments\p0_jacobian_semantics_audit.py`.
- Data/model setting: CPU smoke, d=6, T=100, lag=3, max_iter=120, 8 selected windows; each artifact records its own seed and window indices.
- Diagnostic: concat partial x-only score versus total raw derivative; legacy filtered-coordinate Mamba score versus raw-chain score.
- Frozen five-seed aggregate (population SD): concat partial/total correlation 0.5064 +/- 0.1066; legacy filtered/raw correlation 0.8316 +/- 0.0210; legacy exact-top-k Jaccard 0.7843 +/- 0.0983.
- Manuscript status: five-seed semantic diagnostic. Legacy Mamba cannot support graph-recovery or performance claims.

## Audit limitation

This audit classifies replication status from saved artifacts and generating scripts. It does not convert a single-run diagnostic into a statistical result, and it does not authorize any new run.
