# Phase 9 RTX 4090 Bounded Enhancement Round 1

**Status:** development-only; no manuscript or frozen artifact was modified.

## Executive decision

- **Filtering repair line: STOP.** Static adaptive FIR showed a large stationary-nonlinear gain at 2000 iterations but no stable non-stationary benefit. The contextual gate also failed both non-stationary cells.
- **Gradient-projection repair line: STOP.** Protecting prediction reduced MSE but degraded graph recovery; protecting only historical coverage was nearly equivalent to the standard repair.
- **Audit-generalization line: CONTINUE TO FORMAL PLANNING.** Missing-route, partial-total discrepancy, fixed-target route use, coordinate mixing, and bounded horizon diagnostics were stable across NetSim, MoCap, Mamba, and an active causal TCN auxiliary route.
- These are development results, not submission evidence. A frozen advisor-approved protocol is still required before manuscript use.

## D2 at 500 iterations

| Cell | Method | n | AUROC | AUPRC | MCC | raw MSE |
| --- | --- | --- | --- | --- | --- | --- |
| NS+Linear | adaptive_fir | 3 | 0.8727 | 0.6671 | 0.4163 | 0.01738 |
| NS+Linear | baseline | 3 | 0.8919 | 0.6952 | 0.4731 | 0.01730 |
| NS+Linear | contextual_fir | 3 | 0.8765 | 0.6683 | 0.4163 | 0.01769 |
| NS+Linear | cp_depthwise | 3 | 0.8920 | 0.6981 | 0.5425 | 0.01728 |
| NS+Linear | fixed_fir3 | 3 | 0.8647 | 0.6511 | 0.4163 | 0.01734 |
| NS+Nonlinear | adaptive_fir | 3 | 0.8298 | 0.5892 | 0.3328 | 0.01783 |
| NS+Nonlinear | baseline | 3 | 0.8291 | 0.5853 | 0.4857 | 0.01729 |
| NS+Nonlinear | contextual_fir | 3 | 0.8370 | 0.5916 | 0.4289 | 0.01781 |
| NS+Nonlinear | cp_depthwise | 3 | 0.8242 | 0.5968 | 0.3896 | 0.01731 |
| NS+Nonlinear | fixed_fir3 | 3 | 0.8307 | 0.6291 | 0.4289 | 0.01778 |
| Stat+Linear | adaptive_fir | 3 | 0.8512 | 0.6276 | 0.3595 | 0.01764 |
| Stat+Linear | baseline | 3 | 0.8482 | 0.6342 | 0.4556 | 0.01737 |
| Stat+Linear | contextual_fir | 3 | 0.8552 | 0.6057 | 0.5251 | 0.01761 |
| Stat+Linear | cp_depthwise | 3 | 0.8406 | 0.5955 | 0.3721 | 0.01733 |
| Stat+Linear | fixed_fir3 | 3 | 0.8385 | 0.5727 | 0.3595 | 0.01750 |
| Stat+Nonlinear | adaptive_fir | 3 | 0.8221 | 0.6146 | 0.3721 | 0.01764 |
| Stat+Nonlinear | baseline | 3 | 0.7808 | 0.5389 | 0.4289 | 0.01733 |
| Stat+Nonlinear | contextual_fir | 3 | 0.8225 | 0.6107 | 0.3721 | 0.01761 |
| Stat+Nonlinear | cp_depthwise | 3 | 0.7688 | 0.5210 | 0.3026 | 0.01723 |
| Stat+Nonlinear | fixed_fir3 | 3 | 0.8303 | 0.6167 | 0.5251 | 0.01769 |

Paired AUROC effects for the adaptive candidates:

| Cell | Candidate | Reference | Mean delta | Positive seeds |
| --- | --- | --- | --- | --- |
| NS+Linear | adaptive_fir | baseline | -0.0192 | 0/3 |
| NS+Nonlinear | adaptive_fir | baseline | 0.0007 | 1/3 |
| Stat+Linear | adaptive_fir | baseline | 0.0030 | 2/3 |
| Stat+Nonlinear | adaptive_fir | baseline | 0.0412 | 2/3 |
| NS+Linear | contextual_fir | baseline | -0.0154 | 0/3 |
| NS+Nonlinear | contextual_fir | baseline | 0.0079 | 1/3 |
| Stat+Linear | contextual_fir | baseline | 0.0070 | 1/3 |
| Stat+Nonlinear | contextual_fir | baseline | 0.0417 | 2/3 |
| NS+Linear | adaptive_fir | fixed_fir3 | 0.0080 | 2/3 |
| NS+Nonlinear | adaptive_fir | fixed_fir3 | -0.0009 | 1/3 |
| Stat+Linear | adaptive_fir | fixed_fir3 | 0.0126 | 2/3 |
| Stat+Nonlinear | adaptive_fir | fixed_fir3 | -0.0082 | 1/3 |
| NS+Linear | contextual_fir | fixed_fir3 | 0.0118 | 2/3 |
| NS+Nonlinear | contextual_fir | fixed_fir3 | 0.0063 | 1/3 |
| Stat+Linear | contextual_fir | fixed_fir3 | 0.0166 | 1/3 |
| Stat+Nonlinear | contextual_fir | fixed_fir3 | -0.0078 | 1/3 |

## Phase 8 gradient strategies at 500 iterations

| Method | n | AUROC | AUPRC | MCC | MSE | Coeff. r |
| --- | --- | --- | --- | --- | --- | --- |
| coverage_history_guarded | 3 | 0.9652 | 0.9607 | 0.8639 | 0.00937 | 0.9873 |
| coverage_prediction_guarded | 3 | 0.9265 | 0.8842 | 0.6694 | 0.00806 | 0.9552 |
| coverage_standard | 3 | 0.9738 | 0.9724 | 0.8897 | 0.00943 | 0.9889 |

## Cross-domain audit stability

| Architecture | Dataset | n | Missing route | Partial-total r | Tail median | Mix entropy | mask-c delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| concat | mocap_run | 3 | 0.9218 | 0.6053 | 0.3995 | 0.9796 | 0.6241 |
| concat | mocap_salsa | 3 | 0.9070 | 0.4519 | 0.3991 | 0.9800 | 0.9497 |
| concat | netsim48 | 3 | 0.8233 | 0.3239 | 0.2857 | 0.9677 | 0.8802 |
| concat | netsim49 | 3 | 0.8203 | 0.2962 | 0.2590 | 0.9643 | 0.9716 |
| tcn_concat | mocap_run | 3 | 0.7572 | 0.8118 | 0.3757 | 0.9734 | 0.6437 |
| tcn_concat | mocap_salsa | 3 | 0.7332 | 0.7653 | 0.3810 | 0.9718 | 1.0520 |
| tcn_concat | netsim48 | 3 | 0.4885 | 0.7641 | 0.1573 | 0.9573 | 0.6585 |
| tcn_concat | netsim49 | 3 | 0.5335 | 0.7801 | 0.1904 | 0.9680 | 0.7429 |

## Horizon sensitivity

Using the same target windows, nominal-lag scores were identical at H=32/64/128. H=64 captured about 100% of H=128 off-diagonal mass on NetSim, 99.93% on MoCap-run, and approximately 100% on MoCap-salsa. This supports H=64 for the bounded audit, but does not establish mass beyond H=128.

## Determinism

Same-current-commit NetSim48 concat rerun exact match: **True**.

## Evidence boundary

- NetSim baseline graph recovery is only around random-to-moderate; these runs support route/score diagnostics, not a performance benchmark claim.
- MoCap has no accepted direct graph ground truth here; only prediction interventions and attribution semantics are reported.
- No Phase 7 seeds 4-8, Stage 1b outputs, AutoDL GPU, or KBS manuscript files were accessed or modified.
