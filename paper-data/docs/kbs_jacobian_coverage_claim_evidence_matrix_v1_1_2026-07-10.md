# KBS Jacobian Coverage Audit - Claim Evidence Matrix v1.1

Date: 2026-07-10

Status: documentation-only correction package. This document uses frozen artifacts only. It does not run new experiments, inspect Stage 1b seeds 4-8, revive CP-depthwise as a main method, or replace the canonical KBS manuscript.

## Locked Route-B Narrative

One-sentence argument:

> In Jacobian-regularized neural Granger causality, predictive pathways outside the scored or penalized Jacobian can decouple prediction accuracy from graph knowledge; a Jacobian coverage audit framework exposes this failure mode through controlled diagnostics, score-semantics audits, and a preregistered semantic-repair negative result.

Canonical terms:

| Canonical term | Meaning | Forbidden drift |
|---|---|---|
| Jacobian coverage audit framework | A declaration and diagnostic procedure for checking whether score variables, penalty variables, declared predictive route classes, coordinate identity, and attribution horizon are aligned. | coverage certificate, guarantee, proof of causal identifiability |
| Prediction-knowledge decoupling | The predictor can optimize loss while the graph score loses the intended raw-variable causal meaning. | proof that prediction is useless |
| Auxiliary predictive pathway | A side route such as concat conditioning or external covariates that may influence prediction outside the scored raw-input Jacobian. | all auxiliary inputs are invalid |
| Raw-chain attribution | Derivative of prediction with respect to raw input through differentiable transformations within a declared horizon. | full attribution unless full support is covered |
| CP-depthwise semantic repair | The preregistered coordinate-preserving depthwise candidate evaluated in Stage 1a. | successful repaired method |

## Corrected Audit Framework Logic

The coverage declaration remains:

\[
C = (V_{\mathrm{score}}, V_{\mathrm{penalty}}, P_{\mathrm{pred}}, M_{\mathrm{coord}}, H_{\mathrm{attr}}).
\]

Definitions:

- `V_score`: variables or coordinates used to compute the graph score.
- `V_penalty`: variables or coordinates covered by Jacobian sparsity or causal regularization.
- `P_pred`: architecture-declared predictive route classes. This is a declaration of route classes, not an exact enumeration of every effective neural-network path.
- `M_coord`: mapping from score coordinates to original source-variable identities.
- `H_attr`: attribution horizon or lag support used by the score and semantic audit.

The previous single question, "Do all predictive paths enter either the score or the penalty?", is retired because score coverage and penalty coverage are not interchangeable. The v1.1 framework uses four separate dimensions:

| Dimension | Audit question | Why separate |
|---|---|---|
| A. Score-route completeness | Are all predictive route classes whose effects are interpreted as graph knowledge included in the score attribution? | A route can be penalized but still absent from the graph score. |
| B. Penalty-route completeness | Are all route classes capable of carrying predictive information covered by the sparsity or causal regularization objective, or explicitly declared exempt? | A route can appear in the score but escape regularization. |
| C. Score-penalty alignment | Do the score and penalty cover compatible variables, coordinates, paths, and attribution horizons? | Misalignment can make optimization and reported graph semantics diverge. |
| D. Coordinate and horizon validity | Does each score column map to one original source variable, and does the attribution horizon cover the transformation support? | Coordinate mixing or truncated support can invalidate raw-variable interpretation. |

Audit output labels:

| Label | Meaning |
|---|---|
| `COVERED` | The declared evidence supports the relevant route, coordinate, and horizon coverage for the stated claim. |
| `PARTIALLY COVERED` | Some declared route classes or horizons are covered, but the audit is incomplete for the stated interpretation. |
| `COORDINATE-AMBIGUOUS` | Score coordinates do not map cleanly to original source-variable identities. |
| `HORIZON-TRUNCATED` | Attribution uses a finite horizon and the omitted support remains relevant or insufficiently audited. |
| `UNASSESSED` | The artifact has not yet been audited for the required score semantics or provenance. |

These labels are diagnostics, not mathematical guarantees.

## Main Claims And Evidence

| Claim | Evidence | Exact artifact path | Evidence tier | Supported wording | Unsupported wording |
|---|---|---|---|---|---|
| Auxiliary side channels can create a structural shortcut in JRNGC. | Shortcut theorem; concat setup; d_cond sweep; mask-c/mask-x diagnostics. | `E:\GUOJI\mamba_enhanced\diagnostic_results\exp2_dcond_sweep.json`; `E:\GUOJI\mamba_enhanced\diagnostic_results\mask_shuffle_results.json`; `E:\GUOJI\mamba_enhanced\results\raw\mask_supplement_results.json` | Main-text core after provenance copy/freeze | "can create", "under this architecture", "structural vulnerability" | "always create", "all conditioning is harmful" |
| Prediction loss and graph recovery can decouple under concat conditioning. | d_cond sweep and coefficient recovery. | `E:\GUOJI\mamba_enhanced\diagnostic_results\exp2_dcond_sweep.json`; `E:\GUOJI\mamba_enhanced\diagnostic_results\exp4_coefficient_recovery.json` | Main-text core after provenance copy/freeze | "loss decreased while AUROC or coefficient fidelity degraded in controlled diagnostics" | "monotonic law", "benchmark-wide proof" |
| Expanding Jacobian coverage mitigated the failure in controlled concat diagnostics. | Full auxiliary-Jacobian penalty. | `E:\GUOJI\mamba_enhanced\risk_mitigation_results\full_aux_jacobian_penalty.json`; `E:\GUOJI\mamba_enhanced\risk_mitigation_results\full_aux_jacobian_penalty.csv` | Main-text core after provenance copy/freeze | "expanding Jacobian coverage mitigated the failure in controlled concat diagnostics" | "full penalty solves all cases", "ISTF is superior" |
| Filtered-coordinate scores can diverge from original-input raw-chain attribution when cross-channel transformations destroy source-variable identity. | Legacy ISTF-Mamba filtered-coordinate vs raw-chain mismatch and cross-variable leakage. | `E:\GUOJI\mamba_enhanced\paper-data\p0_jacobian_semantics_audit_2026-07-06.md`; `E:\GUOJI\mamba_enhanced\results\p0_audit\p0_jacobian_semantics_d6_iter120_refactor_seed0.json` through `seed4.json` | Score-semantics failure diagnostic only | Exact permitted claim above | graph-recovery performance, benchmark superiority, ISTF effectiveness, Mamba effectiveness, operating-regime claims |
| Coordinate-preserving repair can pass score-semantics audits without passing performance or novelty gates. | Stage 1a official aggregate. | `E:\GUOJI\kbs_review_packages\phase7_stage1a_901_go_nogo_v1.zip` -> `formal_root/stage1a_aggregate_go_no_go.json` | Main-text core | "semantic gates passed; performance and FixedFIR3 novelty gates failed" | "CP works", "Stage 1a is positive" |
| FixedFIR3 exceeded CP-depthwise in mean AUROC in all four Stage 1a cells, although not in every individual data seed. | Official Stage 1a aggregation. | `E:\GUOJI\kbs_review_packages\phase7_stage1a_901_go_nogo_v1.zip` -> `formal_root/stage1a_aggregate_go_no_go.json` | Main-text core | "in cell-level mean AUROC" | "in every individual seed" |
| EMA full-H dominance no-go did not trigger. | Official `ema_reference_dominance.no_go_triggered=false`. | `E:\GUOJI\kbs_review_packages\phase7_stage1a_901_go_nogo_v1.zip` -> `formal_root/stage1a_aggregate_go_no_go.json` | Supporting precision | "EMA was a full-H reference and did not trigger the dominance no-go" | "EMA is a nominal-lag main causal method" |
| P1 bounded postmortem did not identify a definitive failure class. | P1 decision gates; gradient replay alignment invalid. | `E:\GUOJI\kbs_review_packages\phase7_stage1a_bounded_failure_analysis_v1.zip` -> `ab_decision_gates.json`, `gradient_replay_alignment.json`, `p1_status.json` | Supporting boundary | "bounded postmortem remained inconclusive"; "A3 was uninterpretable and did not pass because `gradient_replay_alignment_valid=false`" | any raw-branch values, tendencies, or conflict-pattern counts |

## Required Artifact-Level Evidence Register

| Evidence item | Exact artifact path | Score type | Model architecture | Audit label | Permitted claim |
|---|---|---|---|---|---|
| d_cond sweep | `E:\GUOJI\mamba_enhanced\diagnostic_results\exp2_dcond_sweep.json`; generating script `E:\GUOJI\mamba_enhanced\experiments\test_shortcut_diagnostics.py` | x-only Jacobian graph score with concat auxiliary dimension sweep | baseline JRNGC and concat-conditioned JRNGC diagnostic variants | `PARTIALLY COVERED` for route audit, because auxiliary routes are intentionally not included in x-only score | Increasing auxiliary conditioning can lower prediction loss while degrading raw-input graph recovery in this controlled concat diagnostic. |
| mask/shuffle | `E:\GUOJI\mamba_enhanced\diagnostic_results\mask_shuffle_results.json`; `E:\GUOJI\mamba_enhanced\results\raw\mask_supplement_results.json` | intervention sensitivity, not a graph score by itself | concat diagnostic models with x-mask, c-mask, and shuffle interventions | `PARTIALLY COVERED` | Auxiliary route usage can be diagnosed by masking or shuffling the auxiliary channel; it supports fallback-path evidence, not a universal claim about all covariates. |
| coefficient recovery | `E:\GUOJI\mamba_enhanced\diagnostic_results\exp4_coefficient_recovery.json`; generating script `E:\GUOJI\mamba_enhanced\experiments\test_shortcut_diagnostics.py` | Jacobian-derived coefficient-fidelity diagnostic against controlled VAR coefficients | baseline JRNGC and concat diagnostic variants | `PARTIALLY COVERED` | Under controlled VAR diagnostics, shortcut routes can reduce coefficient-level fidelity of the graph score. |
| full auxiliary-Jacobian penalty | `E:\GUOJI\mamba_enhanced\risk_mitigation_results\full_aux_jacobian_penalty.json`; `E:\GUOJI\mamba_enhanced\risk_mitigation_results\full_aux_jacobian_penalty.csv`; script `E:\GUOJI\mamba_enhanced\experiments\risk_mitigation_20260515\run_full_aux_penalty.py` | expanded auxiliary-coordinate Jacobian penalty and diagnostics | concat diagnostic with x-only vs full auxiliary-Jacobian penalty variants | `PARTIALLY COVERED` | Expanding Jacobian coverage mitigated the failure in controlled concat diagnostics. |
| root-cause synthetic | Candidate directories: `E:\GUOJI\mamba_enhanced\results_kbs\root_cause\`, `E:\GUOJI\mamba_enhanced\results_kbs\root_cause_v2\`; figure artifacts `E:\GUOJI\mamba_enhanced\paper-data\figures\fig3_root_cause_main_v2.*`, `fig4_checkpoint_dynamics_v2.*`, `fig7_negative_controls_v2.*`; scripts `E:\GUOJI\mamba_enhanced\experiments\run_root_cause.py`, `run_root_cause_v2.py` | pending audit | pending audit of architecture and score coordinate semantics | `UNASSESSED` | Pending semantics/provenance audit; appendix candidate only. It must not remain A-tier main-text core until provenance and score semantics are explicitly audited. |
| legacy ISTF-Mamba filtered-coordinate/raw-chain comparison | `E:\GUOJI\mamba_enhanced\paper-data\p0_jacobian_semantics_audit_2026-07-06.md`; `E:\GUOJI\mamba_enhanced\results\p0_audit\p0_jacobian_semantics_d6_iter120_refactor_seed0.json` through `seed4.json` | filtered-coordinate score vs raw-chain attribution; leakage diagnostics | legacy cross-channel ISTF-Mamba and related semantic-audit variants | `COORDINATE-AMBIGUOUS` | Filtered-coordinate scores can diverge from original-input raw-chain attribution when cross-channel transformations destroy source-variable identity. |
| Stage 1a | `E:\GUOJI\kbs_review_packages\phase7_stage1a_901_go_nogo_v1.zip`; internal file `formal_root/stage1a_aggregate_go_no_go.json`; release commit `65e6ae9afef552c84d8211a9d6e9aa70db48c276`; source manifest SHA `be91dc2d3ee916690ebcd519d42811f3d3698ef4eddf053d96cdf96d0f4cab3d` | nominal raw-chain for baseline, CP-depthwise, FixedFIR3; full-H reference for EMA | baseline JRNGC, CP-depthwise, FixedFIR3, FixedEMA | `COVERED` for the stated boundary table; EMA is full-H reference | Semantic gates passed, CP performance gate failed, FixedFIR3 novelty gate failed, EMA full-H dominance no-go did not trigger, Stage 1b eligibility failed. |
| P1 | `E:\GUOJI\kbs_review_packages\phase7_stage1a_bounded_failure_analysis_v1.zip`; key files `p1_status.json`, `ab_decision_gates.json`, `gradient_replay_alignment.json`, `filter_movement_gate.json` | retrospective score-map, counterfactual substitution, prediction equivalence, filter audit | formal Stage 1a checkpoints and CP counterfactual substitution; no new method training | `PARTIALLY COVERED` for postmortem support; A3 is invalid | P1 ended as `INCONCLUSIVE_BOUNDED_POSTMORTEM`; A3 was uninterpretable and did not pass because `gradient_replay_alignment_valid=false`. |

## Stage 1a Official Boundary Table

| Cell | Baseline AUROC | CP AUROC | FixedFIR3 AUROC | EMA AUROC | CP - Baseline | FIR3 - CP | EMA - CP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stat+Linear | 0.981383 | 0.981467 | 0.984377 | 0.888714 | +0.000084 | +0.002909 | -0.092753 |
| Stat+Nonlinear | 0.966452 | 0.966455 | 0.968040 | 0.845047 | +0.000002 | +0.001586 | -0.121408 |
| NS+Linear | 0.967320 | 0.965096 | 0.973161 | 0.864107 | -0.002223 | +0.008065 | -0.100990 |
| NS+Nonlinear | 0.947989 | 0.945940 | 0.950484 | 0.820996 | -0.002048 | +0.004543 | -0.124945 |

Permitted interpretation:

- CP-vs-baseline gate failed: no qualifying cells and no Delta AUROC near +0.03.
- CP-vs-FIR3 novelty gate failed: no qualifying cells; FixedFIR3 exceeded CP-depthwise in mean AUROC in all four Stage 1a cells, although not in every individual data seed.
- EMA full-H dominance no-go did not trigger; EMA remained a reference, not a nominal-lag causal method.
- Stage 1b eligibility failed.

## Corrected Evidence Tiering

| Evidence | Tier | Score semantics | Permitted claim | Claim it cannot support |
|---|---|---|---|---|
| d_cond sweep | A. Main-text core after provenance copy/freeze | concat x-only score | prediction-knowledge decoupling in controlled concat diagnostics | universal degradation law |
| mask/shuffle | A. Main-text core after provenance copy/freeze | intervention diagnostic | auxiliary fallback pathway | all auxiliary variables are invalid |
| coefficient recovery | A. Main-text core after provenance copy/freeze | VAR coefficient fidelity | score can lose coefficient meaning | real-data benchmark performance |
| full auxiliary-Jacobian penalty | A. Main-text core after provenance copy/freeze | expanded auxiliary-coordinate penalty | missing route coverage can be mitigated in controlled concat diagnostics | ISTF superiority |
| root-cause synthetic diagnostics | Pending semantics/provenance audit; appendix candidate | unverified for current Route-B score semantics | no main-text claim until audited | structural core evidence, deployment-ready method |
| legacy ISTF-Mamba filtered/raw-chain comparison | D. Semantic-failure diagnostic only | filtered-coordinate legacy score vs raw-chain attribution | score-semantics caution | graph-recovery evidence, ISTF effectiveness, Mamba effectiveness |
| Stage 1a CP/Baseline/FIR3/EMA | A. Main-text boundary table | nominal raw-chain for baseline/CP/FIR3, full-H reference for EMA | semantic repair failed performance/novelty gates | repaired method success |
| P1 A1 | B. Supporting | CP learned filter vs identity counterfactual | learned CP often behaved near identity | full cause of failure |
| P1 A2 | B. Supporting | FIR3 substitution gate | no clean parameterization diagnosis | proof that filtering hypotheses are false |
| P1 A3 | E. Must not be used | invalid gradient replay alignment | only: "A3 was uninterpretable and did not pass because `gradient_replay_alignment_valid=false`" | any gradient-conflict claim |
| P1 B | B. Supporting | postmortem decision gates | bounded postmortem inconclusive | definitive D2 hypothesis failure |
| smoothing baselines | C. Appendix-only | EMA/FIR reference | simple filters are important controls | learned filtering novelty |
| old CausalTime benchmark | Removed from active mainline | legacy benchmark | at most a limitation statement that legacy benchmarks cannot validate new raw-variable score-semantics claims | positive, neutral, or negative operating-regime claim |

## Legacy ISTF-Mamba Boundary

Legacy ISTF-Mamba may appear only as:

- one row in the main audit table, and
- at most one compact diagnostic panel if needed.

It may support only:

> Filtered-coordinate scores can diverge from original-input raw-chain attribution when cross-channel transformations destroy source-variable identity.

It may not support:

- graph-recovery performance;
- benchmark superiority;
- ISTF effectiveness;
- Mamba effectiveness;
- operating-regime claims.

Technical details belong in the appendix.

## Active Mainline Exclusions

Do not include old CausalTime benchmark tables or positive, neutral, or negative operating-regime claims in the main manuscript. At most, retain a brief limitation statement explaining that legacy benchmark results cannot validate the new raw-variable score-semantics claims.

## Minimum Main-Text Evidence Package

1. Fig. 1: coverage mismatch schematic.
2. Fig. 2: controlled concat prediction-knowledge decoupling diagnostics.
3. Fig. 3: score-semantics audit table or compact diagnostic panel, including only the allowed legacy ISTF-Mamba row.
4. Table 1: coverage declaration examples and audit labels.
5. Table 2: Stage 1a boundary table, not a benchmark leaderboard.

## Open Evidence-Provenance Items

1. Root-cause synthetic diagnostics require explicit provenance and score-semantics audit before any main-text use.
2. Historical diagnostic JSONs in `diagnostic_results\` and `risk_mitigation_results\` should be copied or frozen under `paper-data\` before final drafting relies on them as main-text evidence.
3. Legacy ISTF-Mamba semantic audit should be reduced to the permitted filtered-coordinate/raw-chain claim, not repurposed as performance evidence.
