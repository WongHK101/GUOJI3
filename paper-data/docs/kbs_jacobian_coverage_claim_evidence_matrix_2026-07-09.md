# KBS Jacobian Coverage Audit - Claim Evidence Matrix

Date: 2026-07-09

Status: planning and writing asset. This document uses frozen artifacts only. It is not a new experiment, does not inspect Stage 1b seeds, and does not revive CP-depthwise as a main method.

## Locked Narrative

One-sentence argument:

> In Jacobian-regularized neural Granger causality, we show that predictive pathways outside the scored or penalized Jacobian can decouple prediction accuracy from graph knowledge, using controlled shortcut diagnostics, score-semantics audits, and a preregistered semantic-repair negative result; the boundary is that this is a reliability/audit contribution, not an ISTF performance claim.

Canonical terms:

| Canonical term | Meaning | Forbidden drift |
|---|---|---|
| Jacobian coverage audit framework | A declaration and diagnostic procedure for checking whether score variables, penalty variables, predictive paths, coordinate identity, and attribution horizon are aligned. | coverage certificate, guaranteed reliability |
| Prediction-knowledge decoupling | The predictor can optimize loss while the graph score loses causal meaning. | proof that prediction is useless |
| Auxiliary predictive pathway | A side path such as concat conditioning or external covariates that may influence prediction outside the scored raw-input Jacobian. | all auxiliary inputs are invalid |
| Raw-chain attribution | Derivative of prediction with respect to raw input through all differentiable transformations within a chosen horizon. | full attribution unless the full support is covered |
| CP-depthwise semantic repair | The preregistered coordinate-preserving depthwise candidate evaluated in Stage 1a. | successful repaired method |

## Main Claims And Evidence

| Claim | Evidence | Artifact / source | Evidence tier | Supported wording | Unsupported wording |
|---|---|---|---|---|---|
| Auxiliary side channels can create a structural shortcut in JRNGC. | Shortcut theorem; concat setup; d_cond sweep; mask-c/mask-x diagnostics. | `istf_kbs.tex` Sections 3 and 5; frozen diagnostic figures/tables. | Main-text core | "can create", "under this architecture", "structural vulnerability" | "always create", "all conditioning is harmful" |
| Prediction loss and graph recovery can decouple under concat conditioning. | d_cond sweep: lower training loss with degraded AUROC at auxiliary dimensions; coefficient fidelity collapse. | Existing controlled diagnostic results in current KBS manuscript and figure sources. | Main-text core | "loss decreased while AUROC/coefficient fidelity degraded in controlled diagnostics" | "monotonic law", "benchmark-wide proof" |
| Graph scores require explicit route coverage. | Partial-vs-total derivative mismatch; full auxiliary-Jacobian penalty improves diagnostics; raw-chain audits. | Risk-mitigation diagnostics; P0/P0.3 semantic audits. | Main-text core | "scores should declare covered variables and paths" | "full penalty solves all cases" |
| Filtered-coordinate scores cannot be used as raw-variable graph evidence when coordinate identity is not preserved. | Legacy ISTF-Mamba filtered-coordinate vs raw-chain mismatch and leakage. | Historical P0 semantic audit and handoff record. | Historical diagnostic only in main text, no performance claim | "legacy results illustrate score-semantics risk" | "legacy ISTF-Mamba improves graph recovery" |
| Coordinate-preserving repair can fix score semantics without improving performance. | Stage 1a: semantic gates passed; CP-vs-baseline and CP-vs-FIR3 performance/novelty gates failed. | `phase7_stage1a_901_go_nogo_v1.zip`, `formal_root/stage1a_aggregate_go_no_go.json`. | Main-text core | "semantic repair is necessary but insufficient" | "CP works", "Stage 1a is positive" |
| FixedFIR3 exceeded CP in cell-level mean AUROC in all four cells. | Official aggregation table: FIR3 - CP AUROC = +0.002909, +0.001586, +0.008065, +0.004543. | Official Stage 1a aggregate. | Main-text core | "in cell-level mean AUROC" | "in every individual seed" |
| EMA dominance no-go did not trigger. | Official `ema_reference_dominance.no_go_triggered=false`; EMA AUROC lower than CP in all four cells. | Official Stage 1a aggregate. | Supporting precision | "EMA full-H reference did not dominate CP in this Stage 1a gate" | "EMA is a main causal method" |
| P1 postmortem did not identify a definitive failure class. | Final decision `INCONCLUSIVE_BOUNDED_POSTMORTEM`; A1 passed, A2 failed, A3 invalid, B incomplete. | `phase7_stage1a_bounded_failure_analysis_v1.zip`. | Supporting boundary | "bounded postmortem remained inconclusive" | "identity-gradient conflict explains failure" |

## Stage 1a Official Cell-Level Table

| Cell | Baseline AUROC | CP AUROC | FixedFIR3 AUROC | EMA AUROC | CP - Baseline | FIR3 - CP | EMA - CP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stat+Linear | 0.981383 | 0.981467 | 0.984377 | 0.888714 | +0.000084 | +0.002909 | -0.092753 |
| Stat+Nonlinear | 0.966452 | 0.966455 | 0.968040 | 0.845047 | +0.000002 | +0.001586 | -0.121408 |
| NS+Linear | 0.967320 | 0.965096 | 0.973161 | 0.864107 | -0.002223 | +0.008065 | -0.100990 |
| NS+Nonlinear | 0.947989 | 0.945940 | 0.950484 | 0.820996 | -0.002048 | +0.004543 | -0.124945 |

Permitted interpretation:

- CP-vs-baseline gate failed: no qualifying cells and no Delta AUROC near +0.03.
- CP-vs-FIR3 novelty gate failed: no qualifying cells; FixedFIR3 was higher in all cell-level mean AUROCs.
- EMA full-H dominance no-go did not trigger; EMA remained a reference, not a nominal-lag causal method.

## Evidence Tiering

| Evidence | Tier | Score semantics | Claim it supports | Claim it cannot support |
|---|---|---|---|---|
| d_cond sweep | A. Main-text core | concat x-only score | prediction-knowledge decoupling | universal degradation law |
| mask/shuffle | A. Main-text core | intervention diagnostic | auxiliary fallback pathway | all auxiliary variables are invalid |
| coefficient recovery | A. Main-text core | VAR coefficient fidelity | score can lose coefficient meaning | real-data performance |
| full auxiliary-Jacobian penalty | A. Main-text core | expanded auxiliary-coordinate penalty | missing route coverage can mitigate shortcut | ISTF superiority |
| root-cause synthetic diagnostics | A. Main-text core | controlled synthetic graphs | structural shortcut mechanism | deployment-ready method |
| Stage 1a CP/Baseline/FIR3/EMA | A. Main-text core | nominal raw-chain for baseline/CP/FIR3, full-H reference for EMA | semantic repair failed performance/novelty gates | repaired method success |
| P1 A1 | B. Main-text supporting | CP learned filter vs identity counterfactual | learned CP was near identity in many runs | full explanation of failure |
| P1 A2 | B. Main-text supporting | FIR3 substitution gate | no clean parameterization diagnosis | filtering hypothesis false |
| P1 A3 | E. Must not be used | invalid gradient replay alignment | only: "A3 was uninterpretable and did not pass because gradient_replay_alignment_valid=false" | any gradient-conflict claim |
| P1 B | B. Main-text supporting | postmortem decision gates | bounded postmortem inconclusive | definitive D2 hypothesis failure |
| smoothing baseline | C. Appendix-only | EMA/FIR reference | simple filters are important controls | learned filtering novelty |
| orthogonality ablation | D. Historical diagnostic only | legacy ISTF semantics | design-risk lesson | final method evidence |
| old CausalTime benchmark | D. Historical diagnostic only / appendix | legacy benchmark | operating-boundary context if carefully caveated | performance claim |
| legacy ISTF-Mamba benchmark | E for performance, D for semantic failure | filtered-coordinate legacy score | score-semantics caution | graph-recovery evidence |

## Manuscript Claim Boundaries

Use:

- "Jacobian coverage audit framework"
- "route-coverage audit"
- "prediction-knowledge decoupling"
- "semantic repair is necessary but not sufficient"
- "preregistered boundary evidence"

Avoid:

- "coverage certificate"
- "ISTF improves neural GC"
- "Mamba/TCN is the main repair"
- "deployment framework"
- "orthogonality certificate"
- "A3 raw branch"
- "FixedFIR3 beats CP in every seed"

## Minimum Main-Text Evidence Package

1. Fig. 1: schematic of auxiliary predictive path vs scored/penalized path.
2. Fig. 2: controlled diagnostics linking prediction loss, graph AUROC, mask sensitivity, and coefficient fidelity.
3. Fig. 3: score-semantics taxonomy and route-coverage audit examples.
4. Table 1: coverage declaration examples across JRNGC, concat, full-aux penalty, legacy filtered-coordinate ISTF, and CP-depthwise.
5. Table 2: Stage 1a official cell-level AUROC and gate interpretation.

## Open Items For Advisor Approval

1. Whether the main title should be the direct audit title or a stronger prediction-knowledge title.
2. Whether legacy ISTF-Mamba appears only as a semantic-failure diagnostic or is removed from the main text.
3. Whether Stage 1a belongs as a main-text boundary table.
4. Whether the manuscript should be rebuilt from the skeleton instead of patching `istf_kbs.tex`.
