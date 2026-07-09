# KBS Jacobian Coverage Audit Claim-Evidence Matrix v1.2

Date: 2026-07-10

Status: full draft v1 evidence register. This document records the claims that
entered the separate Route-B manuscript draft. It uses frozen artifacts only.
It does not authorize new training, Stage 1b, inspection of seeds 4-8, or
replacement of the canonical KBS manuscript.

## Locked Terminology

| Canonical term | Draft usage | Do not use as a substitute |
|---|---|---|
| Jacobian coverage audit framework | A declaration and diagnostic procedure for checking score variables, penalty variables, declared route classes, coordinate identity, and attribution horizon. | coverage certificate; causal-identifiability guarantee |
| Prediction-knowledge decoupling | Prediction loss can improve while a reported graph score loses the declared raw-variable meaning. | proof that prediction is uninformative |
| Raw-chain attribution | Derivative with respect to raw input through declared differentiable transformations within an audited horizon. | full attribution when earlier support was not audited |
| Legacy cross-channel ISTF-Mamba | Historical score-semantics diagnostic. | graph-recovery evidence; Mamba or ISTF performance evidence |
| CP-depthwise semantic repair | Preregistered coordinate-preserving candidate in Stage 1a. | successful method; competitive main method |

## Draft Main-Text Claims

| ID | Draft claim | Exact frozen artifact | Score semantics | Architecture | Draft location | Audit label | Permitted boundary |
|---|---|---|---|---|---|---|---|
| C1 | Auxiliary predictive routes can create prediction-knowledge decoupling when an x-only score or penalty omits them. | E:\GUOJI\mamba_enhanced\diagnostic_results\exp2_dcond_sweep.json; E:\GUOJI\mamba_enhanced\diagnostic_results\mask_shuffle_results.json; E:\GUOJI\mamba_enhanced\results\raw\mask_supplement_results.json | x-only Jacobian graph score; intervention sensitivity | Baseline JRNGC and concat-conditioned JRNGC variants | Sections 3 and 5; Fig. 3a-b | PARTIALLY COVERED | Controlled concat architecture only; not all auxiliary variables or conditioning designs. |
| C2 | In the d_cond sweep, auxiliary capacity lowered training loss while degrading x-only graph recovery. | E:\GUOJI\mamba_enhanced\diagnostic_results\exp2_dcond_sweep.json | x-only Jacobian graph score | Baseline JRNGC and concat-conditioned JRNGC | Section 5.1; Fig. 3a | PARTIALLY COVERED | Not a monotonic law or benchmark-wide claim. |
| C3 | In controlled coefficient recovery, the concat x-only score loses coefficient fidelity. | E:\GUOJI\mamba_enhanced\diagnostic_results\exp4_coefficient_recovery.json | Jacobian-derived coefficient fidelity and graph metrics | Baseline JRNGC and concat-conditioned JRNGC | Section 5.3; Fig. 3c | PARTIALLY COVERED | Controlled VAR diagnostic only. |
| C4 | Expanding Jacobian coverage mitigated the failure in controlled concat diagnostics. | E:\GUOJI\mamba_enhanced\risk_mitigation_results\full_aux_jacobian_penalty.json; E:\GUOJI\mamba_enhanced\risk_mitigation_results\full_aux_jacobian_penalty.csv | x-only graph recovery and coefficient fidelity under expanded auxiliary penalty | Concat x-only and full auxiliary-penalty variants | Sections 4 and 5.4; Fig. 3d | PARTIALLY COVERED | Does not solve all cases or establish filtering superiority. |
| C5 | Filtered-coordinate scores can diverge from original-input raw-chain attribution when cross-channel transformations destroy source-variable identity. | E:\GUOJI\mamba_enhanced\paper-data\p0_jacobian_semantics_audit_2026-07-06.md; E:\GUOJI\mamba_enhanced\results\p0_audit\p0_jacobian_semantics_d6_iter120_refactor_seed0.json through seed4.json | Filtered-coordinate score versus raw-chain attribution; leakage diagnostic | Legacy cross-channel ISTF-Mamba; concat partial/total comparison | Sections 2.2 and 5.5; Fig. 2 | COORDINATE-AMBIGUOUS | Semantic diagnostic only. No graph-recovery performance, benchmark superiority, ISTF effectiveness, Mamba effectiveness, or operating-regime use. |
| C6 | Coordinate-preserving repair passed semantic gates but failed performance and FixedFIR3 novelty gates. | E:\GUOJI\kbs_review_packages\phase7_stage1a_901_go_nogo_v1.zip, internal path formal_root/stage1a_aggregate_go_no_go.json | Nominal raw-chain for baseline, CP-depthwise, and FixedFIR3; full-H reference for FixedEMA | Baseline JRNGC, CP-depthwise, FixedFIR3, FixedEMA | Section 6; Tables 2-3 | COVERED for stated boundary | Boundary table, not a benchmark leaderboard or a successful method evaluation. |
| C7 | FixedFIR3 exceeded CP-depthwise in cell-level mean AUROC in all four Stage 1a cells, though not in every data seed. | E:\GUOJI\kbs_review_packages\phase7_stage1a_901_go_nogo_v1.zip, internal path formal_root/stage1a_aggregate_go_no_go.json | Same as C6 | Same as C6 | Section 6; Table 2 | COVERED for stated aggregate | Do not claim dominance in every individual seed. |
| C8 | The bounded P1 postmortem remained inconclusive. | E:\GUOJI\kbs_review_packages\phase7_stage1a_bounded_failure_analysis_v1.zip, internal files p1_status.json, ab_decision_gates.json, gradient_replay_alignment.json, filter_movement_gate.json | Retrospective score-map, prediction, and substitution diagnostics | Formal Stage 1a checkpoints and CP counterfactual substitutions | Section 6.2; Appendix C | PARTIALLY COVERED | A3 wording is restricted to: A3 was uninterpretable and did not pass because gradient_replay_alignment_valid=false. |

## Excluded or Appendix-Only Material

| Material | Exact location | Status | Reason |
|---|---|---|---|
| Root-cause synthetic diagnostics | E:\GUOJI\mamba_enhanced\results_kbs\root_cause\; E:\GUOJI\mamba_enhanced\results_kbs\root_cause_v2\; E:\GUOJI\mamba_enhanced\paper-data\figures\fig3_root_cause_main_v2.* | UNASSESSED | Provenance and score semantics require a separate read-only audit before any use. |
| Old CausalTime performance and operating-regime material | E:\GUOJI\elsarticle\istf_kbs.tex and legacy result tables | Excluded from active mainline | It cannot validate the new raw-variable score-semantics claims. |
| Legacy ISTF-Mamba benchmark material | Legacy benchmark result tables and figures | Semantic diagnostic only | It cannot support efficacy, graph-recovery, or deployment claims. |

## Absolute Claim Prohibitions

- No claim that ISTF, Mamba, TCN, or CP-depthwise improves neural GC benchmarks.
- No claim that the coverage audit framework is a certificate, guarantee, or identifiability proof.
- No claim that all conditioning or auxiliary variables are harmful.
- No A3 raw-branch quantities, tendencies, conflict counts, or suggested pass status.
- No Stage 1b result, no use of seeds 4-8, and no unregistered new experiment.
