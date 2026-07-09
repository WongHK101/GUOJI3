# Jacobian Coverage Audit Manuscript: Claim-Evidence Matrix v2

## Scope and terminology ledger

This matrix governs the separate Route-B manuscript draft `istf_kbs_jacobian_coverage_draft_v2.tex`. It is not a replacement for the canonical KBS manuscript. All empirical entries below are frozen artifacts. No Stage 1b outputs, seeds 4-8 model-training or model-performance outputs, new training, or GPU runs were used.

| Canonical term | Definition used in v2 | Do not substitute with |
| --- | --- | --- |
| directed Granger-predictive dependency graph | The graph induced by an explicitly stated predictive derivative and aggregation rule. | General causal graph, unless stronger assumptions are stated. |
| Granger graph knowledge | The reported directed Granger-predictive dependency object. | Causal knowledge without qualification. |
| Jacobian coverage audit | A claim-specific audit of declared variables, route classes, coordinate map, and attribution horizon. | Coverage certificate or identifiability proof. |
| route ledger | `P_pred={(r,rho_score(r),rho_penalty(r),exempt(r))}` for architecture-declared route classes. | Enumeration of every activated neural path. |
| CLAIM-COVERED | Support for one declared graph-score claim under the stated architecture, horizon, and evidence. | Model-wide guarantee. |

## Central claims

| ID | Claim | Frozen evidence | Strength and permitted wording | Forbidden reading |
| --- | --- | --- | --- | --- |
| C1 | An auxiliary route omitted from an x-only score or penalty can create prediction-knowledge decoupling in a concat JRNGC architecture. | Auxiliary-capacity sweep, mask/shuffle, coefficient recovery, and auxiliary-route proposition. | Controlled architecture only. Use `can create` and `controlled diagnostic`. | All conditioning is invalid or all auxiliary variables cause shortcuts. |
| C2 | The five-part declaration and route ledger operationalize four distinct audit dimensions. | Formal definition, Algorithm 1, audit table, and reusable template. | Procedural framework. | Complete neural-path enumeration or causal identifiability theorem. |
| C3 | The d_cond sweep lowered training loss from 0.00895 to 0.00211 while x-only AUROC reached 0.350 and was 0.522 at d_cond=16. | `diagnostic_results/exp2_dcond_sweep.json`. | Single-run controlled diagnostic. AUROC is explicitly non-monotonic. | Statistical generality or a monotonic capacity law. |
| C4 | Mask and shuffle interventions support auxiliary-route usage in the declared concat construction. | `results/raw/mask_supplement_results.json`. | Single-run intervention diagnostic, not a graph score. | A performance comparison or universal auxiliary-path conclusion. |
| C5 | The x-only concat score can lose coefficient fidelity in a controlled VAR diagnostic. | `diagnostic_results/exp4_coefficient_recovery.json`. | Single-run controlled coefficient-fidelity evidence. | Benchmark generalization. |
| C6 | Adding the auxiliary route to the regularizer mitigated x-only graph degradation in the controlled concat diagnostic. | `risk_mitigation_results/full_aux_jacobian_penalty.json` and `.csv`. | Five-seed mean +/- SD. Penalty-route coverage expands; the graph score remains x-only, so score-route completeness is partial. | Coverage-complete repair, filtering superiority, or general mitigation guarantee. |
| C7 | Filtered-coordinate scores can diverge from original-input raw-chain attribution when cross-channel transformations destroy source-variable identity. | Five frozen P0 semantic-audit JSON files. | Five-seed semantic diagnostic only. | Legacy ISTF-Mamba graph recovery, benchmark superiority, Mamba effectiveness, or an operating-regime claim. |
| C8 | Semantic correctness did not establish CP-depthwise as a competitive method in the pre-specified and release-locked Stage 1a. | Official Stage 1a aggregate. | Boundary table: semantic gates passed; performance and FIR3 novelty gates failed; EMA dominance no-go did not trigger; Stage 1b ineligible. | Benchmark leaderboard, positive CP result, or a general failure of all filtering hypotheses. |
| C9 | P1 did not resolve the mechanism of the CP failure. | Frozen P1 status and decision files. | `INCONCLUSIVE_BOUNDED_POSTMORTEM`. A3 is uninterpretable and did not pass because `gradient_replay_alignment_valid=false`. | Any A3 raw-branch trend or definitive filtering-hypothesis conclusion. |

## Diagnostic replication and semantic provenance

| Artifact | Exact frozen source | Generating script | Architecture and configuration | Score or diagnostic | Replication status | Permitted claim |
| --- | --- | --- | --- | --- | --- | --- |
| d_cond sweep | `E:\GUOJI\mamba_enhanced\diagnostic_results\exp2_dcond_sweep.json` | `experiments/test_shortcut_diagnostics.py::exp2_dcond_sweep` | VAR(1), d=8, T=300, lag=1, generator seed 42; BaselineJRNGC or concat MambaJRNGC, 3 layers, hidden=32, d_state=4, jacobian_lam=0.01, 1,500 iterations, model seed 42; d_cond={0,1,2,4,8,16}. | x-only Jacobian graph score. | Single-run controlled diagnostic. | Loss-score decoupling under this concat architecture. |
| mask/shuffle | `E:\GUOJI\mamba_enhanced\results\raw\mask_supplement_results.json` | `experiments/test_mask_supplement.py` | NSVAR d=10, lag=7, data seed 0; concat MambaJRNGC and no-aux control, 5 layers, hidden=50, d_cond=4, jacobian_lam=0.01, 2,000 iterations, model seed 0. | Intervention sensitivity: mask x, mask c, mask both, shuffle x. | Single-run controlled diagnostic. | Auxiliary-route usage in the declared concat model. |
| coefficient recovery | `E:\GUOJI\mamba_enhanced\diagnostic_results\exp4_coefficient_recovery.json` | `experiments/test_shortcut_diagnostics.py::exp4_coefficient_recovery` | VAR(1), d=8, T=500, lag=1, generator and model seed 42; baseline and concat MambaJRNGC, 3 layers, hidden=32, d_state=4, jacobian_lam=0.01, 2,000 iterations. | x-only Jacobian coefficient fidelity against known VAR coefficients. | Single-run controlled diagnostic. | Controlled loss of coefficient meaning. |
| full auxiliary penalty | `E:\GUOJI\mamba_enhanced\risk_mitigation_results\full_aux_jacobian_penalty.json`; `...\full_aux_jacobian_penalty.csv` | `experiments/risk_mitigation_20260515/run_full_aux_penalty.py` | VAR(1), d=8, T=500, lag=1, five seeds, 2,000 iterations; concat model uses d_cond=4 and compares x-only versus full auxiliary Jacobian penalty. | x-only graph AUROC and coefficient fidelity with expanded penalty-route coverage. | Five-seed controlled diagnostic; figure uses mean +/- population SD as saved by the artifact. | Controlled mitigation only. |
| P0 score-semantics audit | `E:\GUOJI\mamba_enhanced\results\p0_audit\p0_jacobian_semantics_d6_iter120_refactor_seed{0,1,2,3,4}.json` | `experiments/p0_jacobian_semantics_audit.py` | Controlled CPU smoke, d=6, T=100, lag=3, max_iter=120, 8 fixed windows; concat partial-versus-total and legacy cross-channel Mamba filtered-coordinate-versus-raw-chain comparisons. | Off-diagonal score correlation, exact-top-k Jaccard, and leakage diagnostics. | Five fixed diagnostic seeds; Figure 3 aggregates the same files. | Score-semantics boundary, not performance evidence. |
| Stage 1a | `E:\GUOJI\kbs_review_packages\phase7_stage1a_901_go_nogo_v1.zip`, `formal_root/stage1a_aggregate_go_no_go.json` | Frozen release-locked Stage 1a runner and aggregator. | D2 four cells; data seeds 1,2,3; train seeds 0,1; checkpoint 500; 96 formal runs plus four limited RawChainMamba diagnostics. | Nominal raw-chain tracks for baseline/CP/FIR3; full-H reference for EMA. | Formal pre-specified and release-locked boundary artifact. | Semantic-pass/performance-fail and novelty-fail boundary. |
| P1 | `E:\GUOJI\kbs_review_packages\phase7_stage1a_bounded_failure_analysis_v1.zip` | `experiments/stage1a_bounded_failure_analysis.py` in analysis package. | Retrospective analysis of frozen Stage 1a artifacts and approved development material. | Counterfactual substitutions, score/prediction equivalence, filter audit, and bounded decision gates. | Supporting postmortem only. | Near-identity support and inconclusive classification. |

## Explicit exclusions

- Root-cause synthetic results remain `UNASSESSED` pending semantics and provenance audit and are not used in the Route-B main text.
- Historical benchmark and operating-regime material is excluded from the active mainline.
- Legacy ISTF-Mamba is retained only as a score-semantics diagnostic.
- No Stage 1b outputs or seeds 4-8 model-training or model-performance outputs were inspected or used.

