# Draft v1 Evidence Traceability

Date: 2026-07-10

This register maps every main-text empirical claim to its frozen source. Figure
source data and SHA256 values are additionally recorded in
E:\GUOJI\elsarticle\figures\coverage_audit_draft_v1\figure_data_manifest_draft_v1.json.

| Manuscript location | Claim or value | Exact artifact path | Score semantics | Architecture | Evidence role |
|---|---|---|---|---|---|
| Abstract; Section 5.1; Fig. 3a | Training loss 0.00895 to 0.00211 and AUROC 0.901 to 0.522 across the controlled d_cond sweep. | E:\GUOJI\mamba_enhanced\diagnostic_results\exp2_dcond_sweep.json | x-only Jacobian graph score | Baseline JRNGC and concat-conditioned JRNGC | Controlled route-coverage failure. |
| Section 5.2; Fig. 3b | x-mask, c-mask, both-mask, and shuffle sensitivity values; no-aux x-mask control. | E:\GUOJI\mamba_enhanced\results\raw\mask_supplement_results.json | Intervention sensitivity, not a graph score | Concat-conditioned JRNGC plus matched no-aux comparator | Auxiliary-route diagnostic. |
| Section 5.3; Fig. 3c | Pearson coefficient fidelity, norm ratio, AUROC, and SHD comparison. | E:\GUOJI\mamba_enhanced\diagnostic_results\exp4_coefficient_recovery.json | Jacobian-derived coefficient fidelity and x-only graph metrics | Baseline JRNGC and concat-conditioned JRNGC | Controlled score-meaning failure. |
| Abstract; Sections 4 and 5.4; Fig. 3d | Expanded full auxiliary penalty mitigation. | E:\GUOJI\mamba_enhanced\risk_mitigation_results\full_aux_jacobian_penalty.json; E:\GUOJI\mamba_enhanced\risk_mitigation_results\full_aux_jacobian_penalty.csv | x-only graph recovery and coefficient fidelity under expanded penalty variants | Concat diagnostic variants | Controlled mitigation only. |
| Sections 2.2 and 5.5; Fig. 2 | Partial-total and filtered-coordinate/raw-chain agreement values. | E:\GUOJI\mamba_enhanced\results\p0_audit\p0_jacobian_semantics_d6_iter120_refactor_seed0.json; aggregate interpretation in E:\GUOJI\mamba_enhanced\paper-data\p0_jacobian_semantics_audit_2026-07-06.md | Partial versus total derivative; filtered-coordinate versus raw-chain; leakage diagnostic | Concat diagnostic and legacy cross-channel ISTF-Mamba | Score-semantics diagnostic only. |
| Section 6; Tables 2-3; Appendix C | Formal run count, semantic-gate status, cell means, performance/novelty outcome, EMA no-go status, Stage 1b ineligibility. | E:\GUOJI\kbs_review_packages\phase7_stage1a_901_go_nogo_v1.zip, internal path formal_root/stage1a_aggregate_go_no_go.json | Nominal raw-chain for baseline, CP-depthwise, FixedFIR3; full-H reference for FixedEMA | Baseline JRNGC, CP-depthwise, FixedFIR3, FixedEMA | Preregistered boundary case. |
| Section 6.2; Appendix C | A1 supports near-identity behavior; A2 did not pass; A3 restricted wording; final decision inconclusive. | E:\GUOJI\kbs_review_packages\phase7_stage1a_bounded_failure_analysis_v1.zip, internal files p1_status.json, ab_decision_gates.json, gradient_replay_alignment.json, filter_movement_gate.json | Retrospective score-map, prediction, and substitution analyses | Formal Stage 1a checkpoints and CP predictor substitutions | Bounded postmortem only. |

## Figure Source Files

| Draft figure | Source script | Generated files | Data manifest |
|---|---|---|---|
| Fig. 1 coverage mismatch | E:\GUOJI\mamba_enhanced\tools\generate_coverage_audit_draft_v1_figures.py | E:\GUOJI\elsarticle\figures\coverage_audit_draft_v1\fig1_coverage_mismatch_draft_v1.pdf, .svg, .png | E:\GUOJI\elsarticle\figures\coverage_audit_draft_v1\figure_data_manifest_draft_v1.json |
| Fig. 2 score semantics | E:\GUOJI\mamba_enhanced\tools\generate_coverage_audit_draft_v1_figures.py | E:\GUOJI\elsarticle\figures\coverage_audit_draft_v1\fig3_score_semantics_audit_draft_v1.pdf, .svg, .png | Same manifest. |
| Fig. 3 controlled concat diagnostics | E:\GUOJI\mamba_enhanced\tools\generate_coverage_audit_draft_v1_figures.py | E:\GUOJI\elsarticle\figures\coverage_audit_draft_v1\fig2_controlled_concat_diagnostics_draft_v1.pdf, .svg, .png | Same manifest. |
