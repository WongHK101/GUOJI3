# Route-B Draft v2 Evidence Traceability Register

## Scope

This register is the provenance companion for `istf_kbs_jacobian_coverage_draft_v2.tex`. It records the only empirical artifacts cited by the separate v2 draft. Hashes are recomputed when the review package is assembled. The canonical `E:\GUOJI\elsarticle\istf_kbs.tex` is not an input or output of this draft.

| Manuscript location | Claim or visual element | Frozen artifact | Score coordinate / metric | Architecture | Evidence status |
| --- | --- | --- | --- | --- | --- |
| Abstract; Section 5.1; Fig. 2a | Training loss 0.00895 to 0.00211; AUROC minimum 0.350 and d_cond=16 value 0.522. | `E:\GUOJI\mamba_enhanced\diagnostic_results\exp2_dcond_sweep.json` | x-only Jacobian graph score. | BaselineJRNGC and concat MambaJRNGC. | Single-run controlled diagnostic. |
| Section 5.2; Fig. 2b | x-mask, c-mask, both-mask, shuffle sensitivity, and no-aux control. | `E:\GUOJI\mamba_enhanced\results\raw\mask_supplement_results.json` | Intervention sensitivity, not a graph score. | Concat MambaJRNGC and no-aux input-space control. | Single-run controlled diagnostic. |
| Section 5.3; Fig. 2c | Coefficient Pearson r, norm ratio, AUROC. | `E:\GUOJI\mamba_enhanced\diagnostic_results\exp4_coefficient_recovery.json` | x-only Jacobian coefficient-fidelity and graph metrics. | BaselineJRNGC and concat MambaJRNGC. | Single-run controlled diagnostic. |
| Abstract; Sections 4, 5.4, 7, 8; Fig. 2d | Full auxiliary penalty mitigation and error bars. | `E:\GUOJI\mamba_enhanced\risk_mitigation_results\full_aux_jacobian_penalty.json`; `.csv` | x-only graph AUROC and coefficient fidelity; auxiliary route is added to penalty only. | Concat variants with d_cond=4. | Five-seed controlled diagnostic; score-route completeness remains partial. |
| Section 2; Section 5.5; Fig. 3 | Partial/total and filtered/raw-chain disagreement. | `E:\GUOJI\mamba_enhanced\results\p0_audit\p0_jacobian_semantics_d6_iter120_refactor_seed{0,1,2,3,4}.json` | Partial x-only versus total raw derivative; legacy filtered-coordinate versus raw-chain score. | Concat and legacy cross-channel ISTF-Mamba. | Five-seed semantic diagnostic only. |
| Section 4; Table 1; Algorithm 1; Appendix template | Coverage declaration, route ledger, label taxonomy, and procedure. | Formal manuscript construction; examples constrained by the frozen artifacts above. | Claim-specific audit procedure. | Architecture-declared route classes. | Conceptual/procedural contribution; not a guarantee. |
| Section 6; Tables 2-3 | Semantic pass, performance/no-go outcome, cell-level AUROC and SD. | `E:\GUOJI\kbs_review_packages\phase7_stage1a_901_go_nogo_v1.zip`, `formal_root/stage1a_aggregate_go_no_go.json`. | Nominal raw-chain scores for baseline/CP/FIR3 and full-H reference for EMA. | BaselineJRNGC, CP-depthwise, FixedFIR3, FixedEMA; RawChainMamba limited diagnostic. | Pre-specified and release-locked formal boundary artifact. |
| Section 6.2; Appendix protocol | P1 classification and A3 restriction. | `E:\GUOJI\kbs_review_packages\phase7_stage1a_bounded_failure_analysis_v1.zip`, `p1_status.json`, `p1_ab_decision_gates.json`. | Retrospective analysis only. | Frozen CP checkpoints and approved development material. | Supporting postmortem; `INCONCLUSIVE_BOUNDED_POSTMORTEM`. |

## Non-negotiable interpretation limits

1. The full auxiliary-Jacobian experiment expands penalty-route coverage. Its graph score remains x-only. It is not a coverage-complete repair.
2. Figure 3 uses a five-seed aggregate, not a seed-0 illustration.
3. Legacy cross-channel ISTF-Mamba supports only the score-semantics warning about filtered-coordinate and raw-chain divergence.
4. P1 wording is restricted to: `A3 was uninterpretable and did not pass because gradient_replay_alignment_valid=false.`
5. P1 did not inspect or use Stage 1b model-training or model-performance outputs for seeds 4-8.

