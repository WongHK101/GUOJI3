# Route-B Submission Candidate v2.4 Frozen Evidence Extracts

The review ZIP contains frozen source files needed to inspect the figures and
tables cited in the separate v2.4 candidate. These copies are not new results
and must not be modified, extended, or used to restart experimental
development. V2.4 adds read-only source inspection and frozen-data reporting;
its empirical inputs and numerical evidence are unchanged.

| Package path | Original frozen source | Use and permitted interpretation |
| --- | --- | --- |
| `frozen_evidence/dcond/exp2_dcond_sweep.json` | `mamba_enhanced/diagnostic_results/exp2_dcond_sweep.json` | Fig. 2a; single-run controlled capacity diagnostic. |
| `frozen_evidence/mask/mask_supplement_results.json` | `mamba_enhanced/results/raw/mask_supplement_results.json` | Fig. 2b; single-run auxiliary-route usage diagnostic, not a graph score. |
| `frozen_evidence/coefficient/exp4_coefficient_recovery.json` | `mamba_enhanced/diagnostic_results/exp4_coefficient_recovery.json` | Fig. 2c; single-run controlled coefficient-fidelity diagnostic. |
| `frozen_evidence/full_aux_penalty/full_aux_jacobian_penalty.{json,csv}` | `mamba_enhanced/risk_mitigation_results/` | Fig. 2d and appendix table; five-seed exploratory penalty-route diagnostic with every frozen variant disclosed. |
| `frozen_evidence/concat_posthoc/concat_posthoc_jacobian.json` | `mamba_enhanced/risk_mitigation_results/concat_posthoc_jacobian.json` | Section 5.2; five-seed mean absolute Jacobian-magnitude route-use diagnostic. It is coordinate-scale dependent and is not a graph score, causal contribution, conditional-Granger estimate, benchmark, performance, or confirmatory result. |
| `tools/frozen_provenance/run_concat_posthoc_jacobian.py` | `mamba_enhanced/experiments/risk_mitigation_20260515/run_concat_posthoc_jacobian.py` | Read-only provenance for the preceding frozen post-hoc artifact. |
| `tools/frozen_provenance/{mamba_jrngc_pilot.py,minimal_mamba.py,run_full_aux_penalty.py}` | Source-verified controlled architecture and full-penalty implementation. | Read-only definition of `c=g_phi(X)`, the predictor tensor, detach behavior, exact derivative coordinates, and all penalty variants. |
| `tables/coverage_audit_submission_v2_4/full_aux_penalty_all_variants_v2_4.{csv,tex}` | Deterministic export from the frozen full-penalty JSON. | All-variant appendix table source; no new computation beyond formatting saved summaries. |
| `frozen_evidence/p0/p0_jacobian_semantics_d6_iter120_refactor_seed{0,1,2,3,4}.json` | `mamba_enhanced/results/p0_audit/` | Fig. 3; five-seed score-semantics aggregate. |
| `frozen_evidence/stage1a/stage1a_aggregate_go_no_go.json` | formal root in `phase7_stage1a_901_go_nogo_v1.zip` | Tables 2--3; official release-locked boundary result. |
| `frozen_evidence/p1/{p1_status,ab_decision_gates}.json` | `phase7_stage1a_bounded_failure_analysis_v1.zip` | P1 classification and the restricted A3 wording only. |

The P1 extract deliberately excludes counterfactual score/prediction arrays.
P1 did not inspect or use Stage 1b model-training or model-performance outputs
for seeds 4--8. The package preserves the formal Stage 1a and P1 artifacts as
read-only evidence.
