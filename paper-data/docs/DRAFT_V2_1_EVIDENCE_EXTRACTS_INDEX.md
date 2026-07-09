# Route-B Draft v2.1 Frozen Evidence Extracts

The review ZIP contains only frozen source files needed to inspect the figures
and tables cited in the separate v2.1 manuscript. These copies are not new
results and must not be modified, extended, or used to restart experimental
development.

| Package path | Original frozen source | Use and permitted interpretation |
| --- | --- | --- |
| `frozen_evidence/dcond/exp2_dcond_sweep.json` | `mamba_enhanced/diagnostic_results/exp2_dcond_sweep.json` | Fig. 2a; single-run controlled capacity diagnostic. |
| `frozen_evidence/mask/mask_supplement_results.json` | `mamba_enhanced/results/raw/mask_supplement_results.json` | Fig. 2b; single-run auxiliary-route usage diagnostic, not a graph score. |
| `frozen_evidence/coefficient/exp4_coefficient_recovery.json` | `mamba_enhanced/diagnostic_results/exp4_coefficient_recovery.json` | Fig. 2c; single-run controlled coefficient-fidelity diagnostic. |
| `frozen_evidence/full_aux_penalty/full_aux_jacobian_penalty.{json,csv}` | `mamba_enhanced/risk_mitigation_results/` | Fig. 2d; five-seed controlled penalty-route diagnostic. |
| `frozen_evidence/concat_posthoc/concat_posthoc_jacobian.json` | `mamba_enhanced/risk_mitigation_results/concat_posthoc_jacobian.json` | Section 5.2; five-seed route-usage diagnostic only, not a graph score, conditional-Granger result, or performance result. |
| `tools/frozen_provenance/run_concat_posthoc_jacobian.py` | `mamba_enhanced/experiments/risk_mitigation_20260515/run_concat_posthoc_jacobian.py` | Read-only provenance for the preceding frozen post-hoc artifact. |
| `frozen_evidence/p0/p0_jacobian_semantics_d6_iter120_refactor_seed{0,1,2,3,4}.json` | `mamba_enhanced/results/p0_audit/` | Fig. 3; five-seed score-semantics aggregate. |
| `frozen_evidence/stage1a/stage1a_aggregate_go_no_go.json` | formal root in `phase7_stage1a_901_go_nogo_v1.zip` | Tables 2--3; official release-locked boundary result. |
| `frozen_evidence/p1/{p1_status,ab_decision_gates}.json` | `phase7_stage1a_bounded_failure_analysis_v1.zip` | P1 classification and the restricted A3 wording only. |

The P1 extract deliberately excludes counterfactual score/prediction arrays.
No Stage 1b or seeds 4--8 model-training/model-performance outputs are
included. The package preserves the formal Stage 1a and P1 artifacts as
read-only evidence.
