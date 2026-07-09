# Route-B Draft v2 Frozen Evidence Extracts

The review ZIP contains only the frozen source files needed to inspect the
figures and tables cited in the separate manuscript. These copies are not new
results and must not be modified or extended.

| Package path | Original frozen source | Use in the draft |
| --- | --- | --- |
| `frozen_evidence/dcond/exp2_dcond_sweep.json` | `mamba_enhanced/diagnostic_results/exp2_dcond_sweep.json` | Fig. 2a; single-run controlled capacity diagnostic. |
| `frozen_evidence/mask/mask_supplement_results.json` | `mamba_enhanced/results/raw/mask_supplement_results.json` | Fig. 2b; single-run route-sensitivity diagnostic. |
| `frozen_evidence/coefficient/exp4_coefficient_recovery.json` | `mamba_enhanced/diagnostic_results/exp4_coefficient_recovery.json` | Fig. 2c; single-run coefficient-fidelity diagnostic. |
| `frozen_evidence/full_aux_penalty/full_aux_jacobian_penalty.{json,csv}` | `mamba_enhanced/risk_mitigation_results/` | Fig. 2d; five-seed controlled penalty-route diagnostic. |
| `frozen_evidence/p0/seed0.json` through `seed4.json` | `mamba_enhanced/results/p0_audit/` | Fig. 3; five-seed score-semantics aggregate. |
| `frozen_evidence/stage1a/stage1a_aggregate_go_no_go.json` | formal root in `phase7_stage1a_901_go_nogo_v1.zip` | Tables 2--3; official release-locked boundary result. |
| `frozen_evidence/p1/{p1_status,ab_decision_gates}.json` | `phase7_stage1a_bounded_failure_analysis_v1.zip` | P1 classification and restricted A3 wording. |

The P1 evidence extract deliberately excludes counterfactual score/prediction
arrays. The Route-B draft needs only the formal P1 classification and its
interpretation restriction. No Stage 1b or seeds 4--8 model-training/model-
performance outputs are included.
