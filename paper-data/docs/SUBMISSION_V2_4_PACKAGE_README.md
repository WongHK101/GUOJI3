# Phase 7 Jacobian Coverage Submission Candidate v2.4

## Purpose

This package is a separate KBS submission-readiness candidate built from the
accepted v2.3 science lock. It applies critical correctness and submission-level
editorial revisions only. It does not replace canonical `istf_kbs.tex`.

## Review priorities

1. Confirm the source-verified controlled concat definition, including
   `c=g_phi(X)`, prefix support, predictor tensor, detach points, partial score,
   partial penalty and expanded-penalty coordinates.
2. Confirm that `3.689 +/- 0.750` is called a ratio of blockwise mean absolute
   Jacobian magnitudes and is explicitly coordinate-scale dependent.
3. Confirm disclosure of all frozen full-penalty variants and the exploratory,
   non-optimality framing.
4. Confirm single-run disclosure before abstract numerical values.
5. Confirm the closest-work distinction from general saliency sanity checks,
   faithfulness evaluation and explanation regularization.
6. Confirm the boundary study and retrospective postmortem remain scientifically
   bounded and are not presented as a positive method result.

## Main files

- `istf_kbs_jacobian_coverage_submission_candidate_v2_4.{tex,pdf,log}`
- `CONTROLLED_CONCAT_ARCHITECTURE_V2_4.md`
- `kbs_jacobian_coverage_claim_evidence_matrix_v2_4_2026-07-10.md`
- `SUBMISSION_V2_4_EVIDENCE_TRACEABILITY.md`
- `DIAGNOSTIC_REPLICATION_AUDIT_V2_4.md`
- `BIBLIOGRAPHY_AUDIT_V2_4.md`
- `V2.4_CRITICAL_SCIENTIFIC_CORRECTIONS.md`
- `V2.4_SUBMISSION_EDITORIAL_CHANGELOG.md`
- `V2.4_REMAINING_OPTIONAL_WORK.md`
- `V2.4_JOURNAL_STRATEGY.md`
- `figures/coverage_audit_submission_v2_4/`
- `tables/coverage_audit_submission_v2_4/`
- `tools/` and `frozen_evidence/`

## Locked exclusions

- No training, GPU use, Stage 1b execution or new seed output.
- No inspection of seeds 4--8 model-performance arrays.
- No active CausalTime performance narrative.
- No legacy ISTF-Mamba performance or effectiveness claim.
- No canonical manuscript replacement.
