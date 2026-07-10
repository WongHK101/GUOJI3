# Phase 7 Route-B Draft v2.3 Science-Lock Review Package

## Purpose

This package contains the final tightly bounded Route-B science-lock candidate
for full adversarial paper review and journal-strategy assessment. It is a
separate manuscript and does not replace canonical `istf_kbs.tex`.

## Core review targets

1. Objective-level Remark 3.2 is an existence and objective non-certification
   result, not a claim about optimizer behavior or empirical degradation.
2. The framework consistently uses five audit dimensions.
3. Audit output is a set-valued claim-specific profile, not one forced label.
4. Route score, penalty, and exemption status are recorded independently.
5. P1 uses the exact approved Stage 1b-output wording.
6. Stage 1a and all empirical diagnostics retain their frozen values and claim
   boundaries.

## Main files

- Manuscript: `istf_kbs_jacobian_coverage_draft_v2_3.{tex,pdf,log}`
- Claim-evidence matrix: `kbs_jacobian_coverage_claim_evidence_matrix_v2_3_2026-07-10.md`
- Traceability: `DRAFT_V2_3_EVIDENCE_TRACEABILITY.md`
- Audit template: `DRAFT_V2_3_AUDIT_REPORT_TEMPLATE.md`
- Changelog: `V2.3_SCIENCE_LOCK_CHANGELOG.md`
- Strategic risks: `V2.3_REMAINING_STRATEGIC_RISKS.md`
- Build QA: `DRAFT_V2_3_COMPILE_REPORT.md`,
  `CLEAN_EXTRACTION_COMPILE_REPORT.md`, and
  `DRAFT_V2_3_VISUAL_INSPECTION_REPORT.md`
- Figures and source: `figures/coverage_audit_draft_v2_3/` and
  `tools/generate_coverage_audit_draft_v2_3_figures.py`
- Frozen review evidence: `frozen_evidence/`

## Evidence restrictions

- The five-seed post-hoc `|J_c|/|J_x| = 3.689 +/- 0.750` value is a
  sensitivity-mass route-usage diagnostic only. It is not a graph score,
  conditional-Granger estimate, causal-contribution ratio, Granger-strength
  ratio, shortcut-severity score, benchmark or performance result, or
  confirmatory result.
- Legacy cross-channel ISTF-Mamba is a score-semantics failure diagnostic only.
- Root-cause synthetic and historical CausalTime performance are excluded.
- P1 A3 appears only as uninterpretable and not passed because
  `gradient_replay_alignment_valid=false`.
- P1 did not inspect or use Stage 1b model-training or model-performance outputs
  for seeds 4--8.

## Review endpoint

V2.3 ends local structural manuscript iteration. The requested next action is a
full adversarial paper review and journal-strategy decision.
