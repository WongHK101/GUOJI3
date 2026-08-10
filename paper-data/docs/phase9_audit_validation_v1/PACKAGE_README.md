# Phase 9 Audit-Generality Preregistration Review Package

This is a planning-only package. It contains no new scientific result,
checkpoint, manuscript source, or authorization token.

## Review order

1. `planning/PHASE9_AUDIT_VALIDATION_PREREGISTRATION.md`
2. `planning/PHASE9_AUDIT_CLAIM_GATE_MATRIX.md`
3. `planning/PHASE9_AUDIT_SEED_SUBJECT_FREEZE.json`
4. `planning/PHASE9_AUDIT_RUN_MATRIX.csv`
5. `planning/PHASE9_AUDIT_IMPLEMENTATION_REQUIREMENTS.md`
6. `planning/PHASE9_AUDIT_RUNTIME_ESTIMATE.md`
7. `planning/ADVISOR_REVIEW_REQUEST.md`

The `development_context/` directory contains the bounded RTX 4090 round-one
summary that motivated this protocol. Those values remain development-only.

The `source_snapshot/` directory contains the audit implementation inspected
when drafting the protocol. It is not an approved execution release. A new
release-locked validation runner must be implemented only after approval.

`COMMIT_INFO.txt`, `SOURCE_SHA256_MANIFEST.json`, and
`MANIFEST_SHA256.txt` bind the package to the exact planning commit and payload.

## Hard boundary

Every matrix row is unauthorized. Reviewing or extracting this ZIP does not
authorize Stage A or Stage B execution.
