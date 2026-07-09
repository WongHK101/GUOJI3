# Route-B Draft v2 Changelog

## Scientific and framework corrections

1. Replaced the plain `P_pred` route-class set with an operational route ledger: `(r, rho_score(r), rho_penalty(r), exempt(r))`.
2. Retained the five-part declaration `C=(V_score,V_penalty,P_pred,M_coord,H_attr)` and made the four audit dimensions directly decidable from the ledger.
3. Renamed the positive label from `COVERED` to `CLAIM-COVERED` throughout the separate v2 manuscript, tables, figures, and evidence documents.
4. Added Algorithm 1, a nine-step operational audit workflow, and a reusable appendix audit-report template.
5. Clarified that audit labels are claim-specific diagnostic outcomes, not model-wide guarantees or causal-identifiability certificates.

## Evidence and claim corrections

1. Replaced broad mitigation wording with the bounded claim that expanded penalty-route coverage mitigated x-only graph degradation in a controlled concat diagnostic.
2. Made explicit that the full auxiliary-penalty diagnostic leaves the graph score x-only and hence score-route completeness partial.
3. Corrected abstract and Section 5 d_cond wording: loss falls from 0.00895 to 0.00211, AUROC reaches 0.350 and is 0.522 at the largest tested auxiliary dimension; the AUROC path is non-monotonic.
4. Added replication labels and exact configurations for d_cond, mask/shuffle, coefficient recovery, full auxiliary penalty, and P0 semantic audit.
5. Added mean +/- SD to the five-seed full auxiliary-penalty text and Fig. 2d error bars.
6. Replaced the seed-0 Fig. 3 illustration with the fixed five-seed P0 aggregate used by the prose.
7. Tightened legacy ISTF-Mamba to a score-semantics diagnostic only.

## Stage 1a and P1 corrections

1. Replaced `preregistered` wording with `pre-specified and release-locked` except where historical source names require the original term.
2. Rebuilt the Stage 1a boundary table with mean +/- sample SD across the three data-seed analysis units, deltas rounded to four decimals, and positive-data-seed counts.
3. Added the exact semantic-gate sentence: for each of CP-depthwise, FixedFIR3, and FixedEMA, all 24 formal runs passed the semantic gate.
4. Preserved the EMA full-H dominance no-go outcome as not triggered.
5. Added the required P1 seed restriction and retained only the allowed A3 wording.

## Documentation and provenance

1. Added `DIAGNOSTIC_REPLICATION_AUDIT.md`.
2. Added `BIBLIOGRAPHY_AUDIT.md` with verified and manual-official-check categories.
3. Rebuilt the claim-evidence matrix and traceability register with exact artifact paths, score semantics, architectures, and permitted claims.
4. Added `JACOBIAN_COVERAGE_AUDIT_REPORT_TEMPLATE.md`.

