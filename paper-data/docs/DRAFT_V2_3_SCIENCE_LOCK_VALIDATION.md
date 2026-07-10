# Route-B Draft v2.3 Science-Lock Validation

## Requirement audit

| Requirement | Evidence | Result |
| --- | --- | --- |
| Accepted title preserved | Exact LaTeX title is `Auditing Jacobian Coverage in Neural Granger Causality`; extended subtitle absent. | PASS |
| Objective-level corollary bounded | Remark 3.2 states `R_x=0` for the Proposition 3.1 construction and explicitly excludes optimizer-selection, empirical-degradation, and universal claims. | PASS |
| Five dimensions used | Abstract, Introduction, contribution list, Section 4, Algorithm 1 workflow, Discussion, Conclusion, matrix, traceability, and template distinguish score-route, penalty-route, alignment, coordinate, and horizon validity. | PASS |
| No forced one-label rule | Algorithm 1 constructs a profile and reports every applicable flag; no `assign one` wording remains. | PASS |
| `CLAIM-COVERED` condition | Explicitly requires all applicable dimensions to pass and no required status to be unknown. | PASS |
| Coexisting flags | `PARTIALLY COVERED` may coexist with coordinate and horizon flags; all applicable failure and unassessed flags are retained. | PASS |
| Route annotation semantics | Score, penalty, and exemption status are recorded independently; a route may be scored and penalized. | PASS |
| Table 1 geometry preserved | The v2.2 and v2.3 `tabularx` column declaration is byte-identical. | PASS |
| Stage 1a content preserved | The complete Stage 1a table block is byte-identical between v2.2 and v2.3; both SHA256 values are `99da6be391bc4adf3b2ed57f48033162e56bd1ec6dc69e8e635c77aeccb558eb`. | PASS |
| Exact P1 wording | Main text and appendix both use: `P1 did not inspect or use Stage 1b model-training or model-performance outputs for seeds 4--8.` | PASS |
| Route-usage diagnostic unchanged | `|J_c|/|J_x| = 3.689 +/- 0.750` retained with all forbidden interpretations listed. | PASS |
| Forbidden evidence excluded | No root-cause or CausalTime result enters the manuscript; legacy Mamba remains semantic-only; A3 raw values are absent. | PASS |
| Canonical manuscript unchanged | `git diff -- istf_kbs.tex` is empty; canonical SHA256 is `e47ecd460a8f62bb85c4f79e745c8ddb769e55f34586d6f593d4cc9247cccda2`. | PASS |
| Build and layout | Three-pass compile and 14-page visual inspection pass all hard gates. | PASS |

## Figure reproducibility

The package-layout plotting script was run against only `frozen_evidence/`.
The regenerated PNGs for Figures 1, 2, and 3 were pixel-identical to the
included manuscript exports. No model or experiment was executed.

## Science-lock decision

All requested v2.3 corrections are closed. Further structural manuscript
iteration is outside this package's scope; the next decision belongs to full
adversarial review and journal strategy.
