# Phase 8 KBS v4 Figure Reference Map

## Scope

This record documents the visual literature audit behind the submission-v4
figures. It concerns information hierarchy and scientific-figure grammar only.
No data, panel geometry, icon, or artwork was copied from a reference paper.

## Final Figure 1

Final file:

- `fig1_route_resolved_jacobian_audit_v4.{pdf,svg,png}`
- Generator: `tools/generate_phase8_submission_v4_overview.py`

The v3 manuscript's three text-heavy conceptual figures were replaced by one
asymmetric schematic-led composite. The final figure was informed by the
following specific figures.

| Reference | Figure consulted | Design element abstracted | How v4 differs |
| --- | --- | --- | --- |
| Z. Tao et al., "Active differentiable structure learning for clinical causal discovery," *Knowledge-Based Systems* 327 (2025) 114145 | Fig. 1 | A dense integrated architecture can remain readable when one dominant pipeline is paired with smaller supporting modules. | v4 uses a raw/auxiliary route ledger, Jacobian tensors, and an audit profile; it does not reuse the clinical modules or arrangement. |
| S. Chen et al., "Causal structure learning for high-dimensional non-stationary time series," *Knowledge-Based Systems* 295 (2024) 111868 | Fig. 1 | A strong left-to-right progression from time-series observations through a structural learner to a graph output. | v4 adds two separately declared routes, a fixed raw target, and derivative-coordinate objects. |
| W. Zhou et al., "Jacobian Regularizer-based Neural Granger Causality," *ICML* 2024 | Fig. 1 | The predictor-to-Jacobian-to-directed-graph relationship and target-source score orientation. | v4 explicitly contrasts partial, full-coordinate, and total raw-chain derivatives and adds route/horizon auditing. |
| Y. Cheng et al., "CUTS: Neural Causal Discovery from Irregular Time-Series Data," *ICLR* 2023 | Fig. 1 | Compact repeated glyphs for time-series windows, matrices, and graph-producing stages. | v4 uses original glyph construction and a different route-resolved scientific argument. |
| W. Sun et al., "PCAC: Causal discovery from low-dimensional small-scale time series," *Knowledge-Based Systems* 327 (2025) 114135 | Figs. 1--3 | Consulted for the visual treatment of temporal windows and lag support. | It was not a primary layout source; v4 uses a continuous raw-prefix attribution support curve instead. |

Panel roles in the final figure are:

1. **Panel a:** route-resolved predictor, fixed target, reported x-only Jacobian,
   raw-chain relation, and graph object.
2. **Panel b:** three derivative objects on one computation graph.
3. **Panel c:** route ledger, source-coordinate map, attribution support, and
   claim-specific audit profile.

All connectors are horizontal or vertical, and the visual vocabulary is limited
to muted blue-gray for the raw/scored route, muted brick for the auxiliary
route, and neutral gray/black for shared computation.

## Figures 2--4

The three quantitative figures retain the frozen v3 data and plotting semantics.
They were regenerated into the v4 self-contained directory and renamed to match
their actual main-text numbering:

- `fig2_prediction_knowledge_decoupling_v4.{pdf,svg,png}`
- `fig3_derivative_semantics_v4.{pdf,svg,png}`
- `fig4_graph_prediction_frontier_v4.{pdf,svg,png}`

Their PDF hashes are identical to the corresponding v3 result figures. The
palette is intentionally low saturation and uses shape, line weight, and
luminance in addition to color. The automated saturation audit records mean
foreground saturation of 0.057--0.090 and less than 0.01% high-saturation
foreground pixels for every figure.

## Removed Conceptual Figures

The following v3 concepts were not carried into v4 as standalone figures:

- `fig1_jacobian_coverage_audit_phase8_final`
- `fig2_controlled_concat_architecture_phase8_final`
- `fig3_claim_specific_audit_workflow_phase8_final`

Their valid content was consolidated into final Fig. 1. Removing them avoids a
sequence of similarly styled text-card diagrams and advances quantitative
evidence to Fig. 2.

## Local Review Sources

The visual audit used the four local KBS exemplars in `E:\GUOJI\KBSpaper`, the
local JRNGC PDF at `E:\GUOJI\tmp\kbs_reference_audit\external\jrngc_icml2024.pdf`,
and the local CUTS PDF at
`E:\GUOJI\tmp\kbs_reference_audit\external\cuts_arxiv.pdf`. The temporary
contact sheet is stored at
`E:\GUOJI\tmp\phase8_v4_reference_audit\reference_contact_sheet.png`.
