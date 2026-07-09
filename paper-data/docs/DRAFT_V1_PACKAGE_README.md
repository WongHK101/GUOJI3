# Route-B KBS Full Draft v1 Review Package

## Scope

This package contains a separate full manuscript draft for the Jacobian coverage
audit mainline. It does not replace the canonical KBS manuscript. All empirical
content comes from frozen local artifacts.

## Boundaries

- No new training or GPU execution.
- No Stage 1b.
- No inspection of seeds 4-8 output arrays.
- No new method training.
- No replacement of the canonical istf_kbs.tex.

## Contents

- manuscript: separate LaTeX source, compiled PDF, and compilation log;
- figures: three draft figure sets in PDF, SVG, and PNG formats;
- figure_sources: Python source script and figure data manifest;
- source_data: frozen JSON and CSV files used by the draft figures/tables;
- documentation: claim-evidence matrix, traceability register, changelog, open
  issues, and updated paper/data indexes;
- audit: compile and claim-boundary checks.

## Compile Result

Three pdflatex passes produced a 10-page PDF with zero LaTeX errors, zero
undefined references/citations, zero missing files, and zero overfull hboxes.
Underfull warnings remain in narrow columns and dense tables. The rendered PDF
was inspected page by page; no clipping, overlapping figures, or table bleed was
observed.

## Required Advisor Review

Review the scientific route and claim boundaries, especially:

1. Is the coverage audit framework sufficiently constructive for the KBS route?
2. Are the controlled concat and limited legacy semantic diagnostics calibrated
   correctly?
3. Is the Stage 1a semantic-pass/performance-fail boundary table appropriately
   positioned?
4. Are any claims too strong, too weak, or missing a required provenance
   qualifier?

Do not approve canonical-manuscript replacement until these points are resolved.
