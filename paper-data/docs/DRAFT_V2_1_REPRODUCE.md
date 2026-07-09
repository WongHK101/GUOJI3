# Route-B Draft v2.1 Reproduction Instructions

## Scope

This package reproduces only the separate Route-B manuscript and its figures.
It does not reproduce, rerun, alter, or extend any model-training experiment.
All empirical inputs are frozen copies listed in the evidence-extract index and
the SHA256 manifest.

## Requirements

- A TeX installation with `pdflatex`.
- Python 3 with `matplotlib` and `numpy` only if figure regeneration is wanted.

## Compile the separate manuscript

From the package root, run the following command three times:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error istf_kbs_jacobian_coverage_draft_v2_1.tex
```

The expected outcome is a 14-page PDF with zero LaTeX errors, zero undefined
references/citations, zero missing files, and zero Overfull hboxes. Underfull
box warnings are documented in `DRAFT_V2_1_AUDIT_REPORT.md` and are visually
non-defective.

## Regenerate figures

The figure script uses absolute source paths in the original workstation. For a
portable regeneration, update the source-root constants or run it in the
original project checkout, then verify each source-file hash against
`figure_data_manifest_draft_v2_1.json`. The package already includes the exact
PDF/SVG/PNG exports used by the manuscript, so figure regeneration is not
needed for compilation.

## Integrity checks

1. Verify all package files with `PACKAGE_SHA256SUMS.txt`.
2. Verify figure inputs against `figures/coverage_audit_draft_v2_1/figure_data_manifest_draft_v2_1.json`.
3. Read `DRAFT_V2_1_EVIDENCE_TRACEABILITY.md` before interpreting any figure or
   table. It fixes the allowed score semantics and claim boundaries.

## Prohibited actions

Do not run new training, use GPU, start Stage 1b, inspect seeds 4--8
model-training/model-performance output arrays, revive CP-depthwise, replace
canonical `istf_kbs.tex`, or reinterpret legacy ISTF-Mamba as performance
evidence from this package.
