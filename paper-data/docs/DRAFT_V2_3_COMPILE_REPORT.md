# Route-B Draft v2.3 Compile Report

## Local three-pass build

- Source: `istf_kbs_jacobian_coverage_draft_v2_3.tex`
- Engine: pdfTeX / `pdflatex`, TeX Live 2025
- Passes: 3
- Result: PASS
- PDF pages: 14
- LaTeX errors: 0
- Undefined references: 0
- Undefined citations: 0
- Missing figure files: 0
- Overfull hboxes: 0
- Overfull vboxes: 0
- Underfull hboxes: 82
- Underfull vboxes: 4
- Referenced figure files present: 3/3

The underfull warnings arise from narrow two-column text and dense table cells.
The final PDF render shows no clipping, overlap, table bleed, or text outside a
page boundary.

## Clean-extraction build

The package-level clean-extraction result is recorded in
`CLEAN_EXTRACTION_COMPILE_REPORT.md` and `clean_extraction_compile.log`. The
same hard gates apply: three passes, zero errors, zero undefined references or
citations, zero missing figures, and zero overfull boxes.
