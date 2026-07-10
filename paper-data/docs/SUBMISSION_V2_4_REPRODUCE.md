# Reproduce Submission Candidate v2.4

## Figures and frozen table

From either the repository layout or the extracted package root:

```powershell
python tools/generate_coverage_audit_submission_v2_4_figures.py
python tools/export_full_aux_penalty_submission_v2_4.py
```

These commands only read frozen JSON/CSV files and write figures or formatted
tables. They do not train or evaluate a model.

The scripts auto-detect the repository layout (`../elsarticle`) or the package
layout (`./figures`, `./tables`, and `./frozen_evidence`). Explicit
`--output-dir` and `--input` arguments remain available for audit use.

## Manuscript

```powershell
1..3 | ForEach-Object {
  pdflatex -interaction=nonstopmode -halt-on-error `
    istf_kbs_jacobian_coverage_submission_candidate_v2_4.tex
}
```

The package includes `elsarticle.cls`; the bibliography is embedded as
`thebibliography`, so BibTeX is not required.

## Required validation

- 0 LaTeX errors;
- 0 undefined references or citations;
- 0 missing figures/files;
- 0 overfull boxes;
- 17 rendered pages in the packaged build;
- canonical manuscript SHA256 remains
  `e47ecd460a8f62bb85c4f79e745c8ddb769e55f34586d6f593d4cc9247cccda2`.
