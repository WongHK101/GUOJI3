# Route-B Draft v2 Pre-Package Audit

## Scope checks

- Separate v2 manuscript only: pass.
- Canonical `E:\GUOJI\elsarticle\istf_kbs.tex` intentionally untouched: pass (Git diff is empty for this file).
- Frozen evidence only: pass by source-manifest design.
- No new training, GPU, Stage 1b, or seeds 4-8 model-training/model-performance inspection: pass.

## Content checks

- Route ledger appears in the formal declaration and Table 1: pass.
- Algorithm 1 records all nine requested audit steps: pass.
- `CLAIM-COVERED` replaces the prior bare positive label: pass (final source grep checked).
- Full auxiliary penalty is described as expanded penalty-route coverage while x-only score remains partial: pass.
- d_cond non-monotonicity and exact endpoint wording: pass.
- Single-run versus five-seed disclosure: pass.
- Stage 1a mean +/- SD and release-locked language: pass.
- P1 A3 wording restricted: pass.

## Remaining known issue

The bibliography audit identifies 23 entries that need manual publisher/proceedings verification before submission. v2 neither invents DOI values nor changes those references.

## Final build and visual checks

- Three-pass `pdflatex -interaction=nonstopmode -halt-on-error` completed: 0 LaTeX errors, 0 undefined references, 0 undefined citations, 0 missing files, and 0 Overfull hboxes.
- The separate draft is 12 pages. Rendered-page review found no clipped figures, table bleed, visual overlap, or page-order defect.
- The log retains 69 Underfull hbox and 7 Underfull vbox warnings from ordinary two-column text/table spacing. They were visually reviewed and have no visible layout defect.
- Figure 3 caption was corrected to identify the five-seed P0 aggregate; it no longer describes the panel as a seed-0 result.
