# Route-B Draft v2.2 Scientific and Build Audit

## Scientific scope checks

- Separate V2.2 manuscript only; canonical `istf_kbs.tex` remains untouched.
- Frozen evidence only; no new experiment, GPU, Stage 1b, CP revival, or seeds
  4--8 model-training/model-performance inspection.
- Proposition supports structural score failure only; empirical ranking and
  coefficient degradation are attributed to frozen diagnostics.
- The controlled shortcut retains the predictor-available, score-omitted, and
  penalty-omitted conditions.
- Score-only omission, penalty-only omission, omission from both, coordinate
  ambiguity, and horizon truncation remain distinct.
- Table 1 retains five columns and adds only the requested emphasized graph-
  object sentence.
- The five-seed post-hoc ratio retains all prohibited-interpretation
  boundaries.
- Stage 1a and P1 roles remain unchanged.

## Final build fields

- Local three-pass compile: PASS.
- PDF length: 14 pages.
- LaTeX errors: 0.
- Undefined references/citations: 0.
- Missing figure files: 0.
- Overfull hboxes/vboxes: 0/0.
- Underfull hboxes/vboxes: 84/6. These are non-blocking narrow-column and
  dense-table spacing warnings; no text crosses a column or page boundary.
- Included manuscript figures: 3/3 present.
- Visual inspection: PASS for all 14 pages. Title/abstract, Proposition 3.1,
  Table 1, all three figures, the Stage 1a boundary tables, the P1 paragraph,
  appendix, and references were inspected. No blank page, clipping, overlap,
  displaced caption, or post-bibliography figure was found.
- Canonical manuscript guard: `git diff -- istf_kbs.tex` is empty.
- Clean-extraction compile and package-hash verification are reported in the
  package-level build reports generated during final assembly.
