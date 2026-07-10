# Clean-Extraction Compile Report: v2.4

**Precheck date:** 2026-07-10  
**Input ZIP:** `_phase7_kbs_jacobian_coverage_submission_candidate_v2_4_precheck2.zip`  
**Method:** Extract into a new empty directory; run Python syntax checks; rerun
the packaged figure and table generators; run three consecutive pdflatex passes.

## Result

- Packaged Python generators: passed `py_compile`.
- Figure PNG regeneration: 3/3 SHA256-identical to packaged exports.
- Full-penalty CSV/TeX regeneration: 2/2 SHA256-identical.
- PDF pages: 17.
- Third-pass LaTeX errors: 0.
- Third-pass undefined references/citations: 0.
- Third-pass missing figures/input files: 0.
- Third-pass overfull hbox/vbox: 0.
- Canonical manuscript SHA256 remained
  `e47ecd460a8f62bb85c4f79e745c8ddb769e55f34586d6f593d4cc9247cccda2`.

The final review ZIP is rebuilt from committed source with the final-pass clean
compile log, then extracted and checked once more. First-pass cross-reference
warnings are not represented as final unresolved references.
