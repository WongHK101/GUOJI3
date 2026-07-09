# Route-B Draft v2.1 Final Pre-Package Audit

## Scope checks

- Separate V2.1 manuscript only: pass.
- Canonical `E:\GUOJI\elsarticle\istf_kbs.tex` intentionally untouched: final Git diff check passed.
- Frozen evidence only: pass by source-manifest design.
- No new training, GPU, Stage 1b, or seeds 4--8 model-training/model-performance inspection: pass.

## Content checks

- Controlled shortcut wording requires omission from both x-only score and corresponding penalty: pass.
- Score-only, penalty-only, coordinate, and horizon audit categories remain distinct: pass.
- Table 1 concat rows are claim- and setting-specific: pass.
- Conditional, joint, and transformed auxiliary cases are stated in Table 1 prose and Appendix A: pass.
- Abstract wording is bounded; keyword is `Granger graph reliability`: pass.
- Five-seed post-hoc route-usage artifact has a clear source file, script, configuration, and non-graph-score interpretation: pass.
- Fig. 2 replication tiers and Fig. 3 abbreviation definitions are explicit: pass.
- Main-text figure source order is 1, 2, 3; all figures occur before the bibliography: final source and rendered-PDF check passed.
- 23 previous manual bibliography entries have official-source records: pass.

## Final local build fields

- Final local three-pass build: 14 pages; 0 LaTeX errors; 0 undefined
  references; 0 undefined citations; 0 missing files; and 0 Overfull hboxes.
- The log contains 81 Underfull hboxes and 6 Underfull vboxes. Rendered-page
  inspection found no clipping, overlap, table-column spill, or unreadable
  figure label attributable to these warnings.
- Visual inspection covered the complete contact sheet and the full-resolution
  pages containing Figures 1--3, Tables 2--3, Appendix Tables B.4/D.5, and the
  bibliography. Figure 2 is on page 8, Figure 3 is on page 9, and the
  bibliography begins after all main-text figures.
- Package SHA256 and clean-extraction fields are populated in the packaged
  `CLEAN_EXTRACTION_COMPILE_REPORT.md` after ZIP assembly.
