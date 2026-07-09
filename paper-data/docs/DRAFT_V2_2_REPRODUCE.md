# Route-B Draft v2.2 Reproduction Instructions

From the review-package root, run three times:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error istf_kbs_jacobian_coverage_draft_v2_2.tex
```

Compilation uses only the bundled Elsevier class/style files and the figure
PDFs under `figures/coverage_audit_draft_v2_1/`. Figure regeneration is not
required because V2.2 changes prose only and retains the approved V2.1 exports.

Verify all files against `PACKAGE_SHA256SUMS.txt` before review. Do not run new
training, use GPU, start Stage 1b, inspect seeds 4--8 model-training/model-
performance outputs, revive CP-depthwise, or replace canonical `istf_kbs.tex`.
