# Route-B Draft v2.3 Reproduction Instructions

## Build from a clean extraction

From the extracted package root, run:

```powershell
1..3 | ForEach-Object {
  pdflatex -interaction=nonstopmode -halt-on-error istf_kbs_jacobian_coverage_draft_v2_3.tex
}
```

The expected result is a 14-page PDF with zero LaTeX errors, undefined
references, undefined citations, missing figure files, or overfull boxes.

## Figure regeneration

The package includes frozen evidence and the Python plotting source. To
regenerate the figure exports without training a model:

```powershell
python tools/generate_coverage_audit_draft_v2_3_figures.py --output-dir figures/coverage_audit_draft_v2_3
```

This command reads only the package's frozen source inputs when the script is
run from the original project layout. Figure regeneration is not required to
compile or review the packaged manuscript.

## Frozen boundaries

Do not run new training, use a GPU, start Stage 1b, revive CP-depthwise, replace
canonical `istf_kbs.tex`, or introduce root-cause, CausalTime-performance,
legacy-Mamba-performance, or P1 A3 raw evidence into this draft.

P1 did not inspect or use Stage 1b model-training or model-performance outputs
for seeds 4--8.
