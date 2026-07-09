# Draft v1 Compile and Claim-Boundary Audit

Date: 2026-07-10

## Compile

Command:

    pdflatex -interaction=nonstopmode -halt-on-error istf_kbs_jacobian_coverage_draft_v1.tex

The command was run three times in E:\GUOJI\elsarticle.

| Check | Result |
|---|---:|
| PDF pages | 10 |
| LaTeX errors | 0 |
| Undefined references | 0 |
| Undefined citations | 0 |
| Missing files | 0 |
| Overfull hbox | 0 |
| Underfull hbox | 61 |
| Underfull vbox | 2 |

The remaining underfull warnings originate from dense multi-column tables and
narrow paragraph columns. Every page was rendered at 160 dpi and inspected. No
clipping, visible overlap, figure collision, or table bleed was found.

## Boundary Sweep

| Check | Result |
|---|---:|
| Canonical E:\GUOJI\elsarticle\istf_kbs.tex changed | No |
| Root-cause narrative in main text | No |
| CausalTime performance narrative in main text | No; one limitations sentence states that legacy CausalTime results are not used. |
| Operating-regime performance narrative in main text | No; one limitations sentence excludes legacy operating-regime results. |
| Legacy performance-superiority phrases | 0 |
| A3 raw-branch values or labels | 0 |
| A3 required restricted wording | Present |

## Visual Review

- Page 3: conceptual coverage schematic and caption.
- Page 4: score-semantics audit and caption.
- Page 6: coverage declaration table.
- Page 7: Stage 1a boundary table and gate table.
- Page 9: appendix evidence inventory.
- Page 10: controlled concat diagnostic figure and caption.

All reviewed pages rendered without overlap or clipping.
