# P0 Method Decision Memo

Date: 2026-07-06

Purpose: decide whether the current KBS manuscript should continue with the cross-channel ISTF-Mamba main method, be rescued through a coordinate-preserving ISTF branch, or shift toward a broader method rewrite.

This memo is an internal technical decision record. It should not be copied into the manuscript as a defensive limitations paragraph.

## Executive Decision

The current manuscript should not proceed by language polishing alone. The main risk identified by the external GPT reviews is technically reasonable: the current cross-channel ISTF-Mamba score can differ from a full chain-rule Jacobian with respect to raw input variables.

The project is still salvageable. Local smoke diagnostics support a narrower repair path: move the main method toward coordinate-preserving ISTF, with ordinary depthwise temporal filtering as the current leading candidate. This route preserves the paper's central story, avoids an auxiliary shortcut channel, and greatly improves score semantics.

## What The GPT Critique Got Right

The strongest critique is about score semantics rather than typography or presentation. Current ISTF-Mamba computes graph scores on filtered coordinates, `dY/dx'`. If the filter mixes variables, this does not necessarily match the full raw-input chain-rule score, `dY/dx`.

Local evidence supports this concern:

| Diagnostic | Key result |
| --- | --- |
| ISTF-Mamba current vs raw-chain score | mean score correlation 0.8316; top-k Jaccard 0.7843; leakage 0.0835 |
| Concat partial vs total derivative | mean score correlation 0.5064; top-k Jaccard 0.6913; leakage 0.1195 |
| Depthwise ISTF current vs raw-chain score | mean score correlation 0.999978; top-k Jaccard 1.0000; leakage 0.0053 |

Interpretation: this is not just a wording issue. A reviewer with the code or a careful reading could challenge whether the claimed GC score is truly with respect to the raw variables.

## What The Critique Overstates

The problem does not force abandoning the entire paper. It specifically attacks the current cross-channel filter/scoring semantics. It does not invalidate:

- the auxiliary-channel shortcut concern;
- the need for an input-space repair rather than concat side channels;
- the JRNGC vulnerability story;
- the KBS-level motivation about causal knowledge reliability;
- the possibility of an ISTF design that is constrained enough to support raw-variable scores.

The method can be rescued if the filter is made coordinate-preserving and the manuscript's claims are aligned with that design.

## Local Smoke Evidence For The Rescue Path

Controlled repair smoke, `d=6`, `T=100`, `lag=3`, `max_iter=120`, seeds 0--4:

| Method | Current AUROC | Raw-chain AUROC | Semantic corr | Leakage |
| --- | ---: | ---: | ---: | ---: |
| Baseline JRNGC | 0.6225 | n/a | n/a | n/a |
| ISTF-Mamba | 0.5588 | 0.5768 | 0.8316 | 0.0835 |
| Depthwise ISTF | 0.6325 | 0.6316 | 1.0000 | 0.0053 |
| Depthwise-gated ISTF | 0.6709 | 0.6720 | 0.9999 | 0.0017 |

Factorial D2 smoke, `d=6`, `T=180`, `lag=3`, `max_iter=120`, seeds 0--2:

| Cell | Baseline | ISTF-Mamba | Depthwise ISTF | Depthwise-gated |
| --- | ---: | ---: | ---: | ---: |
| Stat+Linear | 0.8994 | 0.7635 | 0.8736 | 0.7562 |
| Stat+Nonlinear | 0.8366 | 0.7043 | 0.7808 | 0.7505 |
| NS+Linear | 0.7973 | 0.7865 | 0.8030 | 0.6575 |
| NS+Nonlinear | 0.7405 | 0.7135 | 0.7432 | 0.6058 |

Interpretation:

1. Ordinary depthwise ISTF is the most stable coordinate-preserving candidate.
2. Depthwise-gated is not the current main candidate despite one good controlled-smoke result, because it degrades the factorial D2 smoke.
3. Cross-channel ISTF-Mamba no longer has enough evidence to remain the unmodified main method.

## Current Method Decision

Recommended direction:

1. Freeze the current KBS manuscript as a pre-P0 artifact, not as a submission candidate.
2. Promote ordinary depthwise ISTF as the primary rescue branch for formal testing.
3. Keep cross-channel ISTF-Mamba as an ablation/legacy variant only if formal experiments support a clear role.
4. Do not rewrite the manuscript yet; wait for controlled diagnostics and, if warranted, formal benchmark reruns.
5. Do not frame the manuscript as "we discovered a flaw in our old method." The internal flaw motivates the repair, but the paper should present the final method cleanly and selectively.

## Manuscript Framing Guardrail

The manuscript should not proactively expose every internal weakness. The public narrative should be:

- concat/auxiliary channels can create a shortcut around the Jacobian penalty;
- input-space filtering avoids separate side channels;
- the final ISTF implementation preserves variable coordinates, so graph scores remain interpretable with respect to the original variables;
- empirical diagnostics verify both graph recovery and score semantics.

Avoid manuscript language that says:

- "our previous ISTF-Mamba score was invalid";
- "the original method leaked gradients";
- "we found a fatal flaw";
- "we changed method after GPT review";
- "the reviewer may attack this point."

Those are internal audit facts, not publication framing.

## Open Technical Questions

1. Does ordinary depthwise ISTF hold on the full benchmark scale, especially CT-medical, NSVAR, Lorenz, and VAR?
2. Does depthwise ISTF still support the Figure 2 shortcut diagnostics when compared against concat partial and total-derivative scoring?
3. Should the final paper call the method "ISTF" generally, with Mamba as a deprecated ablation, or use a more precise name such as "coordinate-preserving ISTF"?
4. How much of the old cross-channel Mamba result table can be retained as ablation evidence after formal reruns?

## Recommended Next Action

Before starting expensive reruns, ask the external advisor to review this decision:

- Is the coordinate-preserving ISTF pivot methodologically sound?
- Is ordinary depthwise ISTF the right first formal candidate?
- Is the proposed minimal GPU benchmark enough to decide whether to rewrite the KBS manuscript around the repaired method?

