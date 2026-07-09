# Route-B Draft v2.1 Open Issues

## Review questions, not scientific blockers

1. **Working-title comparison.**

| Title | Strength | Risk | Recommendation |
| --- | --- | --- | --- |
| **Jacobian Coverage Audits for Reliable Neural Granger Causality** | Strongest continuity and searchable task framing. | `Reliable` could be read as a method-wide performance claim. | Retain as the default while the title is immediately qualified by the audit framing in the Abstract. |
| Auditing Jacobian Coverage in Neural Granger Causality | Most literal description of the contribution. | Less explicit about the graph-knowledge consequence. | Safest fallback if GPT considers `Reliable` too broad. |
| Jacobian Coverage Audits for Reliable Granger Graph Knowledge | Most precise reliability object. | More specialized and less fluent as a title. | Reserve for a title-specific reviewer objection. |

The current title remains the working title. It was not changed in V2.1 because
the paper audits the interpretation of neural Granger-causality outputs rather
than claiming that the method class is universally reliable.

2. Confirm whether the five-seed post-hoc route-usage ratio should remain in
the main-text prose or move to the appendix if the main text must be shortened.

## Frozen boundaries

- Do not run new training, use GPU, start Stage 1b, inspect seeds 4--8
  model-training/model-performance outputs, or revive CP-depthwise.
- Do not replace canonical `E:\GUOJI\elsarticle\istf_kbs.tex`.
- Do not reintroduce root-cause synthetic, historical CausalTime benchmark,
  operating-regime, or legacy ISTF-Mamba performance narratives.
- Do not use P1 A3 values or trends. The only permitted wording is that A3 was
  uninterpretable and did not pass because
  `gradient_replay_alignment_valid=false`.
