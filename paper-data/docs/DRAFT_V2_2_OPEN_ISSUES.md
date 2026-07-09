# Route-B Draft v2.2 Open Issues

## Review questions, not scientific blockers

1. **Final title length.**

| Title | Strength | Risk | Recommendation |
| --- | --- | --- | --- |
| **Auditing Jacobian Coverage in Neural Granger Causality** | Literal, searchable, and aligned with the audit contribution. | Does not name prediction--knowledge decoupling in the title. | Default working title for v2.2. |
| Auditing Jacobian Coverage in Neural Granger Causality: Prediction--Knowledge Decoupling and Semantic Repair Boundaries | Makes the two empirical case-study roles explicit. | Long and potentially too report-like for the final journal title. | Extended alternative for final editorial selection. |

The previous `Reliable Neural Granger Causality` title is retired because it
could imply a reliability-guaranteed method. The concise title is the default;
the extended alternative should be used only if the empirical boundary needs
to be explicit in the title.

2. Confirm whether the five-seed post-hoc route-usage ratio should remain in
the main-text prose or move to the appendix only if final page pressure requires
it. Its current scientific role is closed: post-hoc sensitivity-mass evidence
for route usage, not a graph score, conditional-Granger estimate, benchmark,
performance result, or confirmatory result.

## Frozen boundaries

- Do not run new training, use GPU, start Stage 1b, inspect seeds 4--8
  model-training/model-performance outputs, or revive CP-depthwise.
- Do not replace canonical `E:\GUOJI\elsarticle\istf_kbs.tex`.
- Do not reintroduce root-cause synthetic, historical CausalTime benchmark,
  operating-regime, or legacy ISTF-Mamba performance narratives.
- Do not use P1 A3 values or trends. The only permitted wording is that A3 was
  uninterpretable and did not pass because
  `gradient_replay_alignment_valid=false`.
