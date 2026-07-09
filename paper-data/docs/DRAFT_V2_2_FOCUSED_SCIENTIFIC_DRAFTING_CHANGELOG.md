# Route-B Draft v2.2 Focused Scientific Drafting Changelog

## Scope

V2.2 is a frozen-evidence scientific writing pass over the separately approved
V2.1 Route-B manuscript. It does not replace canonical `istf_kbs.tex`, alter
any value or artifact, run training, use GPU, start Stage 1b, revive
CP-depthwise, or inspect seeds 4--8 model-training/model-performance outputs.

## Theoretical and empirical claim separation

1. Restricted Proposition 3.1 and the first contribution to a structural
   existence claim: prediction can rely on the omitted auxiliary route while
   the x-only Jacobian score vanishes or becomes uninformative.
2. Assigned observed graph-ranking and coefficient-fidelity degradation to the
   frozen controlled diagnostics rather than to the proposition.
3. Preserved the exact controlled shortcut scope: the auxiliary route is
   available to the predictor and omitted from both the x-only score and its
   corresponding x-only Jacobian penalty.
4. Retained separate audit categories for score-only omission, penalty-only
   omission, omission from both objects, coordinate ambiguity, and horizon
   truncation.

## Constructive framework and narrative flow

1. Added the claim-ordering sentence: **The audit label depends first on the
   declared graph object.** Table 1 was not widened or otherwise redesigned.
2. Reframed Section 4 around the chain from declared graph object to route
   ledger, score/penalty objects, coordinate/horizon checks, and claim-specific
   output label.
3. Added explicit transitions from the structural mechanism to the audit,
   from the audit to frozen diagnostics, and from diagnostics to the Stage 1a
   boundary case.
4. Recast Section 5 subsections as audit questions rather than implementation
   reports while retaining all configurations, seeds, sample sizes, values,
   and evidence-tier disclosures.
5. Recast Stage 1a as the independent question of empirical value after
   semantic validity, without presenting it as a leaderboard or successful
   repair evaluation.

## Title and bounded evidence

1. Changed the working title to **Auditing Jacobian Coverage in Neural Granger
   Causality**.
2. Recorded the extended title alternative in `DRAFT_V2_2_OPEN_ISSUES.md` and
   retired `Reliable Neural Granger Causality` as the default.
3. Retained the five-seed post-hoc ratio
   $|J_c|/|J_x|=3.689\pm0.750$ in the main text. It is explicitly described as
   a post-hoc sensitivity-mass diagnostic, not a graph score,
   conditional-Granger estimate, benchmark, performance, or confirmatory
   result.

## Unchanged scientific assets

- Figures and frozen evidence remain byte-identical to V2.1.
- Table 1 retains its existing five-column structure.
- Stage 1a values, gates, and analysis units are unchanged.
- Bibliography content and metadata are unchanged from the verified V2.1 pass.
- P1 A3 wording remains restricted to the approved alignment-invalid sentence.

## Layout

- Applied local `raggedbottom` only across Sections 6--7 and restored
  `flushbottom` before Section 8. This removes the pre-existing extreme
  vertical stretching around the deferred Stage 1a cross-column table without
  changing float order, table content, or the rest of the Elsevier layout.
