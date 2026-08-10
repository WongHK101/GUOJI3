# Phase 8 KBS v4 Narrative Audit

## Final Argument

Neural Granger predictors can decouple predictive fit from graph fidelity when
their reported Jacobian omits an auxiliary or transformed predictive route. A
route-resolved Jacobian coverage audit makes this mismatch explicit across score
routes, penalty routes, source coordinates, and attribution horizons.

## Structural Revision

The v4 manuscript uses eight first-level sections:

1. Introduction
2. Background and Related Work
3. Structural Shortcut in Auxiliary Predictive Pathways
4. Jacobian Coverage Audit Framework
5. Controlled Evidence for Prediction--Knowledge Decoupling
6. Coverage Interventions Reveal a Graph--Prediction Frontier
7. Discussion
8. Conclusion

The evidence sequence is mechanism -> declaration -> controlled graph evidence
-> route intervention -> coefficient fidelity -> penalty expansion ->
raw-chain semantics -> training-time frontier. This replaces the earlier
ISTF-centered method-development chronology.

## Language Removed or Reframed

The following internal-review language was removed from the active narrative:

- P1 gradient-replay and bounded-postmortem details;
- repeated "not a benchmark leaderboard" and process-gate disclaimers;
- the sentence that no held-out confirmation was executed;
- repeated statements that the framework is not a certificate or proof;
- speculative accounts of why CP-depthwise optimization failed;
- redundant warnings attached to every transformed-coordinate result.

These changes reduce self-interruption and keep each paragraph claim-led. They
do not remove material evidence or convert a failed gate into a positive claim.

## Scientific Boundaries Retained

The following boundaries remain because omitting them would change the meaning
of the evidence:

- the frozen full-penalty field `pred_loss` is a total regularized objective,
  not pure prediction MSE;
- post-hoc total raw-chain scoring did not restore the learned coefficient
  structure in the controlled concat checkpoints;
- the tested full-prefix regularizers traced a graph--prediction frontier and
  did not occupy the pre-specified joint region;
- Stage 1a separates score semantics from empirical method advantage and is
  reported in the appendix;
- the audit does not replace the standard assumptions needed to interpret
  Granger predictability causally.

## Claim Discipline

The v4 main text does not claim that:

- legacy ISTF-Mamba improves graph recovery;
- CP-depthwise is a competitive repair;
- full-prefix regularization is an accepted positive method;
- auxiliary inputs are generally invalid;
- Jacobian coverage establishes causal identifiability.

The positive contribution is the coverage declaration, route-resolved audit,
controlled evidence of prediction--knowledge decoupling, and localization of
where score extraction versus training-time regularization acts.

## Layout Consequences

- Final Fig. 1 is placed at the top of page 2.
- Pages 4 and 6 contain full-width scientific tables; page 6 also contains the
  first quantitative figure.
- Page 8 contains the two remaining quantitative figures.
- The v3 page 13 is eliminated; the final manuscript has 12 pages.
- The last page contains only the balanced tail of the reference list. Its
  lower whitespace is a natural end-of-article condition, not a stranded float
  or broken column.
