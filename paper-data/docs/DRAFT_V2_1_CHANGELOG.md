# Route-B Draft v2.1 Scientific and Bibliography Correction Changelog

## Scientific wording

1. Restricted the shortcut proposition and contribution to the controlled case
   where an auxiliary route is omitted from both the x-only graph score and
   corresponding Jacobian penalty.
2. Distinguished score-only incompleteness, penalty-only incompleteness,
   the both-omitted shortcut construction, and coordinate ambiguity throughout
   the Abstract, Introduction, Related Work, Section 3, Table 1, Discussion,
   Conclusion, and appendix scope statement.
3. Replaced broad abstract claims with bounded wording: coverage auditing is a
   prerequisite for interpreting the stated Jacobian graph knowledge, and the
   Stage 1a boundary case illustrates rather than universally proves that
   semantic correctness alone does not ensure empirical competitiveness.
4. Changed the keyword from `causal graph reliability` to `Granger graph
   reliability`.

## Audit framework and evidence

1. Renamed Table 1 as claim- and setting-specific and renamed both concat rows
   as controlled diagnostics.
2. Added explicit scope for external covariates declared exempt in a conditional
   graph, a joint graph over $(X,c)$, and an auxiliary transform $c=g(X)$.
3. Added the frozen five-seed post-hoc route-usage diagnostic
   ($\lvert J_c\rvert/\lvert J_x\rvert=3.689\pm0.750$) with a strict
   non-graph-score interpretation.
4. Made the evidence tiers explicit in the Fig. 2 caption and retained error
   bars only in the five-seed panel, without cluttering the panel titles.
5. Expanded Fig. 3 axis abbreviations and its caption definitions for P/T,
   F/R, and exact-top-$k$ Jaccard.

## Layout

1. Moved the score-semantics figure from the Background section to its first
   substantive diagnostic discussion in Section 5.
2. Restored numerical source order: Fig. 1 route ledger, Fig. 2 controlled
   diagnostics, Fig. 3 score-semantics audit.
3. Added a targeted float barrier after Section 5 so no main-text figure can
   cross into the bibliography.

## Bibliography

1. Resolved the 23 entries formerly marked manual against official sources.
2. Corrected the substantive metadata errors summarized in
   `BIBLIOGRAPHY_CORRECTION_CHANGELOG.md`.

## Boundaries retained

No new training, GPU work, Stage 1b, seeds 4--8 model-training/model-performance
inspection, CP revival, root-cause main-text use, CausalTime/operating-regime
narrative, legacy ISTF-Mamba performance claim, or canonical-manuscript change
occurred in V2.1.
