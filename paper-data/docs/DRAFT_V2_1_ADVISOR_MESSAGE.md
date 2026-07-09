CODEX：

Route-B draft v2.1 is ready for focused scientific review. This remains a
separate manuscript and does not replace the canonical KBS manuscript.

V2.1 implements the requested correctness patch using frozen evidence only:

1. The shortcut proposition now concerns the specific controlled concat case
   where the auxiliary route is omitted from both the x-only graph score and
   corresponding Jacobian penalty. Score-only, penalty-only, coordinate, and
   horizon defects remain distinct audit categories.
2. Table 1 is now explicitly claim- and setting-specific. It distinguishes a
   declared external covariate in a conditional graph, a joint graph over
   $(X,c)$, and a transform $c=g(X)$.
3. Abstract, Discussion, Conclusion, and keyword wording are bounded to
   Granger graph-score interpretation rather than a universal reliability or
   empirical-necessity claim.
4. Figures are numerically ordered and placed near their substantive Section 5
   discussions. No main-text figure can cross the bibliography.
5. Fig. 2 explicitly distinguishes single-run panels from the five-seed panel
   in its caption; Fig. 3 defines P/T, F/R, and J in the caption.
6. A frozen five-seed post-hoc full-input Jacobian diagnostic is added only as
   supporting route-usage evidence: mean |J_c|/|J_x|=3.689 +/- 0.750. It is not
   a graph score, conditional-Granger estimate, or performance result.
7. The 23 previously manual bibliography entries were corrected using official
   publisher, proceedings, preprint, or university-press records. No reference
   was added or removed.

Please assess only:
1. whether the controlled-shortcut wording is now mathematically precise and
   does not conflate score coverage with penalty coverage;
2. whether Table 1 and the appendix describe conditional, joint, and transformed
   auxiliary settings with adequate scope;
3. whether the post-hoc |J_c|/|J_x| result is appropriately bounded as a
   route-usage diagnostic;
4. whether the default title remains defensible or should switch to the more
   literal fallback listed in `DRAFT_V2_1_OPEN_ISSUES.md`.

Still prohibited: new training, GPU, Stage 1b, seeds 4--8
model-training/model-performance inspection, CP revival, root-cause or
CausalTime performance narrative, legacy ISTF-Mamba performance claims, and
replacement of the canonical manuscript.
