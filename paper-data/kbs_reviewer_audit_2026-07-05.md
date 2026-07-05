# KBS Pre-Submission Reviewer-Style Audit

Date: 2026-07-05

Input scope: current KBS manuscript `E:\GUOJI\elsarticle\istf_kbs.tex`, compiled PDF/log status recorded locally, and manuscript facts extracted from the current text. The manuscript is a full KBS-targeted draft with 7 figures, 33 cited bibliography entries, theory, diagnostics, benchmark evaluation, appendices, limitations, and reproducibility sections.

Assessment boundary: this is a reviewer-style pre-submission assessment, not an editorial decision letter and not an author rebuttal. No new experiments were run. The assessment is grounded in the current manuscript text and local reviewer criteria: originality, scientific importance, interdisciplinary readership, technical soundness, and readability for nonspecialists.

Shared manuscript claim summary: the manuscript argues that auxiliary temporal side channels in Jacobian-regularized neural Granger causality can create a structural shortcut, decoupling prediction loss from Jacobian-based graph recovery. Input-Space Temporal Filtering (ISTF) repairs this vulnerability by confining temporal processing to the original input coordinates with orthogonality-based drift control. ISTF is framed as a regime-dependent structural repair, not as a universal causal discovery booster.

Visible evidence base:
- Formal analysis for the concat shortcut and theoretical constraints for input-space filtering.
- Controlled diagnostics showing loss/AUROC decoupling, intervention sensitivity, coefficient-fidelity degradation under concat, and Jacobian preservation under ISTF.
- Root-cause synthetic experiments and checkpoint dynamics showing prediction-knowledge decoupling.
- Eleven benchmark configurations with four inferential datasets and seven descriptive datasets.
- CT_medical positive inferential evidence: 5/7 Holm-significant metrics, AUROC +0.042 from 0.458 to 0.500.
- Lorenz_F40 near-neutral behavior and VAR_d50 negative boundary.
- Risk-mitigation diagnostics for full auxiliary penalty, smoothing baselines, ISTF ablation, and CT_medical side-channel behavior.
- Reproducibility section describing saved score matrices, deterministic metric computation, top-k protocol, and evidence tiers.

Missing materials affecting confidence:
- No new empirical evidence beyond the current manuscript.
- No external journal-specific policy check was performed in this audit.
- The current text says code/data will be made available upon publication; the audit does not verify an externally accessible artifact package.

## Reviewer 1

Overall assessment: The technical case is substantially more mature than a simple benchmark paper. The strongest contribution is the mechanistic diagnosis of a measurement-pathway failure in auxiliary-channel JRNGC. However, the paper still needs sharper alignment between the controlled shortcut mechanism and the benchmark evidence before the central case is fully established.

Who would be interested in the results, and why: Researchers working on neural Granger causality, time-series causal discovery, robust causal representation, and trustworthy knowledge extraction from predictive models would be interested because the paper identifies a failure mode where better prediction can damage the graph score that the method treats as causal knowledge.

Major strengths:
- The paper separates diagnostic mechanism tests from benchmark claims.
- The evidence-tier design is transparent: four inferential datasets and seven descriptive datasets.
- The limitations section is unusually explicit for a method paper.
- The risk-mitigation appendix directly tests alternative repairs, including full auxiliary-Jacobian penalties and smoothing baselines.

Major concerns:
- The positive benchmark result is narrow. CT_medical is the only inferential dataset with positive evidence, and its AUROC rises only from 0.458 to 0.500. A reviewer may question practical significance even if several metrics are Holm-significant.
- The CT_medical side-channel check weakens a simple mechanism-to-result interpretation: concat does not collapse on CT_medical and full penalty is directionally highest. The manuscript acknowledges this, but the central narrative still risks implying a stronger link than the data support.
- The formal shortcut proof appears strongest for concat. FiLM is positioned as an analogous risk, but the manuscript should avoid letting readers think FiLM has been established to the same level unless the evidence is shown.

Technical failings that need to be addressed before the case is established:
- Clarify that controlled diagnostics establish a structural risk class, while CT_medical supports ISTF as a regime-dependent repair without proving the same shortcut mechanism is active in that dataset.
- Add a short practical-significance statement for CT_medical: why a +0.042 AUROC shift to 0.500 still matters, or why the structural-safeguard claim does not depend on high absolute AUROC.
- Ensure every FiLM mention is bounded as risk/analogy unless supported by matching diagnostics or proof.

Assessment against Nature-style criteria:
- Originality: credible within Jacobian-regularized neural GC; the measurement-pathway framing is the clearest novelty.
- Scientific importance: field-relevant and technically useful, but the broader importance depends on showing the failure mode is common enough to matter beyond the controlled setting.
- Interdisciplinary readership: strongest for machine learning and time-series causal discovery readers; broader reach is limited unless the paper foregrounds "prediction can corrupt knowledge extraction" more clearly.
- Technical soundness: generally strong, with the main weakness being mechanism-to-benchmark linkage rather than implementation or statistics.
- Readability for nonspecialists: improved, but the paper remains dense. Fig. 1 helps, yet the introduction could make the real-world consequence of the shortcut more concrete.

Recommendation posture: promising, but the authors should address mechanism-evidence alignment and practical significance before external review.

## Reviewer 2

Overall assessment: The manuscript is not a standard "new neural model beats baselines" paper, which is a strength. Its novelty lies in defining a structural reliability problem for Jacobian-based graph recovery. The paper is likely more publishable if framed as a knowledge-reliability contribution than as a performance contribution.

Who would be interested in the results, and why: KBS readers interested in reliable knowledge discovery, causal knowledge extraction, and neural time-series systems would care because the work shows that an apparently reasonable architectural extension can produce misleading causal scores.

Major strengths:
- The paper does not overclaim universal superiority and explicitly reports negative and neutral regimes.
- Related work now positions the contribution against neural GC, temporal conditioning, shortcut learning, state-space models, and temporal causal discovery.
- The root-cause synthetic section gives the manuscript a knowledge-reliability dimension beyond ordinary graph AUROC.
- The reproducibility framing around saved score matrices is a strong trust-building feature.

Major concerns:
- The title and abstract may still prime readers for a repair that broadly improves JRNGC, while the evidence shows a narrow and regime-dependent improvement profile.
- The strongest graph-recovery result in the root-cause synthetic experiments belongs to EMA-JRNGC, not ISTF-Mamba. The manuscript explains the prediction-loss tradeoff, but reviewers may ask why ISTF is the proposed framework rather than an EMA-style fixed input-space filter.
- The comparison with non-primary baselines is partly descriptive or legacy. The manuscript labels this, but the method-comparison section should avoid appearing to rank all methods on equal evidentiary footing.

Technical failings that need to be addressed before the case is established:
- Strengthen the explanation of why ISTF is the right general framework when simple causal EMA is competitive in controlled VAR and EMA dominates root-cause graph recovery.
- Make the "framework, not Mamba" positioning even more explicit near the abstract and conclusion.
- Consider whether the title should emphasize "diagnosis and input-space repair" rather than implying broad performance improvement.

Assessment against Nature-style criteria:
- Originality: strong if judged as a structural shortcut diagnosis; weaker if judged as another filter architecture.
- Scientific importance: credible for knowledge reliability in neural causal discovery, but not yet broad unless the authors make the implications for trustworthy AI systems more explicit.
- Interdisciplinary readership: mostly field-local. The broad hook should be that prediction-optimized models can break the measurement path used for knowledge extraction.
- Technical soundness: adequate in diagnostics and statistics; the evidentiary hierarchy is transparent.
- Readability for nonspecialists: acceptable for KBS/ML readers, but still difficult for readers outside neural GC because the causal-score semantics are introduced quickly.

Recommendation posture: promising if positioned as a structural reliability paper; less convincing if positioned as a method-performance paper.

## Reviewer 3

Overall assessment: The manuscript has a coherent story and a useful safeguard principle, but the current presentation is still highly technical and may undersell why KBS readers should care beyond JRNGC specialists. The paper should make the knowledge-system implication clearer: a model can learn to predict better while making its extracted causal knowledge less faithful.

Who would be interested in the results, and why: The immediate audience is neural causal discovery and time-series ML. A broader KBS audience could be interested if the manuscript foregrounds reliability of extracted knowledge under architectural shortcuts.

Major strengths:
- The abstract is candid about neutral and negative regimes.
- The figures and captions now explain the measurement-pathway problem better than earlier workflow-style diagrams.
- The limitations section prevents a reviewer from accusing the paper of hiding weak generalization.
- The paper has a clear non-SOTA identity: structural safeguard, not universal booster.

Major concerns:
- The manuscript is long and conceptually dense. A nonspecialist reader may not understand why Jacobian scores are "knowledge" until too late.
- The practical deployment guidance remains heuristic, and some key thresholds, such as T/d, are explicitly not validated.
- CT_medical is the central positive real-world result, but the absolute AUROC around chance may make the practical value look modest unless contextualized carefully.

Technical failings that need to be addressed before the case is established:
- Add one concise explanatory paragraph early in the introduction that connects Jacobian graph recovery to "knowledge extraction" and explains why a low-loss shortcut is dangerous even when prediction improves.
- Add a reviewer-proof sentence in the benchmark takeaway that the CT_medical result is statistically reliable but not a claim of high absolute causal discovery accuracy.
- Ensure deployment heuristics are never phrased as rules; they should remain guidance derived from present evidence only.

Assessment against Nature-style criteria:
- Originality: clear within the manuscript's immediate field.
- Scientific importance: potentially important for trustworthy knowledge discovery, but the paper needs stronger framing of that implication.
- Interdisciplinary readership: currently limited; can be improved by making the "prediction versus knowledge fidelity" conflict explicit in the first page.
- Technical soundness: mostly sound and transparent, but practical significance is not fully established.
- Readability for nonspecialists: improved but still specialized. The main barrier is the rapid movement from GC/Jacobian details to ISTF without a plain-language statement of the failure's consequence.

Recommendation posture: technically promising, but broad readability and significance framing should be strengthened before submission.

## Cross-review synthesis

Consensus strengths:
- The paper's strongest contribution is the structural shortcut diagnosis in Jacobian-regularized neural GC.
- The evidence hierarchy is transparent and unusually cautious.
- The manuscript is stronger when framed as a reliability and measurement-integrity paper than as a performance paper.
- The current compile/layout state is not the main risk; the main risks are argument alignment and significance framing.

Consensus technical risks:
- CT_medical is the only positive inferential dataset, with modest absolute AUROC.
- The CT_medical side-channel check does not reproduce the synthetic concat-collapse pattern, so the mechanism-to-real-data link must remain carefully bounded.
- FiLM should be kept as an analogous risk unless matching evidence is supplied.
- EMA/simple smoothing results create a fair reviewer question: what exactly does ISTF add beyond input-space smoothing plus drift control?

Where emphasis differs across reviewers:
- Reviewer 1 places greatest weight on technical mechanism-to-benchmark alignment.
- Reviewer 2 places greatest weight on originality and method positioning.
- Reviewer 3 places greatest weight on KBS readership, accessibility, and practical significance.

Broad-interest / significance readout:
- The manuscript has a credible KBS-relevant claim if presented as a reliability problem in causal knowledge extraction: prediction loss can improve while the extracted causal graph becomes less faithful.
- The current evidence does not support a broad claim that ISTF generally improves causal discovery. The manuscript mostly avoids this, but several places should make the boundary even harder to miss.

Most important issues to resolve before a strong case is established:
1. Add explicit language separating "controlled shortcut mechanism established" from "CT_medical positive regime observed".
2. Contextualize the CT_medical AUROC +0.042 improvement to 0.500 so reviewers do not see it as statistically significant but practically weak.
3. Clarify ISTF's added value over EMA/simple input-space smoothing.
4. Keep FiLM wording bounded unless additional proof or diagnostics are added.
5. Strengthen the first-page KBS hook around reliable knowledge extraction, not model performance.

## Risk / unsupported claims

- Unsupported if over-read: CT_medical contains the same auxiliary-channel shortcut failure as the controlled VAR diagnostic. The manuscript explicitly says this should not be claimed.
- Unsupported if over-read: ISTF-Mamba is generally superior to JRNGC or competing causal discovery methods. The manuscript correctly rejects this claim.
- Partly supported but needs careful wording: FiLM-style conditioning is an analogous architectural risk. It is discussed, but the strongest formal and diagnostic evidence is for concat.
- Potential reviewer attack: the only positive inferential real-world result has AUROC 0.500, which may look practically weak despite statistical significance.
- Potential reviewer attack: simple EMA/input-space smoothing is competitive in controlled diagnostics and dominates root-cause graph recovery, so the unique contribution of learnable ISTF must be framed as a constrained framework and deployment repair rather than Mamba-specific superiority.
