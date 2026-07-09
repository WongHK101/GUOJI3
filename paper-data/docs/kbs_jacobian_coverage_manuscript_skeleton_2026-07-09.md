# KBS Manuscript Skeleton - Jacobian Coverage Audit Direction

Date: 2026-07-09

Status: writing skeleton. This is not the active submission manuscript. It is intended to guide a rebuild of the KBS article from the current ISTF-performance draft toward a Jacobian coverage audit paper.

## Working Title

Jacobian Coverage Audits for Reliable Neural Granger Causality

Alternative title if the advisor prefers a more mechanism-led framing:

Prediction-Knowledge Decoupling in Jacobian-Regularized Neural Granger Causality

## One-Sentence Argument

In Jacobian-regularized neural Granger causality, predictive paths outside the scored or penalized Jacobian can decouple prediction accuracy from graph knowledge; a Jacobian coverage audit framework exposes this failure mode and shows why a coordinate-preserving semantic repair can still fail as a performance method.

## Abstract Skeleton

Neural Granger causality turns predictive sensitivity into directed graph knowledge, but this link can break when a model uses auxiliary or transformed pathways that are not covered by the Jacobian score. We study this failure mode in Jacobian-regularized neural Granger causality. First, we formalize an auxiliary-route shortcut in which a concat predictor can lower prediction loss while suppressing the raw-input Jacobian used for graph recovery. Second, we introduce a Jacobian coverage audit framework that declares the scored variables, penalized variables, predictive paths, coordinate mapping, and attribution horizon of a neural GC model. Controlled diagnostics show prediction-knowledge decoupling under auxiliary conditioning and show that expanding Jacobian coverage can mitigate the failure. Finally, a preregistered coordinate-preserving depthwise repair passed semantic gates but failed Stage 1a performance and novelty gates against raw JRNGC and a fixed FIR baseline. These results support a reliability view of neural GC: Jacobian-based graph scores require explicit coverage audits before they can be interpreted as causal knowledge. The work does not claim benchmark superiority for input-space temporal filtering; instead, it defines when such scores are semantically meaningful and where semantic repair alone is insufficient.

## Section Plan

### 1. Introduction

Purpose:

- Frame graph recovery as extracted knowledge, not a prediction by-product.
- Introduce the mismatch between predictive pathways and scored Jacobians.
- State the paper as an audit/reliability contribution.

Reuse:

- Keep selected sentences from old Introduction on GC/JRNGC and the graph as the knowledge object.
- Delete old ISTF performance preview and CT_medical claims.

New writing:

- Open with reliability of learned causal graphs.
- Define the risk as "prediction-knowledge decoupling".
- Preview the audit framework and Stage 1a boundary evidence.

### 2. Background: JRNGC And Score Semantics

Purpose:

- Define the JRNGC predictor, Jacobian penalty, and graph score.
- Introduce score conventions and raw-chain attribution.
- Set up why score coordinates matter.

Reuse:

- Current JRNGC equations for prediction and Jacobian score.

New writing:

- Add a score-semantics taxonomy:
  - x-only raw-input score;
  - auxiliary-coordinate score;
  - filtered-coordinate score;
  - raw-chain score;
  - nominal-lag vs full-H attribution.

### 3. Structural Shortcut In Auxiliary Predictive Pathways

Purpose:

- Present the core failure mechanism.
- Show why concat side channels can bypass x-only penalty/score.

Reuse:

- The auxiliary-channel shortcut theorem and proof sketch.
- Controlled concat descriptions.

New writing:

- Calibrate theorem language: "existence" and "can", not universal occurrence.
- Explain FiLM/external covariates as related risks without overclaiming.

### 4. Jacobian Coverage Audit Framework

Purpose:

- Provide the constructive contribution of the paper.
- Define the route-coverage declaration and audit questions.

Core declaration:

`C = (V_score, V_penalty, P_pred, M_coord, H_attr)`

Audit questions:

1. Path coverage: do all predictive paths enter the score or penalty?
2. Coordinate identity: does each score column map to a unique raw source variable?
3. Horizon completeness: does `H_attr` cover the effective support of filtering or auxiliary transformations?

Tables:

- Table 1: architecture audit examples for JRNGC, concat-JRNGC, full-aux penalty, legacy filtered-coordinate ISTF, CP-depthwise, and EMA full-H reference.

### 5. Diagnostic Evidence For Coverage Failures

Purpose:

- Show the framework catches real failure modes using frozen diagnostics.

Evidence:

- d_cond sweep: loss decreases while graph AUROC degrades.
- mask/shuffle: auxiliary fallback pathway.
- coefficient recovery: concat collapses coefficient fidelity.
- full auxiliary-Jacobian penalty: expanded coverage mitigates shortcut.

New writing:

- Organize evidence by audit question rather than by old ISTF narrative.
- Avoid claiming that ISTF fixes all failures.

### 6. Semantic Repair As A Boundary Case

Purpose:

- Show that semantic correctness is necessary but not sufficient.

Evidence:

- Stage 1a official 96 formal runs.
- CP-depthwise semantic gates passed.
- CP-vs-baseline performance gate failed.
- CP-vs-FIR3 novelty gate failed.
- EMA full-H dominance no-go did not trigger.
- P1 decision remained `INCONCLUSIVE_BOUNDED_POSTMORTEM`.

Main table:

| Cell | Baseline AUROC | CP AUROC | FixedFIR3 AUROC | EMA AUROC | CP - Baseline | FIR3 - CP | EMA - CP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stat+Linear | 0.981383 | 0.981467 | 0.984377 | 0.888714 | +0.000084 | +0.002909 | -0.092753 |
| Stat+Nonlinear | 0.966452 | 0.966455 | 0.968040 | 0.845047 | +0.000002 | +0.001586 | -0.121408 |
| NS+Linear | 0.967320 | 0.965096 | 0.973161 | 0.864107 | -0.002223 | +0.008065 | -0.100990 |
| NS+Nonlinear | 0.947989 | 0.945940 | 0.950484 | 0.820996 | -0.002048 | +0.004543 | -0.124945 |

Required wording:

> A3 was uninterpretable and did not pass because `gradient_replay_alignment_valid=false`.

### 7. Discussion

Purpose:

- Interpret why graph-recovery pipelines need coverage audits.
- Explain why the work is not an ISTF performance paper.
- Discuss implications for auxiliary covariates, learned filters, and causal graph reliability.

Boundaries:

- No Stage 1b.
- No seeds 4/5 confirmatory use.
- No claim that all filtering hypotheses fail.

### 8. Limitations And Conclusion

Purpose:

- State the exact limits of the audit framework and evidence base.

Limitations:

- The framework audits score semantics; it does not guarantee causal identifiability.
- Coverage-complete repair is future work, not current evidence.
- Stage 1a was effect-size triage with data seeds 1-3 and was not a high-power statistical proof.
- Legacy ISTF-Mamba cannot support graph-recovery claims.

Conclusion:

- Jacobian-based graph knowledge requires explicit route coverage.
- Semantic repair alone does not imply performance or novelty.

## Figure And Table Minimum Set

Main figures:

1. Coverage mismatch schematic.
2. Prediction-knowledge decoupling diagnostics.
3. Score-semantics taxonomy and route-coverage audit examples.

Main tables:

1. Coverage declaration examples.
2. Stage 1a go/no-go cell-level summary.
3. Evidence tiering and allowed claims, if space permits.

## Migration Policy From Current `istf_kbs.tex`

Directly reusable:

- JRNGC setup equations.
- Shortcut theorem.
- Controlled diagnostic descriptions and data.
- Related work citations after refocusing.

Rewrite:

- Abstract.
- Introduction.
- Related Work framing.
- Discussion.
- Limitations.
- Conclusion.

Appendix or delete:

- ISTF-Mamba/TCN as main method.
- Operating-regime benchmark narrative.
- CausalTime benchmark.
- orthogonality as certificate.
- deployment framework wording.

## Forbidden Claims

- ISTF improves neural GC.
- Mamba or TCN is the main repaired method.
- CP-depthwise is successful.
- Legacy filtered-coordinate scores are valid graph evidence.
- A3 gradient conflict explains failure.
- FixedFIR3 beats CP in every seed.
- Coverage audit is a certificate or guarantee.
