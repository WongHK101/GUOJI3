# KBS Manuscript Skeleton - Jacobian Coverage Audit Direction v1.1

Date: 2026-07-10

Status: writing skeleton. This is not the active submission manuscript. It is a rebuild guide for changing the KBS article from an ISTF-performance paper into a Jacobian coverage audit paper. It does not replace `E:\GUOJI\elsarticle\istf_kbs.tex`.

## Working Title

Jacobian Coverage Audits for Reliable Neural Granger Causality

Alternative title if the advisor prefers a more mechanism-led framing:

Prediction-Knowledge Decoupling in Jacobian-Regularized Neural Granger Causality

## One-Sentence Argument

In Jacobian-regularized neural Granger causality, predictive route classes outside the scored or penalized Jacobian can decouple prediction accuracy from graph knowledge; a Jacobian coverage audit framework exposes this failure mode and explains why a coordinate-preserving semantic repair can still fail as a performance method.

## Abstract Skeleton

Neural Granger causality turns predictive sensitivity into directed graph knowledge, but this link can break when a predictor uses auxiliary or transformed pathways that are not covered by the Jacobian score. We study this failure mode in Jacobian-regularized neural Granger causality. First, we formalize an auxiliary-route shortcut in which a concat predictor can lower prediction loss while suppressing the raw-input Jacobian used for graph recovery. Second, we introduce a Jacobian coverage audit framework that declares the scored variables, penalized variables, architecture-declared predictive route classes, coordinate mapping, and attribution horizon of a neural GC model. Controlled concat diagnostics show prediction-knowledge decoupling under auxiliary conditioning and show that expanding Jacobian coverage mitigated the failure in controlled concat diagnostics. Finally, a preregistered coordinate-preserving depthwise repair passed semantic gates but failed Stage 1a performance and novelty gates against baseline JRNGC and a fixed FIR baseline. These results support a reliability view of neural GC: Jacobian-based graph scores require explicit coverage audits before they can be interpreted as causal knowledge. The work does not claim benchmark superiority for input-space temporal filtering; instead, it defines when such scores are semantically meaningful and where semantic repair alone is insufficient.

## Section Plan

### 1. Introduction

Purpose:

- Frame graph recovery as extracted knowledge, not a prediction by-product.
- Introduce the mismatch between predictive route classes and scored Jacobians.
- State the paper as an audit and reliability contribution.

Reuse:

- Keep selected sentences from the old Introduction on GC/JRNGC and the graph as the knowledge object.
- Delete old ISTF performance preview, CausalTime operating-regime claims, and CT-medical performance framing.

New writing:

- Open with reliability of learned causal graphs.
- Define the risk as prediction-knowledge decoupling.
- Preview the audit framework and Stage 1a boundary evidence.

### 2. Background and Related Work

Purpose:

- Define JRNGC predictor, Jacobian penalty, and graph score.
- Explain score semantics and raw-chain attribution.
- Position the work in KBS-relevant reliability and knowledge-extraction terms.

Must cover:

- neural Granger causality;
- Jacobian-based graph scoring;
- temporal and auxiliary conditioning;
- shortcut learning;
- attribution and score reliability;
- knowledge-extraction auditing.

Reuse:

- Current JRNGC equations for prediction and Jacobian score.
- Selected related-work citations after refocusing.

New writing:

- Add a score-semantics taxonomy:
  - x-only raw-input score;
  - auxiliary-coordinate score;
  - filtered-coordinate score;
  - raw-chain score;
  - nominal-lag vs full-H attribution.

### 3. Structural Shortcut in Auxiliary Predictive Pathways

Purpose:

- Present the core failure mechanism.
- Show why concat side channels can bypass an x-only penalty or score.

Reuse:

- Auxiliary-channel shortcut theorem and proof sketch.
- Controlled concat descriptions.

New writing:

- Calibrate theorem language: existence and diagnostic claim, not universal law.
- Explain FiLM/external covariates as related route-coverage risks without overclaiming.

### 4. Jacobian Coverage Audit Framework

Purpose:

- Provide the constructive contribution.
- Define the coverage declaration, four audit dimensions, and diagnostic labels.

Core declaration:

`C = (V_score, V_penalty, P_pred, M_coord, H_attr)`

Definitions:

- `V_score`: variables or coordinates used for the graph score.
- `V_penalty`: variables or coordinates covered by sparsity or causal regularization.
- `P_pred`: architecture-declared predictive route classes.
- `M_coord`: mapping from score coordinates to original source-variable identities.
- `H_attr`: attribution horizon or lag support used for the derivative.

Four audit dimensions:

1. Score-route completeness: Are all predictive route classes whose effects are interpreted as graph knowledge included in the score attribution?
2. Penalty-route completeness: Are all route classes capable of carrying predictive information covered by the sparsity or causal regularization objective, or explicitly declared exempt?
3. Score-penalty alignment: Do the score and penalty cover compatible variables, coordinates, paths, and attribution horizons?
4. Coordinate and horizon validity: Does each score column map to one original source variable, and does the attribution horizon cover the transformation support?

Audit labels:

- `COVERED`
- `PARTIALLY COVERED`
- `COORDINATE-AMBIGUOUS`
- `HORIZON-TRUNCATED`
- `UNASSESSED`

These are diagnostic labels, not mathematical guarantees or certificates.

Tables:

- Table 1: architecture audit examples for baseline JRNGC, concat-JRNGC, full auxiliary-Jacobian penalty, legacy filtered-coordinate ISTF-Mamba, CP-depthwise, FixedFIR3, and EMA full-H reference.

### 5. Diagnostic Evidence for Coverage Failures

Purpose:

- Show that the framework catches route and score-semantics failures using frozen diagnostics.

Evidence:

- d_cond sweep: loss decreases while graph AUROC degrades in controlled concat diagnostics.
- mask/shuffle: auxiliary fallback pathway.
- coefficient recovery: concat shortcut can reduce coefficient fidelity.
- full auxiliary-Jacobian penalty: expanding Jacobian coverage mitigated the failure in controlled concat diagnostics.

New writing:

- Organize evidence by audit dimension rather than by old ISTF repair narrative.
- Do not claim that ISTF fixes all failures.
- Keep root-cause synthetic diagnostics out of the active main-text core until their provenance and score semantics are audited. They are a pending appendix candidate.

### 6. Semantic Repair as a Boundary Case

Purpose:

- Show that semantic correctness is necessary but not sufficient.

Placement:

- Stage 1a appears as a main-text boundary table, not as a benchmark leaderboard and not as a successful method evaluation.

Evidence:

- Stage 1a official 96 formal runs.
- CP-depthwise semantic gates passed.
- CP-vs-baseline performance gate failed.
- CP-vs-FIR3 novelty gate failed.
- EMA full-H dominance no-go did not trigger.
- Stage 1b eligibility failed.
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

Forbidden wording:

- any A3 raw-branch values, tendencies, pass status, or conflict-pattern counts;
- any claim that CP-depthwise is successful;
- any claim that FixedFIR3 beat CP in every individual data seed.

### 7. Discussion

Purpose:

- Interpret why graph-recovery pipelines need coverage audits.
- Explain why the work is not an ISTF performance paper.
- Discuss implications for auxiliary covariates, learned filters, smoothing, and causal graph reliability.

Boundaries:

- No Stage 1b.
- No seeds 4-8 inspection or confirmatory use.
- No claim that all filtering hypotheses fail.
- No old CausalTime performance narrative in the active mainline.

### 8. Limitations and Conclusion

Purpose:

- State the exact limits of the framework and evidence base.

Limitations:

- The framework audits score semantics; it does not guarantee causal identifiability.
- Coverage-complete repair is future work, not current evidence.
- Stage 1a was effect-size triage with data seeds 1-3 and was not a high-power statistical proof.
- Legacy ISTF-Mamba cannot support graph-recovery claims.
- Legacy CausalTime benchmark results cannot validate the new raw-variable score-semantics claims.

Conclusion:

- Jacobian-based graph knowledge requires explicit route, coordinate, and horizon coverage.
- Semantic repair alone does not imply performance or novelty.

## Figure And Table Minimum Set

Main figures:

1. Coverage mismatch schematic.
2. Prediction-knowledge decoupling diagnostics.
3. Score-semantics audit schematic or compact diagnostic panel.

Main tables:

1. Coverage declaration examples and audit labels.
2. Stage 1a go/no-go cell-level boundary summary.
3. Evidence tiering and allowed claims, if space permits.

## Migration Policy From Current `istf_kbs.tex`

Directly reusable:

- JRNGC setup equations.
- Shortcut theorem after claim calibration.
- Controlled concat diagnostic descriptions and data after provenance freeze.
- Related-work citations after refocusing.

Rewrite:

- Abstract.
- Introduction.
- Background and Related Work framing.
- Discussion.
- Limitations.
- Conclusion.

Appendix or delete:

- ISTF-Mamba/TCN as main method.
- Old CausalTime benchmark tables.
- Operating-regime positive, neutral, or negative claims.
- Orthogonality as certificate.
- Deployment framework wording.

## Forbidden Claims

- ISTF improves neural GC.
- Mamba or TCN is the main repaired method.
- CP-depthwise is successful.
- Legacy filtered-coordinate scores are valid graph evidence.
- A3 gradient conflict explains failure.
- FixedFIR3 beats CP in every seed.
- Coverage audit is a certificate or guarantee.
- Old CausalTime benchmarks validate raw-variable score-semantics claims.
