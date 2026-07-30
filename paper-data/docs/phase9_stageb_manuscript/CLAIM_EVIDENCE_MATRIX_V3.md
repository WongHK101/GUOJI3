# Jacobian Coverage Audit Manuscript: Claim-Evidence Matrix v3

Date: 2026-07-26

## Governing argument

In neural Granger predictors with auxiliary or transformed predictive routes,
a partial Jacobian graph can omit information actually used for prediction. A
route-resolved Jacobian coverage audit separates route completeness,
score-penalty alignment, source-coordinate validity, and temporal-support
validity. Controlled diagnostics and a preregistered held-out confirmation show
that the resulting missing-route and score-disagreement signals reproduce
across two causal auxiliary architectures and two data domains. A separately
frozen five-data-seed Lorenz-96 confirmation shows that the same score-coverage
diagnosis persists when baseline recovery of the known direct graph is strong.

The paper does not claim that coverage establishes causal identifiability,
that the tested repair succeeds, or that the held-out NetSim results establish
strong graph recovery.

## Terminology ledger

| Canonical term | Definition | Forbidden substitute |
| --- | --- | --- |
| directed Granger-predictive dependency graph | Graph induced by a declared derivative, coordinate system, lag support, and aggregation rule. | Unqualified causal graph. |
| Jacobian coverage audit | Claim-specific diagnostic of score routes, penalty routes, their alignment, coordinate identity, and attribution horizon. | Certificate or identifiability proof. |
| partial nominal score | Jacobian score holding an auxiliary route fixed and aggregating only the declared nominal lags. | Total raw-input attribution. |
| total raw-chain nominal score | End-to-end derivative with respect to original raw variables, aggregated on the declared nominal lags. | Full-prefix graph score. |
| missing-route relative magnitude | Off-diagonal L1 mass of the omitted raw-chain term divided by total raw-chain mass under the frozen coordinate scale. | Causal-contribution ratio or invariant importance. |
| bounded raw-chain attribution | End-to-end raw-variable derivative evaluated through a declared finite horizon. | Complete full-prefix attribution. |
| audit-generality confirmation | Reproduction of preregistered audit signals in held-out units. | Method-performance or graph-recovery confirmation. |

## Central claim matrix

| ID | Claim | Evidence | Status | Permitted wording | Prohibited reading |
| --- | --- | --- | --- | --- | --- |
| C1 | For `F(X)=f(X,g(X))`, total raw attribution equals the direct partial term plus the auxiliary-route chain term. | Chain rule; float64 numerical fixture. | Theoretical identity. | The x-only partial score equals total raw attribution on a declared support when the indirect term is zero there. | Coverage alone identifies a causal graph. |
| C2 | A route omitted from both x-only score and x-only Jacobian penalty can carry prediction while the reported x-only score vanishes or becomes uninformative. | Controlled concat proposition and proof. | Structural existence result. | Low x-only regularized objective does not certify that the x-only score contains all predictive information used by the model. | Optimization always selects this solution; all auxiliary conditioning is invalid. |
| C3 | The audit framework records score-route completeness, penalty-route completeness, score-penalty alignment, coordinate validity, and horizon validity independently. | Formal declaration, reusable report schema, tested fixtures. | Constructive framework. | Audit output is a multi-dimensional diagnostic profile. | A single precedence label or mathematical guarantee. |
| C4 | Controlled concat capacity can improve pure prediction MSE while weakening partial graph ranking. | Phase 8 Track A five-pair replication. | Replicated controlled evidence. | Prediction and reported graph knowledge decoupled under the declared concat construction. | Universal monotonic capacity law or external benchmark result. |
| C5 | Fixed-target interventions show that the auxiliary route contributes to concat predictions. | Corrected Phase 8 interventions and Phase 9 Stage B `mask_c` confirmation. | Controlled plus held-out route-use evidence. | Perturbing the frozen auxiliary route increased fixed-target pure MSE in every held-out Stage B architecture-data unit. | Invariant causal contribution or dominance over the raw route. |
| C6 | Partial scoring can lose coefficient fidelity in a controlled VAR construction. | Phase 8 five-pair coefficient replication. | Replicated controlled evidence. | Partial graph ranking and lag-1 coefficient meaning can degrade even when prediction improves. | Cross-domain coefficient-recovery generalization. |
| C7 | Expanding penalty-route coverage mitigated the controlled concat failure, but the tested full-prefix repair traced a graph-prediction frontier and did not pass the joint gate. | Frozen full auxiliary-penalty variants and Phase 8 final lambda study. | Boundary evidence. | Coverage interventions localize a trade-off between graph recovery and pure prediction fit. | Successful repair, selected lambda, optimality, or method superiority. |
| C8 | Filtered-coordinate scores can diverge from original-input raw-chain attribution when cross-channel transformations destroy source-variable identity. | Five-seed P0 semantic audit. | Semantic diagnostic. | Legacy ISTF-Mamba illustrates coordinate ambiguity. | Legacy graph performance, ISTF effectiveness, or Mamba effectiveness. |
| C9 | Coordinate-wise semantic repair did not establish CP-depthwise as a competitive method. | Official Stage 1a and P1 artifacts. | Negative boundary evidence. | Semantic gates passed; performance and novelty gates failed; the bounded postmortem remained inconclusive. | Failure of all filtering hypotheses. |
| C10 | Omitted-route attribution, fixed-target auxiliary-route use, and partial-total score disagreement reproduced across held-out units, two causal preprocessors, and two domains. | Formal Stage B decision from 36/36 release-locked runs. | Confirmatory bounded generality evidence. | Use the exact allowed claim below. | Graph-performance confirmation, universal architecture validity, or successful repair. |
| C11 | Bounded H=64 attribution captured nearly all measured H=128 mass and did not change nominal scores in Stage B, but mass beyond H=128 was not assessed. | Stage B horizon summaries. | Bounded horizon evidence. | The measured audit signals were stable between H=64 and H=128. | Full-prefix completeness or absence of earlier-history dependence. |
| C12 | The route-coverage discrepancy persists at a strong known-graph operating point. | Fresh Lorenz-96 confirmation: 20/20 release-locked runs, five data-seed units after averaging two train seeds. | Confirmatory strong-operating-point case study. | Baseline mean total nominal AUROC was 0.863; concat lowered pure MSE while its x-only partial score omitted substantial total raw-chain attribution. | Mamba/ISTF superiority, graph-recovery improvement, universal concat invalidity, successful repair, or strong significance from n=5. |

## Exact Stage B claim

> Under a preregistered bounded raw-chain audit, omitted-route attribution,
> fixed-target auxiliary-route use, and partial-versus-total score disagreement
> were reproduced across held-out NetSim subjects, nonoverlapping MoCap
> segments, and two causal auxiliary preprocessors.

## Stage B quantitative support

| Quantity | Mamba concat | Causal TCN concat |
| --- | ---: | ---: |
| Passed architecture/horizon checks | 11/11 | 11/11 |
| Missing-route qualifying units | 6/6 | 6/6 |
| Positive fixed-target `mask_c` units | 6/6 | 6/6 |
| NetSim partial-total discrepancy units | 4/4 | 4/4 |
| Median partial-total Pearson | 0.480 | 0.779 |
| Median coordinate entropy | 0.969 | 0.965 |
| Median temporal tail mass | 0.258 | 0.183 |
| Minimum H64/H128 mass ratio | 0.9999 | 1.0000 |
| Nominal score H64-vs-H128 max difference | 0 | 0 |

Baseline partial and total nominal scores were identical, and baseline
missing-route magnitude was zero. The held-out NetSim baseline median AUROC
was 0.495, so direct graph-performance success is explicitly absent.

## Exact Lorenz-96 claim

> In a fresh five-data-seed Lorenz-96 confirmation with baseline mean AUROC
> 0.863, the auxiliary concat architecture reduced pure prediction MSE while
> an x-only Jacobian score omitted substantial total raw-chain attribution:
> mean missing-route magnitude was 0.561, fixed-target auxiliary masking
> increased MSE by 0.528, and partial-versus-total nominal scores had mean
> Pearson correlation 0.840 and top-k Jaccard 0.658.

## Lorenz-96 quantitative support

| Quantity | Mean | Sample SD | Boundary |
| --- | ---: | ---: | --- |
| Baseline total nominal AUROC | 0.8628 | 0.0177 | Strong known-graph operating point |
| Concat total nominal AUROC | 0.8589 | 0.0149 | Not superior to baseline |
| Concat partial nominal AUROC | 0.8032 | 0.0110 | x-only score diagnostic |
| Concat total-minus-partial AUROC | +0.0557 | 0.0108 | Positive in 5/5 data seeds |
| Concat-vs-baseline relative pure MSE | -0.4156 | 0.1545 | Lower in 5/5 data seeds |
| Missing-route relative magnitude | 0.5614 | 0.0138 | Gate passed in 5/5 data seeds |
| Fixed-target `mask_c` MSE delta | 0.5275 | 0.0255 | Positive in 5/5 data seeds |
| Partial-total Pearson | 0.8401 | 0.0385 | Score-coverage diagnostic |
| Partial-total top-k Jaccard | 0.6577 | 0.0712 | Score-coverage diagnostic |

The statistical unit is the data seed after averaging two training seeds.
Concat total-score AUROC differed from baseline by only -0.0039 on average;
this result therefore strengthens the measurement diagnosis without becoming
method-performance evidence.

## Evidence tiers

| Tier | Evidence |
| --- | --- |
| Core mechanism | Route decomposition, controlled concat proposition, five-dimensional audit declaration. |
| Core controlled evidence | Phase 8 capacity, coefficient, intervention, and full-penalty diagnostics. |
| Core confirmatory evidence | Phase 9 Stage B held-out audit-generality result. |
| Core confirmatory case study | Phase 9 fresh Lorenz-96 strong-known-graph result. |
| Boundary evidence | Phase 8 repair trade-off; Stage 1a semantic-pass/performance-fail result. |
| Appendix diagnostic | Legacy filtered-coordinate/raw-chain disagreement. |
| Excluded | Historical CausalTime performance narrative and unaudited root-cause synthetic figures. |

## Manuscript hard boundaries

1. NetSim is not presented as a successful graph-recovery benchmark.
2. MoCap supports route use, score semantics, stability, and plausibility only.
3. Two architectures and two domains support bounded reproduction, not
   architecture universality.
4. H=64/H=128 agreement does not establish attribution completeness beyond
   H=128.
5. No failed repair is promoted to a positive method contribution.
6. Lorenz-96 supports a score-coverage diagnosis, not auxiliary-model
   superiority; five data-seed units do not support strong significance claims.
7. Every empirical number must map to the traceability register and a frozen
   file hash before submission.
