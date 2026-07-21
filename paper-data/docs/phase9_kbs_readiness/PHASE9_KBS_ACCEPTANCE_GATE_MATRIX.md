# Phase 9 KBS Acceptance Gate Matrix

Status: `PROPOSED FOR ADVISOR REVIEW`

## Decision levels

- `NOT_READY`: any integrity, theory-validity, executable-audit, or provenance
  hard gate fails.
- `KBS_SUBMISSION_READY`: all mandatory gates G0--G4 and G6 pass. This means the
  audit contribution is defensible; it does not guarantee acceptance.
- `KBS_STRONG_READY`: `KBS_SUBMISSION_READY` plus G5 mitigation generalization
  and the prospective diagnostic-validity endpoint pass. This is the target
  corresponding to the user's request for a materially safer KBS submission.
- `METHOD_PLUS_AUDIT_STRETCH`: requires a separately preregistered new repair
  that beats simple controls. It is outside the current plan.

## Current baseline diagnosis

| Dimension | Current Phase 8 status | Gap to strong readiness |
|---|---|---|
| Scope and scientific integrity | Strong | Preserve all frozen claim boundaries |
| Structural mechanism | Strong in controlled concat | Generalize beyond one route construction |
| Theory | One existence proposition plus chain-rule discussion | Add bounded validity, coordinate, horizon, and ranking results |
| Audit framework | Declaration and labels in manuscript | Deliver executable, tested API and machine-readable report |
| Controlled evidence | Strong but mainly synthetic VAR | Add nonlinear/nonstationary and cross-architecture prospective validation |
| External validity | No valid active-mainline external case | Add one known-graph external benchmark and one real no-ground-truth case |
| Mitigation | Controlled full-penalty recovery; final repair no-go | Replicate a fixed mitigation in a second regime without tuning |
| Statistics | Five-pair diagnostics; three data-seed repair pilot | Use data-generating unit, paired uncertainty, and preregistered endpoints |
| Reproducibility | Strong internal release locks | Public archival release, dataset licenses, runnable audit example |

## G0. Integrity and claim-scope hard gates

All must pass.

1. Baseline JRNGC is not described as universally defective. The vulnerability
   is restricted to architectures whose interpreted graph omits a predictive
   route, loses source-coordinate identity, or truncates relevant support.
2. Phase 8 no-go outcomes remain unchanged: no CP revival, no positive
   full-prefix-repair claim, no hidden lambda selection, and no Stage 1b.
3. Legacy ISTF-Mamba remains a score-semantics diagnostic only. Legacy
   CausalTime results remain outside the active performance narrative.
4. Every numeric manuscript claim maps to an immutable artifact and score
   definition. No field named `pred_loss` is interpreted as pure prediction MSE
   unless its decomposition is directly available.
5. Every external dataset has a source URL, license/terms record, raw-file
   SHA256, immutable raw directory, preprocessing manifest, and graph-object
   declaration.
6. Author, funding, conflict, contribution, code URL, data URL, and archival DOI
   placeholders are zero before submission.

## G1. Theory-validity gates

The theory contract must contain all four bounded results below, with explicit
assumptions and no causal-identifiability overclaim.

1. **Route decomposition and local score validity.** For
   `F(X)=f(X,g(X))`, prove `D_X F=partial_X f+partial_c f D_X g` and state that
   the x-only partial score equals total raw attribution on the declared support
   exactly when the indirect term is zero there.
2. **Coordinate-preservation sufficient condition.** Show that a
   variable-separable causal transformation has a block-diagonal raw-input
   Jacobian and therefore preserves source blocks under raw-chain attribution.
   Cross-variable mixing is labelled ambiguous unless an explicit source map is
   supplied and audited.
3. **Horizon closure.** If a predictor consumes `K` transformed lags and each
   transformed point has finite causal support `R`, the raw support is at most
   `K+R-1`. Stateful/infinite-memory transforms require a reported truncation
   rule and omitted-mass audit.
4. **Ranking stability.** If two off-diagonal score vectors differ by at most
   `delta` in sup norm and the exact-top-k boundary margin exceeds `2*delta`,
   the selected top-k edge set is unchanged.

Pass conditions:

- proofs are independently checked by the advisor;
- every result has one passing numerical fixture and one boundary/counterexample
  fixture;
- the manuscript distinguishes local derivative validity, predictive Granger
  interpretation, and causal identifiability;
- no theorem claims that coverage alone establishes a true causal graph.

## G2. Executable-audit gates

The framework must be delivered as reusable software rather than prose alone.

Required outputs:

- coverage declaration and route ledger;
- partial nominal, total nominal, reliable-history, and unrestricted-prefix
  score objects;
- missing-route magnitude, partial-total Pearson/Spearman, exact-top-k Jaccard,
  coordinate leakage, tail mass, eligible-window counts, and uncertainty;
- per-dimension labels: `COVERED`, `PARTIALLY_COVERED`,
  `COORDINATE_AMBIGUOUS`, `HORIZON_TRUNCATED`, or `UNASSESSED`;
- JSON/CSV report, saved score arrays, schema version, code/config hashes, and a
  human-readable summary.

Numerical hard gates:

- finite differences satisfy
  `abs_error <= 1e-5 + 1e-3*max(abs(fd),abs(autograd))`;
- baseline partial and total nominal scores have max absolute difference
  `<=1e-7` in native float32 tests;
- float64 chunked versus unchunked `J_bar` and score differences are `<1e-7`;
- same-seed deterministic score difference is `<1e-7` on CPU;
- exact-top-k edge orientation is `(source,target)` and edge sets match the
  reference exactly;
- detached raw inputs, target leakage, future dependence, nonfinite values, or
  missing route declarations cause a hard failure rather than a warning;
- diagonal entries remain in raw tensors and are excluded only by graph-metric
  adapters.

## G3. Prospective controlled-generality gates

Validation must cover at least:

- two trained data-generating families: linear VAR(1) and nonlinear,
  nonstationary lagged dynamics;
- three architecture route families: auxiliary concat, coordinate-wise finite
  filtering, and cross-channel/stateful transformation;
- matched raw-history baseline and a fixed full-auxiliary-penalty comparator;
- five independent data-generating seeds per formal cell, with two model seeds
  averaged before data-seed-level inference.

Canonical control gates:

- raw baseline: partial-total max difference `<=1e-7`;
- coordinate-wise finite transform: cross-variable leakage `<1e-8`;
- deliberately mixed transform: cross-variable leakage `>=0.10` and
  `COORDINATE_AMBIGUOUS` in every formal data seed;
- finite-memory horizon fixture: omitted mass `<0.01` once the analytically
  sufficient horizon is used;
- deliberately truncated long-memory fixture: tail mass `>0.10` and
  `HORIZON_TRUNCATED` in at least 4/5 data seeds;
- auxiliary-route fixture: missing-route relative magnitude `>=0.10` and either
  partial-total Pearson `<0.95` or exact-top-k Jaccard `<0.90` in at least 4/5
  data seeds.

Prospective diagnostic-validity endpoint for `KBS_STRONG_READY`:

- each audit dimension is assessed against its matching discrepancy, not a
  post-hoc composite risk score;
- across at least 20 independent data-seed-level conditions per dimension,
  Spearman correlation between severity and the matching score disagreement is
  `>=0.50`, with a paired/bootstrap 95% confidence interval whose lower bound is
  above zero;
- no dimension may show a statistically supported effect in the opposite
  preregistered direction;
- association with direct-graph degradation is reported as secondary unless the
  matched graph object and comparator are identical across conditions.

## G4. External-validity gates

At least two external cases are mandatory:

1. **Known-graph benchmark:** DREAM3 in-silico gene networks or NetSim
   simulated-fMRI networks, reconstructed under the Phase 9 score contract.
   Old score arrays cannot be reused as Phase 9 evidence.
2. **Real no-ground-truth application:** human motion capture is preferred
   because it is established in neural Granger literature and already available
   locally. It supports audit status, seed stability, predictive fit, and
   anatomy-informed plausibility only, not true-edge accuracy.

Pass conditions:

- the known-graph benchmark uses at least five independent networks/subjects as
  the analysis units and reports AUROC, AUPRC, exact-top-k F1/MCC/SHD, partial
  versus total scores, and all semantic diagnostics;
- at least one benchmark has baseline AUROC `>=0.60`, preventing an audit claim
  from being based only on random-level graph recovery;
- all methods use identical windows, targets, normalization, graph orientation,
  and off-diagonal handling;
- the real case reports at least five independent training runs, a seed-stability
  matrix, intervention sensitivity, and the complete coverage profile;
- absence of ground truth is stated in the abstract/caption if the real case is
  promoted to the main text.

## G5. Fixed-mitigation generalization gate

This gate does not authorize a new method. It tests a frozen, simple mitigation:
full auxiliary-coordinate Jacobian regularization with the already identified
strong fixed ratio `lambda_c/lambda_x=10`. No graph-truth-based tuning is
allowed.

On at least one new nonlinear/nonstationary controlled family, compared with
paired concat x-only:

- mean delta AUROC `>=0.05`;
- mean delta AUPRC `>=0.05`;
- coefficient correlation improves by `>=0.10` where a coefficient target
  exists;
- graph improvement has the preregistered positive direction in at least 4/5
  data seeds;
- mean fixed-target pure-MSE degradation is `<=10%`, and no data seed exceeds
  `20%`;
- mean AUROC is not more than `0.05` below the matched baseline;
- equal-lambda is retained as a fixed comparator, while lc10 is not called
  optimal.

If G5 fails, the result is retained as boundary evidence. The paper may still
reach `KBS_SUBMISSION_READY`, but not the defined `KBS_STRONG_READY` state.

## G6. Statistics, reproducibility, and release gates

- data-generating seed/network/subject is the statistical unit;
- model seeds are averaged before paired data-unit inference;
- primary endpoints and direction are frozen before formal runs;
- effect sizes and paired 95% bootstrap confidence intervals are reported;
- multiplicity across co-primary graph metrics is controlled with Holm where
  inferential claims are made;
- no claim relies on `n=3` significance testing;
- all runtime, failure, NaN/Inf, and excluded-run records are retained;
- a clean environment can reproduce the audit example, figures, and tables;
- public code and data releases have immutable tags/DOIs, licenses, SHA256
  manifests, and exact commands.

## Final decision rule

Evaluate in order: G0 -> G1 -> G2 -> G3 -> G4 -> G6 -> G5.

- Any G0--G4 or G6 hard failure: `NOT_READY`.
- G0--G4 and G6 pass, G5 fails: `KBS_SUBMISSION_READY_WITH_NEGATIVE_MITIGATION_BOUNDARY`.
- G0--G6 pass: `KBS_STRONG_READY`.
- No failed gate may be relaxed after formal results are viewed.

