# Phase 9 Theory Strengthening Contract

Status: `DOCUMENTATION ONLY; PROOFS REQUIRE ADVISOR REVIEW`

## Scope

The theory concerns the local derivative semantics of a trained predictive
Granger model. It does not establish causal sufficiency, invariance to hidden
confounding, or structural identifiability from observational data.

Let a raw history prefix be `X`, an architecture-declared transformed or
auxiliary route be `c=g(X)`, and the predictor be

\[
\widehat y = f(X,c).
\]

For target coordinate `j`, source coordinate `i`, and raw lag `h`, all score
statements are pointwise before window aggregation.

## Result 1: Route decomposition and local score validity

### Statement

If `f` and `g` are differentiable at the evaluated history, then

\[
D_X\,f(X,g(X))
=\partial_X f(X,g(X))+\partial_c f(X,g(X))D_Xg(X).
\]

Define

\[
J_{\mathrm{partial}}=\partial_X f,\qquad
J_{\mathrm{indirect}}=\partial_cf\,D_Xg,\qquad
J_{\mathrm{total}}=J_{\mathrm{partial}}+J_{\mathrm{indirect}}.
\]

On any declared source-lag support `S`, the partial score is locally equal to
the total raw-coordinate derivative if and only if
`J_indirect|_S=0`.

### Boundary

- Equality is local and first-order. It does not imply global functional
  independence or causal identification.
- A nonzero auxiliary-coordinate Jacobian does not by itself imply a nonzero raw
  indirect term; `D_X g` is also required.
- Cancellation across signed paths may make an aggregated score small. The
  stored tensor must therefore precede absolute-value and lag aggregation.

### Fixtures

- pass: baseline without an auxiliary route;
- pass: transformed route with `partial_c f=0`;
- fail/positive discrepancy: concat route with nonzero `partial_c f D_Xg`;
- boundary: nonzero factors whose signed product cancels at one point.

## Result 2: Source-coordinate preservation under separable transforms

### Sufficient condition

Let transformed coordinate `z_i(t)` depend only on the raw history of variable
`i`:

\[
z_i(t)=g_i(x_i(t),x_i(t-1),\ldots).
\]

Then

\[
\frac{\partial z_i(t)}{\partial x_j(s)}=0\quad\text{for }i\ne j,
\]

so the transform Jacobian is block diagonal by source variable. A raw-chain
derivative through this transform preserves the source-variable partition,
although it may extend temporal support.

### Cross-channel boundary

For `z(t)=g(x_{0:t})` with nonzero off-block derivatives, a transformed-coordinate
column is not automatically one original source variable. It is acceptable only
when either:

1. graph claims are explicitly about transformed coordinates; or
2. an explicit map back to raw variables is supplied and the graph is extracted
   from the raw-chain derivative.

Invertibility alone is not called source-identity preservation: a dense
invertible rotation still mixes source columns.

### Diagnostic consequence

For a declared one-to-one source map, define off-block leakage

\[
L_{\mathrm{coord}}=
\frac{\sum_{i\ne j,t,s}|\partial z_i(t)/\partial x_j(s)|}
{\sum_{i,j,t,s}|\partial z_i(t)/\partial x_j(s)|+\epsilon}.
\]

Variable-separable fixtures must satisfy `L_coord<1e-8`. A deliberately dense
mixing fixture must be detected as `COORDINATE_AMBIGUOUS`.

## Result 3: Causal horizon closure

### Finite-support statement

Suppose the predictor at target time `t` consumes transformed points
`z(t-1),...,z(t-K)`, and each `z(u)` depends only on raw points
`x(u),...,x(u-R+1)`. Then the prediction depends on raw lags no earlier than

\[
H_{\mathrm{raw}}=K+R-1.
\]

This is an upper bound; cancellations or zero coefficients may reduce actual
support.

### Stateful boundary

EMA, recurrent state-space models, and other prefix-stateful transforms may have
nonzero dependence before every fixed finite horizon. Such a model must report:

- evaluated horizon `H`;
- eligible-window count by lag;
- omitted/full-prefix mass where exact full-prefix evaluation is feasible;
- tail mass outside nominal `K`;
- the rule used to label `HORIZON_TRUNCATED`.

No finite `H` is called complete unless support is structurally finite or an
omitted-mass gate is passed against a longer/full-prefix reference.

## Result 4: Exact-top-k stability under score perturbation

Flatten the off-diagonal direct-graph scores into vectors `s` and `s'`, with a
deterministic tie rule. Let `s_(k)` and `s_(k+1)` be the kth and (k+1)th ordered
entries of `s` and define margin `m=s_(k)-s_(k+1)`.

If

\[
\|s-s'\|_\infty\le\delta\quad\text{and}\quad m>2\delta,
\]

then the exact-top-k edge set is unchanged.

### Reporting consequence

Every partial-total and nominal-extended score comparison should report both
`max_abs_difference` and `topk_boundary_margin`. Correlation alone cannot show
that the graph edge set is stable.

### Boundary

When `m<=2*delta`, the theorem is silent; it does not assert that the edge set
must change.

## Score and penalty relationship

The primary direct Granger score remains the total raw-coordinate score on the
declared nominal support `1:K`. A route-coverage penalty may intentionally cover
a broader raw prefix. The correct audit statement is:

- score coordinates: raw-variable aligned;
- primary score horizon: nominal `K`;
- penalty support: declared route-support superset;
- relation: coordinate aligned but not necessarily horizon-identical.

Penalty completeness cannot substitute for score completeness, and score
completeness cannot substitute for penalty coverage during training.

## Required proof and implementation audit

Before manuscript use:

1. a mathematical reviewer checks all quantifiers and support indexing;
2. tensor orientation is fixed as `[target, source, raw_lag]`;
3. each statement has symbolic or finite-difference fixtures;
4. aggregation is explicitly `abs per window -> mean eligible windows per lag ->
   claim-specific lag aggregation`;
5. the current v4 duplicate `Score-route completeness` list item is corrected
   only after Phase 9 manuscript revision is authorized;
6. theorem wording is mirrored in the Chinese review copy in the same session.

