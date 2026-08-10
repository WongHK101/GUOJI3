# Phase 9 Theory and Executable-Audit Contract

Date: 2026-07-26

## Purpose

The theory identifies when a reported Jacobian object is locally aligned with a
declared Granger-predictive graph claim. It does not prove causal
identifiability, causal sufficiency, or recovery under hidden confounding.

## Result T1: route decomposition and local score validity

Let `F(X)=f(X,g(X))`, where `g` is a differentiable causal auxiliary
transformation. Then:

`D_X F = partial_X f + partial_c f D_X g`.

On any declared target-source-lag support, the partial x-only derivative equals
the total raw derivative whenever the indirect term
`partial_c f D_X g` is zero on that support. The converse is stated only for the
sum on the declared support; cancellations can make the total difference zero
without either factor being zero.

Required fixtures:

- float64 autograd-versus-decomposition equality;
- nonzero indirect-route counterexample;
- cancellation boundary example discussed in prose.

## Result T2: coordinate-preservation sufficient condition

If each transformed coordinate `z_i(t)` depends only on the raw history of
source variable `x_i`, then `D_X g` is block diagonal in the source-variable
partition. Raw-chain attribution can therefore be grouped by the original
source blocks without cross-source ambiguity.

This is sufficient, not necessary. A cross-channel map may still have a known
invertible source map, but it must be declared and audited separately.

Required fixtures:

- depthwise finite filter with cross-source leakage below numerical tolerance;
- cross-channel perturbation with nonzero leakage and
  `COORDINATE-AMBIGUOUS`.

## Result T3: finite horizon closure

If the predictor consumes transformed lags `1,...,K` and every transformed
value has finite causal support `R`, the union of raw lag support is bounded by
`K+R-1`.

Stateful or infinite-memory transformations require a declared truncation
horizon and omitted-mass audit. Agreement between two finite horizons does not
prove absence of mass before both horizons.

Required fixtures:

- exact finite-support example with upper bound `K+R-1`;
- EMA boundary showing nonzero mass beyond a finite truncation.

## Result T4: top-k ranking stability

Let two off-diagonal score vectors differ by at most `delta` in sup norm. If the
reference exact-top-k boundary margin exceeds `2 delta`, the selected top-k
edge set is unchanged.

This is a sufficient local stability condition. A failed margin test is not
proof that the edge set must change.

Required fixtures:

- passing margin example with identical top-k set;
- failed sufficient condition with an actual rank crossing.

## Executable report contract

Implementation:

`src/jacobian_coverage_audit.py`

Schema:

`jacobian-coverage-audit/1.0`

Required declaration:

- architecture and graph claim;
- score and penalty variable sets;
- every architecture-declared predictive route;
- score and penalty coverage per route;
- explicit penalty exemptions;
- coordinate map and validity status;
- primary score horizon;
- evaluated attribution horizon and required support when known;
- score-penalty coordinate and horizon relation.

Required profile dimensions:

- score-route completeness;
- penalty-route completeness;
- score-penalty alignment;
- coordinate validity;
- horizon validity.

Allowed labels:

- `COVERED`;
- `PARTIALLY COVERED`;
- `COORDINATE-AMBIGUOUS`;
- `HORIZON-TRUNCATED`;
- `UNASSESSED`.

The labels are diagnostic and are never collapsed into a causal-validity
certificate.

Required machine-readable outputs:

- JSON report with schema version, declaration, profile, diagnostics,
  provenance, and score-file references;
- CSV profile table;
- saved partial nominal, total nominal, bounded-history, and raw Jacobian
  arrays where applicable.

Non-evidentiary schema example:

`paper-data/docs/phase9_stageb_manuscript/example_audit/`

The example includes JSON, CSV, and small illustrative score arrays. It is a
format fixture only and is explicitly excluded from the scientific evidence
base.

Hard failures:

- undeclared predictive route;
- duplicate route identifier;
- invalid score or penalty route reference;
- nonfinite diagnostic;
- missing provenance;
- unsupported schema version;
- detached raw coordinate in a raw-chain evaluation;
- future-target dependence;
- missing score-object file declaration.

## Current verification

`tests/test_jacobian_coverage_audit.py` covers:

- independent dimension labels;
- explicit-exemption validation;
- float64 route-chain decomposition;
- coordinate-preserving and mixed fixtures;
- finite-horizon and long-memory boundary;
- top-k stability and rank-crossing boundary;
- JSON/CSV serialization;
- nonfinite-value rejection.

Existing Phase 8/9 tests retain finite-difference, target-isolation,
orientation, deterministic-window, chunked evaluator, and formal release-lock
coverage.
