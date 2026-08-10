# Phase 9 Executable Jacobian Coverage Audit Specification

Status: `DESIGN LOCK CANDIDATE; NO CODE AUTHORIZED YET`

## Ownership

The implementation should add a new module, tentatively
`src/jacobian_coverage_audit.py`, and a CLI under `experiments/`. Existing
Phase 8 classes and frozen evaluators remain immutable. The new module composes
and generalizes tested helpers from:

- `src/phase8_coverage.py`;
- `src/repaired_istf.py`;
- `src/knowledge_metrics.py`.

It must not change the numerical meaning of any frozen Phase 8 artifact.

## Public API

```python
audit_model(
    model,
    raw_series,
    *,
    graph_claim,
    route_ledger,
    score_spec,
    penalty_spec,
    target_indices,
    true_graph=None,
    source_map=None,
    thresholds=None,
) -> CoverageAuditReport
```

The model adapter must expose prediction from the original raw tensor and, when
applicable, partial-coordinate prediction interfaces. The audit refuses to run
if total attribution cannot be traced back to the supplied raw tensor.

## Required declaration schema

```json
{
  "schema_version": "phase9.coverage-audit.v1",
  "graph_claim": {
    "object": "predictive_granger_direct_graph",
    "target_domain": "raw",
    "source_domain": "raw",
    "nominal_lag_support": [1, 3],
    "diagonal_policy": "retain_in_tensor_exclude_in_metrics"
  },
  "routes": [
    {
      "route_id": "raw_history",
      "origin": "raw",
      "enters_prediction": true,
      "score_status": "included",
      "penalty_status": "included",
      "exemption_reason": null
    }
  ],
  "coordinate_map": {
    "type": "identity|variable_separable|explicit_matrix|unknown",
    "source_axis": "target_source_lag"
  },
  "attribution_horizon": {
    "mode": "finite_exact|truncated|full_prefix",
    "evaluated_max_lag": 32,
    "structural_support": null
  }
}
```

Missing route status, source map, horizon rule, or graph object is a hard input
error, not `UNASSESSED`. `UNASSESSED` is reserved for a declared dimension that
cannot be evaluated after a valid attempt.

## Attribution tensors and graph objects

For derivative object `q`, eligible target windows `U_h`, output target `j`, raw
source `i`, and raw lag `h`:

\[
\bar J^{(q)}_{jih}=
\frac{1}{|U_h|}\sum_{u\in U_h}
\left|\frac{\partial\widehat x_{u,j}}
{\partial x_{u-h,i}}\right|.
\]

Absolute value is taken per window before float64 streaming accumulation.
Lag `h=1` is array index zero. Counts are stored for every lag.

Primary direct graph object:

\[
S^{(q)}_{\mathrm{GC},ji}=\max_{1\le h\le K}\bar J^{(q)}_{jih}.
\]

Secondary reliable-history object:

\[
H_{\mathrm{reliable}}=\{h:|U_h|\ge
\max(20,\lceil0.10|U|\rceil)\},\quad
S^{(q)}_{\mathrm{hist},ji}=\max_{h\in H_{\mathrm{reliable}}}
\bar J^{(q)}_{jih}.
\]

The unrestricted prefix maximum is exploratory and stores its maximizing lag,
window count, and reliable-support status for every edge.

## Metrics

### Missing-route magnitude

The Phase 8-compatible primary definition is

\[
M_{\mathrm{missing}}=
\frac{\sum_{j\ne i,h}\bar J^{(\mathrm{missing})}_{jih}}
{\sum_{j\ne i,h}\bar J^{(\mathrm{total})}_{jih}+\epsilon},
\]

where the missing tensor is accumulated as
`abs(J_total_window-J_partial_window)` before the eligible-window mean. The
quantity can exceed one under signed cancellation and is therefore called a
relative magnitude, not a bounded fraction. If total off-diagonal mass is zero,
the value is `null` with `zero_total_offdiagonal_mass`.

A bounded secondary diagnostic may also be saved:

\[
M_{\mathrm{sym}}=
\frac{\sum|J_{\mathrm{total}}-J_{\mathrm{partial}}|}
{\sum|J_{\mathrm{total}}|+\sum|J_{\mathrm{partial}}|+\epsilon}.
\]

It must not replace the preregistered Phase 8-compatible endpoint after results
are viewed.

### Partial-total agreement

- Pearson and Spearman use the same flattened off-diagonal direct-score vector.
- Constant vectors return `null` plus an `undefined_reason`; NaN is never treated
  as passing.
- exact-top-k uses the known edge count when available, deterministic tie
  breaking, and `(source,target)` edge orientation.
- report score max/mean absolute difference and the kth boundary margin.

### Coordinate leakage

For a one-to-one declared variable map:

\[
L_{\mathrm{coord}}=
\frac{\sum_{i\ne j,t,s}|\partial z_i(t)/\partial x_j(s)|}
{\sum_{i,j,t,s}|\partial z_i(t)/\partial x_j(s)|+\epsilon}.
\]

For grouped variables, replace the off-diagonal mask with an off-declared-block
mask. Coordinate scale affects this L1 quantity; preprocessing units and any
standardization must be fixed and reported. No cross-method magnitude comparison
is allowed under different scaling.

### Temporal tail mass

For each evaluation window:

\[
M_{\mathrm{tail}}(u)=
\frac{\sum_{j\ne i,h>K}|J^{\mathrm{total}}_{ujih}|}
{\sum_{j\ne i,h\ge1}|J^{\mathrm{total}}_{ujih}|+\epsilon}.
\]

Report mean, median, p95, maximum, defined count, undefined count, evaluated
horizon, and omitted-mass comparison where available. All lags contribute to
tail diagnostics even when low-support lags are excluded from reliable-history
ranking.

## Label engine

Labels are diagnostic outcomes, not mathematical certificates.

- `COVERED`: all dimensions applicable to the declared claim pass and none are
  unresolved.
- `PARTIALLY_COVERED`: at least one predictive route interpreted by the claim is
  absent from the score, or a non-exempt predictive route is absent from the
  penalty.
- `COORDINATE_AMBIGUOUS`: the declared source map is missing/non-unique or the
  configured off-block leakage gate fails.
- `HORIZON_TRUNCATED`: structural support exceeds the evaluated horizon or the
  configured tail/omitted-mass gate fails.
- `UNASSESSED`: a declared audit dimension could not be evaluated and has a
  recorded reason.

The report stores dimension-specific statuses separately. A model may be, for
example, penalty-route covered but score-route partial.

## Required tests

1. baseline partial-total equality;
2. chain-rule decomposition on concat and transformed routes;
3. float64 central finite differences for nonzero and near-zero entries;
4. raw-input detach negative test;
5. target isolation and future perturbation;
6. variable-separable structural zero leakage;
7. dense-mixing positive leakage;
8. finite-support horizon closure;
9. stateful tail detection;
10. eligible-window counts and lag-index mapping;
11. chunk sizes 1/4/32/64 versus unchunked float64 reference;
12. exact-top-k orientation, deterministic ties, and margin fixture;
13. zero-denominator/constant-vector handling;
14. diagonal retention versus metric exclusion;
15. same-seed determinism;
16. JSON-schema and save/load round trip;
17. route-ledger missing-field hard failures;
18. adapter parity with the frozen Phase 8 evaluator on an identical fixture.

## Deliverable contract

- versioned Python API and CLI;
- schema and example declaration files;
- unit-test report and coverage summary;
- deterministic example output;
- runtime/RAM/VRAM benchmark;
- user guide showing one covered and three known-failure fixtures;
- no manuscript claims until the tool and prospective validation both pass.

