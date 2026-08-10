# Phase 9 Audit-Generality Prospective Validation Preregistration v1

## Status and purpose

This document freezes a compact two-stage audit-generality validation before
any listed run is executed. It does not authorize execution.

The scientific question is whether the Jacobian coverage audit detects omitted
predictive routes and score-semantic disagreement beyond the Phase 8 controlled
concat fixture, across:

- held-out NetSim subjects with graph annotations;
- nonoverlapping MoCap segments without graph ground truth;
- two distinct causal auxiliary preprocessors: Mamba and TCN.

This is not a new repair-method study. The stopped adaptive-FIR,
contextual-FIR, and gradient-projection lines remain stopped.

## Evidence stages

### Stage A: prospective RTX 4090 validation

- Environment: the dedicated Windows RTX 4090 host.
- Role: internal prospective go/no-go evidence.
- Data units: four held-out NetSim subjects and two nonoverlapping MoCap
  segments.
- Methods: baseline JRNGC, Mamba concat/x-only penalty, and causal TCN
  concat/x-only penalty.
- Training replicates: three fresh seeds per data unit and method.
- Primary training records: `6 data units x 3 methods x 3 seeds = 54`.
- Determinism duplicates: two non-primary runs, one Mamba and one TCN.
- Stage A values cannot enter the manuscript as confirmation evidence.

### Stage B: conditional AutoDL confirmation

- Stage B remains unauthorized until Stage A passes every unlock gate and the
  external advisor explicitly approves execution.
- Data units: four separately sealed NetSim subjects plus the same two frozen
  MoCap segments.
- Methods: the same three methods.
- Training replicates: two fresh seeds.
- Conditional records: `6 x 3 x 2 = 36`.
- No Stage B subject array or output may be inspected before unlock.

The complete prospective matrix therefore contains 90 primary records plus two
Stage A determinism duplicates. Only 54 primary records and two duplicates are
eligible for the first execution authorization.

## Frozen data units

NetSim identifiers were selected without reading arrays or graph metrics.
Subjects 48 and 49, used in development, were excluded. The remaining integer
identifiers 0--47 were sorted by:

`SHA256("phase9-audit-formal-v1|netsim-subject|<integer_id>")`.

The first four are Stage A subjects 19, 8, 44, and 3. The next four are sealed
Stage B subjects 16, 0, 30, and 10. Exact selection and file hashes are in
`PHASE9_AUDIT_SEED_SUBJECT_FREEZE.json`.

MoCap uses:

- run angles `[728, 1228)`, separated from development `[0, 600)`;
- salsa angles `[3000, 3500)`, separated from development `[1000, 1600)`.

Each frozen segment is normalized by per-variable mean and standard deviation
computed on that segment. This is explicitly an in-sample architecture audit,
not forecasting evaluation.

## Frozen model and training configuration

Common configuration:

| Item | Value |
| --- | --- |
| lag `K` | 3 |
| predictor layers / hidden | 2 / 32 |
| dropout | 0 |
| auxiliary dimension | 4 |
| Jacobian lambda | 0.01 |
| optimizer | legacy Adam path |
| learning rate / weight decay | `1e-3 / 0` |
| maximum iterations | 1000 |
| checkpoint | legacy minimum checked total objective |
| checked interval | 50 iterations |
| primary audit horizon | 64 |
| primary audit windows | 32 deterministic windows |
| horizon sensitivity | 32, 64, 128 on identical targets |
| mask value | zero |
| shuffle | time permutation within each source coordinate |
| target policy | original clean raw target fixed |

The Mamba and TCN concat models must receive identical initial predictor
tensors wherever parameter names and shapes coincide. Predictor,
preprocessor, perturbation, and score-window seeds are stored separately. The
baseline is deterministic under its recorded predictor seed but is not claimed
to have shape-identical initialization to concat predictors.

## Audit objects

For target window `w`, prediction target `j`, raw source `i`, and raw lag `h`:

`J_total[w,j,i,h] = d y_hat[w,j] / d x[u_w-h,i]`.

The direct/raw-X partial derivative holds the auxiliary sequence fixed:

`J_partial[w,j,i,h] = partial y_hat[w,j] / partial x[u_w-h,i]`.

Absolute values are taken per window before averaging:

`Jbar_total[j,i,h] = mean_w |J_total[w,j,i,h]|`,

`Jbar_partial[j,i,h] = mean_w |J_partial[w,j,i,h]|`.

The missing-route magnitude is:

`M_missing = sum_{j != i,h} mean_w |J_total-J_partial|`
` / (sum_{j != i,h} Jbar_total + 1e-12)`.

The primary direct graph objects use only the declared nominal lag support:

`S_total_nominal[j,i] = max_{1 <= h <= K} Jbar_total[j,i,h]`,

`S_partial_nominal[j,i] = max_{1 <= h <= K} Jbar_partial[j,i,h]`.

The bounded-history score `max_{1 <= h <= H} Jbar_total[j,i,h]` is a route-use
diagnostic only. It is not a direct Granger graph score.

Partial-versus-total Pearson correlation is computed on the off-diagonal
nominal score vectors. NetSim top-k Jaccard uses the frozen number of
off-diagonal graph edges; diagonal entries are retained in raw tensors and
excluded only at metric evaluation.

Per-window temporal tail mass is:

`sum_{j != i,h > K} |J_total[w,j,i,h]|`
` / (sum_{j != i,h <= H} |J_total[w,j,i,h]| + 1e-12)`.

For auxiliary coordinate `q`, source shares are formed from L1 raw-source
gradient mass across the bounded horizon. Coordinate entropy is normalized by
`log(d)`. It measures source mixing; it does not prove causal invalidity.

## Fixed-target interventions

Every condition retains the original unperturbed raw target:

- `mask_x`: raw-X predictor route is zeroed; clean auxiliary route is retained;
- `mask_c`: auxiliary route is zeroed; clean raw-X route is retained;
- `mask_both`: both routes are zeroed;
- `shuffle_x_only`: raw-X is time-shuffled per variable; clean auxiliary route
  is retained;
- `shuffle_c_only`: auxiliary time positions are shuffled; raw-X is retained;
- `shuffle_both_routes`: raw-X is shuffled and the auxiliary route is
  recomputed from that same shuffled raw-X.

The primary intervention value is pure fixed-target prediction-MSE delta.
Legacy regularized-objective deltas are secondary diagnostics only.

## Statistical unit and aggregation

The data unit, not the training seed, is the statistical unit.

1. Compute every metric per training run.
2. Average training replicates within each data unit and architecture.
3. Apply direction counts and thresholds to the six unit-level values.
4. Report median, range, and all unit-level values; do not make strong
   significance claims from six units.
5. NetSim graph metrics and MoCap no-ground-truth diagnostics remain separate.

## Stage A semantic integrity gates

All must pass:

1. No NaN/Inf in predictions, Jacobians, scores, interventions, or coordinate
   diagnostics.
2. Causality, fixed-target isolation, edge orientation, lag indexing, and raw
   coordinate differentiation tests pass.
3. Baseline partial and total nominal scores differ by at most `1e-7`, and
   baseline `M_missing <= 1e-7`.
4. Mamba and TCN audit profiles both declare an auxiliary predictive route.
5. Mamba/TCN score targets, perturbations, and evaluation windows match within
   each data unit and replicate.
6. Determinism duplicates have loss/score maximum absolute difference
   `<=1e-6`, metric difference `<=1e-6`, and identical top-k edges.

The development runner currently labels the TCN auxiliary-route profile
incorrectly because it tests `method == "concat"`. A new configuration-driven
validation runner must correct this before release.

## Stage A audit-generality gates

Each concat architecture must independently satisfy all of the following:

1. At least 5/6 data units have unit-level mean `M_missing >= 0.20`; in each
   qualifying unit at least 2/3 seeds also satisfy `M_missing >= 0.20`.
2. At least 5/6 units have positive unit-level mean `mask_c` fixed-target MSE
   delta; in each qualifying unit at least 2/3 seeds are positive. The median
   unit-level delta must be at least `0.05`.
3. Median unit-level partial-total nominal Pearson is `<=0.90`. Undefined
   correlations count as failure, not success.
4. At least 3/4 NetSim subjects satisfy either partial-total Pearson `<=0.90`
   or exact-top-k Jaccard `<=0.80`.
5. Median normalized source-coordinate entropy is `>=0.75`.
6. Median temporal tail mass is `>=0.05`.

Both Mamba and TCN must pass. One passing architecture yields only
`PARTIAL_GENERALITY`; it does not unlock cross-architecture confirmation.

## Horizon and reliability gates

For both concat architectures, using exactly the same target indices:

- nominal score maximum absolute difference across H=32/64/128 is `<=1e-7`;
- H=64 cumulative off-diagonal mass relative to H=128 is `>=0.99` in at least
  5/6 data units;
- the minimum H=64/H=128 cumulative mass across units is `>=0.95`.

Failure stops the protocol. The horizon is not silently changed after results.
Passing supports only a bounded H=64 audit. Attribution beyond H=128 remains
unassessed and the label remains `HORIZON-TRUNCATED`.

## Graph-performance interpretation

NetSim AUROC, AUPRC, and exact-top-k metrics are reported for baseline,
partial nominal, and total nominal scores. They are contextual diagnostics, not
success gates for a new method. MoCap has no accepted direct graph truth and
must not receive graph-recovery claims.

If median NetSim baseline AUROC is below 0.60, known-graph evidence is labelled
weak-operating-point and cannot support score-correction claims. Route-use and
partial-total semantic results may still be reported within their own scope.

## Stage A decision

`UNLOCK_STAGE_B` requires:

- all semantic integrity gates;
- both architecture-specific generality gates;
- all horizon gates;
- exact artifact completeness and source/config hash checks;
- total Stage A GPU time below the five-hour hard cap.

Any failure produces a frozen negative validation report. Thresholds,
subjects, segments, seeds, methods, and horizons are not changed after
inspection.

## Stage B confirmation rule

Stage B requires separate advisor authorization and a clean frozen release.
The same gates apply, except each qualifying unit requires both 2/2 training
replicates to meet directional per-run conditions. Both Stage A and Stage B
must pass before manuscript use.

## Allowed claim after Stage A and Stage B pass

The strongest allowed statement is:

> Under a preregistered bounded raw-chain audit, omitted-route attribution,
> fixed-target auxiliary-route use, and partial-versus-total score
> disagreement were reproduced across held-out NetSim subjects,
> nonoverlapping MoCap segments, and two causal auxiliary preprocessors.

This does not establish universal architecture coverage, full-prefix
attribution, improved graph recovery, causal ground truth for MoCap, or a
successful repair method.

## Execution boundary

Before advisor approval:

- do not implement the validation runner;
- do not copy or open selected subject arrays on the 4090 host;
- do not execute any matrix row;
- do not enable AutoDL;
- do not modify either manuscript;
- do not inspect Phase 7 seeds 4--8 or Stage 1b outputs.
