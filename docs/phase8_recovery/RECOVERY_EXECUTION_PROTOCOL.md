# Phase 8 Recovery Execution Protocol

## Track A: independent replication

Track A uses immutable release commit
`dee0d30de66b84db1747d27debfedcfb93a15c93`. It executes only the 30 capacity,
10 fixed-target intervention, and 10 coefficient records. It does not use the
repair estimator and is aggregated under the frozen partial-score semantics.

## Track B: estimator recovery

Track B starts from `dee0d30` and changes only the repair estimator,
preflight diagnostics, validator, and non-evidentiary forensics. Comparator
forward, loss, checkpoint, and score paths remain immutable.

The new release runs:

1. CPU semantic, parity, exact-reference, and schedule tests;
2. fixed-lag float32/float64 forensics against initialization and stopped
   preflight iteration-20/final checkpoints;
3. four 20-iteration infrastructure smokes;
4. one 100-iteration repair benchmark;
5. all 30 pilot records only when every revised preflight gate passes.

## Revised preflight gates

- Nominal `h=1` contribution is finite and nonzero at every reporting point.
- Nominal predictor and preprocessor regularizer gradients are finite and
  nonzero at every reporting point.
- Cumulative 100-step historical contribution is finite and nonzero.
- Cumulative historical predictor and preprocessor gradient norms are finite
  and nonzero.
- B1 and B2 each produce at least one nonzero float32 draw.
- B1, B2, and B3 follow the exact deterministic cycle and frequencies.
- Individual zero historical draws are allowed and disclosed.
- Pure MSE, numerical, determinism, runtime, attribution-time, VRAM, and
  48-hour gates are unchanged.

## Stop boundary

No confirmation token or confirmation run is permitted. A failed revised
preflight stops Track B before all pilot records. Replication outcomes do not
change pilot thresholds or estimator settings.
