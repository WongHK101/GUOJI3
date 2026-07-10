# Phase 8 Implementation CPU Preflight Changelog

Date: 2026-07-11

## Added

- Immutable-comparator adapters for legacy baseline, concat x-only, and frozen
  full-auxiliary equal/lc10 methods.
- A separate `CoverageAlignedRawChainJRNGC` candidate implementing the frozen
  lag-balanced full-prefix raw-chain regularizer.
- Total nominal-lag, partial nominal-lag, reliable-support history, and
  unrestricted-prefix attribution objects with lag-specific float64
  accumulation.
- Fixed-target intervention evaluation with a separate legacy-objective track.
- A deterministic 2-lag x 1-window-per-lag x 2-output estimator schedule.
- Exact-reference, finite-difference, chain-decomposition, second-order,
  causality, score-separation, comparator-parity, and provenance tests.
- A 135-record execution-lock config and dry validator with sealed
  confirmation records.
- Explicit legacy-best versus fixed-final checkpoint-policy helpers.

## Comparator policy

- Baseline, concat x-only, and both full-auxiliary variants retain their legacy
  prediction and training objectives through composition-only adapters.
- Same-weight/same-input parity covers predictions, targets, pure MSE,
  penalties, total objective, partial score, and objective gradients.
- The no-auxiliary input-space comparator is explicitly a new matched control,
  not a replication of the historical value and not graph-recovery evidence.

## Unchanged

- No legacy source file was edited.
- The approved Phase 8 run counts, seeds, score definitions, estimator,
  thresholds, and compute limits were not changed.
- No Phase 7 artifact, Stage 1a/P1 output, v2.4 package, or manuscript was
  changed.
- No GPU or scientific run was executed.
