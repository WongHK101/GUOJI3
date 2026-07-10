# Phase 8 CPU Preflight: Unresolved Risks

The CPU semantic implementation passes its locked tests. The following items
remain unresolved because GPU execution was prohibited in this task.

1. Full-prefix second-order autograd runtime and VRAM at `T=500`, `d=8`, and
   2,000 iterations remain unmeasured.
2. The frozen `H_max/K` normalization may make `lambda=0.01` numerically large;
   only the separately authorized 100-iteration GPU scale preflight may assess
   this. Lambda must not be tuned.
3. The total nominal-lag graph score is semantically implemented but has not
   been evaluated as scientific evidence.
4. The new no-auxiliary input-space control is not historical comparator
   replication and cannot support graph-recovery claims.
5. Importing the frozen full-auxiliary diagnostic module creates its historical
   output directory; that generated directory is ignored and is not packaged.
6. The CPU package implements candidate, training-policy, evaluation, and
   configuration-resolution primitives. GPU orchestration remains blocked by
   release authorization and was not exercised.

None of these risks authorizes changing the estimator, score, lambda,
thresholds, seeds, or run matrix.
