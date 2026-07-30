# Phase 9 Lorenz-96 Formal 901 Confirmation Protocol

## Scientific Role

This confirmation tests whether the Jacobian route-coverage mismatch observed
in the development seed persists across fresh Lorenz-96 realizations while
baseline JRNGC retains strong direct-graph recovery. It is not a repair-method
benchmark and cannot support ISTF or Mamba performance claims.

The protocol was frozen before generating any of the five formal datasets.
The previously observed data seed 0 is excluded from formal aggregation and
is used only for a 20-iteration 901 infrastructure smoke.

## Frozen Design

- environment: AutoDL 901 GPU instance;
- generator: frozen Lorenz-96, `F=40`, `d=10`, `T=500`, nominal lag `K=1`;
- formal data seeds:
  `26073111, 26073123, 26073137, 26073151, 26073167`;
- train seeds: `26073201, 26073219`;
- methods: baseline JRNGC and frozen legacy Mamba concat x-only;
- formal runs: `5 x 2 x 2 = 20`;
- training: 2,000 iterations, Adam `1e-3`, gradient clipping `1.0`;
- checkpoint: restored minimum checked total regularized objective;
- evaluation: 32 deterministic common windows, H=32/64/128,
  primary bounded audit H=64;
- graph object: total raw-chain nominal-lag score at K=1;
- statistical unit: data seed after averaging the two train seeds.

The seed identifiers had no matches in the searched text-based repository,
review-package filenames, or worklog records before freeze. This is a bounded
search statement, not a universal proof of historical non-use. The formal
namespace does not use Phase 7 Stage 1b seed identifiers 4--8.

## Infrastructure Gate

Before formal data generation:

1. check exact release commit, clean worktree, config/matrix SHA256, and every
   critical source hash;
2. run baseline and concat for 20 iterations on development seed 0;
3. independently repeat the same concat smoke;
4. require complete finite CUDA outputs, deterministic algorithms, baseline
   partial/total identity and zero missing route;
5. require training trace, checkpoint, score arrays, metrics, and exact-top-k
   edges to agree within `1e-6`.

The smoke is non-evidentiary and cannot enter any formal aggregate.

## Aggregate Gates

All gates use the five data-seed-level records obtained after averaging the
two train seeds.

Strong baseline:

- mean baseline total nominal AUROC at least 0.85;
- at least 4/5 data seeds have baseline AUROC at least 0.80.

Concat route-coverage signature:

- missing-route magnitude mean at least 0.20 and at least 4/5 seeds at least
  0.20;
- fixed-target `mask_c` MSE delta mean at least 0.05 and at least 4/5 seeds at
  least 0.05;
- at least 4/5 seeds satisfy partial/total Pearson at most 0.90 or exact-top-k
  Jaccard at most 0.80;
- at least 4/5 seeds have coordinate-entropy median at least 0.75;
- at least 4/5 seeds have temporal-tail median at least 0.05;
- all 5 seeds have H64/H128 off-diagonal mass ratio at least 0.95;
- all 5 seeds have nominal H64/H128 maximum absolute difference at most
  `1e-7`.

Prediction adequacy:

- mean concat-versus-baseline relative fixed-target MSE is at most 0.10;
- at least 4/5 data seeds have relative MSE at most 0.15.

All 20 runs must be complete, finite, deterministic, and bound to the exact
release source manifest. Per-run gates are diagnostics; only this aggregate
rule determines evidence eligibility.

## Allowed Conclusion

If every aggregate gate passes:

> A strong-operating-point multi-seed case study shows that a legacy
> auxiliary concat route can carry predictive information while an x-only
> Jacobian score omits part of the total raw-chain attribution.

The result must not be described as graph-recovery improvement, Mamba
effectiveness, a universal architectural failure, full-prefix completeness,
or a successful repair. Phase 7 Stage 1b remains unauthorized.

## Execution Order

1. Build and verify the external release manifest from a clean committed
   worktree.
2. Deploy the exact commit to a new 901 release directory and create a
   dedicated cloned environment.
3. Run release-locked dry validation.
4. Run and validate both smoke roots.
5. Generate all five formal datasets once and freeze their hashes.
6. Run the exact 20-record matrix sequentially with resume support.
7. Aggregate without changing config or thresholds.
8. Copy the complete result tree to local storage and verify a cross-host
   SHA256 manifest.
9. Only after local archival, shut down the 901 instance.

