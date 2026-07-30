# Phase 9 Lorenz 4090 Development Result

## Status

- Date: 2026-07-31
- Code commit: `c60dbd4e42b7b70b163c080b13ec13e5de0c8549`
- Host: `SKY-20221002FGY`
- Device: NVIDIA GeForce RTX 4090
- Dataset: previously observed Lorenz-96 `F=40`, `d=10`, `T=500`,
  data seed 0
- Role: development-only preflight
- Formal result: no
- Manuscript evidence: no

The frozen 20-iteration infrastructure smoke, independent 100-iteration
repeat, and 2,000-iteration development trajectory all completed without
NaN/Inf or deterministic-CUDA failures. The independent 100-iteration concat
runs were bitwise identical for the training trace, checkpoint state,
attribution arrays, audit metrics, and exact-top-k edges.

## Final 2,000-Iteration Results

| Quantity | Baseline JRNGC | Legacy concat x-only |
| --- | ---: | ---: |
| Fixed-target prediction MSE | 0.000208652 | 0.000205372 |
| Total nominal AUROC | 0.907778 | 0.886111 |
| Total nominal AUPRC | 0.849996 | 0.776033 |
| Total nominal exact-top-k F1 | 0.766667 | 0.766667 |
| Partial nominal AUROC | 0.907778 | 0.877222 |
| Missing-route relative magnitude | 0 | 0.539642 |
| Partial/total nominal Pearson | 1 | 0.854739 |
| Partial/total exact-top-k Jaccard | 1 | 0.714286 |
| Fixed-target `mask_c` MSE delta | not applicable | 0.460383 |
| Coordinate-entropy median | not applicable | 0.954961 |
| Temporal-tail median | 0 | 0.205762 |
| H64/H128 off-diagonal mass ratio | not applicable | 1.000000 |
| Nominal H64/H128 max absolute difference | not applicable | 0 |

The baseline passed the pre-registered strong-operating-point condition
(`AUROC >= 0.60`) and the expected partial/total identity checks. The concat
model passed all seven pre-registered route, coordinate, intervention, and
horizon checks at 2,000 iterations.

At 100 iterations, concat already showed a large missing route (0.4651) and
fixed-target auxiliary sensitivity (0.2484), but narrowly missed the
partial/total-discrepancy and temporal-tail gates. This confirms that the
2,000-iteration budget must be retained in any formal confirmation rather
than selecting a shorter budget from the observed result.

## Interpretation Boundary

This result is positive evidence for the feasibility of a formal
strong-operating-point audit, not evidence for a new repair method. It
supports the following development conclusion only:

> On one previously observed nonlinear known-graph fixture with strong
> baseline graph recovery, the legacy concat architecture exhibited a large
> omitted auxiliary route, fixed-target reliance on that route, and
> disagreement between legacy partial and total raw-chain nominal-lag scores.

It does not establish multi-seed robustness, cross-architecture generality,
full-prefix completeness, or manuscript-level graph-recovery effects. Those
claims require a separately frozen formal protocol and fresh data-seed
namespace on the approved 901 environment.

## Artifact Location

The immutable copied outputs are stored at:

`E:\GUOJI\results_kbs\phase9_lorenz_4090_preflight\c60dbd4\`

The directory contains 127 copied files totaling 2,987,158 bytes before the
local manifest was added. `SHA256_MANIFEST.tsv` covers all copied files and
has SHA256:

`fb39ffcfb261a6f11b30e68f10160785e4e97c70d0808dae35ea70360b06d8b6`

