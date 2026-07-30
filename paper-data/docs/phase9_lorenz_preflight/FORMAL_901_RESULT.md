# Phase 9 Lorenz-96 Formal 901 Confirmation Result

## Decision

**PASS: manuscript-evidence eligible within the frozen claim boundary.**

All 20 formal runs completed, all release/integrity checks passed, and every
pre-registered aggregate gate passed. Every seed-coverage gate passed in 5/5
data seeds.

This is positive evidence for the Jacobian coverage-audit paper, not for a
new repair method or for Mamba/ISTF graph-recovery performance.

## Frozen Release

- release commit:
  `2637bb798a2bd09c4f60fe3ead32abbffec3b8ca`;
- config file SHA256:
  `1b1cf90353028c3f6170cb1a769c1b1fbbc6412f2982492f5cc577a6b90fc465`;
- run-matrix SHA256:
  `8357264f5f8bc8d0814a27704ad9f56e4f0758763298cfe54745991f7e380405`;
- source-manifest SHA256:
  `cd5e6b2c1cf3aca6f10f111478214c060f96d8b1b2a2f8ff039417cb23bd44e9`;
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition;
- formal runs: 5 data seeds x 2 train seeds x 2 methods = 20;
- statistical unit: data seed after averaging two train seeds;
- total formal runner wall time: 2,837 seconds (47.3 minutes).

The remote `tests/` suite passed 140 tests. A second release after the
metadata-only smoke-validator fix passed 27 targeted tests before execution.
The final smoke validator reported zero difference in training traces,
checkpoint states, attribution arrays, metrics, and exact-top-k edges.

An earlier release, `01144cb`, exposed a missing
`config_canonical_sha256` status field during non-evidentiary smoke
validation. No formal dataset had been generated. The field was added, the
release was re-locked as `2637bb7`, and both smoke roots were regenerated in
new directories. No result from the superseded smoke was used.

## Aggregate Results

| Quantity | Mean across 5 data seeds | Data-seed SD | Range |
| --- | ---: | ---: | ---: |
| Baseline total nominal AUROC | 0.8628 | 0.0177 | 0.8322--0.8750 |
| Concat total nominal AUROC | 0.8589 | 0.0149 | 0.8475--0.8850 |
| Concat partial nominal AUROC | 0.8032 | 0.0110 | 0.7892--0.8164 |
| Missing-route relative magnitude | 0.5614 | 0.0138 | 0.5446--0.5763 |
| Fixed-target `mask_c` MSE delta | 0.5275 | 0.0255 | 0.5005--0.5571 |
| Partial/total nominal Pearson | 0.8401 | 0.0385 | 0.7770--0.8723 |
| Partial/total top-k Jaccard | 0.6577 | 0.0712 | 0.5395--0.7157 |
| Coordinate-entropy median | 0.9563 | 0.0034 | 0.9518--0.9596 |
| Temporal-tail median | 0.1822 | 0.0185 | 0.1525--0.1984 |
| H64/H128 mass ratio | 1.0000 | <1e-13 | 1.0000--1.0000 |
| Nominal H64/H128 max difference | 0 | 0 | 0--0 |
| Concat-vs-baseline relative pure MSE | -0.4156 | 0.1545 | -0.6863 to -0.3115 |

The mean total-minus-partial concat AUROC was +0.0557 and was positive in
all five data seeds. In contrast, concat total-score AUROC differed from
baseline by only -0.0039 on average and was positive in two seeds and
negative in three. This is therefore a score-coverage result, not a
performance-superiority result.

## Gate Outcome

All aggregate checks passed:

- all 20 runs complete, finite, deterministic, and release-locked;
- baseline mean/seed strong-operating-point gates;
- missing-route mean and 5/5 seed coverage;
- fixed-target auxiliary-sensitivity mean and 5/5 seed coverage;
- partial/total discrepancy in 5/5 data seeds;
- coordinate entropy in 5/5 data seeds;
- temporal-tail mass in 5/5 data seeds;
- H64/H128 mass and nominal-score stability in 5/5 data seeds;
- prediction-MSE mean and 5/5 seed coverage.

The 901 aggregate was repeated in a second directory with byte-identical
CSV and JSON outputs. A separate local recomputation produced byte-identical
CSV files and canonically identical JSON objects.

## Allowed Claim

> In a fresh five-seed Lorenz-96 confirmation with baseline mean AUROC 0.863,
> the auxiliary concat architecture reduced pure prediction MSE while an
> x-only Jacobian score omitted substantial total raw-chain attribution:
> the mean missing-route magnitude was 0.561, fixed-target auxiliary masking
> increased MSE by 0.528, and partial-versus-total nominal scores had mean
> Pearson correlation 0.840 and top-k Jaccard 0.658.

This wording must retain the architecture, scoring, fixed-target, bounded-
horizon, and five-seed qualifiers.

## Prohibited Claims

The result does not support:

- ISTF or Mamba graph-recovery superiority;
- universal invalidity of auxiliary conditioning;
- full-prefix completeness beyond H=128;
- a successful coverage-aligned repair;
- revival of Phase 7 Stage 1b;
- strong significance claims from only five data-seed units.

## Artifact Integrity

Local archive:

`E:\GUOJI\results_kbs\phase9_lorenz_901_confirmation\2637bb7\`

Archive SHA256:

`959916f42e58c61d632d5113b977a1907461b5b89991abac51b3f9baf8a53209`

The archive contains the formal result tree, both valid smoke roots, smoke
validation, dry-run validation, release lock, and exact release source.
The internal manifest covers 832 files; local verification passed 832/832.

After archival and local recomputation, the 901 instance was shut down.
Subsequent SSH verification returned connection refused.
