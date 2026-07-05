# P0 Jacobian Semantics Audit

Date: 2026-07-06

Purpose: assess whether the current ISTF score used in the KBS manuscript is semantically aligned with a full chain-rule Jacobian with respect to the raw input variables.

## Local Audit Scope

Script:

- `experiments/p0_jacobian_semantics_audit.py`

Local outputs:

- `results/p0_audit/p0_jacobian_semantics_d6_iter120_refactor_seed0.json`
- `results/p0_audit/p0_jacobian_semantics_d6_iter120_refactor_seed1.json`
- `results/p0_audit/p0_jacobian_semantics_d6_iter120_refactor_seed2.json`
- `results/p0_audit/p0_jacobian_semantics_d6_iter120_refactor_seed3.json`
- `results/p0_audit/p0_jacobian_semantics_d6_iter120_refactor_seed4.json`

Smoke protocol:

- Controlled VAR-like synthetic process.
- `d=6`, `T=100`, `lag=3`, `max_iter=120`.
- CPU-only, 5 seeds.
- Compared current filtered-coordinate scoring against full raw-input chain-rule scoring.
- Compared concat partial-Jacobian scoring against total derivative through `z(x)`.
- Compared a coordinate-preserving depthwise ISTF filter against its raw-chain score.

These are P0 diagnostic smoke runs, not final benchmark experiments.

## Aggregate Results

| Comparison | Score corr mean | Score corr range | Top-k Jaccard mean | Leakage mean |
| --- | ---: | ---: | ---: | ---: |
| ISTF-Mamba current `dY/dx'` vs raw-chain `dY/dx` | 0.831568 | 0.793710-0.850987 | 0.784299 | 0.083525 |
| Concat partial `dY/dx` vs total derivative through `z(x)` | 0.506416 | 0.382955-0.687991 | 0.691312 | 0.119455 |
| Depthwise ISTF current `dY/dx'` vs raw-chain `dY/dx` | 0.999978 | 0.999958-0.999994 | 1.000000 | 0.005266 |

## Interpretation

1. The current cross-channel ISTF-Mamba score can diverge measurably from a full chain-rule raw-input score, even in small local runs.
2. The depthwise coordinate-preserving filter nearly closes the filtered-coordinate vs raw-chain semantic gap under the same audit protocol.
3. Concat shortcut diagnostics must separate two issues: architectural side-channel risk and partial-Jacobian scoring that omits the auxiliary path derivative.
4. The current KBS manuscript should not proceed as final-submission-ready until the score semantics and coordinate-preservation claims are repaired.

## Method Decision Implication

The strongest rescue path is no longer to defend the existing cross-channel Mamba filter as-is. The next method branch should promote a coordinate-preserving ISTF design, likely depthwise/channel-wise filtering, and then rerun the controlled diagnostics before any large benchmark rerun.

## Controlled Repair Smoke

Script:

- `experiments/p0_controlled_repair_smoke.py`

Local output:

- `results/p0_audit/p0_controlled_repair_smoke_d6_iter120_seed0-4.json`

Protocol:

- Same controlled VAR-like generator as the semantics audit.
- `d=6`, `T=100`, `lag=3`, `max_iter=120`.
- CPU-only, seeds 0--4.
- Compared baseline JRNGC, cross-channel ISTF-Mamba, and coordinate-preserving depthwise ISTF.
- For filter variants, reported both current filtered-coordinate graph scores and raw-chain graph scores on selected windows.

Aggregate results:

| Method | Current AUROC mean | Current AUROC std | Raw-chain AUROC mean | Score corr mean | Top-k Jaccard mean | Leakage mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline JRNGC | 0.6225 | 0.1454 | n/a | n/a | n/a | n/a |
| ISTF-Mamba | 0.5588 | 0.0756 | 0.5768 | 0.8316 | 0.7843 | 0.0835 |
| Depthwise ISTF | 0.6325 | 0.1151 | 0.6316 | 1.0000 | 1.0000 | 0.0053 |

Per-seed AUROC detail:

| Seed | Baseline | ISTF-Mamba current/raw | Depthwise current/raw |
| ---: | ---: | ---: | ---: |
| 0 | 0.8125 | 0.5903 / 0.6181 | 0.8125 / 0.8125 |
| 1 | 0.5926 | 0.6561 / 0.6931 | 0.6720 / 0.6720 |
| 2 | 0.4250 | 0.5700 / 0.6450 | 0.5200 / 0.5200 |
| 3 | 0.7037 | 0.5238 / 0.5344 | 0.5608 / 0.5608 |
| 4 | 0.5787 | 0.4537 / 0.3935 | 0.5972 / 0.5926 |

Interpretation:

1. Depthwise ISTF did not collapse graph recovery in this local smoke run; its current-score AUROC was close to baseline and higher than cross-channel ISTF-Mamba under the same setup.
2. Depthwise current and raw-chain graph scores were effectively identical, supporting it as the leading repair candidate.
3. This smoke result is not a final benchmark. It is sufficient to justify adapting the controlled diagnostic runners to include depthwise ISTF before any AutoDL/GPU-scale rerun.

## Unified Diagnostic Smoke

Script:

- `experiments/p0_unified_diagnostic_smoke.py`

Local output:

- `results/p0_audit/p0_unified_diagnostic_smoke_d6_iter120_seed0-2.json`

Protocol:

- Stable controlled VAR(1), `d=6`, `T=140`, `lag=1`, `max_iter=120`.
- CPU-only, seeds 0--2.
- Compared baseline JRNGC, causal MA(3), causal EMA(0.7), concat partial/total scoring, cross-channel ISTF-Mamba, and depthwise ISTF.
- The generator was made conservative with row-sum coefficient scaling to avoid non-normal transient blow-up in smoke diagnostics.

Aggregate results:

| Method | Primary AUROC mean | Secondary AUROC mean | Semantic corr mean | Leakage mean | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline JRNGC | 0.6621 | n/a | n/a | n/a | raw current score |
| MA(3)-JRNGC | 0.6265 | n/a | n/a | n/a | simple causal smoother |
| EMA(0.7)-JRNGC | 0.5519 | n/a | n/a | n/a | simple causal smoother |
| Concat JRNGC | 0.5821 | 0.6537 | 0.5730 | n/a | partial vs total-raw scores |
| ISTF-Mamba | 0.6798 | 0.6685 | 0.8215 | 0.2598 | current vs raw-chain scores |
| Depthwise ISTF | 0.6242 | 0.6242 | 1.0000 | 0.0117 | current vs raw-chain scores |

Interpretation:

1. On a simple stable VAR(1), depthwise ISTF is semantically clean but does not consistently beat baseline or ISTF-Mamba. This is not surprising because the setting has little non-stationary filtering need.
2. Simple causal smoothers did not explain the ISTF effect in this smoke; MA/EMA were below baseline.
3. Concat partial-vs-total scoring can materially change AUROC and has only moderate score alignment, reinforcing the need to separate shortcut architecture from scoring definition.

## Factorial Depthwise Smoke

Script:

- `experiments/p0_factorial_depthwise_smoke.py`

Local output:

- `results/p0_audit/p0_factorial_depthwise_smoke_D2_d6_iter120_seed0-2.json`

Protocol:

- Existing D2 factorial generator: `{stationary, non-stationary} x {linear, nonlinear}`.
- `d=6`, `T=180`, `lag=3`, `max_iter=120`.
- CPU-only, seeds 0--2.
- Compared baseline JRNGC, cross-channel ISTF-Mamba, and depthwise ISTF.
- Used a 2D summary-max metric to avoid lag-bin undefined-class failures in this small smoke.

Aggregate AUROC and semantic results:

| Cell | Baseline AUROC | ISTF-Mamba AUROC | Depthwise AUROC | Mamba corr/leak | Depthwise corr/leak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stat+Linear | 0.8994 | 0.7635 | 0.8736 | 0.9873 / 0.0419 | 1.0000 / 0.0035 |
| Stat+Nonlinear | 0.8366 | 0.7043 | 0.7808 | 0.9871 / 0.0399 | 1.0000 / 0.0036 |
| NS+Linear | 0.7973 | 0.7865 | 0.8030 | 0.9834 / 0.0474 | 1.0000 / 0.0039 |
| NS+Nonlinear | 0.7405 | 0.7135 | 0.7432 | 0.9796 / 0.0463 | 1.0000 / 0.0037 |

Interpretation:

1. In the small factorial smoke, depthwise ISTF was near baseline in stationary cells and slightly above baseline in both non-stationary cells.
2. Cross-channel ISTF-Mamba did not show a clear advantage over depthwise and remained less semantically clean.
3. This supports continuing the coordinate-preserving ISTF branch through controlled diagnostics before any expensive full benchmark rerun.

## Depthwise-Gated Candidate Check

Code:

- Added `filter_type="depthwise_gated"` as an experimental coordinate-preserving filter.
- The filter uses per-channel causal gated convolutions, so it increases temporal capacity without cross-variable mixing.

Controlled repair smoke:

- Output: `results/p0_audit/p0_controlled_repair_smoke_d6_iter120_seed0-4_gated.json`
- Same controlled setting as the repair smoke (`d=6`, `T=100`, `lag=3`, `max_iter=120`, seeds 0--4).

| Method | Current AUROC mean | Raw-chain AUROC mean | Semantic corr mean | Top-k Jaccard mean | Leakage mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline JRNGC | 0.6225 | n/a | n/a | n/a | n/a |
| ISTF-Mamba | 0.5588 | 0.5768 | 0.8316 | 0.7843 | 0.0835 |
| Depthwise ISTF | 0.6325 | 0.6316 | 1.0000 | 1.0000 | 0.0053 |
| Depthwise-gated ISTF | 0.6709 | 0.6720 | 0.9999 | 1.0000 | 0.0017 |

Factorial D2 smoke with gated:

- Output: `results/p0_audit/p0_factorial_depthwise_smoke_D2_d6_iter120_seed0-2_gated.json`
- Same factorial setting as above (`d=6`, `T=180`, `lag=3`, `max_iter=120`, seeds 0--2).

| Cell | Baseline AUROC | Depthwise AUROC | Depthwise-gated AUROC |
| --- | ---: | ---: | ---: |
| Stat+Linear | 0.8994 | 0.8736 | 0.7562 |
| Stat+Nonlinear | 0.8366 | 0.7808 | 0.7505 |
| NS+Linear | 0.7973 | 0.8030 | 0.6575 |
| NS+Nonlinear | 0.7405 | 0.7432 | 0.6058 |

Interpretation:

1. The gated filter improves the small controlled repair smoke but degrades the factorial D2 smoke, especially in non-stationary cells.
2. The stronger per-channel gated capacity is therefore not the current main repair candidate.
3. Ordinary depthwise ISTF remains the more stable coordinate-preserving branch for the next controlled diagnostic step.
