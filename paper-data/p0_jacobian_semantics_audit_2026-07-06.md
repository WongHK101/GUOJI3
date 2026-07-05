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

