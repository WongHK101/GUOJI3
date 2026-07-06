# P0 Depthwise ISTF Minimal GPU Benchmark Plan

Date: 2026-07-06

Purpose: define the smallest GPU-backed benchmark that can decide whether ordinary depthwise ISTF is strong enough to replace the current cross-channel ISTF-Mamba main method in the KBS manuscript.

This plan should not be launched until the user opens AutoDL/GPU and confirms the run.

## Decision Question

Can coordinate-preserving depthwise ISTF retain the manuscript's core empirical story while fixing the raw-input Jacobian semantics problem?

The benchmark must answer two questions:

1. Performance: does depthwise ISTF achieve competitive or better GC recovery than baseline JRNGC and cross-channel ISTF-Mamba on the key settings?
2. Semantics: do current filtered-coordinate scores and raw-chain scores remain aligned enough to support raw-variable GC interpretation?

## Methods

Required methods:

- `baseline`: original JRNGC.
- `istf_mamba`: current cross-channel ISTF-Mamba, retained as legacy/ablation.
- `istf_depthwise`: ordinary coordinate-preserving depthwise ISTF, current primary candidate.
- `concat`: concat auxiliary-channel JRNGC, only for shortcut diagnostics where needed.

Do not promote `depthwise_gated` in the first GPU run. It degraded the factorial D2 local smoke.

## Stage 1: Controlled Factorial Benchmark

Primary goal: decide whether depthwise ISTF remains viable under the controlled generator that matches the paper's mechanism story.

Configuration:

- Generator: existing D2 factorial generator.
- Cells: `Stat+Linear`, `Stat+Nonlinear`, `NS+Linear`, `NS+Nonlinear`.
- Seeds: 0--4.
- Dimension/time: start with canonical local-to-paper scale, preferably `d=10`, `T=600`, `lag=3`.
- Iterations: use the paper's established controlled-experiment setting where feasible, likely `max_iter=2000`.
- Metrics: summary-max AUROC/AUPRC, top-k SHD/nSHD/MCC, train loss, runtime.
- Semantic check: current vs raw-chain score correlation, top-k Jaccard, and leakage on selected windows.

Pass criteria:

- Depthwise semantic correlation should stay near 1.0 with very low leakage.
- In non-stationary cells, depthwise should be at least competitive with baseline and should not show systematic degradation.
- In stationary cells, mild neutrality is acceptable; large consistent degradation is not.
- Cross-channel ISTF-Mamba should not be used as final main method unless it clearly outperforms depthwise and the semantic issue can be resolved, which is currently unlikely.

## Stage 2: Shortcut Diagnostic Benchmark

Primary goal: verify that the repaired method still supports the shortcut-learning story.

Required diagnostics:

- Concat partial score vs concat total-derivative score.
- Concat side-channel intervention: zero/perturb original input while preserving auxiliary channel, and zero auxiliary channel while preserving original input.
- ISTF-Mamba and depthwise ISTF current-vs-raw-chain score alignment.
- Prediction loss sensitivity under masking/shuffling, but reported carefully as diagnostics rather than final proof by itself.

Pass criteria:

- Concat should show evidence of side-channel dependence or partial-vs-total scoring mismatch.
- Depthwise ISTF should avoid a separate auxiliary path and preserve raw-chain semantic alignment.
- Diagnostics should support the final method's structural claim without requiring us to disclose the internal failed cross-channel version.

## Stage 3: Limited Main Benchmark Probe

Primary goal: decide whether a full manuscript rerun is justified.

Start with a low-cost subset:

- CT-medical if available and not too costly, because it was the clearest positive dataset in the active manuscript.
- NSVAR_d10 because it supports the non-stationary synthetic story.
- Lorenz_F40 and VAR_d50 as boundary checks.

Minimum methods:

- Baseline JRNGC.
- ISTF-Mamba legacy.
- ISTF-depthwise.

Seeds:

- Use the manuscript's canonical seed protocol where available.
- If cost is high, run 3 seeds first and expand only if the result is promising and stable.

Pass criteria:

- Depthwise should not erase the existing positive evidence on CT-medical/NSVAR.
- If depthwise is neutral on stationary datasets, that is acceptable if the paper is framed as a targeted repair rather than a universal booster.
- If depthwise underperforms across most key datasets, do not rewrite the KBS manuscript around it; instead reassess method design before spending more GPU time.

## Run Hygiene

- Do not overwrite existing canonical results.
- Write all outputs to a new timestamped P0 directory, for example:
  `results/p0_depthwise_gpu_YYYYMMDD_HHMM/`
- Save config, commit hash, command line, environment, raw JSON, logs, and aggregation scripts.
- Keep `AGENTS.md` and `WORKLOG.md` local; do not push them.
- Do not modify the KBS manuscript until Stage 1 and Stage 2 pass.

## AutoDL Need

AutoDL/GPU is not needed for more local smoke diagnostics.

Ask the user to start AutoDL/GPU only when we are ready to run Stage 1 with `d=10`, `T=600`, `max_iter=2000`, and 5 seeds. That is the first point where local CPU becomes inefficient and GPU time is justified.

## Immediate Pre-GPU Checklist

Before asking the user to open AutoDL:

1. Prepare a single runner or command set for Stage 1.
2. Make sure `filter_type="depthwise"` is available on the server branch.
3. Confirm no output path overlaps with existing canonical results.
4. Run one local tiny command after any runner edits.
5. Commit and push the exact runner version.

