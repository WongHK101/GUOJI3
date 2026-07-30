# Phase 9 Lorenz Strong-Operating-Point 4090 Preflight

## Role

This is a development-only, non-evidentiary preflight on the previously
observed Lorenz-96 `F=40`, `d=10`, `T=500`, data seed 0 fixture. It cannot be
used in the manuscript and does not authorize 901 execution, formal seed
generation, or modification of frozen Stage B artifacts.

## Purpose

The preflight checks whether the existing Phase 9 baseline/concat audit stack
can be applied to a known-graph nonlinear system with a historically strong
baseline operating point. It verifies:

- immutable dataset hashes and graph orientation;
- deterministic initialization and same-architecture repeatability;
- deterministic legacy training;
- baseline partial/total identity and zero missing route;
- concat missing-route, fixed-target auxiliary sensitivity, nominal
  partial/total disagreement, coordinate entropy, temporal-tail and H64/H128
  diagnostics;
- runtime, VRAM, finite values, and same-seed repeatability.

## Frozen Development Configuration

- methods: baseline JRNGC and legacy Mamba concat x-only;
- nominal lag: 1;
- model: 5 layers, hidden width 50, `d_cond=4`, `d_state=8`;
- optimizer: Adam, learning rate `1e-3`;
- checkpoint: restored minimum checked total objective;
- budgets: 20, 100, and 2,000 iterations;
- audit: 32 deterministic windows, H=32/64/128, primary H=64;
- target: clean raw Lorenz observation;
- all outputs: `development_only=true`, `formal_result=false`,
  `manuscript_evidence=false`.

The numeric gates intentionally reuse the Stage B semantic thresholds and add
the previously declared strong-operating-point requirement:

- baseline total nominal AUROC at least 0.60;
- baseline partial/total max difference and missing route at most `1e-7`;
- concat missing-route magnitude at least 0.20;
- concat fixed-target `mask_c` MSE delta at least 0.05;
- concat Pearson at most 0.90 or exact-top-k Jaccard at most 0.80;
- coordinate-entropy median at least 0.75;
- temporal-tail median at least 0.05;
- H64/H128 off-diagonal mass ratio at least 0.95;
- nominal H64/H128 max difference at most `1e-7`;
- same-seed numerical differences at most `1e-6`.

Passing this preflight permits drafting a separate frozen 901 protocol only.
It does not make the development seed publishable.

Baseline and concat have different input-layer widths because concat adds
auxiliary coordinates. Their full predictor-state hashes are therefore not
claimed to be equal. The same predictor seed is retained, while exact
initialization parity is required only between repeat runs of the same
architecture.
