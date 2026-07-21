# Phase 9 Prospective Validation Preregistration

Status: `DRAFT LOCK FOR ADVISOR REVIEW; EXECUTION FORBIDDEN`

## Scientific questions

1. Can the executable audit distinguish known covered, route-incomplete,
   coordinate-ambiguous, and horizon-truncated fixtures?
2. Do the claim-specific diagnostics generalize from linear stationary VAR to
   nonlinear/nonstationary dynamics?
3. Do the same score semantics remain informative on an external known-graph
   benchmark and a real time-series application?
4. Does the already frozen full-auxiliary-penalty mitigation generalize without
   graph-truth-based tuning?

This is not a new ISTF study. CP-depthwise and coverage-aligned full-prefix
repair remain no-go methods.

## Seed namespaces

The plan does not use Phase 7 data-seed identifiers 4--8.

- controlled development data seeds: `phase9_dev_91001`, `phase9_dev_91002`;
- controlled pilot data seeds: `phase9_pilot_91101..91103`;
- controlled sealed confirmation data seeds: `phase9_confirm_91201..91202`;
- paired model seeds: `phase9_model_0`, `phase9_model_1`;
- MoCap stability model seeds: `phase9_mocap_93001..93005`;
- Jacobian, window, batch, and perturbation seeds use separate namespaced hashes.

Confirmation seeds remain sealed until the CPU release lock, development runs,
and three-seed pilot semantic gates pass. A repository string search is reported
only as bounded evidence of prior use, never as proof that a seed was never used
in every historical computation.

## Stage P9.0: provenance and dataset audit

No model training.

Required records:

- exact source and license/terms for DREAM3, optional NetSim, and MoCap;
- raw archive and extracted-file SHA256;
- shape, missingness, sampling interval, variable names, and graph orientation;
- immutable raw-data directory and separate preprocessing output;
- graph-object contract for each dataset;
- confirmation that old Phase 5--8 score arrays are not imported as Phase 9
  evidence.

Failure to establish redistribution/usage terms prevents packaging the raw data;
the plan must then distribute download and verification scripts only.

## Stage P9.1: CPU semantic fixtures

No scientific result. Required fixtures:

1. raw baseline, expected `COVERED`;
2. concat with derived auxiliary route omitted from score/penalty, expected
   `PARTIALLY_COVERED`;
3. concat with full auxiliary penalty but partial score, expected penalty-route
   covered and score-route partial;
4. coordinate-wise FIR3, expected coordinate-valid finite support;
5. deliberately dense cross-channel linear mixing, expected
   `COORDINATE_AMBIGUOUS`;
6. finite causal convolution evaluated below and at exact support, expected
   fail/pass horizon transition;
7. EMA/stateful fixture at insufficient horizon, expected
   `HORIZON_TRUNCATED`.

All G2 numerical gates must pass before training.

## Stage P9.2: controlled development and runtime preflight

Data-generating families:

| Family | Frozen configuration | Direct graph object |
|---|---|---|
| Linear stationary | VAR(1), `d=8`, `T=500`, fixed sparsity/noise contract | Nonzero off-diagonal lag-1 coefficients |
| Nonlinear/nonstationary | Frozen D2 generator, `d=10`, `T=600`, `lag=3`, `beta=0.5`, `s0=0.075` | Declared transition-Jacobian support over nominal lags |

Architecture profiles:

1. baseline JRNGC;
2. concat x-only score/x-only penalty;
3. concat full auxiliary equal-lambda;
4. concat full auxiliary `lambda_c/lambda_x=10`;
5. coordinate-wise FixedFIR3;
6. cross-channel Mamba transform, semantic diagnostic only.

Unless a dimension-specific incompatibility is found before formal results,
profiles inherit the Phase 8 comparator contract exactly:

| Item | Frozen value |
|---|---|
| Predictor | 3 residual MLP blocks, hidden size 32, dropout 0 |
| Auxiliary width | `d_cond=4` |
| Mamba preprocessor | `d_state=4`, `d_conv=4`, `expand=2`, residual scale 0.1 |
| Optimizer | Adam, learning rate `1e-3`, weight decay 0 |
| Jacobian coefficient | `lambda_x=0.01` |
| Equal auxiliary penalty | `lambda_c=0.01` |
| lc10 auxiliary penalty | `lambda_c=0.10` |
| Training budget | 2,000 iterations, no early stopping, final checkpoint gates |
| Gradient clipping | global norm 1.0 |
| Training dtype | float32 |

The nonlinear/nonstationary family uses nominal lag `K=3`; the linear family
uses `K=1`. Architecture dimensions adapt only to observed `d` and frozen `K`.
Any other required compatibility change must be approved and frozen before the
first formal data seed is run.

The development seeds test code, runtime, and semantics. They do not enter formal
effect estimates. No hyperparameter may be selected by graph truth.

## Stage P9.3: prospective controlled formal study

Run count:

- two families x six profiles x five data seeds x two model seeds = `120`
  formal runs;
- pilot opens the first three data seeds (`72` runs);
- confirmation opens the remaining two (`48` runs) only after the applicable
  gate.

All methods share raw targets, normalization, training windows, evaluation
windows, predictor architecture where applicable, metric orientation, and
checkpoint rule. Method-specific filters and penalty coverage are the only
declared differences.

### Primary graph and prediction objects

- direct graph: total raw-coordinate nominal-lag score `S_GC_total`;
- concat diagnostic: partial nominal score saved in parallel;
- coefficient fidelity: lag-specific total raw-chain derivative against the
  declared lag coefficient/transition derivative;
- prediction: fixed-target pure raw-domain MSE;
- regularized objective and each penalty component: secondary optimization
  diagnostics.

### Statistical unit

Within each family/profile/data seed, average the two model seeds. The five data
seeds are the formal paired units. Report paired effects, population summaries,
and paired bootstrap 95% confidence intervals. Holm correction is applied to
co-primary inferential graph metrics. No strong significance claim is made from
the three-seed pilot.

### Audit-control gates

Use the exact G3 gates in `PHASE9_KBS_ACCEPTANCE_GATE_MATRIX.md`. These gates are
semantic and cannot be replaced by favorable graph performance.

### Mitigation pilot-go gate

The lc10 comparator may unlock its two sealed confirmation seeds only if, on the
nonlinear/nonstationary family:

- all semantic, determinism, completeness, and fixed-target gates pass;
- delta AUROC and delta AUPRC versus concat x-only are positive in at least 2/3
  pilot data seeds;
- mean delta AUROC `>=0.03` and mean delta AUPRC `>=0.03`;
- mean pure-MSE degradation `<=15%` and no seed exceeds `25%`;
- no mean AUROC degradation worse than `0.10` relative to baseline;
- the missing route is nontrivial by the preregistered G3 route gate.

This pilot-go is not positive method evidence.

### Mitigation strong gate

After five data seeds, apply G5 unchanged: delta AUROC/AUPRC `>=0.05`, coefficient
correlation `>=0.10` where defined, positive graph direction in at least 4/5
seeds, mean pure-MSE degradation `<=10%`, maximum `<=20%`, and baseline safety
within `0.05` AUROC.

If the pilot-go fails, mitigation confirmation stops. Other Phase 9 audit
validation may continue because it tests a different claim.

## Stage P9.4: external known-graph benchmark

Primary benchmark: DREAM3 in-silico gene-regulatory networks.

- analysis units: E. coli 1--2 and Yeast 1--3 networks;
- preferred dimension: size 50, subject to CPU/GPU preflight;
- graph truth: official directed regulatory networks;
- profiles: baseline, concat x-only, full auxiliary lc10, and cross-channel
  semantic diagnostic;
- two paired model seeds per network;
- run count: five networks x four profiles x two model seeds = `40`.

Context baselines, if license-compatible official implementations pass parity:
cMLP and cLSTM, two model seeds per network (`20` additional runs). They provide
performance context, not coverage-mitigation evidence.

Primary endpoints:

- `S_GC_total` AUROC, AUPRC, exact-top-k F1/MCC/SHD;
- partial-total disagreement for concat profiles;
- coverage labels, route magnitude, coordinate leakage, tail mass;
- fixed-target pure MSE.

At least one known-graph external benchmark must have baseline mean AUROC
`>=0.60`; otherwise direct graph-performance interpretation is considered too
weak, although semantic audit outputs remain reportable.

NetSim may replace or supplement DREAM3 only through a documented v1.1 protocol;
it is a simulated-fMRI benchmark, not a fully observational real dataset.

## Stage P9.5: real no-ground-truth case

Preferred dataset: human motion-capture `run` and `salsa` sequences, following
the precedent in Neural Granger Causality.

- profiles: baseline, concat x-only, full auxiliary lc10;
- five independent model seeds per sequence;
- run count: two sequences x three profiles x five seeds = `30`;
- no AUROC/AUPRC or true-edge language;
- report raw prediction MSE, per-variable MSE, complete audit profiles,
  pairwise score correlation/Jaccard across seeds, intervention sensitivity, and
  anatomy-informed adjacency enrichment as exploratory only.

The real case passes its reporting gate if all runs are complete and finite, the
audit is reproducible, and instability/ambiguity is reported rather than hidden.
There is no requirement that the graph look anatomically favorable.

## Prospective diagnostic-validity analysis

No composite score will be created after results are observed. Three locked
pairings are used:

| Audit severity | Matching semantic outcome | Expected direction |
|---|---|---|
| Missing-route relative magnitude | `1 - partial/total direct-score Jaccard` | positive |
| Coordinate leakage | `1 - filtered/raw-chain score correlation` | positive |
| Temporal tail mass | `1 - nominal/reliable-history score Jaccard` | positive |

For each pairing, use at least 20 independent data-seed/network-level points,
after averaging model seeds. The `KBS_STRONG_READY` gate is Spearman rho
`>=0.50` with paired/bootstrap 95% CI lower bound above zero. Direct graph AUROC
degradation is a secondary association unless a matched graph object and
covered comparator exist for every point.

## Total proposed evidentiary run count

- controlled formal: `120`;
- DREAM3 audit profiles: `40`;
- DREAM3 context cMLP/cLSTM: up to `20`;
- MoCap: `30`;
- maximum formal total: `210` runs.

Development, CPU fixtures, and infrastructure smokes are explicitly
non-evidentiary and reported separately. The full formal matrix must be generated
and hashed after advisor approval; this document alone does not authorize it.

## Decision outcomes

1. G0--G4 and G6 pass; mitigation fails:
   `KBS_SUBMISSION_READY_WITH_NEGATIVE_MITIGATION_BOUNDARY`.
2. G0--G6 pass:
   `KBS_STRONG_READY`.
3. Audit controls fail or external semantics cannot be reproduced:
   `NOT_READY`; do not patch the manuscript around the failure.
4. No result authorizes a new ISTF/CP performance claim.
