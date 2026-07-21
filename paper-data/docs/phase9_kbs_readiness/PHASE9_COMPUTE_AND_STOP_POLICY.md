# Phase 9 Compute and Stop Policy

Status: `PROPOSED; GPU REMAINS OFF`

## Execution sequence

1. advisor approves the scientific gates and dataset roles;
2. implementation is created in a new release branch/worktree;
3. CPU unit/semantic tests pass;
4. four 20-iteration non-evidentiary infrastructure smokes pass;
5. one 100-iteration GPU benchmark per trained architecture reports runtime,
   VRAM, regularizer/prediction scale, and deterministic replay;
6. extrapolate the exact formal matrix runtime;
7. freeze source/config/run-matrix/schedule hashes;
8. execute development runs;
9. execute three-seed formal pilot;
10. open only those confirmations whose preregistered gates pass.

## Preflight hard gates

- CUDA device and deterministic algorithms enabled;
- duplicate-run loss, checkpoint, and score max absolute difference `<=1e-6`;
- no NaN/Inf, OOM, detached raw chain, future leakage, target leakage, missing
  files, or incomplete audit fields;
- positive finite predictor and preprocessor gradients where required;
- source commit and clean-worktree manifest match the approved release;
- all methods use the same eligible windows and score-window hash;
- full-prefix/reliable-history evaluation reports float64 accumulation and lag
  counts;
- pure MSE decreases during the 100-iteration benchmark and no regularizer
  creates degenerate constant predictions.

## Compute caps

- CPU semantic suite: stop and report if projected runtime exceeds 12 hours on
  the local machine;
- single GPU run: stop and report if projected 2,000-iteration runtime exceeds
  2 hours or peak allocated memory exceeds 80% of the selected GPU;
- three-seed controlled pilot: hard cap 24 GPU-hours;
- all Phase 9 evidentiary runs: hard cap 72 GPU-hours;
- no automatic reduction of horizon, windows, iterations, model seeds, data
  units, or metrics to fit the cap.

If the full 210-run proposal exceeds the cap, stop before formal execution and
return a runtime report. The advisor may remove an entire secondary block (for
example cMLP/cLSTM context baselines) before results are viewed. Individual
failed or slow methods cannot be silently dropped after partial results exist.

## Scientific stop rules

- any G0 provenance or claim-integrity violation stops the entire phase;
- any G2 audit numerical failure stops training;
- failure of controlled semantic gates stops external execution because the
  tool is not validated;
- mitigation pilot failure stops only mitigation confirmation, not the audit
  generalization study;
- external baseline mean AUROC below 0.60 blocks direct graph-recovery
  interpretation but does not erase semantic diagnostics;
- contradictory audit-severity associations are reported and prevent
  `KBS_STRONG_READY`;
- no threshold may be changed after a formal block is opened.

## Environment policy

- Formal GPU work, if eventually approved, runs on one release-locked 901
  environment, not partly on the local machine and partly on another server.
- Raw datasets and old Phase 5--8 outputs are read-only.
- Every run writes to a unique Phase 9 root with status, config snapshot, source
  manifest, environment, stdout/stderr, metrics, scores, and checkpoint hashes.
- Server artifacts are copied locally and verified by a second SHA256 manifest;
  originals are retained.

