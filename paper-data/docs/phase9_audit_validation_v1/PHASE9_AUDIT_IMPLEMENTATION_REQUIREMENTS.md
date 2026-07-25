# Phase 9 Audit Validation Implementation Requirements

Implementation starts only after advisor approval.

## Required new runner

Create a separate configuration-driven validation runner. Do not overwrite
`experiments/phase9_run_audit_generalization.py`, which remains a development
artifact.

The new runner must:

1. Load the frozen JSON and CSV without hardcoded development subjects.
2. Verify every dataset SHA256 before array loading.
3. Reject any row with `execution_authorized=false`.
4. Preserve clean fixed targets for every intervention.
5. Save pure prediction MSE separately from the legacy total objective.
6. Use the same evaluation targets and perturbations across methods within a
   data unit and replicate.
7. Mark both `mamba_concat` and `tcn_concat` as auxiliary-route architectures.
8. Save raw score tensors before diagonal exclusion.
9. Save method, dataset, segment, all seeds, target indices, checkpoint policy,
   environment, runtime, VRAM, commit, config hash, and source-manifest hash.
10. Refuse resume across a source/config mismatch.

## Release lock

Before any Stage A run:

- create an approved code commit;
- require a clean Git worktree;
- create source and config SHA256 manifests;
- bind the authorized Stage A matrix hash to the runner;
- verify deterministic CUDA settings and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- run CPU semantic tests and non-evidentiary 20-iteration GPU smokes;
- inspect no scientific metric during smoke acceptance.

Stage B requires a separate release token bound to:

- the same scientific configuration;
- the sealed Stage B matrix;
- the passed Stage A decision artifact;
- the approved AutoDL source commit.

## Required tests

- finite-difference total raw-chain parity;
- direct plus auxiliary chain decomposition;
- source/target orientation and lag indexing;
- future perturbation and raw-target isolation;
- baseline partial-total equivalence;
- Mamba and TCN auxiliary-route profile labels;
- fixed-target intervention route isolation;
- same-window/same-perturbation schedule parity;
- horizon 32/64/128 same-target extraction;
- score aggregation `abs -> mean windows -> max nominal lag`;
- coordinate-mixing entropy fixture;
- exact determinism duplicate comparator;
- negative tests for modified config, source hash, data hash, and unauthorized
  Stage B execution.

## Aggregator

The aggregator must:

- validate the frozen config and matrix hashes;
- average train replicates within each data unit first;
- keep NetSim and MoCap scopes separate;
- evaluate every gate mechanically;
- emit both machine-readable and human-readable decisions;
- refuse a broad claim when only one architecture passes;
- never select subjects, checkpoints, horizons, or thresholds from results.

## Known P0 correction

The development runner currently passes:

`has_auxiliary_route = (method == "concat")`

to the audit profile. This incorrectly marks `tcn_concat` as lacking an
auxiliary route. The validation runner must use:

`has_auxiliary_route = method in {"mamba_concat", "tcn_concat"}`.

This is a profile-label correction, not a change to the stored development
Jacobians or metrics.
