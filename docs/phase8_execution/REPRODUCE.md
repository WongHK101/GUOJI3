# Phase 8 Execution Reproduction

Use only the release commit and manifests supplied in the review package. On
the approved CUDA host:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python experiments/execute_phase8_stage.py \
  --stage preflight \
  --config configs/phase8/phase8_execution_lock.json \
  --run-matrix configs/phase8/phase8_run_matrix.csv \
  --release-lock-dir RELEASE_LOCK_DIR \
  --authorization AUTHORIZATION_JSON \
  --output-root RESULTS_ROOT

python experiments/validate_phase8_gpu_preflight.py \
  --root RESULTS_ROOT \
  --config configs/phase8/phase8_execution_lock.json \
  --run-matrix configs/phase8/phase8_run_matrix.csv \
  --cpu-preflight-summary RELEASE_LOCK_DIR/cpu_preflight_summary.json
```

Only after the validator exits zero:

```bash
python experiments/execute_phase8_stage.py --stage replication \
  --config configs/phase8/phase8_execution_lock.json \
  --run-matrix configs/phase8/phase8_run_matrix.csv \
  --release-lock-dir RELEASE_LOCK_DIR --authorization AUTHORIZATION_JSON \
  --output-root RESULTS_ROOT

python experiments/aggregate_phase8.py --stage replication \
  --root RESULTS_ROOT --config configs/phase8/phase8_execution_lock.json \
  --run-matrix configs/phase8/phase8_run_matrix.csv

python experiments/execute_phase8_stage.py --stage pilot \
  --config configs/phase8/phase8_execution_lock.json \
  --run-matrix configs/phase8/phase8_run_matrix.csv \
  --release-lock-dir RELEASE_LOCK_DIR --authorization AUTHORIZATION_JSON \
  --output-root RESULTS_ROOT

python experiments/aggregate_phase8.py --stage pilot \
  --root RESULTS_ROOT --config configs/phase8/phase8_execution_lock.json \
  --run-matrix configs/phase8/phase8_run_matrix.csv \
  --preflight-report RESULTS_ROOT/gpu_preflight_validation.json \
  --replication-report RESULTS_ROOT/replication_aggregate_and_gates.json
```

There is deliberately no authorized confirmation command in this release.
