# Phase 8 Recovery Reproduction Commands

All commands below assume a clean checkout of the release commit recorded in
the package manifest. They do not authorize confirmation execution.

## CPU regression

```powershell
$env:CUDA_VISIBLE_DEVICES="-1"
python -m pytest tests/test_phase8_coverage.py tests/test_phase8_execution.py -q
python experiments/phase8_cpu_preflight.py `
  --config configs/phase8/phase8_execution_lock.json `
  --run-matrix configs/phase8/phase8_run_matrix.csv `
  --output-dir results_kbs/phase8_recovery_cpu_preflight
```

The CPU runner must reject CUDA and must reproduce the exact-reference and
formal schedule hashes recorded in its report.

## Numerical forensics

On the approved GPU release host, with the frozen stopped-preflight root still
read-only:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python experiments/phase8_numerical_forensics.py \
  --stopped-preflight-root /root/autodl-tmp/GUOJI/phase8_results_dee0d30 \
  --output-dir /root/autodl-tmp/GUOJI/phase8_recovery_forensics \
  --device cuda
```

## Revised preflight

Run the four `infrastructure_smoke` rows and the single
`repair_scale_benchmark` row in a fresh result root, then validate with:

```bash
python experiments/validate_phase8_gpu_preflight.py \
  --root "$PREFLIGHT_ROOT" \
  --config configs/phase8/phase8_execution_lock.json \
  --run-matrix configs/phase8/phase8_run_matrix.csv \
  --cpu-preflight-summary "$CPU_PREFLIGHT_SUMMARY"
```

The 30 pilot records may run only if this validator returns zero and Track A's
independent replication report has `passed_execution_completeness=true`.
Confirmation remains prohibited regardless of pilot outcome.
