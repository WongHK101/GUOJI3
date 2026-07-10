# Reproduce the Phase 8 CPU Preflight

Run from the repository root at the approved implementation commit.

PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES="-1"
$env:PYTHONPATH="src"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"

python -m py_compile `
  src/phase8_coverage.py `
  src/phase8_protocol.py `
  src/phase8_training.py `
  experiments/phase8_cpu_preflight.py `
  tests/test_phase8_coverage.py

python -m pytest tests -q --disable-warnings

python experiments/phase8_cpu_preflight.py `
  --device cpu `
  --config configs/phase8/phase8_execution_lock.json `
  --run-matrix configs/phase8/phase8_run_matrix.csv `
  --output-root results_kbs/phase8_cpu_preflight/reproduction
```

Expected conditions:

- all tests pass;
- `formal_scientific_runs_executed=0`;
- `gpu_used=false`;
- `execution_device=cpu`;
- 135 records resolve, including 50 sealed confirmation records;
- all report-level `passed` fields are true.

Do not run any record from the matrix during CPU-preflight reproduction.
