# Stage 1a GPU Infrastructure Smoke Commands

These commands are for after P0.3d package approval. They do not start during
P0.3d local closure.

```powershell
cd E:\GUOJI\mamba_enhanced

# First five-method GPU infrastructure smoke:
# Baseline, CP-depthwise, FixedFIR3, FixedEMA, RawChainMamba limited.
python experiments\stage1a_gpu_benchmark.py `
  --config configs\stage1a_gpu_infrastructure_smoke_config.json `
  --output-root results_kbs\stage1a_gpu_smoke\smoke_a `
  --device cuda `
  --smoke

# Second identical smoke root. The CP run here is the duplicate-B comparison.
python experiments\stage1a_gpu_benchmark.py `
  --config configs\stage1a_gpu_infrastructure_smoke_config.json `
  --output-root results_kbs\stage1a_gpu_smoke\smoke_b `
  --device cuda `
  --smoke

python experiments\validate_gpu_infrastructure_smoke.py `
  --smoke-root-a results_kbs\stage1a_gpu_smoke\smoke_a `
  --smoke-root-b results_kbs\stage1a_gpu_smoke\smoke_b `
  --output results_kbs\stage1a_gpu_smoke\gpu_smoke_validation.json `
  --tol 1e-6
```

Acceptance: both 5-method smoke roots complete with `formal_result=false`, no
NaN/Inf, full output completeness, CUDA memory/time fields recorded,
deterministic algorithms enabled, release lock metadata matched, and the CP
duplicate comparison passes.
