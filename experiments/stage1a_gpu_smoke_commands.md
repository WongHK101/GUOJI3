# Stage 1a GPU Infrastructure Smoke Commands

These commands are for after P0.3d package approval. They do not start during
P0.3d local closure.

```powershell
cd E:\GUOJI\mamba_enhanced

# Five-method GPU infrastructure smoke:
# Baseline, CP-depthwise, FixedFIR3, FixedEMA, RawChainMamba limited.
python experiments\stage1a_gpu_benchmark.py `
  --config configs\stage1a_gpu_infrastructure_smoke_config.json `
  --output-root results_kbs\stage1a_gpu_smoke\five_method `
  --device cuda `
  --smoke

# CP duplicate determinism run A.
python experiments\stage1a_gpu_benchmark.py `
  --config configs\stage1a_gpu_infrastructure_smoke_config.json `
  --output-root results_kbs\stage1a_gpu_smoke\cp_duplicate_a `
  --device cuda `
  --smoke

# CP duplicate determinism run B.
python experiments\stage1a_gpu_benchmark.py `
  --config configs\stage1a_gpu_infrastructure_smoke_config.json `
  --output-root results_kbs\stage1a_gpu_smoke\cp_duplicate_b `
  --device cuda `
  --smoke

python experiments\compare_stage1a_determinism.py `
  --run-a results_kbs\stage1a_gpu_smoke\cp_duplicate_a\runs\stage1a__cp_depthwise__NS_Nonlinear__data1__train0 `
  --run-b results_kbs\stage1a_gpu_smoke\cp_duplicate_b\runs\stage1a__cp_depthwise__NS_Nonlinear__data1__train0 `
  --output results_kbs\stage1a_gpu_smoke\cp_duplicate_determinism_report.json `
  --tol 1e-6
```

Acceptance: all five smoke runs complete with `formal_result=false`, no NaN/Inf,
full output completeness, CUDA memory/time fields recorded, and the CP duplicate
determinism report passes.
