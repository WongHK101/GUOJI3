# Phase 8 Final Reproduction Commands

## Validate and aggregate the final lambda stage

```powershell
$env:PYTHONPATH = "src"
python experiments/aggregate_phase8_final.py `
  --stage lambda_tradeoff `
  --root <phase8_final_lambda_tradeoff_root> `
  --frozen-reference-root <phase8_recovery_execution_root> `
  --config configs/phase8_final/phase8_lambda_tradeoff_config.json `
  --run-matrix configs/phase8_final/phase8_lambda_tradeoff_matrix.csv `
  --cpu-preflight-report <cpu_preflight_summary.json> `
  --output <local_reaggregate.json>
```

The resulting JSON object must match the packaged server aggregate at every key
and leaf. LF/CRLF differences may change the byte hash on different operating
systems.

## Generate figures and source data

```powershell
python tools/generate_phase8_final_figures.py `
  --track-a-root <phase8_trackA_replication_root> `
  --p0-audit-dir <p0_audit_json_directory> `
  --final-aggregate <lambda_tradeoff_aggregate_and_decision.json> `
  --output-dir <manuscript_figures_directory> `
  --source-data-dir <manuscript_source_data_directory>
```

## Compile the independent manuscript

```powershell
Set-Location <manuscript_directory>
1..3 | ForEach-Object {
  pdflatex -interaction=nonstopmode -halt-on-error `
    istf_kbs_jacobian_coverage_phase8_final.tex
}
```

Required log gates: zero LaTeX errors, undefined references/citations, missing
figures, and overfull hboxes/vboxes.

