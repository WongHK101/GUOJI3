# Result Consistency Audit — Phase 0-A (P0-0)

Performed 2026-05-10. Traces every numeric value in the ISTF-JRNGC paper (Table I, III, IV) back to its source file, script, and seeds.

## Table I: 11-Config AUROC Matrix

### Synthetic Configs

| Config | Method | Paper Value | Source JSON | Script | Seeds | Verified |
|--------|--------|-------------|-------------|--------|-------|----------|
| VAR_d50 | Baseline | 0.7178±0.033 | multiseed_synthetic_results.json | test_multiseed_synthetic.py | 5 (0-4) | ✓ |
| VAR_d50 | ISTF-Mamba | 0.7237±0.026 | multiseed_synthetic_results.json | test_multiseed_synthetic.py | 5 (0-4) | ✓ |
| VAR_d50 | ISTF-TCN | 0.6989±0.017 | final_consolidated.json / var50_tcn_results.json | test_tcn_backfill.py | 3 (0-2) | ✓ |
| Lorenz_F40 | Baseline | 0.9417±0.012 | multiseed_synthetic_results.json | test_multiseed_synthetic.py | 5 (0-4) | ✓ |
| Lorenz_F40 | ISTF-Mamba | 0.9462±0.009 | multiseed_synthetic_results.json | test_multiseed_synthetic.py | 5 (0-4) | ✓ |
| Lorenz_F40 | ISTF-TCN | 0.8619±0.010 | tcn_backfill_results.json | test_tcn_backfill.py | 3 (0-2) | ✓ |
| NSVAR_d10 | Baseline | 0.9279±0.029 | multiseed_synthetic_results.json | test_multiseed_synthetic.py | 5 (0-4) | ✓ |
| NSVAR_d10 | ISTF-Mamba | 0.9269±0.034 | multiseed_synthetic_results.json | test_multiseed_synthetic.py | 5 (0-4) | ✓ |
| NSVAR_d10 | ISTF-TCN | 0.9465±0.027 | final_consolidated.json | mixed (factorial + backfill) | 5+ | ⚠ mixed source |
| NSVAR_d50_PlanA | Baseline | 0.6476±0.005 | multiseed_synthetic_results.json | test_multiseed_synthetic.py | 3 (0-2) | ✓ |
| NSVAR_d50_PlanA | ISTF-Mamba | 0.6406±0.025 | multiseed_synthetic_results.json | test_multiseed_synthetic.py | 3 (0-2) | ✓ |
| NSVAR_d50_PlanA | ISTF-TCN | 0.6511±0.013 | tcn_backfill_results.json | test_tcn_backfill.py | 3 (0-2) | ✓ |

### DREAM3 Configs

All DREAM3 values verified against `dream3_backfill_results.json` (script: `test_dream3_backfill.py`, 3 subjects each). All match.

| Config | Method | Source | Verified |
|--------|--------|--------|----------|
| DREAM3_d10 | All 3 methods | dream3_backfill_results.json | ✓ |
| DREAM3_d50 | All 3 methods | dream3_backfill_results.json | ✓ |
| DREAM3_d100 | All 3 methods | dream3_backfill_results.json | ✓ |

### CausalTime Configs

| Config | Method | Paper Value | Source JSON | Seeds | Verified |
|--------|--------|-------------|-------------|-------|----------|
| CT_medical | Baseline | 0.4741±0.024 | ct_medical_3seed_results.json | 3 (0-2) | ✓ |
| CT_medical | ISTF-Mamba | 0.5136±0.033 | ct_medical_3seed_results.json | 3 (0-2) | ✓ |
| CT_medical | ISTF-TCN | 0.4609 | final_consolidated.json | 1 | ✓ |
| CT_pm25 | All | As listed | causaltime_results.json | 1 | ✓ |
| CT_traffic | All | As listed | causaltime_results.json | 1 | ✓ |

### fMRI Config

| Config | Method | Paper Value | Source JSON | Subjects | Verified |
|--------|--------|-------------|-------------|----------|----------|
| fMRI_d15 | Baseline | 0.5255±0.048 | fmri_3subj_results.json | 3 | ✓ |
| fMRI_d15 | ISTF-Mamba | 0.4439±0.032 | fmri_3subj_results.json | 3 | ✓ |
| fMRI_d15 | ISTF-TCN | 0.5459±0.075 | final_consolidated.json | 3 | ✓ |

### cMLP and PCMCI+ Columns

**No JSON result files exist** for cMLP or PCMCI+ values. These numbers were likely entered directly from script stdout:

- **cMLP**: Likely from `test_additional_baselines.py` (runs cMLP from Neural-GC code)
- **PCMCI+**: Likely from `test_pcmci.py` (runs PCMCI+ via Tigramite)

**Action required**: Add these values to JSON result files and create reproducible scripts that save structured output.

## Table III: Mask Intervention (P1-5)

All values verified against `mask_supplement_results.json` (script: `test_mask_supplement.py`, NSVAR d=10 seed 0).

| Intervention | Paper Value | JSON Source | Verified |
|-------------|-------------|-------------|----------|
| mask_x_only (Concat) | +1.68 | mask_supplement_results.json | ✓ |
| mask_c_only (Concat) | +0.44 | mask_supplement_results.json | ✓ |
| mask_both (Concat) | +1.66 | mask_supplement_results.json | ✓ |
| mask_x_only (ISTF) | +29.15 | mask_supplement_results.json | ✓ |
| mask_both (ISTF) | +29.15 | mask_supplement_results.json | ✓ |
| shuffle_x (Concat) | +3.04 | mask_supplement_results.json | ✓ |
| shuffle_x (ISTF) | +2.88 | mask_supplement_results.json | ✓ |
| c0_Δloss (Concat) | +0.05 | mask_supplement_results.json | ✓ |

## Table IV: 2×2 Factorial Ablation

### ❌ VERDICT: UNTRACEABLE — Paper values must be invalidated

**Full provenance audit**: See [factorial_provenance_audit.md](factorial_provenance_audit.md).

The paper's Table IV values do NOT appear in any JSON, log, or checkpoint file. The experiment script (`test_interaction_ablation.py`) was run with **d=10 for all cells** (not d=50 as paper claims for Stat+Linear), producing AUROCs ~0.92-0.97 with **near-zero Mamba Δ across all cells**. The paper's claimed S main effect (+1.05pp) and S×N interaction (−4.77pp) are contradicted by the actual experiment output (S main effect = −0.0018, S×N interaction = −0.013).

**Decision**: Paper Table IV and all associated quantitative claims (Section 5.4) are frozen pending redesign and rerun of a clean unified-protocol 2×2 factorial experiment.

## Edge Count Verification (per `compute_metrics`)

`compute_metrics` workflow: `gc_true_3d → gc_true[:, :, 0] → remove_self_connection → n_edges_true`

| Config | gc 3D shape | Raw sum | gc[:,:,0] sum | Self-loops removed | n_edges_true |
|--------|-------------|---------|---------------|-------------------|--------------|
| CT_medical | (40,40,1) | 153 | 153 | 20 | 133 |
| Lorenz_F40 | (10,10,1) | 40 | 40 | 10 | 30 |
| VAR_d50 seed0 | (50,50,5) | 500 | 113 | 0 | 113 |
| NSVAR_d10 seed0 | (10,10,7) | 20 | 4 | 0 | 4 |
| NSVAR_d50_PlanA seed0 | (50,50,14) | 1681 | 124 | 0 | 124 |

Note: VAR and NSVAR gc matrices have all-zero diagonals at lag 0, so no self-connections are removed.

## Key Findings

1. **✅ All main results traceable**: 100% of Table I baseline/ISTF-Mamba/ISTF-TCN values are verified against JSON files
2. **✅ Mask results consistent**: Table III values match mask_supplement_results.json
3. **⚠️ Factorial ablation inconsistent**: Table IV values don't match interaction_ablation_results.json — needs investigation
4. **❌ cMLP/PCMCI+ not reproducible**: No structured output files exist for these baselines — must save to JSON
5. **⚠️ TCN NSVAR_d10 source mixed**: Values come from multiple experiment files (factorial + backfill)

## Next Steps

1. Resolve Table IV factorial discrepancy (highest priority audit item)
2. Add JSON output to cMLP and PCMCI+ baseline scripts
3. Add script-to-JSON manifest mapping each paper value to exact file and key path
4. Consider generating paper tables programmatically from JSON (postprocess_p1.py)
