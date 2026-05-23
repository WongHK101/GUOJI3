# Risk Mitigation Experiments — 2026-05-15

Controlled experiments to strengthen TNNLS review robustness. NOT part of frozen v2 benchmark.

## Experiments (P0→P3)

| Priority | Script | Setting | Seeds | Est. Time |
|----------|--------|---------|-------|-----------|
| P0 | `run_full_aux_penalty.py` | VAR(1) d=8 T=500 | 5 | ~15 min |
| P1 | `run_smoothing_baselines.py` | VAR(1) d=8 T=500 | 5 | ~20 min |
| P2 | `run_principle_ablation.py` | VAR(1) d=8 T=500 | 5 | ~20 min |
| P3 | `run_ct_medical_shortcut.py` | CT_medical d=40 T=1200 | 5 | ~30 min |

## Output

All results go to `../../risk_mitigation_results/`.

## Key Changes

### P0: Fixed MambaConcatFullPenaltyJRNGC bug
- Bug: `compute_loss()` called inherited `compute_jacobian_loss()` which only penalized x_orig
- Fix: Override `compute_jacobian_loss()` to penalize full (d+d_cond) input
- Added: separate lam_x, lam_c; |J_x| and |J_c| tracking

### Comparison methods
- JRNGC Baseline (no auxiliary channel)
- Concat x-only (shortcut collapse expected)
- Concat full same-λ (direct alternative — penalizes c too)
- Concat full budget-normalized (λ scaled by d/(d+d_cond))
- Concat full λc/λx ∈ {0.1, 10} (penalty balancing sensitivity)
- ISTF-Mamba (input-space repair)

## Data Provenance
- Controlled VAR data generated with fixed seed × 100 + 42
- Ground truth: known sparse A matrix (spectral radius 0.8, sparsity 0.3)
- CT_medical: pre-existing frozen data from canonical benchmark
- Seed policy: torch.manual_seed(seed), np.random.seed(seed) before each run

## Hyperparameters
- max_iter: 2000 (P0-P2) / 3000 (P3)
- lr: 1e-3
- jacobian_lam (λ_x base): 0.01
- layers: 3, hidden: 32 (P0-P2) / layers: 3, hidden: 50 (P3)
- d_cond: 4, d_state: 4
- lag: 1 (P0-P2) / lag: 10 (P3)
- ortho_lam: 0.05, residual_scale: 0.1 (ISTF variants)
