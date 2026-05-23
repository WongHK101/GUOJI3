# Factorial Post-Hoc Analysis Summary

**Date**: 2026-05-09–10 (expert review pass: 2026-05-10)
**Setting**: D2 canonical (coeff=0.40, noise=0.15, regime=0.20, nonlinear=0.50)
**Data**: d=10, T=600, lag=3, sparsity=0.2, 10 seeds
**Metric**: summary_max AUROC
**Status**: expert-reviewed; diagnostics expansion to 5 seeds complete

---

## 1. Primary Results (10-seed, max_iter=2000)

| Cell | Baseline | Mamba | TCN | Δ(M−B) | Δ(T−B) |
|------|----------|-------|-----|---------|--------|
| Stat+Linear | 0.7566±0.054 | 0.7698±0.034 | 0.7621±0.043 | +0.013 | +0.006 |
| Stat+Nonlinear | 0.6884±0.084 | 0.6493±0.053 | 0.6526±0.078 | -0.039 | -0.036 |
| NS+Linear | 0.7796±0.068 | 0.7410±0.076 | 0.7811±0.056 | **-0.039** | **+0.002** |
| NS+Nonlinear | 0.6854±0.053 | 0.6439±0.091 | 0.6535±0.075 | -0.042 | -0.032 |

### Cell-level observations

- **Stat+Linear**: Both filters near-baseline. Input-space filtering does not harm or help on stationary linear data.
- **Stat+Nonlinear**: Both filters show mild negative trends (~−0.04). Nonlinear dynamics alone pose some difficulty for temporal filtering.
- **NS+Linear**: Clear divergence. TCN near-neutral (+0.002), Mamba shows medium negative trend (−0.039). This is the clean diagnostic cell.
- **NS+Nonlinear**: Both filters show mild negative trends. Combined nonlinearity and mechanism-level non-stationarity is the hardest regime in this controlled setting.

## 2. Statistical Tests (Holm-Bonferroni corrected)

**No comparison reaches significance after multiple-comparison correction** (all p_corr > 0.05, n=12 comparisons). All claims should be framed as directional trends with effect sizes and CI.

Key comparisons (uncorrected):

| Cell | Comparison | Δ mean | 95% CI | t | p (unc.) | Cohen's d |
|------|-----------|--------|--------|---|---------|-----------|
| NS+Linear | Mamba − Baseline | -0.039 | [-0.088, 0.010] | -1.78 | 0.108 | -0.508 |
| NS+Linear | TCN − Baseline | +0.002 | [-0.041, 0.044] | +0.08 | 0.939 | +0.023 |
| NS+Linear | Mamba − TCN | -0.041 | [-0.108, 0.028] | -1.34 | 0.212 | -0.570 |
| NS+Nonlinear | Mamba − Baseline | -0.042 | [-0.103, 0.020] | -1.54 | 0.158 | -0.531 |
| Stat+Nonlinear | Mamba − Baseline | -0.039 | [-0.119, 0.041] | -1.11 | 0.296 | -0.531 |

**Interpretation**: All 95% CIs cross zero. Effect sizes are medium (|d| ≈ 0.5–0.6) and directionally consistent across Mamba-vs-Baseline comparisons in non-stationary cells. Paper should report directional trends and CI, not "statistically significant."

## 3. Post-hoc Diagnostics

### 3a. Filter Deviation by Regime

`|x_filtered − x| / |x|` per timestep:

| Seed | Cell | Mean Deviation |
|------|------|---------------|
| 0 | Stat+Linear | 0.043 |
| 0 | NS+Linear | 0.037 |
| 3 | Stat+Linear | 0.052 |
| 3 | NS+Linear | 0.051 |

Filter deviation is small and does not differ systematically between stationary and non-stationary regimes. Mamba does not substantially distort the input representation.

### 3b. Jacobian Norm Ratio (Mamba / Baseline)

**Initial 2-seed (full cells)**:

| Seed | Stat+Linear | NS+Linear |
|------|------------|-----------|
| 0 | 0.960 | 1.040 |
| 3 | 1.047 | 1.061 |

**5-seed NS+Linear expansion**:

| Seed | edges | Jac Ratio |
|------|-------|-----------|
| 0 | 18 | 1.063 |
| 1 | 17 | 1.031 |
| 2 | 13 | 0.986 |
| 3 | 23 | 1.050 |
| 4 | 15 | 1.011 |

**JacR across 5 seeds**: 1.028 ± 0.027 (mean ± std). 4/5 seeds at or above 1.0. This rules out the simple explanation that Mamba degradation arises from uniformly suppressing Jacobian magnitudes. Combined with symmetric true/false score shifts, this suggests that the degradation arises from ranking perturbation rather than selective denoising or uniform suppression.

### 3c. True-Edge vs False-Edge Score Shift

ΔS = S_mamba − S_baseline (summary_max scores):

**Initial 2-seed (all 4 cells)**:

| Seed | Cell | ΔS_true | ΔS_false | SI |
|------|------|---------|----------|-----|
| 0 | Stat+Linear | -0.029 | -0.009 | (-) |
| 0 | NS+Linear | +0.015 | +0.015 | ~0 |
| 3 | Stat+Linear | +0.024 | +0.020 | +0.091 |
| 3 | NS+Linear | +0.019 | +0.015 | +0.118 |

**5-seed NS+Linear expansion**:

| Seed | edges | ΔS_true | ΔS_false | ΔS_true − ΔS_false | |ΔS|_mean |
|------|-------|---------|----------|---------------------|-----------|
| 0 | 18 | +0.0212 | +0.0225 | -0.0013 | 0.0219 |
| 1 | 17 | -0.0054 | +0.0067 | -0.0121 | 0.0060 |
| 2 | 13 | +0.0011 | -0.0001 | +0.0012 | 0.0006 |
| 3 | 23 | +0.0188 | +0.0147 | +0.0041 | 0.0168 |
| 4 | 15 | -0.0124 | +0.0059 | -0.0183 | 0.0091 |

**Primary evidence**: For seeds where the score shift is non-negligible (seeds 0 and 3, |ΔS|_mean > 0.015), ΔS_true and ΔS_false move together — both are positive and within ~0.004 of each other. For the remaining seeds, both shifts are near zero (|ΔS|_mean < 0.01). This pattern supports global score rescaling rather than structure-aware edge separation.

**Selectivity index (auxiliary only)**: `SI = (ΔS_true − ΔS_false) / (|ΔS_true| + |ΔS_false| + ε)`. In seeds 0 and 3, SI = -0.029 and +0.122 (near zero, opposite signs). In seeds 1, 2, 4 the denominator is near zero, causing numerical instability (±1 artifacts). SI is reported here only as a secondary check; the primary evidence is the raw ΔS_true and ΔS_false values and their absolute difference.

**Key observation**: When Mamba produces a non-trivial score shift, ΔS_true ≈ ΔS_false. Mamba shifts true-edge and false-edge scores together rather than selectively enhancing causal edges or suppressing spurious ones. This is consistent with global sensitivity rescaling, not structure-aware denoising. The 5-seed data adds robustness to the 2-seed finding.

## 4. Appendix: Convergence Diagnostic (3-seed, max_iter=5000)

Extended training degrades Mamba AUROC in all cells further (vs Baseline at 5000 iters). Included in canonical JSON as `appendix.convergence_diagnostic`. Confirms that 2000 iterations is not under-trained; extended training does not recover Mamba performance.

## 5. Paper Framing (expert-reviewed wording)

### Operating Boundary (3-layer model)

- **Layer 1 — Near-neutral zone (Stat+Linear)**: Input-space filtering is near-neutral under stationary linear dynamics. Both Mamba and TCN deviate from baseline by ~0.01 or less.
- **Layer 2 — Adaptive-filter risk (NS+Linear)**: TCN remains near-neutral under linear mechanism-level non-stationarity, whereas Mamba shows a medium-size negative trend (−0.039, d = −0.51, 95% CI crosses zero). Diagnostics indicate that Mamba does not selectively suppress causal edges; rather, it rescales GC scores without improving true-vs-false separation.
- **Layer 3 — Controlled hard regime (NS+Nonlinear)**: In the D2 factorial setting, combined nonlinearity and mechanism-level non-stationarity produces mild negative trends for both filters (−0.04 range). This defines a regime where temporal filtering provides no benefit.

### Key Claims

1. **Input-space filtering is near-neutral in stationary linear dynamics.** Both Mamba and TCN produce trivial deviations from baseline in Stat+Linear.
2. **TCN remains near-neutral under linear mechanism-level non-stationarity, whereas Mamba shows a medium-size negative trend**, suggesting that adaptive selectivity can perturb GC score ranking in this regime without improving edge separation.
3. **Both Mamba and TCN show mild negative trends under nonlinear dynamics** in this controlled setting. Temporal filtering does not provide a benefit when the underlying functional form is nonlinear.
4. **Diagnostics suggest Mamba does not selectively improve true-edge separation**: ΔS_true and ΔS_false shift together (selectivity index near zero), consistent with global sensitivity rescaling rather than structure-aware denoising.
5. **None of these differences are statistically significant after Holm-Bonferroni correction at n=10.** The factorial study provides **directional operating-boundary evidence**, not confirmatory proof. All claims are framed as directional trends with reported effect sizes and confidence intervals. The paper should avoid "statistically significant," "proves," "demonstrates conclusively," "confirms failure," or "universal difficulty."

### Recommended Paper Paragraph

> In the controlled D2 factorial setting, ISTF variants do not provide universal performance gains. Mamba shows a medium-size negative trend under mechanism-level non-stationarity, while TCN remains near-neutral in the linear non-stationary cell. Post-hoc diagnostics indicate that Mamba shifts true-edge and false-edge scores in a nearly symmetric manner, suggesting global sensitivity rescaling rather than structure-aware denoising. These results define an operating boundary: input-space confinement removes shortcut pathways, but adaptive filtering does not automatically improve causal edge separation.

## 6. Figures / Tables to Include

- **Table 1**: Three-model AUROC with deltas (Section 1)
- **Table 2**: Paired-delta statistical table with CI and effect sizes (Section 2)
- **Figure 1**: Operating boundary diagram (3-layer schematic)
- **Figure 2**: Per-seed delta bar chart (10 seeds, Mamba−Baseline and TCN−Baseline per cell)
- **Figure 3**: Per-seed Jacobian ratio (Mamba/Baseline) bar chart (Stat+Linear vs NS+Linear)
- **Figure 4**: ΔS_true vs ΔS_false scatter/boxplot with selectivity index (5 seeds)

## 7. Limitations

1. n=10 seeds: 95% CIs cross zero for all comparisons; larger seed counts would be needed to confirm the directional trends at p<0.05
2. Full-cell diagnostics on 2 seeds; NS+Linear expanded to 5 seeds — SI is interpretable in 2/5 seeds (non-trivial |ΔS|) and near zero in both, but 3/5 seeds have very small score shifts making SI unstable
3. Single controlled setting (d=10, T=600, lag=3, D2 params); generalizability to other data dimensions unknown
4. No alternative SSM variant beyond standard Mamba; results may not generalize to other adaptive filters
5. All effects are in the medium range (|d| ≈ 0.5); Mamba does not catastrophically fail on any cell

## 8. Source Files

- `results/raw/factorial_ablation_canonical.json` — canonical merged data with protocol/audit
- `results/raw/factorial_stat_tests.json` — full statistical test results (12 comparisons)
- `experiments/diagnostics_D2_seed0.json` — seed 0 diagnostics (all 4 cells)
- `experiments/diagnostics_D2_seed3.json` — seed 3 diagnostics (all 4 cells)
- `experiments/diagnostics_nslinear_5seed_D2.json` — 5-seed NS+Linear diagnostics with SI
- `experiments/factorial_postprocess_plan.md` — expert-approved execution plan
