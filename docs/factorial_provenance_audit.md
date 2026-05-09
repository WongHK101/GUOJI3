# P0-0b: Factorial Ablation Provenance Audit

Performed 2026-05-10. Traces every value in paper Table IV back to experimental records.

## Verdict: Paper Table IV is UNTRACEABLE

**No JSON, log, or checkpoint file contains the values reported in the paper.** The paper's operating-regime analysis (Section 5.4, S main effect, N main effect, S×N interaction) is built on numbers that do not match any experiment output.

## Evidence

### 1. Experiment Script: `test_interaction_ablation.py`

- **Version**: "P2 #4 v2"
- **Location**: `/root/autodl-tmp/GUOJI/mamba_enhanced/test_interaction_ablation.py`
- **Protocol**: d=10 for ALL cells, max_iter=5000, 3 seeds
- **Data generators**:
  - Stat+Linear: `var_stable(d=10, t=500, lag=3, sparsity=0.1, beta_value=0.5)` — fresh generation
  - Stat+Nonlinear: `lorenz_96(d=10, t=500, f=40)` — fresh generation
  - NS+Linear: NSVAR d=10, lag=7 — pre-generated data
  - NS+Nonlinear: NSVAR d=10 + tanh nonlinear link
- **Mamba config**: default d_state=4, ortho_lam=0.05, residual_scale=0.1 (line 92-93)

### 2. Actual Experiment Output (from interaction_ablation.log on cloud)

```
PAPER-READY TABLE (from script's own output):
Cell                 Baseline        TCN      Mamba    Mamba Δ
Stat+Linear          0.9728    0.9670    0.9656   -0.0072
Stat+Nonlinear       0.9165    0.8652    0.9115   -0.0050
NS+Linear            0.9477    0.9481    0.9452   -0.0025
NS+Nonlinear         0.9511    0.9396    0.9378   -0.0133
```

**Key observations from actual output:**
- All baseline AUROCs > 0.91 (d=10 is an easy task with 2-4 edges)
- Mamba Δ is near-zero and SLIGHTLY NEGATIVE across all cells (−0.001 to −0.013)
- S main effect (stationary→NS) = −0.0018 (NEGLIGIBLE, not +1.05pp)
- S×N interaction = −0.0130 (not −4.77pp)
- Filter provides NO benefit in any cell at max_iter=5000

### 3. Paper's Claimed Values (Table IV, Section 5.4)

| Cell | Paper Baseline | Paper ISTF-Mamba | Paper Δ |
|------|---------------|-----------------|---------|
| Stat+Linear | 0.7145 | 0.6963 | −1.8pp |
| Stat+Nonlinear | 0.9350 | 0.9374 | +0.2pp |
| NS+Linear | 0.9296 | 0.9457 | +1.6pp |
| NS+Nonlinear | 0.9209 | 0.9099 | −1.1pp |

From these, the paper derives:
- S main effect = +1.05pp → "filter benefits from non-stationarity"
- S×N interaction = −4.77pp → "strong negative interaction when both co-occur"

### 4. Discrepancy Analysis

| Aspect | Paper Claims | Actual Experiment | Match? |
|--------|-------------|-------------------|--------|
| Stat+Linear d | 50 (VAR d=50, lag 3) | 10 (var_stable d=10) | ❌ |
| Stat+Linear n_edges | ~113 (d=50) | 2-4 | ❌ |
| Stat+Linear baseline | 0.7145 | 0.9728 | ❌ |
| S main effect | +1.05pp | -0.0018 | ❌ |
| S×N interaction | -4.77pp | -0.013 | ❌ |
| Mamba benefit | Mixed (+1.6 to −1.8pp) | Uniformly ~0 (all within ±0.013) | ❌ |
| max_iter | 2000 | 5000 | ❌ |

### 5. Search for Alternative Sources

- Searched ALL JSON files for values 0.7145, 0.6963, 0.9350, 0.9374, 0.9296, 0.9457, 0.9209, 0.9099 → **No matches found**
- Checked `interaction_ablation_checkpoint.json` on cloud → Same values as results JSON (0.97 range)
- Checked `var50_3seed_results.json` → VAR d=50 baseline mean ≈ 0.714 (close to 0.7145!) but only has baseline+mamba (no factorial structure)
- Checked `nsvar50_3seed_results.json` → NSVAR d=50 baseline mean ≈ 0.650 (not matching any paper factorial cell)

**Hypothesis**: The paper's Stat+Linear value (0.7145) may have been copied from VAR_d50 main results (which has baseline ~0.718), and the other cells may have been estimated or constructed from separate single experiments rather than from a unified factorial protocol. The factorial effects (S main effect, S×N interaction) may have been computed from these hand-assembled numbers rather than from a single experiment.

### 6. Root Cause

The script `test_interaction_ablation.py` was explicitly written as "v2" using "existing proven data loaders" with d=10 for all cells. The paper was then written describing a DIFFERENT protocol (d=50 for Stat+Linear, max_iter=2000) that was never run as a unified factorial experiment.

The paper text describes (Section 5.4):
> Stat+Linear (VAR d=50, lag 3), Stat+Nonlinear (Lorenz-96 F=40), NS+Linear (NSVAR d=10, lag 7), NS+Nonlinear (NSVAR d=10 with tanh link)

This mixed-d design (d=50 for one cell, d=10 for others) makes the factorial analysis invalid even if run — the S factor would be confounded with d change.

## Recommendations

### Immediate (Blocking Paper Publication)

1. **Retract Table IV and all operating-regime quantitative conclusions** from the paper.
2. **Replace with placeholder** acknowledging that the factorial analysis will be redone under a unified protocol.
3. **Remove all claims** about S main effect, N main effect, S×N interaction, and factorial-based filter benefit attribution.

### Next Steps

1. **Design and run a clean 2×2 factorial** with unified protocol:
   - All cells: same d (recommend d=10 for computational efficiency)
   - Same T (500 or 600), same lag (3 or 5), same edge density
   - 10 seeds per cell
   - Single generator family, differing only in stationarity flag and nonlinearity flag
   - max_iter=2000 (early phase, when filter effects are visible) AND max_iter=5000 (convergence)
   
2. **Save as canonical JSON**: `results/raw/factorial_ablation_canonical.json` with full protocol metadata.

3. **Regenerate paper Table IV and operating-regime analysis** from canonical JSON only.

### Files to Update

- `docs/result_consistency_audit.md` — add this factual finding and cross-reference
- `IEEE-Transactions-LaTeX2e-templates-and-instructions/istf_jrngc.tex` — replace Section 5.4 quantitative claims with placeholder
- `WORKLOG.md` — record this as a critical finding
