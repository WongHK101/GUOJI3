# Dataset Metadata Audit — Phase 0-A (P0-0)

Performed 2026-05-10. Verified all dataset dimensions against actual data files on AutoDL cloud (`/root/autodl-tmp/GUOJI/JRNGC/data/` and `/root/autodl-tmp/GUOJI/mamba_enhanced/data/`).

## Summary of Discrepancies

**5 T values in paper Table I are wrong** — 3 CausalTime datasets, VAR_d50, NSVAR_d50_PlanA.

## Per-Dataset Verification

### VAR_d50
- **Paper**: d=50, T=500, T/d=10
- **Actual**: x.shape=(50, 600), T=600 (train), x_eval.shape=(50, 600), lag=5, gc.shape=(50,50,5)
- **n_edges_true**: 113±14 across 5 seeds (varies per random realization)
- **Discrepancy**: ❌ T should be 600 (T/d=12), not 500
- **Root cause**: Paper T=500 likely from original JRNGC data generator; our data uses different generation parameters

### Lorenz_F40
- **Paper**: d=10, T=500, T/d=50
- **Actual**: x.shape=(10, 500) ✓, x_eval.shape=(10, 100), gc.shape=(10,10,1), raw edges=40
- **n_edges_true**: 30 (all 5 seeds: 40 edges − 10 self-connections = 30)
- **Verdict**: ✓ All correct

### NSVAR_d10
- **Paper**: d=10, T=500, T/d=50
- **Actual**: x.shape=(10, 500) ✓, lag=7, gc.shape=(10,10,7), raw edges=20, lag0 edges=4
- **n_edges_true**: 1–4 per seed (mean 2.2)
- **Verdict**: ✓ All correct

### NSVAR_d50_PlanA
- **Paper**: d=50, T=600, T/d=12
- **Actual**: x.shape=(50, 500), T=500, lag=14, gc.shape=(50,50,14), raw edges=1681, lag0 edges=124
- **n_edges_true**: 124–132 across 3 seeds
- **Discrepancy**: ❌ T should be 500 (T/d=10), not 600
- **Root cause**: Paper and data generator out of sync; generator produces T=500

### DREAM3_d10 / d50 / d100
- **Paper**: d=10/50/100, T=21, T/d=2.1/0.4/0.2
- **Actual**: Generated on-the-fly by `dream3_trajectories()`, always T=21 per trajectory
- **Note**: No data files on cloud (directory structure exists but is empty); data is loaded via JRNGC API
- **Verdict**: ✓ Correct

### CT_medical
- **Paper**: d=40, T=384, T/d=9.6
- **Actual**: x.shape=(40, 1200) ❌, T=1200, gc.shape=(40,40,1), raw edges=153
- **n_edges_true**: 133 (153 − 20 diagonal self-connections)
- **d_state**: 8 (confirmed from FILTER_KWARGS in test_causaltime.py)
- **Discrepancy**: ❌ T should be 1200 (T/d=30.0), not 384
- **Edge count correctness**: 153 → 133 is correct. gc is 3D (40,40,1), `compute_metrics` extracts `gc[:,:,0]` as 2D (40,40) with sum=153, then `remove_self_connection` removes 20 non-zero diagonal entries → n_edges_true=133

### CT_traffic
- **Paper**: d=40, T=672, T/d=16.8
- **Actual**: x.shape=(40, 1200) ❌, gc.shape=(40,40,1), raw edges=82
- **Discrepancy**: ❌ T should be 1200 (T/d=30.0), not 672

### CT_pm25
- **Paper**: d=72, T=584, T/d=8.1
- **Actual**: x.shape=(72, 1200) ❌, gc.shape=(72,72,1), raw edges=354
- **Discrepancy**: ❌ T should be 1200 (T/d=16.7), not 584

### fMRI_d15
- **Paper**: d=15, T=200, T/d=13.3
- **Actual**: x.shape=(15, 200) ✓, gc.shape=(15,15,1), raw edges=33
- **Verdict**: ✓ Correct

## Self-Connection Removal Logic

Verified `remove_self_connection()` in `JRNGC/tgc/metrics/causal.py:5`:

```python
def remove_self_connection(gc):
    # gc: [d, d, t] or [d, d]
    if 2 == len(gc.shape):
        gc = gc[:, :, np.newaxis]  # add lag dim
    idx = np.array(tuple(np.ndindex(gc.shape)))
    idx.resize(gc.shape + (3,))
    flat = gc[idx[:, :, :, 0] != idx[:, :, :, 1]]  # keep only where i != j
    return flat
```

`compute_metrics` first collapses 3D gc to 2D via `gc_true[:, :, 0]` (first lag only), then calls `remove_self_connection`. This is consistent across all experiments.

For CT_medical: 153 total → gc 3D (40,40,1) → gc[:,:,0] sum=153 → 20 non-zero diagonal → 133 after removal. **Math verified.**

## d_state Parameter Audit

| Experiment | d_state | Confirmed |
|-----------|---------|-----------|
| CT_medical 1-seed (original) | 8 | From FILTER_KWARGS in test_causaltime.py |
| CT_medical 3-seed (P1-1) | 8 | Line 43 of test_ct_medical_3seed.py |
| All other synthetic (P1-3) | 4 | Line 49 of test_multiseed_synthetic.py |
| Mask supplement (P1-5) | 4 | Consistent with synthetic default |

## Action Items

1. **Fix paper Table I**: Update T values for VAR_d50 (600), NSVAR_d50_PlanA (500), CT_medical/CT_traffic/CT_pm25 (all 1200). Recalculate T/d ratios.
2. **Update result JSONs**: Add explicit `T` field checked against actual data.
3. **Add data-loading assertion**: In experiment scripts, assert `x.shape[1]` matches expected T.
