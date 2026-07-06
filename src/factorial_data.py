"""Unified factorial data generator for {Stationary, Non-Stationary} x {Linear, Nonlinear}.

All 4 cells share the same ground-truth GC graph per seed (same VAR coefficient sparsity).
Only the specified axes vary: regime_shift_strength (non-stationarity) and nonlinear_strength.

Usage:
    from factorial_data import generate_factorial_cell, FACTORIAL_SETTINGS
    x, gc = generate_factorial_cell(d=10, T=600, lag=3, seed=0,
                                     stationary=True, linear=True,
                                     coeff_scale=0.25, noise_scale=0.20,
                                     regime_shift_strength=0.0, nonlinear_strength=0.0)
"""
import numpy as np
import os

# Expert-specified pilot calibration settings
FACTORIAL_SETTINGS = {
    "A": {"coeff_scale": 0.25, "noise_scale": 0.20, "regime_shift_strength": 0.30, "nonlinear_strength": 0.50},
    "B": {"coeff_scale": 0.20, "noise_scale": 0.30, "regime_shift_strength": 0.40, "nonlinear_strength": 0.75},
    "C": {"coeff_scale": 0.22, "noise_scale": 0.25, "regime_shift_strength": 0.60, "nonlinear_strength": 0.50},
    # D2: canonical setting (selected after 2 rounds of pilot calibration)
    # All 4 cells have baseline AUROC in 0.68-0.84 range with summary_max metric.
    "D2": {"coeff_scale": 0.40, "noise_scale": 0.15, "regime_shift_strength": 0.20, "nonlinear_strength": 0.50},
}

# 2x2 cell definitions
FACTORIAL_CELLS = [
    ("Stat+Linear",     True,   True),
    ("Stat+Nonlinear",  True,   False),
    ("NS+Linear",       False,  True),
    ("NS+Nonlinear",    False,  False),
]


def generate_factorial_cell(
    d=10, T=600, lag=3, seed=0,
    stationary=True, linear=True,
    coeff_scale=0.25, noise_scale=0.20,
    regime_shift_strength=0.0, nonlinear_strength=0.0,
    sparsity=0.2,
):
    """Generate one cell of the 2x2 factorial design.

    All cells within a seed share the same base VAR coefficients and GC graph.
    Only stationary/linear flags and their strength parameters vary.

    Args:
        d: number of variables
        T: time series length
        lag: max lag order
        seed: random seed (determines base graph and noise)
        stationary: if True, coefficients are constant; if False, they drift
        linear: if True, dynamics are linear; if False, nonlinear tanh link
        coeff_scale: magnitude of VAR coefficients
        noise_scale: standard deviation of observation noise
        regime_shift_strength: how much coefficients drift (0 for stationary)
        nonlinear_strength: mix-in weight for tanh nonlinearity (0 for linear)
        sparsity: fraction of non-zero GC edges

    Returns:
        x: np.ndarray (d, T) float32
        gc: np.ndarray (d, d, lag) float32
    """
    rng = np.random.RandomState(seed * 1000 + 42)

    # ---- 1. Generate base GC graph (SHARED across all 4 cells for this seed) ----
    gc = np.zeros((d, d, lag), dtype=np.float32)
    for i in range(d):
        for j in range(d):
            if i != j and rng.rand() < sparsity:
                k = rng.randint(0, lag)  # random lag for each edge
                gc[i, j, k] = 1.0

    # ---- 2. Generate base coefficient matrices A_k(0) from gc ----
    A_base = []
    for k in range(lag):
        A_k = np.zeros((d, d), dtype=np.float32)
        for i in range(d):
            for j in range(d):
                if gc[i, j, k] > 0:
                    A_k[i, j] = coeff_scale * rng.uniform(0.3, 1.0) * rng.choice([-1, 1])
        A_base.append(A_k)

    # ---- 3. Coefficient drift for non-stationary cells ----
    if not stationary and regime_shift_strength > 0:
        # Generate smooth random walk for each coefficient
        drift_scale = regime_shift_strength * coeff_scale
        A_drift = []
        for k in range(lag):
            # Random walk then smooth
            raw = np.cumsum(rng.randn(T, d, d) * drift_scale / np.sqrt(T), axis=0)
            window = max(5, T // 30)
            smoothed = np.zeros_like(raw)
            for t in range(T):
                lo = max(0, t - window)
                smoothed[t] = raw[lo:t + 1].mean(axis=0)
            A_drift.append(smoothed)
    else:
        A_drift = [np.zeros((T, d, d), dtype=np.float32) for _ in range(lag)]

    # ---- 4. Generate time series ----
    x = np.zeros((d, T), dtype=np.float32)
    noise = rng.randn(d, T).astype(np.float32) * noise_scale

    # Initialize with noise
    for t in range(lag):
        x[:, t] = noise[:, t]

    for t in range(lag, T):
        # Linear predictor
        pred = np.zeros(d, dtype=np.float32)
        for k in range(lag):
            A_k_t = A_base[k] + A_drift[k][t]
            pred += A_k_t @ x[:, t - k - 1]

        # Apply nonlinearity if specified
        if not linear and nonlinear_strength > 0:
            # pred_nl = (1-α)·pred + α·s·tanh(pred/s)  where s = std(pred)
            # Smooth interpolation: identity for small pred, saturation for large pred
            s = float(np.std(pred)) + 1e-8
            pred = (1.0 - nonlinear_strength) * pred + nonlinear_strength * s * np.tanh(pred / s)

        x[:, t] = pred + noise[:, t]

    return x.astype(np.float32), gc


def generate_all_factorial_cells(
    setting="A", d=10, T=600, lag=3, seed=0, sparsity=0.2,
):
    """Generate all 4 cells for a given setting and seed.

    Returns dict: cell_name -> (x, gc)
    """
    params = FACTORIAL_SETTINGS[setting].copy()
    cells = {}
    for name, stationary, linear in FACTORIAL_CELLS:
        regime = params["regime_shift_strength"] if not stationary else 0.0
        nl = params["nonlinear_strength"] if not linear else 0.0
        x, gc = generate_factorial_cell(
            d=d, T=T, lag=lag, seed=seed,
            stationary=stationary, linear=linear,
            coeff_scale=params["coeff_scale"],
            noise_scale=params["noise_scale"],
            regime_shift_strength=regime,
            nonlinear_strength=nl,
            sparsity=sparsity,
        )
        cells[name] = (x, gc)
    return cells


if __name__ == "__main__":
    # Quick smoke test
    for setting in ["A", "B", "C"]:
        print(f"\nSetting {setting}: {FACTORIAL_SETTINGS[setting]}")
        for seed in range(3):
            cells = generate_all_factorial_cells(setting=setting, seed=seed)
            gc0 = cells["Stat+Linear"][1]
            for name, (x, gc_n) in cells.items():
                assert np.array_equal(gc0, gc_n), f"GC mismatch: {name} vs Stat+Linear"
                print(f"  seed={seed} {name}: x.shape={x.shape}, edges={int(gc_n.sum())}, "
                      f"std={np.std(x):.3f}, range=[{x.min():.3f}, {x.max():.3f}]")
    print("\nAll checks passed.")
