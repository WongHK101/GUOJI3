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
    return_metadata=False,
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

        return_metadata: if True, additionally return generator metadata,
            including coefficients and support audit diagnostics.

    Returns:
        x: np.ndarray (d, T) float32
        gc: np.ndarray (d, d, lag) float32
        metadata: optional dict when return_metadata=True
    """
    # Separate RNG streams keep the factorial pairing fixed across cells:
    # graph/A_base, drift, and observation noise never depend on call order or
    # on whether the current cell is stationary/linear.
    graph_rng = np.random.RandomState(seed * 1000 + 42)
    coef_rng = np.random.RandomState(seed * 1000 + 43)
    drift_rng = np.random.RandomState(seed * 1000 + 44)
    noise_rng = np.random.RandomState(seed * 1000 + 45)

    # ---- 1. Generate base GC graph (SHARED across all 4 cells for this seed) ----
    gc = np.zeros((d, d, lag), dtype=np.float32)
    for i in range(d):
        for j in range(d):
            if i != j and graph_rng.rand() < sparsity:
                k = graph_rng.randint(0, lag)  # random lag for each edge
                gc[i, j, k] = 1.0

    # ---- 2. Generate base coefficient matrices A_k(0) from gc ----
    A_base = []
    for k in range(lag):
        A_k = np.zeros((d, d), dtype=np.float32)
        for i in range(d):
            for j in range(d):
                if gc[i, j, k] > 0:
                    A_k[i, j] = coeff_scale * coef_rng.uniform(0.3, 1.0) * coef_rng.choice([-1, 1])
        np.fill_diagonal(A_k, 0.0)
        A_base.append(A_k)
    A_base_arr = np.stack(A_base, axis=0).astype(np.float32)  # (lag,d,d)

    # ---- 3. Coefficient drift for non-stationary cells ----
    if not stationary and regime_shift_strength > 0:
        # Generate smooth random walk only on the declared graph support.
        # This preserves the fixed ground-truth graph in non-stationary cells.
        drift_scale = regime_shift_strength * coeff_scale
        A_drift = []
        for k in range(lag):
            # Random walk then smooth
            raw = np.cumsum(drift_rng.randn(T, d, d) * drift_scale / np.sqrt(T), axis=0)
            window = max(5, T // 30)
            smoothed = np.zeros_like(raw)
            for t in range(T):
                lo = max(0, t - window)
                smoothed[t] = raw[lo:t + 1].mean(axis=0)
            smoothed *= gc[:, :, k][None, :, :]
            for t in range(T):
                np.fill_diagonal(smoothed[t], 0.0)
            A_drift.append(smoothed)
    else:
        A_drift = [np.zeros((T, d, d), dtype=np.float32) for _ in range(lag)]
    A_drift_arr = np.stack(A_drift, axis=1).astype(np.float32)  # (T,lag,d,d)
    A_t = A_base_arr[None, :, :, :] + A_drift_arr
    for t in range(T):
        for k in range(lag):
            np.fill_diagonal(A_t[t, k], 0.0)

    # ---- 4. Generate time series ----
    x = np.zeros((d, T), dtype=np.float32)
    noise = noise_rng.randn(d, T).astype(np.float32) * noise_scale

    # Initialize with noise
    for t in range(lag):
        x[:, t] = noise[:, t]

    for t in range(lag, T):
        # Linear predictor
        pred = np.zeros(d, dtype=np.float32)
        for k in range(lag):
            A_k_t = A_t[t, k]
            pred += A_k_t @ x[:, t - k - 1]

        # Apply nonlinearity if specified
        if not linear and nonlinear_strength > 0:
            # pred_nl = (1-α)·pred + α·s·tanh(pred/s)  where s = std(pred)
            # Smooth interpolation: identity for small pred, saturation for large pred
            s = float(np.std(pred)) + 1e-8
            pred = (1.0 - nonlinear_strength) * pred + nonlinear_strength * s * np.tanh(pred / s)

        x[:, t] = pred + noise[:, t]

    x = x.astype(np.float32)
    if return_metadata:
        metadata = {
            "seed": int(seed),
            "rng_streams": {
                "graph": int(seed * 1000 + 42),
                "coefficients": int(seed * 1000 + 43),
                "drift": int(seed * 1000 + 44),
                "noise": int(seed * 1000 + 45),
            },
            "A_base": A_base_arr.astype(np.float32),
            "A_drift": A_drift_arr.astype(np.float32),
            "A_t": A_t.astype(np.float32),
            "noise": noise.astype(np.float32),
            "support_audit": audit_factorial_support(gc, A_base_arr, A_drift_arr, A_t, x),
        }
        return x, gc, metadata
    return x, gc


def audit_factorial_support(gc, A_base, A_drift, A_t, x=None, eps=1e-12):
    """Audit that generated coefficients obey the declared graph support."""
    gc_bool_lag_first = np.transpose(np.asarray(gc).astype(bool), (2, 0, 1))
    A_base_arr = np.asarray(A_base)
    A_drift_arr = np.asarray(A_drift)
    A_t_arr = np.asarray(A_t)
    lag, d, _ = A_base_arr.shape
    off_support = ~gc_bool_lag_first
    diag_mask = np.eye(d, dtype=bool)
    base_off_support_max = float(np.max(np.abs(A_base_arr[off_support]))) if np.any(off_support) else 0.0
    drift_off_support_max = float(np.max(np.abs(A_drift_arr[:, off_support]))) if np.any(off_support) else 0.0
    actual_support = np.any(np.abs(A_t_arr) > eps, axis=0)  # (lag,d,d)
    declared_support = gc_bool_lag_first
    spectral_radii = []
    for t in range(A_t_arr.shape[0]):
        companion = np.zeros((d * lag, d * lag), dtype=np.float64)
        companion[:d, :d * lag] = np.concatenate([A_t_arr[t, k] for k in range(lag)], axis=1)
        if lag > 1:
            companion[d:, :-d] = np.eye(d * (lag - 1))
        spectral_radii.append(float(np.max(np.abs(np.linalg.eigvals(companion)))))
    return {
        "base_off_support_max_abs": base_off_support_max,
        "drift_off_support_max_abs": drift_off_support_max,
        "actual_support_subset_declared": bool(np.all(~actual_support | declared_support)),
        "actual_any_time_support_equals_declared": bool(np.array_equal(actual_support, declared_support)),
        "base_diagonal_max_abs": float(np.max(np.abs(A_base_arr[:, diag_mask]))),
        "drift_diagonal_max_abs": float(np.max(np.abs(A_drift_arr[:, :, diag_mask]))),
        "A_t_diagonal_max_abs": float(np.max(np.abs(A_t_arr[:, :, diag_mask]))),
        "series_has_nan_or_inf": bool(False if x is None else (not np.isfinite(x).all())),
        "spectral_radius_max": float(np.max(spectral_radii)),
        "spectral_radius_mean": float(np.mean(spectral_radii)),
        "spectral_radius_p95": float(np.percentile(spectral_radii, 95)),
    }


def generate_all_factorial_cells(
    setting="A", d=10, T=600, lag=3, seed=0, sparsity=0.2,
    return_metadata=False,
):
    """Generate all 4 cells for a given setting and seed.

    Returns dict: cell_name -> (x, gc)
    """
    params = FACTORIAL_SETTINGS[setting].copy()
    cells = {}
    for name, stationary, linear in FACTORIAL_CELLS:
        regime = params["regime_shift_strength"] if not stationary else 0.0
        nl = params["nonlinear_strength"] if not linear else 0.0
        generated = generate_factorial_cell(
            d=d, T=T, lag=lag, seed=seed,
            stationary=stationary, linear=linear,
            coeff_scale=params["coeff_scale"],
            noise_scale=params["noise_scale"],
            regime_shift_strength=regime,
            nonlinear_strength=nl,
            sparsity=sparsity,
            return_metadata=return_metadata,
        )
        cells[name] = generated
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
