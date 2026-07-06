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
    "A": {"coeff_scale": 0.25, "noise_scale": 0.20, "regime_shift_strength": 0.30, "nonlinear_strength": 0.50, "nonlinear_scale": 0.50},
    "B": {"coeff_scale": 0.20, "noise_scale": 0.30, "regime_shift_strength": 0.40, "nonlinear_strength": 0.75, "nonlinear_scale": 0.50},
    "C": {"coeff_scale": 0.22, "noise_scale": 0.25, "regime_shift_strength": 0.60, "nonlinear_strength": 0.50, "nonlinear_scale": 0.50},
    # D2: canonical setting (selected after 2 rounds of pilot calibration)
    # All 4 cells have baseline AUROC in 0.68-0.84 range with summary_max metric.
    "D2": {"coeff_scale": 0.40, "noise_scale": 0.15, "regime_shift_strength": 0.20, "nonlinear_strength": 0.50, "nonlinear_scale": 0.075},
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
    nonlinear_scale=0.50,
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
        nonlinear_scale: fixed scalar s0 for coordinate-wise nonlinearity.
            It must not depend on the current pred vector.

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

    nonlinear_records = []
    for t in range(lag, T):
        # Linear predictor
        pred = np.zeros(d, dtype=np.float32)
        for k in range(lag):
            A_k_t = A_t[t, k]
            pred += A_k_t @ x[:, t - k - 1]

        # Apply coordinate-wise fixed-scale nonlinearity if specified.
        if not linear and nonlinear_strength > 0:
            pred_linear = pred.copy()
            pred = coordinatewise_nonlinearity(pred, nonlinear_strength, nonlinear_scale)
            nonlinear_records.append(nonlinear_diagnostics(pred_linear, pred, nonlinear_scale))

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
            "nonlinear_scale": float(nonlinear_scale),
            "nonlinear_diagnostics": summarize_nonlinear_records(nonlinear_records),
            "support_audit": audit_factorial_support(gc, A_base_arr, A_drift_arr, A_t, x),
            "transition_jacobian_audit": audit_transition_jacobian_support(
                gc,
                A_t,
                x,
                linear=linear,
                nonlinear_strength=nonlinear_strength,
                nonlinear_scale=nonlinear_scale,
            ),
        }
        return x, gc, metadata
    return x, gc


def coordinatewise_nonlinearity(pred, beta, s0):
    """Coordinate-wise fixed-scale nonlinear transition."""
    pred_arr = np.asarray(pred)
    out_dtype = pred_arr.dtype if np.issubdtype(pred_arr.dtype, np.floating) else np.float64
    pred = pred_arr.astype(np.float64, copy=False)
    s0 = float(s0)
    if s0 <= 0:
        raise ValueError("nonlinear_scale must be positive")
    beta = float(beta)
    out = (1.0 - beta) * pred + beta * s0 * np.tanh(pred / s0)
    return out.astype(out_dtype, copy=False)


def coordinatewise_nonlinearity_derivative(pred, beta, s0):
    pred = np.asarray(pred, dtype=np.float64)
    z = pred / float(s0)
    return (1.0 - float(beta)) + float(beta) * (1.0 - np.tanh(z) ** 2)


def nonlinear_diagnostics(pred_linear, pred_nl, s0):
    pred_linear = np.asarray(pred_linear, dtype=np.float64)
    pred_nl = np.asarray(pred_nl, dtype=np.float64)
    z = pred_linear / float(s0)
    denom = np.mean(np.abs(pred_linear)) + 1e-12
    per_variable_relative = np.abs(pred_nl - pred_linear) / (np.abs(pred_linear) + 1e-12)
    return {
        "relative_l1_deviation": float(np.mean(np.abs(pred_nl - pred_linear)) / denom),
        "mean_abs_pre_activation_over_s0": float(np.mean(np.abs(z))),
        "max_abs_pre_activation_over_s0": float(np.max(np.abs(z))),
        "saturated_fraction_abs_z_gt_2": float(np.mean(np.abs(z) > 2.0)),
        "near_identity_fraction_abs_z_lt_0_1": float(np.mean(np.abs(z) < 0.1)),
        "per_variable_relative_l1_deviation": per_variable_relative.tolist(),
        "per_variable_abs_pre_activation_over_s0": np.abs(z).tolist(),
        "per_variable_saturated_fraction_abs_z_gt_2": (np.abs(z) > 2.0).astype(np.float64).tolist(),
        "per_variable_near_identity_fraction_abs_z_lt_0_1": (np.abs(z) < 0.1).astype(np.float64).tolist(),
    }


def summarize_nonlinear_records(records):
    if not records:
        return {
            "enabled": False,
            "relative_l1_deviation_mean": 0.0,
            "relative_l1_deviation_max": 0.0,
            "saturated_fraction_abs_z_gt_2_mean": 0.0,
            "near_identity_fraction_abs_z_lt_0_1_mean": 0.0,
        }
    out = {"enabled": True}
    for key in records[0].keys():
        vals = np.asarray([r[key] for r in records], dtype=np.float64)
        if vals.ndim == 1:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_max"] = float(np.max(vals))
            out[f"{key}_p95"] = float(np.percentile(vals, 95))
        else:
            out[f"{key}_mean_per_variable"] = np.mean(vals, axis=0).tolist()
            out[f"{key}_max_per_variable"] = np.max(vals, axis=0).tolist()
            out[f"{key}_p95_per_variable"] = np.percentile(vals, 95, axis=0).tolist()
            out[f"{key}_pooled_mean"] = float(np.mean(vals))
            out[f"{key}_pooled_max"] = float(np.max(vals))
            out[f"{key}_pooled_p95"] = float(np.percentile(vals, 95))
    return out


def transition_jacobian(A_t_at_time, history, linear=True, nonlinear_strength=0.0, nonlinear_scale=0.50):
    """Analytic one-step transition Jacobian D[target, source, lag]."""
    A = np.asarray(A_t_at_time, dtype=np.float64)  # (lag,d,d)
    lag = A.shape[0]
    d = A.shape[1]
    pred = np.zeros(d, dtype=np.float64)
    for k in range(lag):
        pred += A[k] @ np.asarray(history[k], dtype=np.float64)
    if linear or nonlinear_strength <= 0:
        scales = np.ones(d, dtype=np.float64)
    else:
        scales = coordinatewise_nonlinearity_derivative(pred, nonlinear_strength, nonlinear_scale)
    D = np.zeros((d, d, lag), dtype=np.float64)
    for k in range(lag):
        D[:, :, k] = scales[:, None] * A[k]
    return D


def deterministic_transition(A_t_at_time, history, linear=True, nonlinear_strength=0.0, nonlinear_scale=0.50):
    A = np.asarray(A_t_at_time, dtype=np.float64)
    pred = np.zeros(A.shape[1], dtype=np.float64)
    for k in range(A.shape[0]):
        pred += A[k] @ np.asarray(history[k], dtype=np.float64)
    if not linear and nonlinear_strength > 0:
        pred = coordinatewise_nonlinearity(pred, nonlinear_strength, nonlinear_scale)
    return np.asarray(pred, dtype=np.float64)


def audit_transition_jacobian_support(
    gc,
    A_t,
    x,
    linear=True,
    nonlinear_strength=0.0,
    nonlinear_scale=0.50,
    times=None,
    eps=1e-10,
):
    """Audit one-step transition Jacobian support against declared GC.

    The returned fields are intended to be directly usable as pass/fail gates
    for the P0.3b nonlinear ground-truth support audit.
    """
    gc_arr = np.asarray(gc).astype(bool)  # (d,d,lag)
    A_t_arr = np.asarray(A_t, dtype=np.float64)  # (T,lag,d,d)
    x_arr = np.asarray(x, dtype=np.float64)
    d, T = x_arr.shape
    lag = gc_arr.shape[2]
    if times is None:
        raw_times = [lag, max(lag, T // 3), max(lag, 2 * T // 3), T - 1]
        times = sorted(set(int(t) for t in raw_times if lag <= t < T))
    declared = gc_arr
    off_support = ~declared
    diag_off_support = np.zeros_like(declared, dtype=bool)
    diag = np.eye(d, dtype=bool)
    for k in range(lag):
        diag_off_support[:, :, k] = diag & off_support[:, :, k]

    max_off = 0.0
    max_diag_off = 0.0
    min_declared = np.inf
    linear_alignment = []
    actual_support_any = np.zeros_like(declared, dtype=bool)
    per_time = []
    declared_values = {tuple(idx): [] for idx in np.argwhere(declared)}
    ratio_values = []
    sign_mismatches = []
    coefficient_crossing_zero_edges = set()
    for t in times:
        history = [x_arr[:, t - k - 1] for k in range(lag)]
        D = transition_jacobian(
            A_t_arr[t],
            history,
            linear=linear,
            nonlinear_strength=nonlinear_strength,
            nonlinear_scale=nonlinear_scale,
        )
        A_as_d = np.transpose(A_t_arr[t], (1, 2, 0))
        actual_support_any |= np.abs(D) > eps
        if np.any(off_support):
            max_off = max(max_off, float(np.max(np.abs(D[off_support]))))
        if np.any(diag_off_support):
            max_diag_off = max(max_diag_off, float(np.max(np.abs(D[diag_off_support]))))
        if np.any(declared):
            min_declared = min(min_declared, float(np.min(np.abs(D[declared]))))
            for edge in declared_values:
                target, source, k = edge
                declared_values[edge].append(float(abs(D[target, source, k])))
                a_val = float(A_as_d[target, source, k])
                d_val = float(D[target, source, k])
                if abs(a_val) <= 1e-6:
                    coefficient_crossing_zero_edges.add(edge)
                else:
                    ratio_values.append(abs(d_val / a_val))
                    if np.sign(d_val) != np.sign(a_val):
                        sign_mismatches.append((int(target), int(source), int(k), int(t), d_val, a_val))
        if linear or nonlinear_strength <= 0:
            linear_alignment.append(float(np.max(np.abs(D - A_as_d))))
        per_time.append({
            "time": int(t),
            "nonzero_derivative_entries": int(np.sum(np.abs(D) > eps)),
            "off_support_entries": int(np.sum((np.abs(D) > eps) & off_support)),
            "declared_min_abs_derivative": float(np.min(np.abs(D[declared]))) if np.any(declared) else 0.0,
        })
    edge_medians = {}
    edge_maxima = {}
    for edge, values in declared_values.items():
        arr = np.asarray(values, dtype=np.float64)
        key = f"{edge[0]}<-{edge[1]}@lag{edge[2]}"
        edge_medians[key] = float(np.median(arr)) if arr.size else 0.0
        edge_maxima[key] = float(np.max(arr)) if arr.size else 0.0
    ratio_arr = np.asarray(ratio_values, dtype=np.float64)
    ratio_min = float(np.min(ratio_arr)) if ratio_arr.size else 1.0
    ratio_max = float(np.max(ratio_arr)) if ratio_arr.size else 1.0
    declared_edge_median_min = float(min(edge_medians.values())) if edge_medians else 0.0
    declared_edge_max_min = float(min(edge_maxima.values())) if edge_maxima else 0.0
    support_subset = bool(np.all(~actual_support_any | declared))
    any_lag_equal = bool(np.array_equal(np.any(actual_support_any, axis=2), np.any(declared, axis=2)))
    lag_specific_equal = bool(np.array_equal(actual_support_any, declared))
    off_support_gate = bool(max_off < 1e-8)
    ratio_gate = bool((len(sign_mismatches) == 0) and ratio_min >= 0.5 - 1e-6 and ratio_max <= 1.0 + 1e-6)
    edge_strength_gate = bool(declared_edge_median_min > 1e-3 and declared_edge_max_min > 1e-3)
    return {
        "times": [int(t) for t in times],
        "max_abs_off_support_derivative": float(max_off),
        "max_abs_diagonal_off_support_derivative": float(max_diag_off),
        "actual_support_subset_declared": support_subset,
        "actual_any_lag_support_equals_declared": any_lag_equal,
        "actual_lag_specific_support_equals_declared": lag_specific_equal,
        "declared_min_abs_derivative": float(0.0 if np.isinf(min_declared) else min_declared),
        "declared_min_abs_derivative_threshold": 1e-8,
        "supported_derivative_over_A_min": ratio_min,
        "supported_derivative_over_A_max": ratio_max,
        "supported_derivative_over_A_gate": ratio_gate,
        "supported_sign_mismatch_count": int(len(sign_mismatches)),
        "supported_sign_mismatches": sign_mismatches[:20],
        "declared_edge_median_abs_derivative_min": declared_edge_median_min,
        "declared_edge_max_abs_derivative_min": declared_edge_max_min,
        "declared_edge_median_abs_derivative": edge_medians,
        "declared_edge_max_abs_derivative": edge_maxima,
        "coefficient_crossing_zero_edges": [
            {"target": int(e[0]), "source": int(e[1]), "lag": int(e[2])}
            for e in sorted(coefficient_crossing_zero_edges)
        ],
        "off_support_gate_passed": off_support_gate,
        "support_gate_passed": bool(support_subset and any_lag_equal and lag_specific_equal),
        "declared_edge_strength_gate_passed": edge_strength_gate,
        "transition_jacobian_gate_passed": bool(
            off_support_gate
            and support_subset
            and any_lag_equal
            and lag_specific_equal
            and ratio_gate
            and edge_strength_gate
        ),
        "linear_jacobian_A_t_max_abs_diff": float(max(linear_alignment) if linear_alignment else 0.0),
        "per_time": per_time,
    }


def transition_jacobian_fd_spot_check(
    gc,
    A_t,
    x,
    linear=True,
    nonlinear_strength=0.0,
    nonlinear_scale=0.50,
    times=None,
    entries_per_time=8,
    eps=1e-6,
):
    """Deterministic central finite-difference spot-check for transition Jacobians."""
    gc_arr = np.asarray(gc).astype(bool)
    A_t_arr = np.asarray(A_t, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    d, T = x_arr.shape
    lag = gc_arr.shape[2]
    if times is None:
        n_times = min(10, max(1, T - lag))
        times = np.linspace(lag, T - 1, num=n_times, dtype=int).tolist()
    times = sorted(set(int(t) for t in times if lag <= int(t) < T))
    declared = np.argwhere(gc_arr)
    off_support = np.argwhere(~gc_arr & ~np.eye(d, dtype=bool)[:, :, None])
    if declared.size == 0:
        declared = np.empty((0, 3), dtype=int)
    if off_support.size == 0:
        off_support = np.empty((0, 3), dtype=int)

    checks = []
    failures = []
    max_abs_diff = 0.0
    max_allowed = 0.0
    for pos, t in enumerate(times):
        history = [x_arr[:, t - k - 1].copy() for k in range(lag)]
        D = transition_jacobian(
            A_t_arr[t],
            history,
            linear=linear,
            nonlinear_strength=nonlinear_strength,
            nonlinear_scale=nonlinear_scale,
        )
        candidates = []
        half = max(1, entries_per_time // 2)
        for arr, n_take in [(declared, half), (off_support, entries_per_time - half)]:
            if len(arr) == 0:
                continue
            for j in range(n_take):
                candidates.append(tuple(int(v) for v in arr[(pos * n_take + j) % len(arr)]))
        for target, source, lag_pos in candidates:
            h_plus = [h.copy() for h in history]
            h_minus = [h.copy() for h in history]
            h_plus[lag_pos][source] += eps
            h_minus[lag_pos][source] -= eps
            f_plus = deterministic_transition(
                A_t_arr[t],
                h_plus,
                linear=linear,
                nonlinear_strength=nonlinear_strength,
                nonlinear_scale=nonlinear_scale,
            )[target]
            f_minus = deterministic_transition(
                A_t_arr[t],
                h_minus,
                linear=linear,
                nonlinear_strength=nonlinear_strength,
                nonlinear_scale=nonlinear_scale,
            )[target]
            fd = (f_plus - f_minus) / (2.0 * eps)
            auto = float(D[target, source, lag_pos])
            allowed = 1e-6 + 1e-3 * max(abs(fd), abs(auto))
            diff = abs(fd - auto)
            max_abs_diff = max(max_abs_diff, float(diff))
            max_allowed = max(max_allowed, float(allowed))
            row = {
                "time": int(t),
                "target": int(target),
                "source": int(source),
                "lag": int(lag_pos),
                "fd": float(fd),
                "analytic": auto,
                "abs_diff": float(diff),
                "allowed": float(allowed),
                "passed": bool(diff <= allowed),
            }
            checks.append(row)
            if not row["passed"]:
                failures.append(row)
    return {
        "times": [int(t) for t in times],
        "n_checks": int(len(checks)),
        "entries_per_time": int(entries_per_time),
        "max_abs_diff": max_abs_diff,
        "max_allowed_tolerance": max_allowed,
        "passed": bool(len(failures) == 0 and len(checks) > 0),
        "failures": failures[:20],
        "checks": checks,
    }


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
            nonlinear_scale=params.get("nonlinear_scale", 0.50),
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
