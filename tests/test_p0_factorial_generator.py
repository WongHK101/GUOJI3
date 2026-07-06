"""P0.2 tests for the factorial generator support contract."""

import os
import sys

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from factorial_data import (  # noqa: E402
    FACTORIAL_CELLS,
    FACTORIAL_SETTINGS,
    audit_transition_jacobian_support,
    deterministic_transition,
    generate_factorial_cell,
    transition_jacobian,
    transition_jacobian_fd_spot_check,
)


def _generate_all(seed=0):
    params = FACTORIAL_SETTINGS["D2"]
    out = {}
    for name, stationary, linear in FACTORIAL_CELLS:
        x, gc, meta = generate_factorial_cell(
            d=6,
            T=80,
            lag=3,
            seed=seed,
            stationary=stationary,
            linear=linear,
            coeff_scale=params["coeff_scale"],
            noise_scale=params["noise_scale"],
            regime_shift_strength=0.0 if stationary else params["regime_shift_strength"],
            nonlinear_strength=0.0 if linear else params["nonlinear_strength"],
            nonlinear_scale=params["nonlinear_scale"],
            sparsity=0.2,
            return_metadata=True,
        )
        out[name] = {"x": x, "gc": gc, "meta": meta}
    return out


def test_factorial_cells_share_graph_base_and_noise():
    cells = _generate_all(seed=0)
    ref = cells["Stat+Linear"]
    for name, payload in cells.items():
        np.testing.assert_array_equal(payload["gc"], ref["gc"], err_msg=f"gc mismatch for {name}")
        np.testing.assert_allclose(payload["meta"]["A_base"], ref["meta"]["A_base"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(payload["meta"]["noise"], ref["meta"]["noise"], atol=0.0, rtol=0.0)


def test_nonstationary_drift_respects_declared_support_and_diagonal():
    cells = _generate_all(seed=1)
    for name in ["NS+Linear", "NS+Nonlinear"]:
        payload = cells[name]
        gc = payload["gc"]
        meta = payload["meta"]
        audit = meta["support_audit"]
        assert audit["drift_off_support_max_abs"] == 0.0
        assert audit["base_off_support_max_abs"] == 0.0
        assert audit["base_diagonal_max_abs"] == 0.0
        assert audit["drift_diagonal_max_abs"] == 0.0
        assert audit["A_t_diagonal_max_abs"] == 0.0
        assert audit["actual_support_subset_declared"]
        assert audit["actual_any_time_support_equals_declared"]
        support = np.any(np.abs(meta["A_t"]) > 1e-12, axis=0)  # (lag,d,d)
        np.testing.assert_array_equal(np.transpose(gc.astype(bool), (2, 0, 1)), support)


def test_factorial_series_finite_and_stability_diagnostics_present():
    cells = _generate_all(seed=2)
    for name, payload in cells.items():
        assert np.isfinite(payload["x"]).all(), name
        audit = payload["meta"]["support_audit"]
        assert not audit["series_has_nan_or_inf"]
        assert np.isfinite(audit["spectral_radius_max"])
        assert np.isfinite(audit["spectral_radius_mean"])
        assert np.isfinite(audit["spectral_radius_p95"])


def test_transition_jacobian_support_matches_declared_graph_all_cells():
    cells = _generate_all(seed=3)
    for name, payload in cells.items():
        audit = payload["meta"]["transition_jacobian_audit"]
        assert audit["max_abs_off_support_derivative"] < 1e-8, name
        assert audit["max_abs_diagonal_off_support_derivative"] < 1e-8, name
        assert audit["actual_support_subset_declared"], name
        assert audit["actual_any_lag_support_equals_declared"], name
        assert audit["actual_lag_specific_support_equals_declared"], name
        assert audit["declared_min_abs_derivative"] > audit["declared_min_abs_derivative_threshold"], name
        if "Linear" in name:
            assert audit["linear_jacobian_A_t_max_abs_diff"] < 1e-12, name


def test_transition_jacobian_matches_central_finite_difference():
    cells = _generate_all(seed=5)
    eps = 1e-6
    for name, payload in cells.items():
        meta = payload["meta"]
        linear = "Nonlinear" not in name
        nonlinear_strength = 0.0 if linear else FACTORIAL_SETTINGS["D2"]["nonlinear_strength"]
        t = 35
        history = [payload["x"][:, t - k - 1].astype(np.float64) for k in range(3)]
        D = transition_jacobian(
            meta["A_t"][t],
            history,
            linear=linear,
            nonlinear_strength=nonlinear_strength,
            nonlinear_scale=FACTORIAL_SETTINGS["D2"]["nonlinear_scale"],
        )
        for target, source, lag_pos in [(0, 1, 0), (1, 2, 1), (3, 0, 2), (4, 5, 1)]:
            h_plus = [h.copy() for h in history]
            h_minus = [h.copy() for h in history]
            h_plus[lag_pos][source] += eps
            h_minus[lag_pos][source] -= eps
            f_plus = deterministic_transition(
                meta["A_t"][t],
                h_plus,
                linear=linear,
                nonlinear_strength=nonlinear_strength,
                nonlinear_scale=FACTORIAL_SETTINGS["D2"]["nonlinear_scale"],
            )[target]
            f_minus = deterministic_transition(
                meta["A_t"][t],
                h_minus,
                linear=linear,
                nonlinear_strength=nonlinear_strength,
                nonlinear_scale=FACTORIAL_SETTINGS["D2"]["nonlinear_scale"],
            )[target]
            fd = (f_plus - f_minus) / (2.0 * eps)
            auto = D[target, source, lag_pos]
            tol = 1e-6 + 1e-3 * max(abs(fd), abs(auto))
            assert abs(fd - auto) <= tol, (name, fd, auto, tol)


def test_nonlinear_is_coordinatewise_fixed_scale_and_nontrivial():
    cells = _generate_all(seed=4)
    for name in ["Stat+Nonlinear", "NS+Nonlinear"]:
        diag = cells[name]["meta"]["nonlinear_diagnostics"]
        assert diag["enabled"], name
        assert cells[name]["meta"]["nonlinear_scale"] == FACTORIAL_SETTINGS["D2"]["nonlinear_scale"]
        assert diag["relative_l1_deviation_mean"] > 1e-4, name
        assert diag["saturated_fraction_abs_z_gt_2_mean"] < 0.25, name


def test_p0_3b_stage1_generator_calibration_gates_seeds_0_to_5():
    params = FACTORIAL_SETTINGS["D2"]
    assert params["nonlinear_scale"] == 0.075
    for seed in range(6):
        for name, stationary, linear in FACTORIAL_CELLS:
            x, gc, meta = generate_factorial_cell(
                d=10,
                T=600,
                lag=3,
                seed=seed,
                stationary=stationary,
                linear=linear,
                coeff_scale=params["coeff_scale"],
                noise_scale=params["noise_scale"],
                regime_shift_strength=0.0 if stationary else params["regime_shift_strength"],
                nonlinear_strength=0.0 if linear else params["nonlinear_strength"],
                nonlinear_scale=params["nonlinear_scale"],
                sparsity=0.2,
                return_metadata=True,
            )
            assert np.isfinite(x).all(), (seed, name)
            assert not meta["support_audit"]["series_has_nan_or_inf"], (seed, name)
            audit = audit_transition_jacobian_support(
                gc,
                meta["A_t"],
                x,
                linear=linear,
                nonlinear_strength=0.0 if linear else params["nonlinear_strength"],
                nonlinear_scale=params["nonlinear_scale"],
                times=range(3, 600),
            )
            assert audit["max_abs_off_support_derivative"] < 1e-8, (seed, name)
            assert audit["actual_lag_specific_support_equals_declared"], (seed, name)
            assert audit["supported_sign_mismatch_count"] == 0, (seed, name)
            assert audit["supported_derivative_over_A_min"] >= 0.5 - 1e-6, (seed, name)
            assert audit["supported_derivative_over_A_max"] <= 1.0 + 1e-6, (seed, name)
            assert audit["declared_edge_median_abs_derivative_min"] > 1e-3, (seed, name)
            assert audit["declared_edge_max_abs_derivative_min"] > 1e-3, (seed, name)
            assert audit["transition_jacobian_gate_passed"], (seed, name, audit)

            fd = transition_jacobian_fd_spot_check(
                gc,
                meta["A_t"],
                x,
                linear=linear,
                nonlinear_strength=0.0 if linear else params["nonlinear_strength"],
                nonlinear_scale=params["nonlinear_scale"],
                times=np.linspace(3, 599, num=10, dtype=int).tolist(),
                entries_per_time=8,
            )
            assert fd["passed"], (seed, name, fd)

            if not linear:
                diag = meta["nonlinear_diagnostics"]
                assert 0.05 <= diag["relative_l1_deviation_mean"] <= 0.20, (seed, name, diag)
                assert diag["near_identity_fraction_abs_z_lt_0_1_mean"] < 0.50, (seed, name, diag)
                assert diag["saturated_fraction_abs_z_gt_2_mean"] < 0.10, (seed, name, diag)
                assert "per_variable_relative_l1_deviation_mean_per_variable" in diag
                assert len(diag["per_variable_relative_l1_deviation_mean_per_variable"]) == 10
                assert np.isfinite(np.asarray(diag["per_variable_relative_l1_deviation_mean_per_variable"])).all()


if __name__ == "__main__":
    tests = [
        obj for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
