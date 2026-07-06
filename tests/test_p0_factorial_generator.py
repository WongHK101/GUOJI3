"""P0.2 tests for the factorial generator support contract."""

import os
import sys

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from factorial_data import FACTORIAL_CELLS, FACTORIAL_SETTINGS, generate_factorial_cell  # noqa: E402


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


if __name__ == "__main__":
    tests = [
        obj for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
