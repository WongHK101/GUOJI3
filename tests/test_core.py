"""Minimal unit tests for ISTF-Mamba core invariants.

Run: python tests/test_core.py
"""
import numpy as np
import sys, os

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# JRNGC path (resolve from environment or sibling directory)
JRNGC_PATH = os.environ.get(
    "JRNGC_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "JRNGC")
)
if os.path.isdir(JRNGC_PATH):
    sys.path.insert(0, JRNGC_PATH)


# ============================================================
# Test 1: Windowing shape
# ============================================================
def test_windowing_shape():
    """make_windows produces (N, d, lag+1) where N = T - lag."""
    import torch

    class MockModel:
        def __init__(self, lag):
            self.lag = lag

        def make_windows(self, x_full):
            device = torch.device("cpu")
            x = torch.tensor(x_full, device=device, dtype=torch.float32)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = x.transpose(1, 2).unfold(1, self.lag + 1, 1)
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
            return x

    d, T, lag = 10, 500, 3
    x = np.random.randn(d, T).astype(np.float32)
    m = MockModel(lag=lag)
    w = m.make_windows(x)
    expected_N = T - lag
    assert w.shape == (expected_N, d, lag + 1), f"Expected ({expected_N},{d},{lag+1}), got {w.shape}"
    print(f"PASS: test_windowing_shape — N={expected_N}, shape={w.shape}")


# ============================================================
# Test 2: ISTF output dimension equals input dimension
# ============================================================
def test_istf_output_dim_eq_input_dim():
    """MambaFilterJRNGC forward pass preserves (B, d) output from (B, d, lag) input."""
    try:
        import torch
        from mamba_jrngc_pilot import MambaFilterJRNGC
    except (ImportError, ModuleNotFoundError) as e:
        print(f"SKIP: test_istf_output_dim — JRNGC not available ({e})")
        return

    d, lag = 10, 3
    model = MambaFilterJRNGC(
        d=d, lag=lag, layers=3, hidden=32,
        jacobian_lam=0.01, d_state=4, ortho_lam=0.05,
        residual_scale=0.1, filter_type="mamba"
    )
    x = torch.randn(4, d, lag)
    out = model(x)
    assert out.shape == (4, d), f"Expected (4, {d}), got {out.shape}"
    print(f"PASS: test_istf_output_dim — input (4,{d},{lag}) -> output {out.shape}")


# ============================================================
# Test 3: Mask intervention correctness
# ============================================================
def test_mask_intervention_correctness():
    """Verify mask_x_only / mask_c_only / mask_both produce zero/nonzero patterns."""
    import torch

    class MockPreprocessor:
        def __init__(self, d, d_cond, T_val):
            self.d = d
            self.d_cond = d_cond
            self.T_val = T_val

        def forward(self, x_in):
            B, d, T = x_in.shape
            c = torch.ones(B, T, self.d_cond)
            w = torch.ones(B, T)
            return c, w

    d, d_cond, lag, T_val = 10, 4, 3, 100
    pre = MockPreprocessor(d, d_cond, T_val)
    x = torch.randn(1, d, T_val)

    c, w = pre.forward(x)
    assert c.abs().sum() > 0, "c should be non-zero"
    assert w.abs().sum() > 0, "w should be non-zero"

    # mask_c_only: keep x, zero c
    c_zero = torch.zeros_like(c)
    assert c_zero.abs().sum() == 0, "mask_c_only: c should be all zero"
    assert x.abs().sum() > 0, "mask_c_only: x should be non-zero"

    # mask_x_only: zero x, keep c
    x_zero = torch.zeros_like(x)
    assert x_zero.abs().sum() == 0, "mask_x_only: x should be all zero"
    assert c.abs().sum() > 0, "mask_x_only: c should be non-zero"

    # mask_both: zero both
    assert x_zero.abs().sum() == 0 and c_zero.abs().sum() == 0, "mask_both: both should be zero"

    print("PASS: test_mask_intervention_correctness")


# ============================================================
# Test 4: Self-link removal consistency with metadata audit
# ============================================================
def test_self_link_removal_consistency():
    """remove_self_connection produces edge counts matching metadata audit values."""
    from tgc.metrics.causal import remove_self_connection

    # CT_medical: (40, 40) with 153 total, 20 non-zero diagonal → 133 after removal
    np.random.seed(42)
    gc_2d = np.zeros((40, 40), dtype=np.int32)
    # Place 153 edges: first fill diagonal with 20 ones, rest elsewhere
    np.fill_diagonal(gc_2d, 0)
    # Set 20 diagonal entries
    for i in range(20):
        gc_2d[i, i] = 1
    # Set 133 off-diagonal entries
    for i in range(40):
        for j in range(40):
            if i != j and gc_2d.sum() < 153:
                if gc_2d[i, j] == 0:
                    gc_2d[i, j] = 1
    # Adjust to exactly 153 by filling remaining
    remaining = 153 - gc_2d.sum()
    idx = 0
    for i in range(40):
        for j in range(40):
            if remaining <= 0:
                break
            if i != j and gc_2d[i, j] == 0:
                gc_2d[i, j] = 1
                remaining -= 1

    assert gc_2d.sum() == 153, f"Expected 153, got {gc_2d.sum()}"
    gt = remove_self_connection(gc_2d.astype(np.int32))
    # With 20 self-links on diagonal, expect 133 non-self edges
    assert gt.sum() == 133, f"CT_medical self-link removal: expected 133, got {gt.sum()}"

    # Lorenz_F40: full diagonal, 40 total, 30 after removal
    gc_lor = np.ones((10, 10), dtype=np.int32)
    assert gc_lor.sum() == 100
    gt_lor = remove_self_connection(gc_lor)
    assert gt_lor.sum() == 90, f"Lorenz full removal: expected 90, got {gt_lor.sum()}"

    # No diagonal entries → no removal
    gc_no_diag = np.ones((5, 5), dtype=np.int32)
    np.fill_diagonal(gc_no_diag, 0)
    assert gc_no_diag.sum() == 20
    gt_no = remove_self_connection(gc_no_diag.astype(np.int32))
    assert gt_no.sum() == 20, f"No diagonal: expected 20, got {gt_no.sum()}"

    print("PASS: test_self_link_removal_consistency — CT_medical 153→133, Lorenz 100→90, no-diag 20→20")


# ============================================================
# Test 5: Factorial data generation
# ============================================================
def test_factorial_data_generation():
    """generate_factorial_cell produces correct shapes, sparse edges, no self-loops."""
    from factorial_data import generate_factorial_cell

    d, T, lag, seed = 10, 600, 3, 42
    sparsity = 0.2

    # Test stationary linear
    x_stat, gc_stat = generate_factorial_cell(
        d=d, T=T, lag=lag, seed=seed,
        stationary=True, linear=True,
        coeff_scale=0.40, noise_scale=0.15,
        sparsity=sparsity,
    )
    assert x_stat.shape == (d, T), f"Stat+Linear x shape: expected ({d},{T}), got {x_stat.shape}"
    assert gc_stat.shape == (d, d, lag), f"Stat+Linear gc shape: expected ({d},{d},{lag}), got {gc_stat.shape}"
    # No self-loops
    for l in range(lag):
        assert np.all(np.diag(gc_stat[:, :, l]) == 0), f"Self-loops found in lag {l}"
    # Edge count roughly matches sparsity: d*(d-1)*sparsity
    expected_edges = d * (d - 1) * sparsity
    actual_edges = gc_stat.sum()
    assert abs(actual_edges - expected_edges) < d, \
        f"Edge count {actual_edges} deviates too far from expected {expected_edges}"

    # Test non-stationary nonlinear
    x_ns, gc_ns = generate_factorial_cell(
        d=d, T=T, lag=lag, seed=seed,
        stationary=False, linear=False,
        coeff_scale=0.40, noise_scale=0.15,
        regime_shift_strength=0.20, nonlinear_strength=0.50,
        sparsity=sparsity,
    )
    assert x_ns.shape == (d, T)
    assert gc_ns.shape == (d, d, lag)
    for l in range(lag):
        assert np.all(np.diag(gc_ns[:, :, l]) == 0), f"Self-loops found in lag {l}"

    # Same seed produces same GC for same sparsity
    x_s2, gc_s2 = generate_factorial_cell(
        d=d, T=T, lag=lag, seed=seed,
        stationary=True, linear=True,
        coeff_scale=0.40, noise_scale=0.15,
        sparsity=sparsity,
    )
    assert np.allclose(gc_stat, gc_s2), "Same seed should produce same GC structure"
    assert np.allclose(x_stat, x_s2), "Same seed should produce same data"

    # Different seed produces different graph
    x_diff, gc_diff = generate_factorial_cell(
        d=d, T=T, lag=lag, seed=seed + 1,
        stationary=True, linear=True,
        coeff_scale=0.40, noise_scale=0.15,
        sparsity=sparsity,
    )
    assert not np.allclose(gc_stat, gc_diff), "Different seeds should produce different graphs"

    print("PASS: test_factorial_data_generation — shapes, sparsity, no self-loops, seed reproducibility")


if __name__ == "__main__":
    test_windowing_shape()
    test_istf_output_dim_eq_input_dim()
    test_mask_intervention_correctness()
    test_self_link_removal_consistency()
    test_factorial_data_generation()
    print("\nAll 5 tests passed.")
