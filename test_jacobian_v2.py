"""Jacobian stability test v2 — element-wise verification.

Tests autodiff Jacobian correctness at small scale with precise comparison.
"""
import torch
import torch.nn as nn
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimal_mamba import MambaBlock


def test_jacobian_elementwise(d_model=4, d_state=4, seq_len=4, batch=1):
    """Element-wise Jacobian comparison on a tiny model."""
    print(f"{'='*60}")
    print(f"  Jacobian Element-wise Test")
    print(f"  d={d_model}, state={d_state}, L={seq_len}, batch={batch}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    torch.manual_seed(42)
    model = MambaBlock(d_model=d_model, d_state=d_state, d_conv=2, expand=2).to(device)
    model.eval()

    # Small input with controlled values
    x = torch.randn(batch, seq_len, d_model, device=device) * 0.3

    # 1. Autodiff Jacobian — all elements
    print("\n[1] Computing full autodiff Jacobian...")
    x_req = x.detach().clone().requires_grad_(True)
    y = model(x_req)
    y_flat = y.reshape(-1)
    x_flat = x_req.reshape(-1)

    n_out = y_flat.shape[0]
    n_in = x_flat.shape[0]
    print(f"  n_in={n_in}, n_out={n_out}")

    J_auto = torch.zeros(n_out, n_in, device=device)
    t0 = time.time()
    for j in range(n_out):
        grad_output = torch.zeros_like(y_flat)
        grad_output[j] = 1.0
        grad = torch.autograd.grad(y_flat, x_req, grad_outputs=grad_output,
                                   retain_graph=True, create_graph=False)[0]
        J_auto[j] = grad.reshape(-1)
    t_auto = time.time() - t0
    print(f"  Done in {t_auto:.1f}s")

    # 2. Numerical Jacobian — all elements
    print("[2] Computing full numerical Jacobian...")
    eps = 1e-3
    J_num = torch.zeros(n_out, n_in, device=device)
    t0 = time.time()
    for i in range(n_in):
        e_i = torch.zeros_like(x)
        e_i_flat = e_i.reshape(-1)
        e_i_flat[i] = 1.0

        y_plus = model(x + eps * e_i).reshape(-1)
        y_minus = model(x - eps * e_i).reshape(-1)
        J_num[:, i] = (y_plus - y_minus) / (2 * eps)
    t_num = time.time() - t0
    print(f"  Done in {t_num:.1f}s")

    # 3. Comparison
    print("[3] Comparing...")
    diff = J_auto - J_num
    abs_diff = torch.abs(diff)
    rel_diff = abs_diff / (torch.abs(J_num) + 1e-8)

    max_abs_err = float(abs_diff.max())
    mean_abs_err = float(abs_diff.mean())
    max_rel_err = float(rel_diff[torch.abs(J_num) > 1e-6].max()) if (torch.abs(J_num) > 1e-6).any() else 0
    mean_rel_err = float(rel_diff[torch.abs(J_num) > 1e-6].mean()) if (torch.abs(J_num) > 1e-6).any() else 0

    # Cosine similarity
    cos_sim = float(torch.dot(J_auto.reshape(-1), J_num.reshape(-1)) /
                    (torch.norm(J_auto) * torch.norm(J_num) + 1e-8))

    # Condition numbers
    U_a, S_a, V_a = torch.svd(J_auto.float())
    U_n, S_n, V_n = torch.svd(J_num.float())
    S_a_pos = S_a[S_a > 1e-10]
    S_n_pos = S_n[S_n > 1e-10]
    cond_auto = float(S_a_pos.max() / S_a_pos.min()) if len(S_a_pos) > 0 else float('inf')
    cond_num = float(S_n_pos.max() / S_n_pos.min()) if len(S_n_pos) > 0 else float('inf')

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Max absolute error:  {max_abs_err:.2e}")
    print(f"  Mean absolute error: {mean_abs_err:.2e}")
    print(f"  Max relative error:  {max_rel_err:.2e}")
    print(f"  Mean relative error: {mean_rel_err:.2e}")
    print(f"  Cosine similarity:   {cos_sim:.6f}")
    print(f"  Cond (autodiff):     {cond_auto:.2e}")
    print(f"  Cond (numerical):    {cond_num:.2e}")

    # Pass/Fail
    PASS_COS, PASS_COND = 0.98, 100.0
    cos_pass = cos_sim > PASS_COS
    cond_pass = cond_auto < PASS_COND
    err_pass = max_rel_err < 0.1

    print(f"\n  Thresholds: cos > {PASS_COS}, cond < {PASS_COND}, max_rel_err < 0.1")
    print(f"  Cosine similarity:  {'PASS' if cos_pass else 'FAIL'}")
    print(f"  Condition number:   {'PASS' if cond_pass else 'FAIL'}")
    print(f"  Relative error:     {'PASS' if err_pass else 'FAIL'}")
    print(f"  OVERALL:            {'PASS' if cos_pass and cond_pass and err_pass else 'FAIL'}")

    # Show some element comparisons
    print(f"\n  Sample Jacobian elements (first 5x5):")
    print(f"    Autodiff:   {J_auto[:5, :5].tolist()}")
    print(f"    Numerical:  {J_num[:5, :5].tolist()}")

    return cos_sim, cond_auto, cos_pass and cond_pass and err_pass


def test_multi_scale():
    """Test at multiple scales to check stability."""
    configs = [(4, 4, 4, 1), (8, 8, 8, 1), (16, 8, 16, 2)]
    all_ok = True
    for d, state, L, B in configs:
        print(f"\n{'='*60}")
        cos, cond, ok = test_jacobian_elementwise(d, state, L, B)
        if not ok:
            all_ok = False
    print(f"\n{'='*60}")
    print(f"  ALL CONFIGS PASS: {all_ok}")
    return all_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi", action="store_true")
    args = parser.parse_args()

    if args.multi:
        test_multi_scale()
    else:
        test_jacobian_elementwise()
