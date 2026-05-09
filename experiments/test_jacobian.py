"""Jacobian stability test for Mamba SSM block.

Computes input-output Jacobian via:
1. torch.autograd (vjp-based reconstruction)
2. Finite-difference numerical approximation

Compares cosine similarity and condition number.
Pass threshold: cosine_sim > 0.98 and condition_number < 100.

Upload to /root/autodl-tmp/GUOJI/mamba_enhanced/ and run on cloud.
"""
import torch
import torch.nn as nn
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimal_mamba import MambaBlock, SelectiveSSM


def compute_jacobian_autodiff(model, x):
    """Compute input-output Jacobian via autograd.

    For each output dimension j, compute ∂y_j / ∂x via vjp.
    Returns (n_outputs_sampled, n_inputs) Jacobian matrix.
    """
    batch, seq_len, d_model = x.shape

    x_req = x.detach().clone().requires_grad_(True)
    y = model(x_req)  # (B, L, D)
    y_flat = y.reshape(-1)  # (B*L*D,)

    n_in = x_req.numel()
    n_out = y_flat.shape[0]
    n_sample = min(n_out, 200)
    jac = torch.zeros(n_sample, n_in, device=x.device)

    for j in range(n_sample):
        grad_output = torch.zeros_like(y_flat)
        grad_output[j] = 1.0
        grad = torch.autograd.grad(y_flat, x_req, grad_outputs=grad_output,
                                   retain_graph=True, create_graph=False)[0]
        jac[j] = grad.reshape(-1)

    return jac


def compute_jacobian_numerical(model, x, eps=1e-4):
    """Compute input-output Jacobian via symmetric finite differences."""
    batch, seq_len, d_model = x.shape
    x_flat = x.reshape(-1)
    n_inputs = x_flat.shape[0]

    y0 = model(x).reshape(-1)  # (B*L*D,)
    n_outputs = min(y0.shape[0], 200)
    jac = torch.zeros(n_outputs, n_inputs, device=x.device)

    for i in range(min(n_inputs, 100)):  # Sample ~100 input dims for efficiency
        e_i = torch.zeros_like(x)
        e_i_flat = e_i.reshape(-1)
        e_i_flat[i] = 1.0

        x_plus = x + eps * e_i
        x_minus = x - eps * e_i

        y_plus = model(x_plus).reshape(-1)[:n_outputs]
        y_minus = model(x_minus).reshape(-1)[:n_outputs]

        jac[:, i] = (y_plus - y_minus) / (2 * eps)

    return jac


def cosine_similarity(J_auto, J_num):
    """Compute mean cosine similarity between corresponding rows."""
    sims = []
    for j in range(min(J_auto.shape[0], J_num.shape[0])):
        a = J_auto[j]
        b = J_num[j]
        cos = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)
        sims.append(cos.item())
    return float(np.mean(sims)), float(np.std(sims))


def condition_number(J):
    """Compute condition number (ratio of largest to smallest singular value)."""
    U, S, V = torch.svd(J.float())
    S = S[S > 1e-10]
    if len(S) == 0:
        return float('inf')
    return float(S.max() / S.min())


def test_jacobian_stability(d_model=16, d_state=8, seq_len=32, batch=2):
    """Main Jacobian stability test."""
    print(f"{'='*60}")
    print(f"  Mamba Jacobian Stability Test")
    print(f"  d_model={d_model}, d_state={d_state}, seq_len={seq_len}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    torch.manual_seed(42)
    model = MambaBlock(d_model=d_model, d_state=d_state, d_conv=4, expand=2).to(device)
    model.eval()

    x = torch.randn(batch, seq_len, d_model, device=device) * 0.5

    # Autodiff Jacobian
    print("\n[1/3] Computing autodiff Jacobian...")
    t0 = time.time()
    J_auto = compute_jacobian_autodiff(model, x)
    t_auto = time.time() - t0
    print(f"  Done in {t_auto:.1f}s, shape={J_auto.shape}")

    # Numerical Jacobian
    print("[2/3] Computing numerical Jacobian...")
    t0 = time.time()
    J_num = compute_jacobian_numerical(model, x, eps=1e-3)
    t_num = time.time() - t0
    print(f"  Done in {t_num:.1f}s, shape={J_num.shape}")

    # Comparison
    print("[3/3] Computing similarity metrics...")
    cos_mean, cos_std = cosine_similarity(J_auto, J_num)
    cond_auto = condition_number(J_auto)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Cosine similarity (autodiff vs numerical): {cos_mean:.4f} ± {cos_std:.4f}")
    print(f"  Condition number (autodiff Jacobian):       {cond_auto:.2f}")
    print(f"  Autodiff time:  {t_auto:.1f}s")
    print(f"  Numerical time: {t_num:.1f}s")

    # Pass/Fail
    PASS_COS, PASS_COND = 0.98, 100.0
    cos_pass = cos_mean > PASS_COS
    cond_pass = cond_auto < PASS_COND

    print(f"\n  Thresholds: cosine > {PASS_COS}, condition < {PASS_COND}")
    print(f"  Cosine similarity: {'PASS ✓' if cos_pass else 'FAIL ✗'}")
    print(f"  Condition number:  {'PASS ✓' if cond_pass else 'FAIL ✗'}")
    print(f"  OVERALL:           {'PASS ✓' if cos_pass and cond_pass else 'FAIL ✗'}")

    return cos_mean, cond_auto, cos_pass and cond_pass


def test_jacobian_varying_dims():
    """Test Jacobian stability across different model sizes and sequence lengths."""
    configs = [
        (8, 4, 16),
        (16, 8, 32),
        (32, 16, 64),
        (64, 16, 128),
    ]

    print(f"\n{'='*60}")
    print(f"  Multi-Scale Jacobian Stability")
    print(f"{'='*60}")

    all_pass = True
    for d_model, d_state, seq_len in configs:
        print(f"\n--- d={d_model}, state={d_state}, L={seq_len} ---")
        cos, cond, ok = test_jacobian_stability(d_model, d_state, seq_len)
        if not ok:
            all_pass = False
            print(f"  WARNING: Config FAILED!")

    print(f"\n{'='*60}")
    print(f"  ALL CONFIGS PASS: {all_pass}")
    print(f"{'='*60}")
    return all_pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi", action="store_true", help="Test multiple dimensions")
    args = parser.parse_args()

    if args.multi:
        test_jacobian_varying_dims()
    else:
        test_jacobian_stability()
