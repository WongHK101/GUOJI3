"""Numerical verification of theoretical bounds (Theorem 1-3).

Tests:
  1. Perturbation sweep: Granger deviation vs residual_scale (Theorem 1)
  2. Gradient coupling: Jacobian penalty gradient O(eps) for filter vs O(1) for concat (Theorem 2)
  3. Orthogonality certificate: L_ortho vs Granger deviation during training (Theorem 3)
  4. Lipschitz constant estimation for MambaBlock and TCNBlock

Run on cloud: /root/autodl-tmp/GUOJI/mamba_enhanced/
"""
import torch
import torch.nn as nn
import numpy as np
import sys, os, json, time
from collections import defaultdict

torch.backends.cudnn.enabled = False

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path
_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)
# project root now resolved via _PROJ_ROOT

from minimal_mamba import MambaBlock, TCNBlock
from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                MambaJRNGC, train_model, compute_metrics)

device = torch.device("cuda")


# ============================================================
# Test 1: Perturbation Sweep (Theorem 1)
# ============================================================
def test_perturbation_sweep():
    """Verify ||G_eps - G_0|| scales linearly with epsilon."""
    print("=" * 60)
    print("TEST 1: Perturbation Sweep (Theorem 1)")
    print("=" * 60)

    # Load small dataset
    data_dir = "" + os.path.join(_PROJ_ROOT, "data", "nonstationary_var"
    x = np.load(os.path.join(data_dir, "num_nodes_10", "true_lag_7",
                             "noise_scale_1", "seed_0", "_x.npy"))
    gc = np.load(os.path.join(data_dir, "num_nodes_10", "true_lag_7",
                              "noise_scale_1", "seed_0", "_gc.npy"))
    d = x.shape[0]
    print(f"  Data: d={d}, T={x.shape[1]}")

    eps_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    results = {"mamba": {}, "tcn": {}}

    for filter_type in ["mamba", "tcn"]:
        print(f"\n  --- {filter_type.upper()} Filter ---")
        G0 = None
        for eps in eps_values:
            torch.manual_seed(0)
            np.random.seed(0)
            m = MambaFilterJRNGC(d=d, lag=7, layers=5, hidden=50,
                                 jacobian_lam=0.01, d_state=8,
                                 ortho_lam=0.05, residual_scale=eps,
                                 filter_type=filter_type).to(device)
            m, loss = train_model(m, x, max_iter=2000, lr=1e-3, verbose=False)
            G = m.get_gc_matrix(x)
            if G.ndim == 3:
                G = np.max(np.abs(G), axis=2)

            if eps == 0.0:
                G0 = G
                results[filter_type][eps] = {"auroc": None, "deviation": 0.0}
            else:
                deviation = np.max(np.abs(G - G0))
                results[filter_type][eps] = {"deviation": float(deviation)}

            auroc = compute_metrics(gc, G)["auroc"]
            results[filter_type][eps]["auroc"] = float(auroc)
            print(f"    eps={eps:.3f}: AUROC={auroc:.4f}, "
                  f"||G-G0||_max={results[filter_type][eps]['deviation']:.6f}")
            del m
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n  {'eps':>8}  {'Mb deviation':>14}  {'Mb AUROC':>10}  "
          f"{'TCN deviation':>14}  {'TCN AUROC':>10}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*10}  {'-'*14}  {'-'*10}")
    for eps in eps_values:
        if eps == 0.0:
            print(f"  {eps:8.3f}  {'(ref)':>14}  {results['mamba'][eps]['auroc']:10.4f}  "
                  f"{'(ref)':>14}  {results['tcn'][eps]['auroc']:10.4f}")
        else:
            print(f"  {eps:8.3f}  {results['mamba'][eps]['deviation']:14.6f}  "
                  f"{results['mamba'][eps]['auroc']:10.4f}  "
                  f"{results['tcn'][eps]['deviation']:14.6f}  "
                  f"{results['tcn'][eps]['auroc']:10.4f}")

    # Verify linear scaling: fit ||G-G0|| vs eps
    eps_arr = np.array([e for e in eps_values if e > 0])
    for ft in ["mamba", "tcn"]:
        dev_arr = np.array([results[ft][e]["deviation"] for e in eps_arr])
        if np.any(dev_arr > 0):
            slope = np.polyfit(eps_arr, dev_arr, 1)[0]
            r2 = np.corrcoef(eps_arr, dev_arr)[0, 1] ** 2
            print(f"\n  {ft}: deviation ≈ {slope:.4f} × ε  (R² = {r2:.4f})")

    return results


# ============================================================
# Test 2: Gradient Coupling (Theorem 2)
# ============================================================
def test_gradient_coupling():
    """Verify filter gradient to Jacobian penalty is O(eps) vs O(1) for concat."""
    print("\n" + "=" * 60)
    print("TEST 2: Gradient Coupling (Theorem 2)")
    print("=" * 60)

    d, lag, T = 10, 7, 200
    x = torch.randn(d, T).to(device)

    # --- Input Filter model ---
    m_filter = MambaFilterJRNGC(d=d, lag=lag, layers=3, hidden=32,
                                jacobian_lam=0.01, d_state=4,
                                ortho_lam=0.05, residual_scale=0.1,
                                filter_type="mamba").to(device)

    # --- Concat model ---
    m_concat = MambaJRNGC(d=d, lag=lag, layers=3, hidden=32,
                          jacobian_lam=0.01, d_state=4, d_cond=4,
                          use_time_weight_loss=False).to(device)

    for model_name, model in [("Filter", m_filter), ("Concat", m_concat)]:
        print(f"\n  --- {model_name} Model ---")
        model.train()

        # Forward: get loss
        loss = model.compute_loss(x.cpu().numpy())
        loss.backward()

        # Measure gradient norms for different parameter groups
        total_grad_norm = 0.0
        jac_related_grad_norm = 0.0  # params that affect Jacobian path
        n_params = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                gnorm = param.grad.norm().item()
                total_grad_norm += gnorm ** 2
                n_params += param.numel()

        total_grad_norm = total_grad_norm ** 0.5
        print(f"    Total grad norm: {total_grad_norm:.6f}")
        print(f"    Avg per-param grad: {total_grad_norm / max(n_params, 1):.8f}")

        # Key measurement: gradient of filter/preprocessor params
        filter_grad_norm = 0.0
        mlp_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                gnorm = param.grad.norm().item()
                if 'filter' in name or 'preprocessor' in name:
                    filter_grad_norm += gnorm ** 2
                elif 'inputgate' in name or 'outputgate' in name or 'encoders' in name:
                    mlp_grad_norm += gnorm ** 2

        filter_grad_norm = filter_grad_norm ** 0.5
        mlp_grad_norm = mlp_grad_norm ** 0.5
        print(f"    Filter/preprocessor grad norm: {filter_grad_norm:.6f}")
        print(f"    MLP (predictor) grad norm:      {mlp_grad_norm:.6f}")
        if mlp_grad_norm > 0:
            print(f"    Ratio filter/MLP:               {filter_grad_norm/mlp_grad_norm:.6f}")
            print(f"    Interpretation: ", end="")
            if filter_grad_norm / mlp_grad_norm < 0.1:
                print("Filter gradient is O(eps) smaller ✓ (Theorem 2)")
            else:
                print("Filter gradient comparable to MLP — needs larger eps gap")

        model.zero_grad()
        del model
        torch.cuda.empty_cache()

    # Also test at init (no training) — gradient should be truly O(eps)
    print(f"\n  --- Gradient at Init (no training) ---")
    for model_name, model_cls, kwargs in [
        ("Filter", MambaFilterJRNGC,
         {"d": d, "lag": lag, "layers": 3, "hidden": 32, "jacobian_lam": 0.01,
          "d_state": 4, "ortho_lam": 0.05, "residual_scale": 0.1,
          "filter_type": "mamba"}),
        ("Concat", MambaJRNGC,
         {"d": d, "lag": lag, "layers": 3, "hidden": 32, "jacobian_lam": 0.01,
          "d_state": 4, "d_cond": 4, "use_time_weight_loss": False}),
    ]:
        m = model_cls(**kwargs).to(device)
        loss = m.compute_loss(x.cpu().numpy())

        # Isolate Jacobian penalty gradient
        m.zero_grad()
        windows = None
        if hasattr(m, 'make_filtered_windows'):
            windows, x_orig_t, x_filt_t = m.make_filtered_windows(x.cpu().numpy())
            jac_x = windows[:min(len(windows), 50), :, :lag].detach().clone()
        elif hasattr(m, 'preprocess_and_windowing'):
            if model_name == "Filter":
                continue
            xz_win, t_weights = m.preprocess_and_windowing(x.cpu().numpy())
            jac_x = xz_win[:min(len(xz_win), 50), :d, :lag].detach().clone()
            x_cond = xz_win[:min(len(xz_win), 50), d:, :lag].detach()
        else:
            windows = m.make_windows(x.cpu().numpy())
            jac_x = windows[:min(len(windows), 50), :, :lag].detach().clone()

        jac_x = jac_x.to(device).requires_grad_(True)

        if model_name == "Concat":
            x_cond = x_cond.to(device)
            x_cat = torch.cat([jac_x, x_cond], dim=1).flatten(start_dim=1)
        else:
            x_cat = jac_x.flatten(start_dim=1)

        y = m(x_cat)
        jac_loss = torch.tensor(0.0, device=device)
        for j_idx in range(y.shape[1]):
            grad = torch.autograd.grad(y[:, j_idx], jac_x,
                                       grad_outputs=torch.ones_like(y[:, j_idx]),
                                       create_graph=True, retain_graph=True)[0]
            jac_loss = jac_loss + torch.mean(torch.abs(grad))
        jac_loss.backward()

        filter_grad_norm = 0.0
        for name, param in m.named_parameters():
            if param.grad is not None:
                if 'filter' in name or 'preprocessor' in name:
                    filter_grad_norm += param.grad.norm().item() ** 2
        filter_grad_norm = filter_grad_norm ** 0.5
        print(f"    {model_name}: ||∂L_jac/∂θ_filter|| = {filter_grad_norm:.8f}")
        del m
        torch.cuda.empty_cache()


# ============================================================
# Test 3: Orthogonality Certificate (Theorem 3)
# ============================================================
def test_orthogonality_certificate():
    """Track L_ortho vs Granger deviation during training."""
    print("\n" + "=" * 60)
    print("TEST 3: Orthogonality Certificate (Theorem 3)")
    print("=" * 60)

    data_dir = "" + os.path.join(_PROJ_ROOT, "data", "nonstationary_var"
    x = np.load(os.path.join(data_dir, "num_nodes_10", "true_lag_7",
                             "noise_scale_1", "seed_0", "_x.npy"))
    gc = np.load(os.path.join(data_dir, "num_nodes_10", "true_lag_7",
                              "noise_scale_1", "seed_0", "_gc.npy"))
    d, T = x.shape
    lag = 7

    # Train baseline (eps=0) for reference G0
    print("  Training baseline (eps=0)...")
    torch.manual_seed(0)
    np.random.seed(0)
    m0 = MambaFilterJRNGC(d=d, lag=lag, layers=5, hidden=50,
                          jacobian_lam=0.01, d_state=8,
                          ortho_lam=0.05, residual_scale=0.0,
                          filter_type="mamba").to(device)
    m0, _ = train_model(m0, x, max_iter=2000, lr=1e-3, verbose=False)
    G0 = m0.get_gc_matrix(x)
    if G0.ndim == 3:
        G0 = np.max(np.abs(G0), axis=2)
    G0_norm = np.max(np.abs(G0))
    del m0
    torch.cuda.empty_cache()

    # Train filter model with logging every N iterations
    print("  Training filter model (eps=0.1) with logging...")
    torch.manual_seed(0)
    np.random.seed(0)
    m = MambaFilterJRNGC(d=d, lag=lag, layers=5, hidden=50,
                         jacobian_lam=0.01, d_state=8,
                         ortho_lam=0.05, residual_scale=0.1,
                         filter_type="mamba").to(device)

    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
    log_every = 50
    logs = []

    for it in range(2000):
        m.train()
        optimizer.zero_grad()

        windows, x_orig_t, x_filt_t = m.make_filtered_windows(x)
        x_input = windows[:, :, :lag]
        x_target = windows[:, :, -1]
        pred = m(x_input)
        pred_loss = m.loss_fn(pred, x_target)

        # Ortho loss (same as in compute_loss)
        diff_sq = torch.mean((x_filt_t - x_orig_t) ** 2)
        x_orig_norm = torch.sqrt(torch.mean(x_orig_t ** 2) + 1e-8)
        x_filt_norm = torch.sqrt(torch.mean(x_filt_t ** 2) + 1e-8)
        loss_ortho = m.ortho_lam * diff_sq / (x_orig_norm * x_filt_norm + 1e-8)

        # Jacobian loss
        jac_x = windows[:min(len(windows), 100), :, :lag].detach().clone()
        jac_x.requires_grad_(True)
        y = m(jac_x)
        jac_loss_val = torch.tensor(0.0, device=device)
        for j in range(y.shape[1]):
            grad = torch.autograd.grad(y[:, j], jac_x,
                                       grad_outputs=torch.ones_like(y[:, j]),
                                       create_graph=True)[0]
            jac_loss_val = jac_loss_val + torch.mean(torch.abs(grad))
        jac_loss_val = m.jacobian_lam * jac_loss_val

        total_loss = pred_loss + jac_loss_val + loss_ortho
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()

        if it % log_every == 0:
            G = m.get_gc_matrix(x)
            if G.ndim == 3:
                G = np.max(np.abs(G), axis=2)
            deviation = np.max(np.abs(G - G0))
            logs.append({
                "iter": it,
                "loss_ortho": float(loss_ortho.item()),
                "deviation": float(deviation),
                "pred_loss": float(pred_loss.item()),
                "jac_loss": float(jac_loss_val.item()),
            })
            if it % 200 == 0:
                print(f"    iter {it:4d}: ortho={loss_ortho.item():.6f}, "
                      f"||G-G0||={deviation:.6f}")

    # Verify correlation — segment by training phase (Theorem 3)
    ortho_vals = np.array([l["loss_ortho"] for l in logs])
    dev_vals = np.array([l["deviation"] for l in logs])

    half = len(logs) // 2
    ortho_conv = ortho_vals[half:]
    dev_conv = dev_vals[half:]

    corr_all = np.corrcoef(ortho_vals, dev_vals)[0, 1]
    corr_conv = np.corrcoef(ortho_conv, dev_conv)[0, 1] if len(ortho_conv) > 5 else corr_all

    print(f"\n  Correlation(L_ortho, ||G-G0||):")
    print(f"    Full trajectory:  r = {corr_all:.4f}  (MLP convergence dominates early)")
    print(f"    Post-convergence: r = {corr_conv:.4f}  (filter drift effect isolated)")
    print(f"  Interpretation: ", end="")
    if corr_conv > 0.5:
        print("L_ortho strongly predicts Granger deviation ✓ (Theorem 3)")
    elif corr_conv > 0.2:
        print("L_ortho moderately predicts Granger deviation ~")
    else:
        print("L_ortho weakly predicts Granger deviation — check ortho_lam")

    del m
    torch.cuda.empty_cache()
    return logs


# ============================================================
# Test 4: Lipschitz Constant Estimation
# ============================================================
def estimate_lipschitz(model_fn, input_shape, n_iters=50):
    """Estimate Lipschitz constant via power iteration on Jacobian.

    For a function f: R^n → R^m, the Lipschitz constant (spectral norm of Jacobian)
    can be estimated by power iteration: repeatedly apply J^T·J to a random vector.

    We use finite differences to approximate J·v, avoiding explicit Jacobian.
    """
    x = torch.randn(*input_shape, device=device)
    x.requires_grad_(True)

    # Initial random direction
    v = torch.randn_like(x)
    v = v / v.norm()

    for i in range(n_iters):
        # J·v ≈ (f(x + δ·v) - f(x)) / δ
        with torch.no_grad():
            delta = 1e-3
            x_plus = x + delta * v
            x.requires_grad_(True)

        fx = model_fn(x)
        fx_plus = model_fn(x_plus)

        Jv = (fx_plus - fx) / delta

        # J^T·(J·v) via backward
        loss = (fx * Jv.detach()).sum()
        grad_outputs = torch.autograd.grad(loss, x, create_graph=False)[0]

        v_new = grad_outputs
        v = v_new / (v_new.norm() + 1e-8)

        x = x.detach().requires_grad_(True)

    # Final estimate
    with torch.no_grad():
        delta = 1e-3
        x_plus = x + delta * v
        fx = model_fn(x)
        fx_plus = model_fn(x_plus)
        Jv = (fx_plus - fx) / delta
        lip_est = Jv.norm() / v.norm()

    return float(lip_est)


def test_lipschitz_estimation():
    """Estimate Lipschitz constants for MambaBlock and TCNBlock."""
    print("\n" + "=" * 60)
    print("TEST 4: Lipschitz Constant Estimation")
    print("=" * 60)

    d_model = 32
    B, L = 1, 64

    for block_name, block_cls, kwargs in [
        ("MambaBlock", MambaBlock,
         {"d_model": d_model, "d_state": 8, "d_conv": 4, "expand": 2,
          "residual_scale": 0.1}),
        ("TCNBlock", TCNBlock,
         {"d_model": d_model, "kernel_size": 3, "dilation": 2,
          "residual_scale": 0.1}),
    ]:
        print(f"\n  --- {block_name} ---")
        block = block_cls(**kwargs).to(device)

        # At initialization
        def model_fn(x):
            return block(x)

        lip_init = estimate_lipschitz(model_fn, (B, L, d_model), n_iters=20)
        print(f"    Lipschitz at init: {lip_init:.4f}")

        # Expected: at init, MambaBlock should have small Lipschitz due to
        # zero-init out_proj + residual_scale=0.1. The residual connection
        # adds identity, so L ≈ 1 + ε·L_H ≈ 1 + 0.1·(small).

        # After some random updates (simulate training effect)
        opt = torch.optim.Adam(block.parameters(), lr=1e-3)
        for _ in range(100):
            opt.zero_grad()
            x_in = torch.randn(B, L, d_model, device=device)
            y = block(x_in)
            loss = y.norm()
            loss.backward()
            opt.step()

        lip_trained = estimate_lipschitz(model_fn, (B, L, d_model), n_iters=20)
        print(f"    Lipschitz after 100 steps: {lip_trained:.4f}")

        del block
        torch.cuda.empty_cache()


# ============================================================
# Main
# ============================================================
def main():
    print("THEORETICAL BOUND VERIFICATION")
    print("=" * 60)
    print(f"Device: {device}")
    print()

    all_results = {}

    # Test 1: Perturbation sweep
    try:
        r1 = test_perturbation_sweep()
        all_results["perturbation"] = r1
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Gradient coupling
    try:
        test_gradient_coupling()
    except Exception as e:
        print(f"Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Orthogonality certificate
    try:
        r3 = test_orthogonality_certificate()
        all_results["orthogonality"] = r3
    except Exception as e:
        print(f"Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Lipschitz estimation
    try:
        test_lipschitz_estimation()
    except Exception as e:
        print(f"Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Save
    with open("theory_verification_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to theory_verification_results.json")


if __name__ == "__main__":
    main()
