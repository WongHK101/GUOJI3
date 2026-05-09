"""P1 #1: Shortcut Learning Mechanism Proof — 4 Diagnostic Experiments.

Experiments:
  Exp 1: Gradient path analysis — track ∂L/∂x_orig vs ∂L/∂z across architectures
  Exp 2: Controlled d_cond sweep — AUROC vs auxiliary dimension
  Exp 3: Loss decomposition trajectory — (pred_loss, jac_loss) phase portrait
  Exp 4: Coefficient recovery — known VAR(1) A matrix reconstruction

Run on cloud: /root/autodl-tmp/GUOJI/mamba_enhanced/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os, json, time
from collections import defaultdict

torch.backends.cudnn.enabled = False

sys.path.insert(0, "/root/autodl-tmp/GUOJI/JRNGC")
sys.path.insert(0, "/root/autodl-tmp/GUOJI/mamba_enhanced")

from mamba_jrngc_pilot import (BaselineJRNGC, MambaJRNGC, MambaJRNGC_FiLM,
                                MambaFilterJRNGC, train_model, compute_metrics)

device = torch.device("cuda")
OUT = "diagnostic_results/"
os.makedirs(OUT, exist_ok=True)


# ============================================================
# Synthetic Data: VAR(1) with known A matrix
# ============================================================
def generate_var1_data(d=8, T=300, noise_scale=0.1, sparsity=0.3, seed=42):
    """Generate VAR(1) data x_t = A·x_{t-1} + ε_t with known sparse A."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Sparse A with controlled spectral radius < 1 (stable)
    A = torch.randn(d, d) * (torch.rand(d, d) < sparsity).float()
    # Scale so spectral radius = 0.8 (stable but clearly non-zero)
    sr = torch.linalg.eigvals(A).abs().max()
    A = A * (0.8 / sr)
    A = A.numpy()

    # Generate time series
    x = np.zeros((d, T))
    x[:, 0] = np.random.randn(d) * 0.1
    for t in range(1, T):
        x[:, t] = A @ x[:, t-1] + np.random.randn(d) * noise_scale

    # Ground truth GC: |A[i,j]| > 0
    gc = (np.abs(A) > 0.01).astype(np.float64)
    n_edges = int(gc.sum())

    print(f"  VAR(1) data: d={d}, T={T}, edges={n_edges}, "
          f"spectral_radius={float(sr)*0.8:.3f}")
    return x, gc, A


# ============================================================
# Exp 1: Gradient Path Analysis
# ============================================================
def exp1_gradient_path_analysis():
    """Track gradient norms through original vs auxiliary paths during training."""
    print("\n" + "=" * 60)
    print("EXP 1: Gradient Path Analysis")
    print("=" * 60)

    d, T, lag = 8, 300, 1
    x, gc, A_true = generate_var1_data(d=d, T=T, seed=42)

    architectures = {
        "Baseline": (BaselineJRNGC,
                     {"d": d, "lag": lag, "layers": 3, "hidden": 32, "jacobian_lam": 0.01}),
        "Concat": (MambaJRNGC,
                   {"d": d, "lag": lag, "layers": 3, "hidden": 32, "jacobian_lam": 0.01,
                    "d_state": 4, "d_cond": 4, "use_time_weight_loss": False}),
        "FiLM": (MambaJRNGC_FiLM,
                 {"d": d, "lag": lag, "layers": 3, "hidden": 32, "jacobian_lam": 0.01,
                  "d_state": 4, "d_cond": 4, "use_time_weight_loss": False}),
        "Filter": (MambaFilterJRNGC,
                   {"d": d, "lag": lag, "layers": 3, "hidden": 32, "jacobian_lam": 0.01,
                    "d_state": 4, "ortho_lam": 0.05, "residual_scale": 0.1,
                    "filter_type": "mamba"}),
    }

    results = {}
    max_iter = 1500
    log_every = 30

    for arch_name, (ModelClass, kwargs) in architectures.items():
        print(f"\n  --- {arch_name} ---")
        torch.manual_seed(42)
        np.random.seed(42)
        model = ModelClass(**kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        log = []
        for it in range(max_iter):
            model.train()
            optimizer.zero_grad()

            # Forward based on architecture type
            if arch_name == "Baseline":
                windows = model.make_windows(x)
                x_input = windows[:, :, :lag]
                x_target = windows[:, :, -1]
                x_input.requires_grad_(True)
                pred = model(x_input)
                pred_loss = model.loss_fn(pred, x_target)

                # Jacobian loss
                jac_loss_val = torch.tensor(0.0, device=device)
                for j_idx in range(d):
                    grad = torch.autograd.grad(pred[:, j_idx], x_input,
                                               grad_outputs=torch.ones_like(pred[:, j_idx]),
                                               create_graph=True)[0]
                    jac_loss_val = jac_loss_val + torch.mean(torch.abs(grad))
                jac_loss_val = model.jacobian_lam * jac_loss_val
                total_loss = pred_loss + jac_loss_val

            elif arch_name == "Concat":
                xz_win, t_weights = model.preprocess_and_windowing(x)
                xz_full = xz_win[:, :, :lag]
                x_orig = xz_full[:, :d, :].detach().clone().requires_grad_(True)
                x_cond = xz_full[:, d:, :].detach()
                x_cat = torch.cat([x_orig, x_cond], dim=1).flatten(start_dim=1)
                x_target = xz_win[:, :d, -1]

                pred = model(x_cat)
                pred_loss = F.mse_loss(pred, x_target)

                # Jacobian only on x_orig
                jac_loss_val = torch.tensor(0.0, device=device)
                for j_idx in range(d):
                    grad = torch.autograd.grad(pred[:, j_idx], x_orig,
                                               grad_outputs=torch.ones_like(pred[:, j_idx]),
                                               create_graph=True, retain_graph=True)[0]
                    jac_loss_val = jac_loss_val + torch.mean(torch.abs(grad))
                jac_loss_val = model.jacobian_lam * jac_loss_val
                total_loss = pred_loss + jac_loss_val

            elif arch_name == "FiLM":
                x_win, z_avg, t_weights = model.preprocess_and_windowing(x)
                x_input = x_win[:, :, :lag].detach().clone().requires_grad_(True)
                x_target = x_win[:, :, -1]

                pred = model(x_input.flatten(start_dim=1), z_avg.detach())
                pred_loss = F.mse_loss(pred, x_target)

                # Jacobian on x_input
                jac_loss_val = torch.tensor(0.0, device=device)
                for j_idx in range(d):
                    grad = torch.autograd.grad(pred[:, j_idx], x_input,
                                               grad_outputs=torch.ones_like(pred[:, j_idx]),
                                               create_graph=True, retain_graph=True)[0]
                    jac_loss_val = jac_loss_val + torch.mean(torch.abs(grad))
                jac_loss_val = model.jacobian_lam * jac_loss_val
                total_loss = pred_loss + jac_loss_val

            elif arch_name == "Filter":
                windows, x_orig_t, x_filt_t = model.make_filtered_windows(x)
                x_input = windows[:, :, :lag].detach().clone().requires_grad_(True)
                x_target = windows[:, :, -1]

                pred = model(x_input)
                pred_loss = model.loss_fn(pred, x_target)

                # Ortho loss
                diff_sq = torch.mean((x_filt_t - x_orig_t) ** 2)
                x_orig_norm = torch.sqrt(torch.mean(x_orig_t ** 2) + 1e-8)
                x_filt_norm = torch.sqrt(torch.mean(x_filt_t ** 2) + 1e-8)
                loss_ortho = model.ortho_lam * diff_sq / (x_orig_norm * x_filt_norm + 1e-8)

                # Jacobian loss
                jac_loss_val = torch.tensor(0.0, device=device)
                for j_idx in range(d):
                    grad = torch.autograd.grad(pred[:, j_idx], x_input,
                                               grad_outputs=torch.ones_like(pred[:, j_idx]),
                                               create_graph=True)[0]
                    jac_loss_val = jac_loss_val + torch.mean(torch.abs(grad))
                jac_loss_val = model.jacobian_lam * jac_loss_val
                total_loss = pred_loss + jac_loss_val + loss_ortho

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if it % log_every == 0:
                # Measure gradient norms per parameter group
                param_groups = {"orig_path": 0.0, "aux_path": 0.0, "filter": 0.0}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gnorm = param.grad.norm().item()
                        if any(k in name for k in ["inputgate", "outputgate", "encoders",
                                                     "linear_1", "linear_2", "linear_res"]):
                            param_groups["orig_path"] += gnorm ** 2
                        elif any(k in name for k in ["cond_proj", "film_gamma", "film_beta"]):
                            param_groups["aux_path"] += gnorm ** 2
                        elif any(k in name for k in ["preprocessor", "filter_mamba",
                                                     "filter_", "weight_head"]):
                            if arch_name == "Concat" or arch_name == "FiLM":
                                param_groups["aux_path"] += gnorm ** 2
                            else:
                                param_groups["filter"] += gnorm ** 2

                for k in param_groups:
                    param_groups[k] = np.sqrt(param_groups[k])

                log.append({
                    "iter": it,
                    "pred_loss": float(pred_loss.item()),
                    "jac_loss": float(jac_loss_val.item()),
                    "total_loss": float(total_loss.item()),
                    "grad_orig": float(param_groups["orig_path"]),
                    "grad_aux": float(param_groups["aux_path"]),
                    "grad_filter": float(param_groups["filter"]),
                })

        # Compute final GC matrix
        gc_pred = model.get_gc_matrix(x)
        metrics = compute_metrics(gc, gc_pred)

        results[arch_name] = {
            "final_auroc": metrics["auroc"],
            "final_shd": metrics["shd"],
            "final_auprc": metrics["auprc"],
            "log": log
        }

        # Key metric: gradient ratio aux/orig at end of training
        if len(log) > 0:
            final = log[-1]
            ratio = final["grad_aux"] / max(final["grad_orig"], 1e-10)
            results[arch_name]["grad_ratio_aux_orig"] = float(ratio)
            print(f"    AUROC={metrics['auroc']:.4f}, grad_aux/orig={ratio:.4f}, "
                  f"jac_loss={final['jac_loss']:.6f}")

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n  {'Architecture':<12} {'AUROC':>8} {'grad_aux/orig':>14} {'jac_loss':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*14} {'-'*10}")
    for arch in ["Baseline", "Concat", "FiLM", "Filter"]:
        if arch in results:
            r = results[arch]
            print(f"  {arch:<12} {r['final_auroc']:>8.4f} "
                  f"{r.get('grad_ratio_aux_orig', 0):>14.4f} "
                  f"{r['log'][-1]['jac_loss']:>10.6f}")

    return results


# ============================================================
# Exp 2: Controlled d_cond Sweep
# ============================================================
def exp2_dcond_sweep():
    """AUROC vs d_cond for concat architecture."""
    print("\n" + "=" * 60)
    print("EXP 2: Controlled d_cond Sweep")
    print("=" * 60)

    d, T, lag = 8, 300, 1
    x, gc, A_true = generate_var1_data(d=d, T=T, seed=42)

    d_cond_values = [0, 1, 2, 4, 8, 16]
    results = {}

    for d_cond in d_cond_values:
        torch.manual_seed(42)
        np.random.seed(42)

        if d_cond == 0:
            # Use baseline (no auxiliary path)
            model = BaselineJRNGC(d=d, lag=lag, layers=3, hidden=32,
                                  jacobian_lam=0.01).to(device)
            label = "Baseline (d_cond=0)"
        else:
            model = MambaJRNGC(d=d, lag=lag, layers=3, hidden=32,
                               jacobian_lam=0.01, d_state=4, d_cond=d_cond,
                               use_time_weight_loss=False).to(device)
            label = f"Concat d_cond={d_cond}"

        print(f"  Training {label}...")
        model, loss = train_model(model, x, max_iter=1500, lr=1e-3, verbose=False)
        gc_pred = model.get_gc_matrix(x)
        metrics = compute_metrics(gc, gc_pred)
        results[label] = {
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "shd": metrics["shd"],
            "train_loss": float(loss),
            "d_cond": d_cond,
        }
        print(f"    AUROC={metrics['auroc']:.4f}, SHD={metrics['shd']}, "
              f"loss={loss:.6f}")

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n  {'d_cond':>8} {'AUROC':>8} {'SHD':>6}")
    print(f"  {'-'*8} {'-'*8} {'-'*6}")
    for label, r in results.items():
        print(f"  {r['d_cond']:>8} {r['auroc']:>8.4f} {r['shd']:>6}")

    return results


# ============================================================
# Exp 3: Loss Decomposition Trajectory (Phase Portrait)
# ============================================================
def exp3_loss_trajectory():
    """Track (pred_loss, jac_loss) over training for concat vs filter."""
    print("\n" + "=" * 60)
    print("EXP 3: Loss Decomposition Trajectory")
    print("=" * 60)

    d, T, lag = 8, 300, 1
    x, gc, A_true = generate_var1_data(d=d, T=T, seed=42)

    configs = [
        ("Concat", MambaJRNGC,
         {"d": d, "lag": lag, "layers": 3, "hidden": 32, "jacobian_lam": 0.01,
          "d_state": 4, "d_cond": 8, "use_time_weight_loss": False}),
        ("Filter", MambaFilterJRNGC,
         {"d": d, "lag": lag, "layers": 3, "hidden": 32, "jacobian_lam": 0.01,
          "d_state": 4, "ortho_lam": 0.05, "residual_scale": 0.1,
          "filter_type": "mamba"}),
    ]

    max_iter = 2000
    log_every = 20
    results = {}

    for arch_name, ModelClass, kwargs in configs:
        print(f"\n  --- {arch_name} ---")
        torch.manual_seed(42)
        np.random.seed(42)
        model = ModelClass(**kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        trajectory = []
        for it in range(max_iter):
            model.train()
            optimizer.zero_grad()

            if arch_name == "Concat":
                xz_win, t_weights = model.preprocess_and_windowing(x)
                xz_full = xz_win[:, :, :lag]
                x_orig = xz_full[:, :d, :].detach().clone().requires_grad_(True)
                x_cond = xz_full[:, d:, :].detach()
                x_cat = torch.cat([x_orig, x_cond], dim=1).flatten(start_dim=1)
                x_target = xz_win[:, :d, -1]
                pred = model(x_cat)
                pred_loss = F.mse_loss(pred, x_target)

                jac_loss_val = torch.tensor(0.0, device=device)
                for j_idx in range(d):
                    grad = torch.autograd.grad(pred[:, j_idx], x_orig,
                                               grad_outputs=torch.ones_like(pred[:, j_idx]),
                                               create_graph=True, retain_graph=True)[0]
                    jac_loss_val = jac_loss_val + torch.mean(torch.abs(grad))
                jac_loss_val = model.jacobian_lam * jac_loss_val
                total_loss = pred_loss + jac_loss_val

            else:  # Filter
                windows, x_orig_t, x_filt_t = model.make_filtered_windows(x)
                x_input = windows[:, :, :lag].detach().clone().requires_grad_(True)
                x_target = windows[:, :, -1]
                pred = model(x_input)
                pred_loss = model.loss_fn(pred, x_target)

                diff_sq = torch.mean((x_filt_t - x_orig_t) ** 2)
                x_orig_norm = torch.sqrt(torch.mean(x_orig_t ** 2) + 1e-8)
                x_filt_norm = torch.sqrt(torch.mean(x_filt_t ** 2) + 1e-8)
                loss_ortho = model.ortho_lam * diff_sq / (x_orig_norm * x_filt_norm + 1e-8)

                jac_loss_val = torch.tensor(0.0, device=device)
                for j_idx in range(d):
                    grad = torch.autograd.grad(pred[:, j_idx], x_input,
                                               grad_outputs=torch.ones_like(pred[:, j_idx]),
                                               create_graph=True)[0]
                    jac_loss_val = jac_loss_val + torch.mean(torch.abs(grad))
                jac_loss_val = model.jacobian_lam * jac_loss_val
                total_loss = pred_loss + jac_loss_val + loss_ortho

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if it % log_every == 0:
                trajectory.append({
                    "iter": it,
                    "pred_loss": float(pred_loss.item()),
                    "jac_loss": float(jac_loss_val.item()),
                })

        gc_pred = model.get_gc_matrix(x)
        metrics = compute_metrics(gc, gc_pred)
        results[arch_name] = {
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "shd": metrics["shd"],
            "trajectory": trajectory,
        }

        # Key metrics: how much jac_loss decreased independently of pred_loss
        if len(trajectory) >= 2:
            pred_drop = trajectory[0]["pred_loss"] - trajectory[-1]["pred_loss"]
            jac_drop = trajectory[0]["jac_loss"] - trajectory[-1]["jac_loss"]
            # "Efficiency": jac_loss reduction per unit pred_loss reduction
            efficiency = jac_drop / max(pred_drop, 1e-10)
            results[arch_name]["jac_reduction_efficiency"] = float(efficiency)
            print(f"    pred_loss: {trajectory[0]['pred_loss']:.5f} → "
                  f"{trajectory[-1]['pred_loss']:.5f}")
            print(f"    jac_loss:  {trajectory[0]['jac_loss']:.5f} → "
                  f"{trajectory[-1]['jac_loss']:.5f}")
            print(f"    jac/pred efficiency: {efficiency:.4f} "
                  f"({'SHORTCUT' if efficiency > 5 else 'coupled'})")

        del model
        torch.cuda.empty_cache()

    return results


# ============================================================
# Exp 4: Coefficient Recovery
# ============================================================
def exp4_coefficient_recovery():
    """Compare recovered A matrix with ground truth for concat vs filter."""
    print("\n" + "=" * 60)
    print("EXP 4: Coefficient Recovery")
    print("=" * 60)

    d, T, lag = 8, 500, 1  # More data for better recovery
    x, gc, A_true = generate_var1_data(d=d, T=T, seed=42)

    configs = [
        ("Baseline", BaselineJRNGC,
         {"d": d, "lag": lag, "layers": 3, "hidden": 32, "jacobian_lam": 0.01}),
        ("Concat", MambaJRNGC,
         {"d": d, "lag": lag, "layers": 3, "hidden": 32, "jacobian_lam": 0.01,
          "d_state": 4, "d_cond": 4, "use_time_weight_loss": False}),
        ("Filter", MambaFilterJRNGC,
         {"d": d, "lag": lag, "layers": 3, "hidden": 32, "jacobian_lam": 0.01,
          "d_state": 4, "ortho_lam": 0.05, "residual_scale": 0.1,
          "filter_type": "mamba"}),
    ]

    results = {}
    for arch_name, ModelClass, kwargs in configs:
        print(f"\n  --- {arch_name} ---")
        torch.manual_seed(42)
        np.random.seed(42)
        model = ModelClass(**kwargs).to(device)
        model, loss = train_model(model, x, max_iter=2000, lr=1e-3, verbose=False)

        gc_pred = model.get_gc_matrix(x)
        if gc_pred.ndim == 3:
            gc_pred = gc_pred[:, :, 0]  # lag=1

        # Compare with ground truth A
        A_recovered = gc_pred  # (d, d)
        # Remove diagonal (self-connections are not causal edges)
        np.fill_diagonal(A_recovered, 0)
        A_true_nodiag = A_true.copy()
        np.fill_diagonal(A_true_nodiag, 0)

        # Correlation between recovered and true coefficients
        recovered_flat = A_recovered.flatten()
        true_flat = np.abs(A_true_nodiag).flatten()
        # Only consider off-diagonal entries
        mask = ~np.eye(d, dtype=bool).flatten()
        corr = np.corrcoef(recovered_flat[mask], true_flat[mask])[0, 1]

        # Shrinkage: ratio of ||A_recovered|| / ||A_true||
        shrinkage = np.linalg.norm(recovered_flat[mask]) / max(
            np.linalg.norm(true_flat[mask]), 1e-10)

        metrics = compute_metrics(gc, A_true_nodiag)
        results[arch_name] = {
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "shd": metrics["shd"],
            "coefficient_correlation": float(corr),
            "coefficient_shrinkage": float(shrinkage),
            "recovered_norm": float(np.linalg.norm(recovered_flat[mask])),
            "true_norm": float(np.linalg.norm(true_flat[mask])),
        }

        print(f"    AUROC={metrics['auroc']:.4f}, Corr={corr:.4f}, "
              f"Shrinkage={shrinkage:.4f} "
              f"({'SHORTCUT' if shrinkage < 0.3 else 'PRESERVED'})")

        del model
        torch.cuda.empty_cache()

    return results


# ============================================================
# Toy Theorem: Formal Statement
# ============================================================
TOY_THEOREM = """
Toy Theorem: Shortcut Existence in Auxiliary-Channel Architectures

Consider a simplified 2-layer linear model:
  y = W_1·[x; z]  (first layer, concatenated input)
  y = W_2·h        (second layer, output)

where x ∈ R^d is the original input, z ∈ R^{d_aux} is the auxiliary feature,
and the Jacobian penalty λ·||∂y/∂x||_1 is applied only to x.

Let W_1 = [W_x | W_z] be the partitioned first-layer weights.

Claim 1 (Shortcut Existence). For any target mapping y*(x), there exist
weights (W_x → 0, W_z → W_z*) such that:
  - Prediction error ||y - y*|| → 0  (through W_z capturing all information)
  - Jacobian penalty ||∂y/∂x||_1 → 0  (since W_x → 0)

Proof: The auxiliary features z = f_aux(x) are computed from x via a
learned function f_aux. If f_aux is sufficiently expressive (e.g., an SSM),
it can encode the predictive information from x into z. The model can then
set W_x ≈ 0 and W_z ≈ W_z*, achieving both low prediction error and near-zero
Jacobian penalty. The gradient descent dynamics favor this solution because
the Jacobian penalty creates a "downward pressure" on W_x that is not
counterbalanced by prediction error (since W_z can compensate).

Claim 2 (Filter Immunity). For the input-filtering architecture:
  x' = x + ε·h(x)
  y = W_1·x'
  y = W_2·h

any reduction in ||W_1|| (and thus ||∂y/∂x'||_1) directly increases
prediction error because x' is the ONLY input. Formally:

  ∂L_pred/∂||W_1|| < 0  whenever ||W_1|| is below the optimal value for
  predicting y from x'.

The gradient dynamics CANNOT decouple prediction error reduction from
Jacobian penalty reduction — the two objectives are structurally coupled.

Corollary. In the concat/FiLM architectures, the Jacobian penalty can be
driven to zero while maintaining low prediction error. In the input-filtering
architecture, the Jacobian penalty at convergence reflects true predictive
reliance on each input variable, enabling valid Granger causal inference.
"""


# ============================================================
# Main
# ============================================================
def main():
    print("SHORTCUT LEARNING DIAGNOSTIC EXPERIMENTS")
    print("=" * 60)
    print(f"Device: {device}")
    print(TOY_THEOREM)

    all_results = {"toy_theorem": TOY_THEOREM}

    # Exp 1: Gradient path analysis
    try:
        r1 = exp1_gradient_path_analysis()
        all_results["exp1_gradient_paths"] = r1
        with open(OUT + "exp1_gradient_paths.json", "w") as f:
            json.dump(r1, f, indent=2, default=str)
    except Exception as e:
        print(f"Exp 1 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Exp 2: d_cond sweep
    try:
        r2 = exp2_dcond_sweep()
        all_results["exp2_dcond_sweep"] = r2
        with open(OUT + "exp2_dcond_sweep.json", "w") as f:
            json.dump(r2, f, indent=2, default=str)
    except Exception as e:
        print(f"Exp 2 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Exp 3: Loss trajectory
    try:
        r3 = exp3_loss_trajectory()
        all_results["exp3_loss_trajectory"] = r3
        with open(OUT + "exp3_loss_trajectory.json", "w") as f:
            json.dump(r3, f, indent=2, default=str)
    except Exception as e:
        print(f"Exp 3 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Exp 4: Coefficient recovery
    try:
        r4 = exp4_coefficient_recovery()
        all_results["exp4_coefficient_recovery"] = r4
        with open(OUT + "exp4_coefficient_recovery.json", "w") as f:
            json.dump(r4, f, indent=2, default=str)
    except Exception as e:
        print(f"Exp 4 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Final summary
    with open(OUT + "all_diagnostic_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")

    # Print consolidated findings
    if "exp1_gradient_paths" in all_results:
        print("\nExp 1 — Gradient Path Ratio (aux/orig):")
        for arch, r in all_results["exp1_gradient_paths"].items():
            if isinstance(r, dict) and "grad_ratio_aux_orig" in r:
                print(f"  {arch}: {r['grad_ratio_aux_orig']:.4f}")

    if "exp2_dcond_sweep" in all_results:
        print("\nExp 2 — AUROC vs d_cond:")
        for label, r in sorted(all_results["exp2_dcond_sweep"].items(),
                                key=lambda x: x[1].get("d_cond", 0)):
            print(f"  {label}: AUROC={r['auroc']:.4f}")

    if "exp3_loss_trajectory" in all_results:
        print("\nExp 3 — Jacobian Reduction Efficiency (higher = more shortcut):")
        for arch, r in all_results["exp3_loss_trajectory"].items():
            if isinstance(r, dict) and "jac_reduction_efficiency" in r:
                print(f"  {arch}: {r['jac_reduction_efficiency']:.2f}")

    if "exp4_coefficient_recovery" in all_results:
        print("\nExp 4 — Coefficient Shrinkage (lower = more shortcut):")
        for arch, r in all_results["exp4_coefficient_recovery"].items():
            if isinstance(r, dict) and "coefficient_shrinkage" in r:
                print(f"  {arch}: shrinkage={r['coefficient_shrinkage']:.4f}, "
                      f"corr={r['coefficient_correlation']:.4f}")

    print(f"\nAll results saved to {OUT}")


if __name__ == "__main__":
    main()
