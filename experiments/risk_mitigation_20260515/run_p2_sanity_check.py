"""P2 Sanity Check: verify no-near-id ablation actually changed filter init.

Checks:
1. Iteration 0 filter drift: Full ISTF vs No near-id init
2. Filter parameter difference at init
3. Early drift trajectory (first 100 iters)
4. Whether orthogonality loss pulls random init back to near-identity

Runs single seed on VAR(1) d=8 T=500.
"""
import torch
import torch.nn as nn
import numpy as np
import sys, os, json, time

torch.backends.cudnn.enabled = False

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, os.path.join(_PROJ_ROOT, "src"))

from src.minimal_mamba import MambaBlock
from src.mamba_jrngc_pilot import (
    BaselineJRNGC, MambaFilterJRNGC, MambaJRNGC,
    train_model
)
from experiments.risk_mitigation_20260515.run_principle_ablation import (
    MambaFilterNoNearIdentity, MambaFilterNoOrtho,
    get_filter_drift, generate_var1_data
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(_PROJ_ROOT, "risk_mitigation_results")
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def get_filter_params(model):
    """Extract filter parameters for comparison."""
    params = {}
    if hasattr(model, 'filter_mamba'):
        fm = model.filter_mamba
        for name, param in fm.named_parameters():
            params[f"filter.{name}"] = param.detach().cpu().clone()
    return params


def param_distance(p1, p2):
    """Compute total L2 distance between two parameter dicts."""
    total = 0.0
    for k in p1:
        if k in p2:
            total += torch.norm(p1[k] - p2[k], p=2).item() ** 2
    return np.sqrt(total)


def record_drift_trajectory(model, x, max_iters=100, record_every=10):
    """Record filter drift during early training iterations."""
    model.train()
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trajectory = []
    for it in range(max_iters + 1):
        if it % record_every == 0:
            model.eval()
            with torch.no_grad():
                drift = get_filter_drift(model, x)
            trajectory.append({"iteration": it, "drift": drift})
            model.train()

        optimizer.zero_grad()
        loss = model.compute_loss(x_tensor)
        if hasattr(model, 'compute_orthogonality_loss'):
            ortho_loss = model.compute_orthogonality_loss()
            loss = loss + getattr(model, 'ortho_lam', 0.05) * ortho_loss
        loss.backward()
        optimizer.step()

    return trajectory


def main():
    log("=" * 70)
    log("P2 SANITY CHECK: No-Near-Identity Ablation Verification")
    log("=" * 70)

    d, T, seed = 8, 500, 42
    x, gc, A_true = generate_var1_data(d=d, T=T, seed=seed)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    n_edges = int(gc.sum())
    log(f"  d={d}, T={T}, edges={n_edges}, seed={seed}")

    # ================================================================
    # 1. Iteration 0 drift comparison
    # ================================================================
    log("\n--- Check 1: Iteration 0 Filter Drift ---")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model_full = MambaFilterJRNGC(
        d=d, lag=1, layers=3, hidden=32,
        jacobian_lam=0.01, d_state=4,
        ortho_lam=0.05, residual_scale=0.1,
        filter_type="mamba"
    ).to(device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model_no_ni = MambaFilterNoNearIdentity(
        d=d, lag=1, layers=3, hidden=32,
        jacobian_lam=0.01, d_state=4,
        ortho_lam=0.05, residual_scale=0.1,
        filter_type="mamba"
    ).to(device)

    model_full.eval()
    model_no_ni.eval()
    with torch.no_grad():
        drift_full_0 = get_filter_drift(model_full, x)
        drift_no_ni_0 = get_filter_drift(model_no_ni, x)
    log(f"  Full ISTF iter=0 drift:     {drift_full_0:.6f}")
    log(f"  No near-id iter=0 drift:    {drift_no_ni_0:.6f}")

    # ================================================================
    # 2. Parameter difference at init
    # ================================================================
    log("\n--- Check 2: Filter Parameter Difference at Init ---")
    params_full = get_filter_params(model_full)
    params_no_ni = get_filter_params(model_no_ni)
    dist = param_distance(params_full, params_no_ni)
    log(f"  L2 distance between filter params: {dist:.6f}")

    for k in sorted(params_full.keys()):
        d_param = torch.norm(params_full[k] - params_no_ni[k], p=2).item()
        norm_full = torch.norm(params_full[k], p=2).item()
        norm_no_ni = torch.norm(params_no_ni[k], p=2).item()
        log(f"  {k}: |full|={norm_full:.6f}, |no-ni|={norm_no_ni:.6f}, Δ={d_param:.6f}")

    if dist < 1e-6:
        log("  WARNING: Filter parameters are IDENTICAL — ablation did NOT work!")
        log("  The reset_parameters() call may not have changed the filter.")
    else:
        log("  OK: Filter parameters differ at init")

    # ================================================================
    # 3. Early drift trajectory
    # ================================================================
    log("\n--- Check 3: Early Drift Trajectory (first 100 iters) ---")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model_full2 = MambaFilterJRNGC(
        d=d, lag=1, layers=3, hidden=32,
        jacobian_lam=0.01, d_state=4,
        ortho_lam=0.05, residual_scale=0.1,
        filter_type="mamba"
    ).to(device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model_no_ni2 = MambaFilterNoNearIdentity(
        d=d, lag=1, layers=3, hidden=32,
        jacobian_lam=0.01, d_state=4,
        ortho_lam=0.05, residual_scale=0.1,
        filter_type="mamba"
    ).to(device)

    log("  Recording Full ISTF trajectory...")
    traj_full = record_drift_trajectory(model_full2, x, max_iters=100, record_every=10)

    log("  Recording No near-id trajectory...")
    traj_no_ni = record_drift_trajectory(model_no_ni2, x, max_iters=100, record_every=10)

    log("\n  Iteration  Full ISTF drift  No near-id drift")
    log("  " + "-" * 45)
    for tf, tn in zip(traj_full, traj_no_ni):
        log(f"  {tf['iteration']:>9}  {tf['drift']:>15.6f}  {tn['drift']:>17.6f}")

    # ================================================================
    # 4. Check if ortho pulls random init back to near-identity
    # ================================================================
    log("\n--- Check 4: Does ortho pull random init back? ---")
    drift_full_final = traj_full[-1]["drift"]
    drift_no_ni_final = traj_no_ni[-1]["drift"]

    log(f"  Full ISTF drift at iter 100:     {drift_full_final:.6f}")
    log(f"  No near-id drift at iter 100:    {drift_no_ni_final:.6f}")
    log(f"  Initial drift difference:        {abs(drift_no_ni_0 - drift_full_0):.6f}")
    log(f"  Final drift difference:          {abs(drift_no_ni_final - drift_full_final):.6f}")

    if abs(drift_no_ni_0 - drift_full_0) > 0.01 and abs(drift_no_ni_final - drift_full_final) < 0.01:
        log("  → Ortho regularization pulled random init toward near-identity within 100 iters")
        log("    This means near-identity init is redundant because ortho enforces it anyway")
    elif abs(drift_no_ni_0 - drift_full_0) < 0.001:
        log("  → Initial drift already identical — near-id init not actually disabled")
        log("    CRITICAL: Ablation may be broken, need to fix MambaFilterNoNearIdentity")
    else:
        log("  → Drifts differ at both init and iter 100 — need further investigation")

    # Save results
    output = {
        "experiment": "p2_no_near_id_sanity_check",
        "seed": seed,
        "check_1_iter0_drift": {
            "full_istf": drift_full_0,
            "no_near_id": drift_no_ni_0,
        },
        "check_2_param_distance": {
            "l2_distance": dist,
            "per_param": {k: float(torch.norm(params_full[k] - params_no_ni[k], p=2).item())
                          for k in params_full},
        },
        "check_3_early_trajectory": {
            "full_istf": traj_full,
            "no_near_id": traj_no_ni,
        },
        "check_4_ortho_pullback": {
            "drift_full_initial": drift_full_0,
            "drift_no_ni_initial": drift_no_ni_0,
            "drift_full_iter100": drift_full_final,
            "drift_no_ni_iter100": drift_no_ni_final,
        }
    }

    json_path = os.path.join(OUT_DIR, "p2_sanity_check.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nSaved to {json_path}")

    # Cleanup
    del model_full, model_no_ni, model_full2, model_no_ni2
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
