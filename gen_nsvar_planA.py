"""Generate non-stationary VAR via Plan A: stable VAR + shared random walk trend.

Plan A (per advisor): stable stationary VAR(P) + cumulative random walk trend.
This avoids coefficient drift instability while still introducing non-stationarity.

Key: VAR coefficients are FIXED and stable (eigenvalues < 0.95).
Non-stationarity comes from an EXOGENOUS shared trend, not from changing dynamics.
The GC ground truth is the sparsity pattern of the fixed VAR coefficients.

Usage: python gen_nsvar_planA.py  (run on cloud in mamba_enhanced/)
"""
import numpy as np
import os


def generate_stable_var_coefficients(d, lag, sparsity, rng):
    """Generate stable VAR coefficient matrices.

    Strategy: generate random sparse A_k, then scale down to ensure
    max eigenvalue of companion matrix < 0.95.
    """
    # Generate sparse coefficient matrices
    A = np.zeros((lag, d, d))
    gc = np.zeros((d, d, lag))

    for k in range(lag):
        for i in range(d):
            for j in range(d):
                if i != j and rng.rand() < sparsity:
                    # Random coefficient, magnitude decays with lag
                    val = rng.randn() * 0.3 / (k + 1)
                    A[k, i, j] = val
                    gc[i, j, k] = 1.0

    # Build companion matrix to check stability
    companion = np.zeros((d * lag, d * lag))
    companion[:d, :] = np.hstack([A[k] for k in range(lag)])
    for k in range(1, lag):
        companion[k*d:(k+1)*d, (k-1)*d:k*d] = np.eye(d)

    max_ev = np.max(np.abs(np.linalg.eigvals(companion)))
    if max_ev > 0.95:
        scale = 0.95 / max_ev * 0.9  # Scale to well within unit circle
        A *= scale

    return A, gc


def generate_nonstationary_var_planA(
    d=50, lag=14, T=500, n_seeds=5,
    trend_scale=0.05, sparsity=0.05,
    noise_scale=1.0,
    data_dir="/root/autodl-tmp/GUOJI/mamba_enhanced/data/nonstationary_var_planA"
):
    """Generate non-stationary VAR using Plan A.

    x_t = Σ A_k x_{t-k} + μ_t + ε_t

    where:
    - A_k: FIXED stable VAR coefficients (ensures stable dynamics)
    - μ_t: shared random walk trend (the non-stationarity source)
    - ε_t: i.i.d. Gaussian noise

    The trend is a d-dimensional cumulative sum of small Gaussian steps,
    shared across variables with different weights.

    Args:
        d: number of variables
        lag: VAR lag order
        T: time series length
        n_seeds: number of seeds
        trend_scale: magnitude of shared trend steps
        sparsity: fraction of non-zero edges in GC graph
        noise_scale: std of observation noise
        data_dir: output directory
    """
    os.makedirs(data_dir, exist_ok=True)

    for seed in range(n_seeds):
        print(f"  Generating seed {seed}...")
        rng = np.random.RandomState(seed * 100 + 42)

        # Step 1: Generate stable VAR coefficients
        A_k, gc = generate_stable_var_coefficients(d, lag, sparsity, rng)

        # Parameters
        burn = 200  # Burn-in period
        total_len = T + lag + burn

        # Step 2: Generate shared random walk trend μ(t)
        # Each variable gets a random loading on the shared trend
        trend_loadings = rng.randn(d) * 0.5 + 1.0  # Mean ~1, some variation
        shared_rw = np.cumsum(rng.randn(total_len) * trend_scale)
        mu = np.outer(trend_loadings, shared_rw)  # (d, total_len)

        # Step 3: Generate stationary VAR process
        x = np.zeros((d, total_len))
        noise = rng.randn(d, total_len) * noise_scale

        for t in range(lag, total_len):
            ar_term = np.zeros(d)
            for k in range(lag):
                ar_term += A_k[k] @ x[:, t - 1 - k]
            x[:, t] = ar_term + mu[:, t] + noise[:, t]

        # Discard burn-in
        x = x[:, burn+lag:]  # (d, T)
        assert x.shape == (d, T), f"Expected ({d},{T}), got {x.shape}"

        # Check value range
        print(f"    x range: [{x.min():.2f}, {x.max():.2f}], "
              f"std={x.std():.2f}, gc_edges={int(gc.sum())}")

        # Save
        subdir = os.path.join(data_dir,
                            f"num_nodes_{d}",
                            f"true_lag_{lag}",
                            f"noise_scale_{int(noise_scale)}",
                            f"seed_{seed}")
        os.makedirs(subdir, exist_ok=True)
        np.save(os.path.join(subdir, "_x.npy"), x.astype(np.float32))
        np.save(os.path.join(subdir, "_gc.npy"), gc.astype(np.float32))

    print(f"\nSaved to: {data_dir}")


if __name__ == "__main__":
    # d=50, lag=14, T=500, 3 seeds (for Priority 1)
    generate_nonstationary_var_planA(
        d=50, lag=14, T=500, n_seeds=3,
        trend_scale=0.05, sparsity=0.05,
    )
    print("\nDone. Ready for Priority 1 testing.")
