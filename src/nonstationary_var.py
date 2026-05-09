"""Non-stationary VAR data generator for pilot experiment.

Generates VAR(p) with controlled trend-type non-stationarity:
- Shared random walk trend injected into all variables
- Drifting VAR coefficient matrix across time
- Produces (x, gc_true) pairs compatible with JRNGC data format.
"""
import numpy as np
import sys, os

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_data_dir


def generate_nonstationary_var(
    d=10, lag=7, T=500, n_seeds=5,
    trend_strength=0.3, coefficient_drift=0.1,
    sparsity=0.2, noise_scale=1.0,
    data_dir=None
):
    if data_dir is None:
        data_dir = os.path.join(resolve_data_dir(), "nonstationary_var")
    """Generate non-stationary VAR data with trend injection.

    The non-stationarity comes from:
    1. Shared random walk trend μ(t) added to all variables
    2. VAR coefficients A_k drift slowly over time (continuous non-stationarity)
       A_k(t) = A_k(0) + drift * B_k(t) where B_k(t) is a smoothed random walk

    Model: x_t = Σ_{k=1}^{lag} A_k(t) * x_{t-k} + μ(t) + ε_t

    Args:
        d: number of variables
        lag: maximum lag order
        T: time series length
        n_seeds: number of random seeds
        trend_strength: magnitude of shared trend
        coefficient_drift: magnitude of coefficient drift over time
        sparsity: fraction of non-zero edges in gc
        noise_scale: standard deviation of noise
        data_dir: output directory

    Returns:
        Dictionary mapping seed -> (x, gc)
    """
    os.makedirs(data_dir, exist_ok=True)

    results = {}
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed * 100 + 42)
        x, gc = _generate_one(d, lag, T, rng, trend_strength, coefficient_drift, sparsity, noise_scale)

        # Save to disk for JRNGC compatibility
        subdir = os.path.join(data_dir,
                            f"num_nodes_{d}",
                            f"true_lag_{lag}",
                            f"noise_scale_{int(noise_scale)}",
                            f"seed_{seed}")
        os.makedirs(subdir, exist_ok=True)
        np.save(os.path.join(subdir, "_x.npy"), x)
        np.save(os.path.join(subdir, "_gc.npy"), gc)

        results[seed] = (x, gc)
        print(f"  seed={seed}: x.shape={x.shape}, gc.shape={gc.shape}, "
              f"gc_edges={int(gc.sum())}, T={T}, trend_std={np.std(x, axis=1).mean():.3f}")

    print(f"\nSaved to: {data_dir}")
    return results


def _generate_one(d, lag, T, rng, trend_strength, coefficient_drift, sparsity, noise_scale):
    """Generate a single realization."""
    # 1. Generate sparse ground-truth GC graph
    gc = np.zeros((d, d, lag))
    for i in range(d):
        for j in range(d):
            if i != j and rng.rand() < sparsity:
                # Random lag for this edge
                k = rng.randint(0, lag)
                gc[i, j, k] = 1.0

    # 2. Generate base coefficient matrices A_k(0)
    A_base = []
    for k in range(lag):
        A_k = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if gc[i, j, k] > 0:
                    A_k[i, j] = rng.uniform(0.2, 0.6) * rng.choice([-1, 1])
        A_base.append(A_k)

    # 3. Generate coefficient drift series B_k(t) - smoothed random walks
    drift_scale = coefficient_drift * 0.1
    B_drift = []
    for k in range(lag):
        B_k_raw = np.cumsum(rng.randn(T, d, d) * drift_scale / np.sqrt(T), axis=0)
        # Smooth with moving average
        window = 20
        B_k = np.zeros_like(B_k_raw)
        for t in range(T):
            start = max(0, t - window)
            B_k[t] = B_k_raw[start:t+1].mean(axis=0)
        B_drift.append(B_k)

    # 4. Generate shared trend (random walk)
    trend = np.cumsum(rng.randn(T) * trend_strength / np.sqrt(T))
    trend = trend - trend.mean()  # Zero-center

    # 5. Generate time series
    x = np.zeros((d, T))
    noise = rng.randn(d, T) * noise_scale

    # Initialize with noise
    for t in range(lag):
        x[:, t] = noise[:, t]

    # Simulate
    for t in range(lag, T):
        x_t = np.zeros(d)
        for k in range(lag):
            A_k_t = A_base[k] + B_drift[k][t]
            x_t += A_k_t @ x[:, t - k - 1]
        x_t += trend[t]  # Add shared trend
        x_t += noise[:, t]
        x[:, t] = x_t

    return x.astype(np.float32), gc.astype(np.float32)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--lag", type=int, default=7)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trend", type=float, default=0.3)
    parser.add_argument("--drift", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Generating non-stationary VAR: d={args.d}, lag={args.lag}, T={args.T}, seeds={args.seeds}")
    print(f"  trend_strength={args.trend}, coefficient_drift={args.drift}")
    generate_nonstationary_var(
        d=args.d, lag=args.lag, T=args.T, n_seeds=args.seeds,
        trend_strength=args.trend, coefficient_drift=args.drift
    )
