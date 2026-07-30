"""Frozen Lorenz-96 generator for the Phase 9 formal confirmation.

The numerical recipe matches the historical JRNGC fixture generator while
using an isolated RandomState so data generation cannot alter model-training
randomness.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.integrate import odeint


def lorenz96_derivative(
    state: np.ndarray,
    _time: float,
    forcing: float,
) -> np.ndarray:
    """Return the Lorenz-96 ODE derivative."""
    d = len(state)
    derivative = np.zeros(d, dtype=np.float64)
    for target in range(d):
        derivative[target] = (
            (
                state[(target + 1) % d]
                - state[(target - 2) % d]
            )
            * state[(target - 1) % d]
            - state[target]
            + forcing
        )
    return derivative


def lorenz96_direct_graph(d: int) -> np.ndarray:
    """Return target-by-source direct nominal-lag support."""
    graph = np.zeros((d, d, 1), dtype=np.int64)
    for target in range(d):
        graph[target, target, 0] = 1
        graph[target, (target + 1) % d, 0] = 1
        graph[target, (target - 1) % d, 0] = 1
        graph[target, (target - 2) % d, 0] = 1
    return graph


def generate_lorenz96(
    *,
    d: int,
    t: int,
    t_eval: int,
    forcing: float,
    seed: int,
    delta_t: float = 0.1,
    observation_noise_sd: float = 0.1,
    burn_in: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate normalized train/evaluation series and direct graph.

    The integration grid, burn-in, float32 cast, and population-standard-
    deviation normalization intentionally match the frozen historical code.
    """
    if d < 4:
        raise ValueError("Lorenz-96 requires d >= 4 for this graph contract")
    if t <= 0 or t_eval < 0 or burn_in < 0:
        raise ValueError("Invalid sequence lengths")
    if delta_t <= 0 or observation_noise_sd < 0:
        raise ValueError("Invalid numerical parameters")

    rng = np.random.RandomState(int(seed))
    total = int(t + t_eval + burn_in)
    initial_state = rng.normal(scale=0.01, size=d)
    time_grid = np.linspace(0, total * delta_t, total)
    series = odeint(
        lorenz96_derivative,
        initial_state,
        time_grid,
        args=(float(forcing),),
    )
    series += rng.normal(
        scale=float(observation_noise_sd),
        size=(total, d),
    )

    series = np.swapaxes(series[burn_in:].astype(np.float32), 0, 1)
    variable_mean = np.mean(series, axis=1, keepdims=True)
    variable_std = np.std(series, axis=1, keepdims=True)
    if np.any(variable_std <= 0) or not np.all(np.isfinite(variable_std)):
        raise RuntimeError("Degenerate Lorenz-96 normalization statistics")
    normalized = (series - variable_mean) / variable_std
    graph = lorenz96_direct_graph(d)
    return normalized[:, :t], normalized[:, t:], graph

