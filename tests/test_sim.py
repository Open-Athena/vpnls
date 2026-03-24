"""Tests for simulation utilities."""

import numpy as np

from vpnls.sim import simulate_isoflop_data


def test_simulate_isoflop_data_shape():
    N, D, L = simulate_isoflop_data(
        alpha=0.34,
        beta=0.28,
        A=406.4,
        B=410.7,
        E=1.69,
        compute_budgets=np.array([1e18, 1e19, 1e20]),
        n_points_per_budget=10,
        noise_std=0,
    )
    assert len(N) == len(D) == len(L) == 30
    assert np.all(N > 0) and np.all(D > 0)


def test_simulate_isoflop_data_defaults():
    N, D, L = simulate_isoflop_data(alpha=0.34, beta=0.28, A=406.4, B=410.7, E=1.69)
    # 5 default budgets × 15 points
    assert len(N) == 75
