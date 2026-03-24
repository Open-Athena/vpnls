"""IsoFLOP sampling and synthetic data generation."""

from __future__ import annotations

import numpy as np

from vpnls.types import LossSurface


def compute_center_offset(
    C: float,
    compute_budgets: np.ndarray,
    drift_rate: float,
    center_scale: float,
) -> float:
    """Compute sampling center offset combining drift and scale."""
    offset = 0.0
    if center_scale != 1.0:
        offset -= np.log10(center_scale)
    if drift_rate != 0.0:
        log_C = np.log10(C)
        log_C_min = np.log10(compute_budgets.min())
        log_C_max = np.log10(compute_budgets.max())
        fraction = (log_C - log_C_min) / (log_C_max - log_C_min) if log_C_max > log_C_min else 0.0
        offset -= drift_rate * fraction
    return offset


def isoflop_sample(
    C: float,
    n_points: int,
    log_range: float,
    center_offset: float,
    surface: LossSurface,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (N, D, L) along an IsoFLOP contour with C = 6ND."""
    log_N_center = np.log10(surface.N_opt(C)) + center_offset
    N = np.logspace(log_N_center - log_range, log_N_center + log_range, n_points)
    D = C / (6 * N)
    L = np.array([surface.loss(n, d) for n, d in zip(N, D)])
    return N, D, L


def generate_isoflop_data(
    surface: LossSurface,
    *,
    compute_budgets: np.ndarray = np.array([1e17, 1e18, 1e19, 1e20, 1e21]),
    n_points_per_budget: int = 15,
    log_range: float = 1.0,
    noise_std: float = 0.002,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic IsoFLOP data with optional Gaussian noise on losses."""
    all_N, all_D, all_L = [], [], []
    for C in compute_budgets:
        N, D, L = isoflop_sample(
            C, n_points_per_budget, log_range, center_offset=0.0, surface=surface
        )
        all_N.append(N)
        all_D.append(D)
        all_L.append(L)
    N = np.concatenate(all_N)
    D = np.concatenate(all_D)
    L = np.concatenate(all_L)
    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        L = L + rng.normal(0.0, noise_std, size=L.shape)
    return N, D, L
