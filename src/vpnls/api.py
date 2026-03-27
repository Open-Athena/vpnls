"""Unified entry point for VPNLS fitting."""

from __future__ import annotations

import numpy as np

from vpnls.sim import simulate_isoflop_data
from vpnls.types import (
    LBFGSBOptions,
    LossFunction,
    LossType,
    SurfaceBounds,
    VPNLSResult,
)

__all__ = [
    "bounds",
    "fit_vpnls",
    "huber",
    "simulate_isoflop_data",
]


def bounds(
    *,
    E: tuple[float, float] = (1e-6, 10.0),
    A: tuple[float, float] = (1e-6, 1e6),
    B: tuple[float, float] = (1e-6, 1e6),
    alpha: tuple[float, float] = (0.01, 0.99),
    beta: tuple[float, float] = (0.01, 0.99),
) -> SurfaceBounds:
    """Create optimization bounds for surface parameters."""
    return SurfaceBounds(E=E, A=A, B=B, alpha=alpha, beta=beta)


def huber(delta: float = 1.0) -> LossFunction:
    """Create a Huber loss function with the given delta."""
    return LossFunction(type=LossType.HUBER, huber_delta=delta)


def fit_vpnls(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    method: str = "grid",
    # Common params
    bounds: SurfaceBounds | None = None,
    loss: LossFunction | None = None,
    max_irls_iter: int = 10,
    # Grid params (method="grid")
    resolution: float = 0.01,
    num_workers: int = 1,
    # Scipy/JAX params (method="scipy" or "jax")
    options: LBFGSBOptions | None = None,
    # JAX-only params (method="jax")
    enable_x64: bool = True,
) -> VPNLSResult:
    """Fit L(N,D) = E + A*N^{-alpha} + B*D^{-beta} via variable projection.

    Dispatches to the solver specified by `method`.

    Args:
        N: Parameter counts.
        D: Token counts.
        L: Loss values (same length as N, D).
        method: Solver to use — "grid", "scipy", or "jax".
        bounds: Optimization bounds for (E, A, B, alpha, beta).
        loss: Loss function (MSE or Huber).
        max_irls_iter: Max IRLS iterations for Huber loss.
        resolution: Grid step size in exponent space. For "grid", this is the
            final resolution. For "scipy"/"jax", this is the coarse grid used
            for L-BFGS-B initialization.
        num_workers: Number of parallel processes for grid search (grid only).
        options: L-BFGS-B optimizer options (scipy/jax).
        enable_x64: Enable JAX float64 precision (jax only).

    Returns:
        GridResult, ScipyResult, or JaxResult depending on method.
    """
    common = dict(bounds=bounds, loss=loss, max_irls_iter=max_irls_iter)

    if method == "grid":
        from vpnls.grid import fit_vpnls_grid

        return fit_vpnls_grid(N, D, L, resolution=resolution, num_workers=num_workers, **common)
    elif method == "scipy":
        from vpnls.scipy import fit_vpnls_scipy

        return fit_vpnls_scipy(N, D, L, resolution=resolution, options=options, **common)
    elif method == "jax":
        from vpnls.jax import fit_vpnls_jax

        return fit_vpnls_jax(
            N,
            D,
            L,
            resolution=resolution,
            options=options,
            enable_x64=enable_x64,
            **common,
        )
    else:
        raise ValueError(f"Unknown method {method!r}, expected 'grid', 'scipy', or 'jax'")
