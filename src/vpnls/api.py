"""Unified entry point for VPNLS fitting."""

from __future__ import annotations

import numpy as np

from vpnls.types import (
    LBFGSBOptions,
    LossFunction,
    SurfaceBounds,
    VPNLSResult,
)


def fit_vpnls(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    method: str = "scipy",
    # Common params
    bounds: SurfaceBounds | None = None,
    loss: LossFunction | None = None,
    max_irls_iter: int = 10,
    # Grid params (method="grid")
    resolution: float = 0.01,
    num_workers: int = 1,
    # Scipy/JAX params (method="scipy" or "jax")
    grid_resolution: float = 0.03,
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
        resolution: Grid step size in exponent space (grid only).
        num_workers: Number of parallel processes for grid search (grid only).
        grid_resolution: Coarse grid resolution for L-BFGS-B initialization (scipy/jax).
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

        return fit_vpnls_scipy(N, D, L, grid_resolution=grid_resolution, options=options, **common)
    elif method == "jax":
        from vpnls.jax import fit_vpnls_jax

        return fit_vpnls_jax(
            N,
            D,
            L,
            grid_resolution=grid_resolution,
            options=options,
            enable_x64=enable_x64,
            **common,
        )
    else:
        raise ValueError(f"Unknown method {method!r}, expected 'grid', 'scipy', or 'jax'")
