"""Python wrapper for the Cython grid solver."""

from __future__ import annotations

import math

import numpy as np

from vpnls._core import grid_search
from vpnls.types import (
    DEFAULT_SURFACE_BOUNDS,
    FitStatus,
    GridResult,
    LossFunction,
    LossType,
    NonFiniteFitError,
    SurfaceBounds,
)


def fit_vpnls_grid(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    resolution: float = 0.01,
    bounds: SurfaceBounds | None = None,
    loss: LossFunction | None = None,
    max_irls_iter: int = 10,
) -> GridResult:
    """Fit L(N,D) = E + A*N^{-alpha} + B*D^{-beta} via exhaustive grid search."""
    N = np.asarray(N, dtype=np.float64).ravel()
    D = np.asarray(D, dtype=np.float64).ravel()
    L = np.asarray(L, dtype=np.float64).ravel()

    if not (len(N) == len(D) == len(L)):
        raise ValueError("N, D, L must have the same length")
    if len(N) < 4:
        raise ValueError("Need at least 4 data points")
    if not (np.all(np.isfinite(N)) and np.all(N > 0)):
        raise ValueError("N must be positive and finite")
    if not (np.all(np.isfinite(D)) and np.all(D > 0)):
        raise ValueError("D must be positive and finite")
    if not np.all(np.isfinite(L)):
        raise ValueError("L must be finite")

    if bounds is None:
        bounds = DEFAULT_SURFACE_BOUNDS
    if loss is None:
        loss = LossFunction()

    loss_type_int = 0 if loss.type == LossType.MSE else 1
    log_N = np.log(N)
    log_D = np.log(D)

    result = grid_search(
        log_N,
        log_D,
        L,
        bounds.alpha[0],
        bounds.alpha[1],
        bounds.beta[0],
        bounds.beta[1],
        resolution,
        loss_type_int,
        loss.huber_delta,
        max_irls_iter,
    )

    (
        best_E,
        best_A,
        best_B,
        best_alpha,
        best_beta,
        best_obj,
        best_rss,
        clamped_mask,
        best_ai,
        best_bi,
        n_alpha,
        n_beta,
    ) = result

    # Non-finite check
    for name, val in [
        ("E", best_E),
        ("A", best_A),
        ("B", best_B),
        ("alpha", best_alpha),
        ("beta", best_beta),
    ]:
        if not math.isfinite(val):
            raise NonFiniteFitError(f"Non-finite fitted parameter: {name}={val}")

    # Status checks
    status = FitStatus.CONVERGED
    messages: list[str] = []

    # Grid-edge hit
    if best_ai == 0 or best_ai == n_alpha - 1:
        messages.append(
            f"alpha={best_alpha:.4f} at grid edge [{bounds.alpha[0]:.4f}, {bounds.alpha[1]:.4f}]"
        )
    if best_bi == 0 or best_bi == n_beta - 1:
        messages.append(
            f"beta={best_beta:.4f} at grid edge [{bounds.beta[0]:.4f}, {bounds.beta[1]:.4f}]"
        )

    # NNLS clamping
    clamped_names = []
    if clamped_mask & 1:
        clamped_names.append("E")
    if clamped_mask & 2:
        clamped_names.append("A")
    if clamped_mask & 4:
        clamped_names.append("B")
    if clamped_names:
        messages.append(f"NNLS clamped {', '.join(clamped_names)} to zero")

    if messages:
        status = FitStatus.BOUND_HIT

    return GridResult(
        E=best_E,
        A=best_A,
        B=best_B,
        alpha=best_alpha,
        beta=best_beta,
        loss_value=best_obj,
        rss=best_rss,
        n_points=len(N),
        loss_function=loss,
        status=status,
        status_message="; ".join(messages),
        resolution=resolution,
    )
