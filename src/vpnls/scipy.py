"""Scipy L-BFGS-B solver for VPNLS with envelope-theorem gradients."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from vpnls.grid import fit_vpnls_grid
from vpnls.types import (
    DEFAULT_LBFGSB_OPTIONS,
    DEFAULT_SURFACE_BOUNDS,
    FitStatus,
    LBFGSBOptions,
    LossFunction,
    LossType,
    NonFiniteFitError,
    ScipyResult,
    SurfaceBounds,
)

# =============================================================================
# Inner solves
# =============================================================================


def _vpnls_rss_and_params_ols(
    alpha: float,
    beta: float,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Inner OLS solve for fixed (alpha, beta). Returns (rss, params[E, A, B])."""
    N_neg_alpha = np.exp(-alpha * log_N)
    D_neg_beta = np.exp(-beta * log_D)
    X = np.column_stack([np.ones(len(L)), N_neg_alpha, D_neg_beta])
    params, _, _, _ = np.linalg.lstsq(X, L, rcond=None)
    resid = L - X @ params
    return float(resid @ resid), params


def _vpnls_objective_and_grad(
    x: np.ndarray,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
) -> tuple[float, np.ndarray]:
    """RSS objective + envelope-theorem gradient for L-BFGS-B (jac=True)."""
    alpha, beta = x
    N_neg_alpha = np.exp(-alpha * log_N)
    D_neg_beta = np.exp(-beta * log_D)
    X = np.column_stack([np.ones(len(L)), N_neg_alpha, D_neg_beta])
    params, _, _, _ = np.linalg.lstsq(X, L, rcond=None)
    E, A, B = params
    resid = L - (E + A * N_neg_alpha + B * D_neg_beta)
    rss = float(resid @ resid)
    grad_alpha = float(2 * A * (resid @ (log_N * N_neg_alpha)))
    grad_beta = float(2 * B * (resid @ (log_D * D_neg_beta)))
    return rss, np.array([grad_alpha, grad_beta])


# =============================================================================
# Huber inner solves
# =============================================================================


def _huber_weights(abs_r: np.ndarray, delta: float) -> np.ndarray:
    """Huber weights: 1.0 where |r| <= delta, else delta / |r|."""
    with np.errstate(divide="ignore"):
        return np.where(abs_r <= delta, 1.0, delta / abs_r)


def _vpnls_huber_irls_and_params(
    alpha: float,
    beta: float,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
    delta: float,
    max_irls_iter: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Huber inner solve via IRLS. Returns (huber_obj, rss, params[E,A,B], weights)."""
    N_neg_alpha = np.exp(-alpha * log_N)
    D_neg_beta = np.exp(-beta * log_D)
    X = np.column_stack([np.ones(len(L)), N_neg_alpha, D_neg_beta])

    # OLS initialization
    params, _, _, _ = np.linalg.lstsq(X, L, rcond=None)

    # IRLS loop with convergence check
    for _ in range(max_irls_iter):
        resid = L - X @ params
        abs_r = np.abs(resid)
        w = _huber_weights(abs_r, delta)
        XtW = (X * w[:, None]).T
        new_params = np.linalg.solve(XtW @ X, XtW @ L)
        if np.max(np.abs(new_params - params)) < 1e-14:
            params = new_params
            break
        params = new_params

    resid = L - X @ params
    abs_r = np.abs(resid)
    w = _huber_weights(abs_r, delta)

    # Huber objective = mean(huber(r_i, delta))
    huber_vals = np.where(abs_r <= delta, 0.5 * resid**2, delta * (abs_r - 0.5 * delta))
    huber_obj = float(np.mean(huber_vals))
    rss = float(resid @ resid)

    return huber_obj, rss, params, w


def _vpnls_huber_objective_and_grad(
    x: np.ndarray,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
    delta: float,
    max_irls_iter: int,
) -> tuple[float, np.ndarray]:
    """Huber objective + weighted envelope-theorem gradient for L-BFGS-B."""
    alpha, beta = x
    huber_obj, _, params, w = _vpnls_huber_irls_and_params(
        alpha, beta, log_N, log_D, L, delta, max_irls_iter
    )
    E, A, B = params
    N_neg_alpha = np.exp(-alpha * log_N)
    D_neg_beta = np.exp(-beta * log_D)
    resid = L - (E + A * N_neg_alpha + B * D_neg_beta)
    n = len(L)
    wr = w * resid
    grad_alpha = float(A / n * (wr @ (log_N * N_neg_alpha)))
    grad_beta = float(B / n * (wr @ (log_D * D_neg_beta)))
    return huber_obj, np.array([grad_alpha, grad_beta])


# =============================================================================
# Post-fit checks
# =============================================================================


def _check_at_bounds(name: str, val: float, lo: float, hi: float) -> str | None:
    tol = 1e-4 * (hi - lo)
    if val - lo < tol:
        return f"{name}={val:.6f} at lower bound {lo:.4f}"
    if hi - val < tol:
        return f"{name}={val:.6f} at upper bound {hi:.4f}"
    return None


def _check_positive(name: str, val: float) -> str | None:
    if val <= 0:
        return f"{name}={val:.2e} is non-positive"
    return None


# =============================================================================
# Main solver
# =============================================================================


def fit_vpnls_scipy(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    grid_resolution: float = 0.01,
    bounds: SurfaceBounds | None = None,
    options: LBFGSBOptions | None = None,
    loss: LossFunction | None = None,
    max_irls_iter: int = 10,
) -> ScipyResult:
    """Fit L(N,D) = E + A*N^{-alpha} + B*D^{-beta} via grid init + L-BFGS-B."""
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
    if options is None:
        options = DEFAULT_LBFGSB_OPTIONS
    if loss is None:
        loss = LossFunction()

    log_N, log_D = np.log(N), np.log(D)

    # Grid initialization
    grid_result = fit_vpnls_grid(
        N,
        D,
        L,
        resolution=grid_resolution,
        bounds=bounds,
        loss=loss,
        max_irls_iter=max_irls_iter,
    )
    x0 = np.array([grid_result.alpha, grid_result.beta])

    # L-BFGS-B refinement
    if loss.type == LossType.HUBER:
        result = minimize(
            _vpnls_huber_objective_and_grad,
            x0=x0,
            args=(log_N, log_D, L, loss.huber_delta, max_irls_iter),
            jac=True,
            method="L-BFGS-B",
            bounds=[bounds.alpha, bounds.beta],
            options=options.to_dict(),
        )
    else:
        result = minimize(
            _vpnls_objective_and_grad,
            x0=x0,
            args=(log_N, log_D, L),
            jac=True,
            method="L-BFGS-B",
            bounds=[bounds.alpha, bounds.beta],
            options=options.to_dict(),
        )

    assert result is not None
    alpha, beta = float(result.x[0]), float(result.x[1])

    # Extract final params
    if loss.type == LossType.HUBER:
        huber_obj, rss, params, _ = _vpnls_huber_irls_and_params(
            alpha, beta, log_N, log_D, L, loss.huber_delta, max_irls_iter
        )
        loss_value = huber_obj
    else:
        rss, params = _vpnls_rss_and_params_ols(alpha, beta, log_N, log_D, L)
        loss_value = rss

    E, A, B = params

    # Non-finite check
    for name, val in [("E", E), ("A", A), ("B", B), ("alpha", alpha), ("beta", beta)]:
        if not math.isfinite(val):
            raise NonFiniteFitError(f"Non-finite fitted parameter: {name}={val}")

    # Status from optimizer
    if result.success:
        status = FitStatus.CONVERGED
        status_message = ""
    elif result.nit >= options.maxiter:
        status = FitStatus.MAX_ITER
        status_message = f"Hit maxiter ({options.maxiter}): {result.message}"
    else:
        status = FitStatus.ABNORMAL
        status_message = f"Optimization failed: {result.message}"

    # Post-fit bound checks
    if status == FitStatus.CONVERGED:
        bound_msgs = [
            m
            for m in [
                _check_at_bounds("alpha", alpha, *bounds.alpha),
                _check_at_bounds("beta", beta, *bounds.beta),
                _check_positive("E", E),
                _check_positive("A", A),
                _check_positive("B", B),
            ]
            if m is not None
        ]
        if bound_msgs:
            status = FitStatus.BOUND_HIT
            status_message = "; ".join(bound_msgs)

    return ScipyResult(
        E=float(E),
        A=float(A),
        B=float(B),
        alpha=alpha,
        beta=beta,
        loss_value=loss_value,
        rss=rss,
        n_points=len(N),
        loss_function=loss,
        status=status,
        status_message=status_message,
        n_iter=int(result.nit),
        scipy_message=str(result.message),
    )
