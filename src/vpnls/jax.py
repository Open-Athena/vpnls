"""JAX autodiff + scipy L-BFGS-B solver for VPNLS."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from vpnls.grid import fit_vpnls_grid
from vpnls.types import (
    DEFAULT_LBFGSB_OPTIONS,
    DEFAULT_SURFACE_BOUNDS,
    FitStatus,
    JaxResult,
    LBFGSBOptions,
    LossFunction,
    LossType,
    NonFiniteFitError,
    SurfaceBounds,
)

# =============================================================================
# JAX objective functions
# =============================================================================


def _huber_loss_jax(residuals, delta):
    abs_r = jnp.abs(residuals)
    return jnp.mean(jnp.where(abs_r <= delta, 0.5 * residuals**2, delta * (abs_r - 0.5 * delta)))


def _vpnls_objective_mse(x, log_N, log_D, L):
    alpha, beta = x
    N_neg_alpha = jnp.exp(-alpha * log_N)
    D_neg_beta = jnp.exp(-beta * log_D)
    X = jnp.column_stack([jnp.ones(len(L)), N_neg_alpha, D_neg_beta])
    params, _, _, _ = jnp.linalg.lstsq(X, L, rcond=None)
    resid = L - X @ params
    return jnp.sum(resid**2)


def _vpnls_objective_huber(x, log_N, log_D, L, delta):
    alpha, beta = x
    N_neg_alpha = jnp.exp(-alpha * log_N)
    D_neg_beta = jnp.exp(-beta * log_D)
    X = jnp.column_stack([jnp.ones(len(L)), N_neg_alpha, D_neg_beta])
    params, _, _, _ = jnp.linalg.lstsq(X, L, rcond=None)
    resid = L - X @ params
    return _huber_loss_jax(resid, delta)


# =============================================================================
# Post-fit checks (same as scipy solver)
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


def fit_vpnls_jax(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    resolution: float = 0.01,
    bounds: SurfaceBounds | None = None,
    options: LBFGSBOptions | None = None,
    loss: LossFunction | None = None,
    max_irls_iter: int = 10,
    enable_x64: bool = True,
) -> JaxResult:
    """Fit L(N,D) = E + A*N^{-alpha} + B*D^{-beta} via grid init + JAX autodiff + L-BFGS-B."""
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
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

    log_N_np, log_D_np = np.log(N), np.log(D)

    # Grid initialization
    grid_result = fit_vpnls_grid(
        N,
        D,
        L,
        resolution=resolution,
        bounds=bounds,
        loss=loss,
        max_irls_iter=max_irls_iter,
    )
    x0 = np.array([grid_result.alpha, grid_result.beta])

    # Convert to JAX arrays
    log_N_jax = jnp.array(log_N_np)
    log_D_jax = jnp.array(log_D_np)
    L_jax = jnp.array(L)

    # Build value_and_grad function
    if loss.type == LossType.HUBER:
        delta = loss.huber_delta

        def obj_fn(x):
            return _vpnls_objective_huber(x, log_N_jax, log_D_jax, L_jax, delta)
    else:

        def obj_fn(x):
            return _vpnls_objective_mse(x, log_N_jax, log_D_jax, L_jax)

    vg_fn = jax.value_and_grad(obj_fn)

    def func(x_np):
        x_jax = jnp.array(x_np)
        val, grad = vg_fn(x_jax)
        return float(val), np.array(grad, dtype=np.float64)

    # L-BFGS-B optimization
    # fmin_l_bfgs_b uses factr (not ftol) and pgtol (not gtol)
    # factr = ftol / machine_eps
    factr = options.ftol / np.finfo(float).eps
    x_opt: np.ndarray
    info: dict
    x_opt, _, info = fmin_l_bfgs_b(
        func,
        x0,
        bounds=[bounds.alpha, bounds.beta],
        maxiter=options.maxiter,
        factr=factr,
        pgtol=options.gtol,
    )

    alpha, beta = float(x_opt[0]), float(x_opt[1])

    # Extract final params via numpy OLS
    N_neg_alpha = np.exp(-alpha * log_N_np)
    D_neg_beta = np.exp(-beta * log_D_np)
    X = np.column_stack([np.ones(len(L)), N_neg_alpha, D_neg_beta])
    params, _, _, _ = np.linalg.lstsq(X, L, rcond=None)
    E, A, B = params
    resid = L - X @ params
    rss = float(resid @ resid)

    # loss_value: RSS for MSE, Huber objective for Huber
    if loss.type == LossType.HUBER:
        abs_r = np.abs(resid)
        d = loss.huber_delta
        huber_vals = np.where(abs_r <= d, 0.5 * resid**2, d * (abs_r - 0.5 * d))
        loss_value = float(np.mean(huber_vals))
    else:
        loss_value = rss

    # Non-finite check
    for name, val in [("E", E), ("A", A), ("B", B), ("alpha", alpha), ("beta", beta)]:
        if not math.isfinite(val):
            raise NonFiniteFitError(f"Non-finite fitted parameter: {name}={val}")

    # Status from optimizer
    warnflag = info["warnflag"]
    if warnflag == 0:
        status = FitStatus.CONVERGED
        status_message = ""
    elif warnflag == 1:
        status = FitStatus.MAX_ITER
        status_message = f"Hit maxiter ({options.maxiter})"
    else:
        status = FitStatus.ABNORMAL
        status_message = f"Optimization failed: {info.get('task', 'unknown')}"

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

    return JaxResult(
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
        n_iter=int(info.get("nit", 0)),
    )
