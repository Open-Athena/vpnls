"""Python wrapper for the Cython grid solver."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

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


@dataclass
class RawGridResult:
    """Unpacked grid_search output."""

    E: float
    A: float
    B: float
    alpha: float
    beta: float
    obj: float
    rss: float
    clamped_mask: int
    ai: int
    bi: int
    n_alpha: int
    n_beta: int

    @staticmethod
    def from_tuple(t: tuple) -> RawGridResult:
        return RawGridResult(*t)


@dataclass(frozen=True)
class GridSearchArgs:
    """Shared arguments for grid_search calls."""

    log_N: np.ndarray
    log_D: np.ndarray
    L: np.ndarray
    beta_lo: float
    beta_hi: float
    resolution: float
    loss_type_int: int
    huber_delta: float
    max_irls_iter: int

    def run(self, alpha_lo: float, alpha_hi: float) -> RawGridResult:
        return RawGridResult.from_tuple(
            grid_search(
                self.log_N,
                self.log_D,
                self.L,
                alpha_lo,
                alpha_hi,
                self.beta_lo,
                self.beta_hi,
                self.resolution,
                self.loss_type_int,
                self.huber_delta,
                self.max_irls_iter,
            )
        )


def _run_chunk(args: GridSearchArgs, alpha_lo: float, alpha_hi: float) -> RawGridResult:
    """Top-level function for picklability in ProcessPoolExecutor."""
    return args.run(alpha_lo, alpha_hi)


def _run_parallel(
    args: GridSearchArgs, alpha_lo: float, n_alpha_total: int, num_workers: int
) -> RawGridResult:
    """Split alpha range into chunks and run grid_search in parallel."""
    chunks = [ch for ch in np.array_split(range(n_alpha_total), num_workers) if len(ch) > 0]
    res = args.resolution
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            (
                ch[0],
                pool.submit(
                    _run_chunk,
                    args,
                    alpha_lo=alpha_lo + ch[0] * res,
                    alpha_hi=alpha_lo + ch[-1] * res,
                ),
            )
            for ch in chunks
        ]

    best: RawGridResult | None = None
    for offset, future in futures:
        r = future.result()
        r.ai += offset
        if best is None or r.obj < best.obj:
            best = r

    assert best is not None
    best.n_alpha = n_alpha_total  # chunk has local count; override with global
    return best


def _check_status(raw: RawGridResult, bounds: SurfaceBounds) -> tuple[FitStatus, str]:
    """Check for grid-edge hits and NNLS clamping."""
    messages: list[str] = []

    if raw.ai == 0 or raw.ai == raw.n_alpha - 1:
        messages.append(
            f"alpha={raw.alpha:.4f} at grid edge [{bounds.alpha[0]:.4f}, {bounds.alpha[1]:.4f}]"
        )
    if raw.bi == 0 or raw.bi == raw.n_beta - 1:
        messages.append(
            f"beta={raw.beta:.4f} at grid edge [{bounds.beta[0]:.4f}, {bounds.beta[1]:.4f}]"
        )

    clamped = [name for bit, name in enumerate(["E", "A", "B"]) if raw.clamped_mask & (1 << bit)]
    if clamped:
        messages.append(f"NNLS clamped {', '.join(clamped)} to zero")

    if messages:
        return FitStatus.BOUND_HIT, "; ".join(messages)
    return FitStatus.CONVERGED, ""


def fit_vpnls_grid(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    resolution: float = 0.001,
    bounds: SurfaceBounds | None = None,
    loss: LossFunction | None = None,
    max_irls_iter: int = 10,
    num_workers: int = 1,
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

    alpha_lo, alpha_hi = bounds.alpha
    args = GridSearchArgs(
        log_N=np.log(N),
        log_D=np.log(D),
        L=L,
        beta_lo=bounds.beta[0],
        beta_hi=bounds.beta[1],
        resolution=resolution,
        loss_type_int=0 if loss.type == LossType.MSE else 1,
        huber_delta=loss.huber_delta,
        max_irls_iter=max_irls_iter,
    )
    n_alpha_total = int((alpha_hi - alpha_lo) / resolution) + 1

    if num_workers > 1 and n_alpha_total >= num_workers:
        raw = _run_parallel(args, alpha_lo, n_alpha_total, num_workers)
    else:
        raw = args.run(alpha_lo, alpha_hi)

    for name in ("E", "A", "B", "alpha", "beta"):
        if not math.isfinite(getattr(raw, name)):
            raise NonFiniteFitError(f"Non-finite fitted parameter: {name}={getattr(raw, name)}")

    status, status_message = _check_status(raw, bounds)

    return GridResult(
        E=raw.E,
        A=raw.A,
        B=raw.B,
        alpha=raw.alpha,
        beta=raw.beta,
        loss_value=raw.obj,
        rss=raw.rss,
        n_points=len(N),
        loss_function=loss,
        status=status,
        status_message=status_message,
        resolution=resolution,
    )
