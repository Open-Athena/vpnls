"""Tests for the Cython grid solver."""

import numpy as np
import pytest
from conftest import CHINCHILLA, SURFACES
from scipy.optimize import least_squares

from vpnls.grid import fit_vpnls_grid
from vpnls.sim import generate_isoflop_data
from vpnls.types import FitStatus, LossFunction, LossSurface, LossType

LOSS_FUNCTIONS = [
    pytest.param(None, id="mse"),
    pytest.param(LossFunction(LossType.HUBER, huber_delta=0.01), id="huber"),
]


# ── Parameter recovery (noise-free) ─────────────────────────────────────────


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("loss", LOSS_FUNCTIONS)
def test_parameter_recovery(surface: LossSurface, loss):
    N, D, L = generate_isoflop_data(surface, noise_std=0.0)
    result = fit_vpnls_grid(N, D, L, resolution=0.005, loss=loss)

    assert result.alpha == pytest.approx(surface.alpha, rel=1e-2)
    assert result.beta == pytest.approx(surface.beta, rel=1e-2)
    assert result.A == pytest.approx(surface.A, rel=5e-2)
    assert result.B == pytest.approx(surface.B, rel=5e-2)
    assert result.E == pytest.approx(surface.E, rel=5e-2)


# ── Cross-validation against scipy.optimize.least_squares ────────────────────


def _design_matrix(result, N, D):
    log_N, log_D = np.log(N), np.log(D)
    return np.column_stack(
        [np.ones(len(N)), np.exp(-result.alpha * log_N), np.exp(-result.beta * log_D)]
    )


def test_mse_matches_scipy_least_squares():
    N, D, L = generate_isoflop_data(CHINCHILLA, noise_std=0.0)
    result = fit_vpnls_grid(N, D, L, resolution=0.005)

    X = _design_matrix(result, N, D)
    x0 = np.array([result.E, result.A, result.B])
    scipy_result = least_squares(lambda p: X @ p - L, x0, loss="linear")

    np.testing.assert_allclose([result.E, result.A, result.B], scipy_result.x, rtol=1e-10)
    assert result.rss == pytest.approx(np.sum(scipy_result.fun**2), abs=1e-20)


def test_huber_matches_scipy_least_squares():
    delta = 0.01
    N, D, L = generate_isoflop_data(CHINCHILLA, noise_std=0.0)
    loss = LossFunction(LossType.HUBER, huber_delta=delta)
    result = fit_vpnls_grid(N, D, L, resolution=0.005, loss=loss)

    X = _design_matrix(result, N, D)
    x0 = np.array([result.E, result.A, result.B])
    scipy_result = least_squares(lambda p: X @ p - L, x0, loss="huber", f_scale=delta)

    np.testing.assert_allclose([result.E, result.A, result.B], scipy_result.x, rtol=1e-10)
    resid = scipy_result.fun
    ar = np.abs(resid)
    scipy_obj = np.mean(np.where(ar <= delta, 0.5 * resid**2, delta * (ar - 0.5 * delta)))
    assert result.loss_value == pytest.approx(scipy_obj, abs=1e-20)


# ── NNLS clamping ────────────────────────────────────────────────────────────


def test_clamped_rss_matches_returned_params():
    """When NNLS clamps params to zero, RSS must match the clamped (not unconstrained) solution."""
    # E=0 surface + noise → unconstrained OLS finds E<0, NNLS clamps to 0
    surface = LossSurface(alpha=0.40, beta=0.30, A=500, B=500, E=0.0)
    N, D, L = generate_isoflop_data(surface, noise_std=0.02, seed=2)

    result = fit_vpnls_grid(N, D, L, resolution=0.005)

    # Clamping must have occurred and be reported as BOUND_HIT
    assert result.status == FitStatus.BOUND_HIT
    assert "clamped" in result.status_message.lower()
    assert result.E == 0.0

    # Recompute RSS independently from returned (clamped) params
    pred = result.E + result.A * N ** (-result.alpha) + result.B * D ** (-result.beta)
    rss_recomputed = float(np.sum((L - pred) ** 2))
    assert result.rss == pytest.approx(rss_recomputed, rel=1e-12)

    # Verify unconstrained OLS at same (alpha, beta) has a negative param
    X = _design_matrix(result, N, D)
    ols_params = np.linalg.lstsq(X, L, rcond=None)[0]
    assert ols_params[0] < 0, "Unconstrained OLS should have negative E"
