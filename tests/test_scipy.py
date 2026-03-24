"""Tests for the scipy L-BFGS-B VPNLS solver."""

import numpy as np
import pytest
from conftest import NOISELESS, SURFACES

from vpnls.scipy import (
    _vpnls_huber_objective_and_grad,
    _vpnls_objective_and_grad,
    fit_vpnls_scipy,
)
from vpnls.sim import generate_isoflop_data
from vpnls.types import LossFunction, LossType

# =============================================================================
# Parameter recovery
# =============================================================================


@pytest.mark.parametrize("surface", SURFACES)
def test_parameter_recovery_mse(surface):
    N, D, L = generate_isoflop_data(surface, NOISELESS)
    result = fit_vpnls_scipy(N, D, L)
    assert result.alpha == pytest.approx(surface.alpha, rel=1e-6)
    assert result.beta == pytest.approx(surface.beta, rel=1e-6)
    assert result.E == pytest.approx(surface.E, rel=1e-6)
    assert result.A == pytest.approx(surface.A, rel=1e-6)
    assert result.B == pytest.approx(surface.B, rel=1e-6)


@pytest.mark.parametrize("surface", SURFACES)
def test_parameter_recovery_huber(surface):
    N, D, L = generate_isoflop_data(surface, NOISELESS)
    loss = LossFunction(LossType.HUBER, huber_delta=0.01)
    result = fit_vpnls_scipy(N, D, L, loss=loss)
    assert result.alpha == pytest.approx(surface.alpha, rel=1e-6)
    assert result.beta == pytest.approx(surface.beta, rel=1e-6)
    assert result.E == pytest.approx(surface.E, rel=1e-6)
    assert result.A == pytest.approx(surface.A, rel=1e-6)
    assert result.B == pytest.approx(surface.B, rel=1e-6)


# =============================================================================
# RSS near-zero
# =============================================================================


@pytest.mark.parametrize("surface", SURFACES)
def test_rss_near_zero_mse(surface):
    N, D, L = generate_isoflop_data(surface, NOISELESS)
    result = fit_vpnls_scipy(N, D, L)
    assert result.rss < 1e-17


# =============================================================================
# Gradient vs finite differences
# =============================================================================


def _assert_gradient_matches_fd(obj_and_grad_fn, x0, args, rtol, h=None):
    """Compare analytical gradient against central finite differences."""
    if h is None:
        h = np.finfo(float).eps ** (1.0 / 3.0)
    _, grad = obj_and_grad_fn(x0, *args)
    fd_grad = np.zeros_like(x0)
    for i in range(len(x0)):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[i] += h
        x_minus[i] -= h
        f_plus = obj_and_grad_fn(x_plus, *args)[0]
        f_minus = obj_and_grad_fn(x_minus, *args)[0]
        fd_grad[i] = (f_plus - f_minus) / (2 * h)
    np.testing.assert_allclose(grad, fd_grad, rtol=rtol)


@pytest.mark.parametrize("surface", SURFACES)
def test_gradient_mse(surface):
    N, D, L = generate_isoflop_data(surface, NOISELESS)
    log_N, log_D = np.log(N), np.log(D)
    x0 = np.array([0.25, 0.35])
    _assert_gradient_matches_fd(_vpnls_objective_and_grad, x0, (log_N, log_D, L), rtol=1e-7)


@pytest.mark.parametrize("surface", SURFACES)
def test_gradient_huber(surface):
    N, D, L = generate_isoflop_data(surface, NOISELESS)
    log_N, log_D = np.log(N), np.log(D)
    x0 = np.array([0.25, 0.35])
    _assert_gradient_matches_fd(
        _vpnls_huber_objective_and_grad,
        x0,
        (log_N, log_D, L, 0.1, 100),
        rtol=1e-7,
    )
