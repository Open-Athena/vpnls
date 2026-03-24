"""Tests for the JAX autodiff VPNLS solver."""

import pytest
from conftest import CHINCHILLA, NOISELESS, SURFACES

from vpnls.jax import fit_vpnls_jax
from vpnls.scipy import fit_vpnls_scipy
from vpnls.sim import generate_isoflop_data
from vpnls.types import LossFunction, LossType

# =============================================================================
# Parameter recovery
# =============================================================================


@pytest.mark.parametrize("surface", SURFACES)
def test_parameter_recovery_mse(surface):
    N, D, L = generate_isoflop_data(surface, NOISELESS)
    result = fit_vpnls_jax(N, D, L)
    assert result.alpha == pytest.approx(surface.alpha, rel=1e-6)
    assert result.beta == pytest.approx(surface.beta, rel=1e-6)
    assert result.E == pytest.approx(surface.E, rel=1e-6)
    assert result.A == pytest.approx(surface.A, rel=1e-6)
    assert result.B == pytest.approx(surface.B, rel=1e-6)


@pytest.mark.parametrize("surface", SURFACES)
def test_parameter_recovery_huber(surface):
    N, D, L = generate_isoflop_data(surface, NOISELESS)
    loss = LossFunction(LossType.HUBER, huber_delta=0.01)
    result = fit_vpnls_jax(N, D, L, loss=loss)
    assert result.alpha == pytest.approx(surface.alpha, rel=1e-6)
    assert result.beta == pytest.approx(surface.beta, rel=1e-6)
    assert result.E == pytest.approx(surface.E, rel=1e-6)
    assert result.A == pytest.approx(surface.A, rel=1e-6)
    assert result.B == pytest.approx(surface.B, rel=1e-6)


# =============================================================================
# Agreement with scipy
# =============================================================================


def test_agreement_with_scipy():
    N, D, L = generate_isoflop_data(CHINCHILLA, NOISELESS)
    jax_result = fit_vpnls_jax(N, D, L)
    scipy_result = fit_vpnls_scipy(N, D, L)
    assert jax_result.alpha == pytest.approx(scipy_result.alpha, rel=1e-4)
    assert jax_result.beta == pytest.approx(scipy_result.beta, rel=1e-4)
    assert jax_result.E == pytest.approx(scipy_result.E, rel=1e-4)
    assert jax_result.A == pytest.approx(scipy_result.A, rel=1e-4)
    assert jax_result.B == pytest.approx(scipy_result.B, rel=1e-4)
