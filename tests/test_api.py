"""Tests for the unified fit_vpnls API."""

import pytest
from conftest import NOISELESS, SURFACES

from vpnls.api import fit_vpnls
from vpnls.sim import generate_isoflop_data
from vpnls.types import GridResult, JaxResult, LossFunction, LossType, ScipyResult

LOSS_FUNCTIONS = [
    pytest.param(None, id="mse"),
    pytest.param(LossFunction(LossType.HUBER, huber_delta=0.01), id="huber"),
]


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("loss", LOSS_FUNCTIONS)
def test_scipy_jax_agreement(surface, loss):
    """Scipy and JAX solvers produce equivalent results for both MSE and Huber."""
    N, D, L = generate_isoflop_data(surface, NOISELESS)

    scipy_result = fit_vpnls(N, D, L, method="scipy", loss=loss)
    jax_result = fit_vpnls(N, D, L, method="jax", loss=loss)

    assert isinstance(scipy_result, ScipyResult)
    assert isinstance(jax_result, JaxResult)

    assert jax_result.alpha == pytest.approx(scipy_result.alpha, rel=1e-8)
    assert jax_result.beta == pytest.approx(scipy_result.beta, rel=1e-8)
    assert jax_result.E == pytest.approx(scipy_result.E, rel=1e-8)
    assert jax_result.A == pytest.approx(scipy_result.A, rel=1e-8)
    assert jax_result.B == pytest.approx(scipy_result.B, rel=1e-8)


def test_dispatch_returns_correct_types():
    """Each method returns its specific result type."""
    from conftest import CHINCHILLA

    N, D, L = generate_isoflop_data(CHINCHILLA, NOISELESS)

    assert isinstance(fit_vpnls(N, D, L, method="grid"), GridResult)
    assert isinstance(fit_vpnls(N, D, L, method="scipy"), ScipyResult)
    assert isinstance(fit_vpnls(N, D, L, method="jax"), JaxResult)


def test_invalid_method():
    from conftest import CHINCHILLA

    N, D, L = generate_isoflop_data(CHINCHILLA, NOISELESS)

    with pytest.raises(ValueError, match="Unknown method"):
        fit_vpnls(N, D, L, method="bogus")
