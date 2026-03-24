"""Tests for the unified fit_vpnls API."""

import numpy as np
import pytest
from conftest import NOISELESS, SURFACES

from vpnls.api import fit_vpnls
from vpnls.sim import generate_isoflop_data
from vpnls.types import GridResult, JaxResult, LossFunction, LossType, ScipyResult

METHODS = ["grid", "scipy", "jax"]

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


@pytest.mark.parametrize("method", METHODS)
class TestInputValidation:
    N = np.array([1e9, 1e10, 1e11, 1e12, 1e13])
    D = np.array([1e10, 1e11, 1e12, 1e13, 1e14])
    L = np.array([2.0, 1.9, 1.8, 1.7, 1.6])

    def test_mismatched_lengths(self, method):
        with pytest.raises(ValueError, match="same length"):
            fit_vpnls(self.N, self.D[:3], self.L, method=method)

    def test_too_few_points(self, method):
        with pytest.raises(ValueError, match="at least 4"):
            fit_vpnls(self.N[:3], self.D[:3], self.L[:3], method=method)

    def test_non_positive_N(self, method):
        bad_N = self.N.copy()
        bad_N[0] = -1
        with pytest.raises(ValueError, match="N must be positive"):
            fit_vpnls(bad_N, self.D, self.L, method=method)

    def test_non_positive_D(self, method):
        bad_D = self.D.copy()
        bad_D[0] = 0
        with pytest.raises(ValueError, match="D must be positive"):
            fit_vpnls(self.N, bad_D, self.L, method=method)

    def test_non_finite_L(self, method):
        bad_L = self.L.copy()
        bad_L[0] = np.nan
        with pytest.raises(ValueError, match="L must be finite"):
            fit_vpnls(self.N, self.D, bad_L, method=method)
