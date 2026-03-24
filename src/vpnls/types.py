"""Types for the VPNLS package."""

import enum
from dataclasses import dataclass

import numpy as np

# =============================================================================
# Fit status and exceptions
# =============================================================================


class FitStatus(enum.Enum):
    """Outcome of a surface-fitting procedure."""

    CONVERGED = "converged"
    """Optimizer reported success and all diagnostics passed."""

    MAX_ITER = "max_iter"
    """Optimizer exhausted its iteration budget before convergence."""

    ABNORMAL = "abnormal"
    """Optimizer reported failure (e.g. line-search breakdown)."""

    BOUND_HIT = "bound_hit"
    """Fitted parameters at or near a bound or grid edge."""


class FitError(Exception):
    """A fitting procedure failed -- no usable result is available."""


class NonFiniteFitError(FitError):
    """A fitted parameter is NaN or Inf."""


# =============================================================================
# Loss configuration
# =============================================================================


class LossType(enum.Enum):
    MSE = "mse"
    HUBER = "huber"


@dataclass(frozen=True)
class LossFunction:
    """Loss function specification."""

    type: LossType = LossType.MSE
    huber_delta: float = 1.0


# =============================================================================
# Loss surface model
# =============================================================================


@dataclass(frozen=True)
class LossSurface:
    """Loss function L(N, D) = E + A/N^alpha + B/D^beta."""

    alpha: float
    beta: float
    A: float
    B: float
    E: float

    @property
    def a(self) -> float:
        """N* scaling exponent: beta/(alpha+beta)."""
        return self.beta / (self.alpha + self.beta)

    @property
    def b(self) -> float:
        """D* scaling exponent: alpha/(alpha+beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def G(self) -> float:
        """Scaling constant: (alpha*A / beta*B)^(1/(alpha+beta))."""
        return (self.alpha * self.A / (self.beta * self.B)) ** (1.0 / (self.alpha + self.beta))

    def loss(self, N: float, D: float) -> float:
        """Compute L(N, D) = E + A/N^alpha + B/D^beta."""
        return self.E + self.A / N**self.alpha + self.B / D**self.beta

    def N_opt(self, C: float) -> float:
        """Optimal parameter count for compute budget C."""
        return self.G * (C / 6) ** self.a

    def D_opt(self, C: float) -> float:
        """Optimal token count for compute budget C."""
        return (1 / self.G) * (C / 6) ** self.b


# =============================================================================
# Grid and optimizer configuration
# =============================================================================


@dataclass(frozen=True)
class ExponentGrid:
    """Initialization grid for 2D searches over (alpha, beta)."""

    alpha: np.ndarray
    beta: np.ndarray

    @property
    def total_size(self) -> int:
        return len(self.alpha) * len(self.beta)


@dataclass(frozen=True)
class SurfaceBounds:
    """Optimization bounds for all 5 surface parameters."""

    E: tuple[float, float] = (1e-6, 10.0)
    A: tuple[float, float] = (1e-6, 1e6)
    B: tuple[float, float] = (1e-6, 1e6)
    alpha: tuple[float, float] = (0.01, 0.99)
    beta: tuple[float, float] = (0.01, 0.99)

    def to_list(self) -> list[tuple[float, float]]:
        """Return bounds in [E, A, B, alpha, beta] order for scipy."""
        return [self.E, self.A, self.B, self.alpha, self.beta]


@dataclass(frozen=True)
class LBFGSBOptions:
    """Options for the L-BFGS-B optimizer."""

    ftol: float = 1e-15
    gtol: float = 1e-15
    maxiter: int = 10_000

    def to_dict(self) -> dict:
        return {"ftol": self.ftol, "gtol": self.gtol, "maxiter": self.maxiter}


# =============================================================================
# Result types
# =============================================================================


@dataclass(frozen=True)
class VPNLSResult:
    """Base result -- shared across all solvers."""

    E: float
    A: float
    B: float
    alpha: float
    beta: float
    loss_value: float
    rss: float
    n_points: int
    loss_function: LossFunction
    status: FitStatus
    status_message: str = ""

    @property
    def a(self) -> float:
        """N* scaling exponent: beta/(alpha+beta)."""
        return self.beta / (self.alpha + self.beta)

    @property
    def b(self) -> float:
        """D* scaling exponent: alpha/(alpha+beta)."""
        return self.alpha / (self.alpha + self.beta)

    def to_loss_surface(self) -> LossSurface:
        return LossSurface(alpha=self.alpha, beta=self.beta, A=self.A, B=self.B, E=self.E)


@dataclass(frozen=True)
class GridResult(VPNLSResult):
    """Result from fit_vpnls_grid."""

    resolution: float = 0.0


@dataclass(frozen=True)
class ScipyResult(VPNLSResult):
    """Result from fit_vpnls_scipy."""

    n_iter: int = 0
    scipy_message: str = ""


@dataclass(frozen=True)
class JaxResult(VPNLSResult):
    """Result from fit_vpnls_jax."""

    n_iter: int = 0


# =============================================================================
# Default instances
# =============================================================================

DEFAULT_EXPONENT_GRID = ExponentGrid(
    alpha=np.linspace(0.01, 0.99, 128),
    beta=np.linspace(0.01, 0.99, 128),
)

DEFAULT_SURFACE_BOUNDS = SurfaceBounds()

DEFAULT_LBFGSB_OPTIONS = LBFGSBOptions()
