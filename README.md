# vpnls

[![CI](https://github.com/Open-Athena/vpnls/actions/workflows/ci.yml/badge.svg)](https://github.com/Open-Athena/vpnls/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Open-Athena/vpnls/branch/main/graph/badge.svg)](https://codecov.io/gh/Open-Athena/vpnls)

Variable Projection with Nonlinear Least Squares (VPNLS) for fitting compute-optimal scaling laws of the form:

$$L(N, D) = E + A \cdot N^{-\alpha} + B \cdot D^{-\beta}$$

## Installation

```bash
pip install vpnls            # Grid solver only (Cython, no runtime deps)
pip install vpnls[scipy]     # + scipy L-BFGS-B solver
pip install vpnls[jax]       # + JAX autodiff solver
```

## Solvers

| Method | Backend | Use case |
|--------|---------|----------|
| `grid` | Cython grid search | Fast brute-force search over exponent space; supports multiprocess parallelism |
| `scipy` | L-BFGS-B with analytical gradients | Continuous optimization via envelope-theorem gradients, as in the [VPNLS paper](https://arxiv.org/abs/2503.04715) |
| `jax` | JAX autodiff + L-BFGS-B | Same approach as [ml-scalefit](https://github.com/apple/ml-scalefit); autodiff through `lstsq` instead of hand-derived gradients |

All three support MSE and Huber loss.

## Usage

```python
import numpy as np
from vpnls.api import fit_vpnls, simulate_isoflop_data

# Generate synthetic data (8 budgets x 16 points = 128 samples)
N, D, L = simulate_isoflop_data(
    alpha=0.34, beta=0.28, A=406.4, B=410.7, E=1.69,
    compute_budgets=np.geomspace(1e17, 1e22, 8), n_points_per_budget=16, noise_std=0,
)

# 2-digit exponent (alpha/beta) precision (~25ms)
result = fit_vpnls(N, D, L, method="grid", resolution=0.01)

# 3-digit precision, 10 processes (~250ms on M4 Pro; 4-digit takes ~16s)
result = fit_vpnls(N, D, L, method="grid", resolution=0.001, num_workers=10)
# -> alpha=0.34, beta=0.28, E=1.6900, A=406.40, B=410.70

# L-BFGS-B refinement from dense grid search above
result = fit_vpnls(N, D, L, method="jax")  # or "scipy"
```
