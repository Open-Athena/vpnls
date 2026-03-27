# vpnls

[![arXiv](https://img.shields.io/badge/arXiv-2603.22339-b31b1b.svg)](https://arxiv.org/abs/2603.22339)
[![CI](https://github.com/Open-Athena/vpnls/actions/workflows/ci.yml/badge.svg)](https://github.com/Open-Athena/vpnls/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Open-Athena/vpnls/graph/badge.svg?token=6GD640V7I7)](https://codecov.io/gh/Open-Athena/vpnls)

Variable Projection with Nonlinear Least Squares (VPNLS) for fitting compute-optimal scaling laws of the form:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

## Installation

```bash
pip install vpnls            # Grid solver only (Cython, no runtime deps)
pip install vpnls[scipy]     # + scipy L-BFGS-B solver
pip install vpnls[jax]       # + JAX autodiff solver
```

## Solvers

| Method | Backend | Use case |
|--------|---------|----------|
| `grid` | Cython grid search | Fast brute-force search over 2D exponent space for 5D inference; supports multiprocess parallelism |
| `scipy` | L-BFGS-B with analytical gradients | 2D continuous optimization via envelope-theorem gradients, as in the [VPNLS paper](https://arxiv.org/abs/2603.22339) |
| `jax` | JAX autodiff + L-BFGS-B | Same 2D optimization as scipy but with autodiff through `lstsq`; same approach as [ml-scalefit](https://github.com/apple/ml-scalefit) |

All three support MSE and Huber loss.

## Usage

### Simulated data

```python
import numpy as np
from vpnls.api import fit_vpnls, simulate_isoflop_data

# Generate synthetic IsoFLOP data (8 budgets × 16 points = 128 samples)
N, D, L = simulate_isoflop_data(
    alpha=0.34, beta=0.28, A=406.4, B=410.7, E=1.69,
    compute_budgets=np.geomspace(1e17, 1e22, 8), n_points_per_budget=16, noise_std=0,
)

# 2-digit exponent (alpha/beta) precision (~25ms)
result = fit_vpnls(N, D, L, resolution=0.01)
# -> alpha=0.34, beta=0.28, E=1.6900, A=406.40, B=410.70 (recovery is exact)

# 3-digit precision, 10 processes (~250ms on M4 Pro; 4-digit takes ~16s)
result = fit_vpnls(N, D, L, resolution=0.001, num_workers=10)
# -> alpha=0.340, beta=0.280, E=1.6900, A=406.40, B=410.70 (still exact)
```

Passing `method="jax"` or `method="scipy"` runs L-BFGS-B refinement from the dense grid search results above. This is not typically necessary on real data and included here primarily for parity with other methods.

### Real data

Fit Chinchilla scaling law parameters on data from [open-athena/isoflop-experiments](https://huggingface.co/datasets/open-athena/isoflop-experiments):

```python
from datasets import load_dataset
from vpnls.api import fit_vpnls, huber

df = load_dataset("open-athena/isoflop-experiments", split="train").to_pandas()
data = df[df["experiment"] == "ml_scalefit__massivetext__chinchilla"]

N = data["params"].values.copy() / 1e6   # normalize to millions
D = data["tokens"].values.copy() / 1e9   # normalize to billions
L = data["loss"].values.copy()

result = fit_vpnls(N, D, L, resolution=0.01, loss=huber(1e-3))
# -> alpha=0.3600, beta=0.4000, E=1.8608, A=4.1479, B=1.0546
```

Comparison to [ml-scalefit](https://github.com/apple/ml-scalefit) across 5 experiments from the same dataset showing better fits in less time:

```
                        ── Huber loss (×10⁶) ─────────  ── Time: first run (s)
      experiment     n     vpnls  scalefit           Δ   vpnls  scalefit  speedup
────────────────  ────  ────────  ────────  ──────────  ──────  ────────  ───────
      chinchilla   124      13.1      13.4       -0.35   0.106     1.672    15.8x
          llama3   133       2.3       2.4       -0.16   0.117     0.536     4.6x
     marin/comma    85      22.9      23.5       -0.63   0.074     0.474     6.4x
      marin/dclm    85      22.9      23.4       -0.45   0.076     0.288     3.8x
  marin/nemotron    88      22.6      24.2       -1.57   0.079     0.486     6.1x
────────────────  ────  ────────  ────────  ──────────  ──────  ────────  ───────
           total   515      83.8      87.0       -3.16   0.453     3.457     7.6x
```

<details>
<summary>Avg of 10 subsequent runs (7.6x → 3.2x speedup, scalefit loss averaged across seeds)</summary>

```
                        ── Huber loss (×10⁶) ─────────  ── Time: avg of 10 runs (s)
      experiment     n     vpnls  scalefit           Δ   vpnls  scalefit  speedup
────────────────  ────  ────────  ────────  ──────────  ──────  ────────  ───────
      chinchilla   124      13.1      13.6       -0.49   0.107     0.386     3.6x
          llama3   133       2.3       2.4       -0.10   0.117     0.325     2.8x
     marin/comma    85      22.9      24.2       -1.34   0.075     0.256     3.4x
      marin/dclm    85      22.9      23.5       -0.57   0.076     0.249     3.3x
  marin/nemotron    88      22.6      23.8       -1.21   0.079     0.252     3.2x
────────────────  ────  ────────  ────────  ──────────  ──────  ────────  ───────
           total   515      83.8      87.5       -3.70   0.453     1.467     3.2x
```

</details>

Both minimize Huber loss (δ=0.001); evaluation uses [`scalefit.optim.huber_loss`](https://github.com/apple/ml-scalefit/blob/ac4664af5db6c94e6ac7521a61dd3bbb0d91cc3a/src/scalefit/optim.py#L88-L106). Neither method uses bootstrap resamples, which is one way these marginal execution times can start to matter. See [scripts/usage.py](scripts/usage.py) to reproduce.

## Citation

```bibtex
@article{openathena2026approach2,
  title={Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits},
  author={Czech, Eric and Xu, Zhiwei and Elmatad, Yael and Wang, Yixin and Held, William},
  journal={arXiv preprint arXiv:2603.22339},
  year={2026}
}
```
