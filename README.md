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
                        ── Huber loss ────────────────  ── Time: first run (s)
      experiment     n     vpnls  scalefit           Δ   vpnls  scalefit  speedup
────────────────  ────  ────────  ────────  ──────────  ──────  ────────  ───────
      chinchilla   124      13.1      13.2       -0.13   0.103     1.711    16.6x
          llama3   133       2.3       2.4       -0.08   0.113     0.628     5.5x
     marin/comma    85      22.9      23.4       -0.51   0.073     0.490     6.7x
      marin/dclm    85      22.9      25.0       -2.07   0.075     0.257     3.4x
  marin/nemotron    88      22.6      24.2       -1.59   0.079     0.476     6.0x
────────────────  ────  ────────  ────────  ──────────  ──────  ────────  ───────
           total   515      83.8      88.2       -4.39   0.443     3.561     8.0x
```

<details>
<summary>Avg of 10 subsequent runs (8.0x → 3.5x speedup)</summary>

```
                        ── Huber loss ────────────────  ── Time: avg of 10 runs (s)
      experiment     n     vpnls  scalefit           Δ   vpnls  scalefit  speedup
────────────────  ────  ────────  ────────  ──────────  ──────  ────────  ───────
      chinchilla   124      13.1      13.2       -0.13   0.104     0.390     3.7x
          llama3   133       2.3       2.4       -0.08   0.113     0.388     3.4x
     marin/comma    85      22.9      23.4       -0.51   0.073     0.257     3.5x
      marin/dclm    85      22.9      25.0       -2.07   0.074     0.243     3.3x
  marin/nemotron    88      22.6      24.2       -1.59   0.077     0.265     3.4x
────────────────  ────  ────────  ────────  ──────────  ──────  ────────  ───────
           total   515      83.8      88.2       -4.39   0.441     1.542     3.5x
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
