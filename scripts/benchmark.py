"""Benchmark grid solver: serial vs parallel across resolutions."""

import time

import numpy as np

from vpnls.grid import fit_vpnls_grid
from vpnls.sim import generate_isoflop_data
from vpnls.types import LossSurface

# Simulated data: 8 compute budgets × 16 points = 128 data points
surface = LossSurface(alpha=0.34, beta=0.28, A=406.4, B=410.7, E=1.69)
budgets = np.geomspace(1e17, 1e22, 8)
N, D, L = generate_isoflop_data(
    surface, noise_std=0.002, compute_budgets=budgets, n_points_per_budget=16
)

# Log-spaced resolutions: 0.01 to 0.0001 (2 decades, 11 points)
resolutions = np.geomspace(0.01, 0.0001, 11)
num_workers = 10

print(f"Data: {len(N)} points (8 budgets × 16 pts), Chinchilla surface, noise_std=0.002")
print(f"Workers: {num_workers}")
print()
print(f"{'res':>10s} {'grid pts':>12s} {'serial':>10s} {f'w={num_workers}':>10s} {'speedup':>8s}")
print("-" * 56)

for res in resolutions:
    n_alpha = int((0.99 - 0.01) / res) + 1
    grid_pts = n_alpha * n_alpha

    t0 = time.perf_counter()
    fit_vpnls_grid(N, D, L, resolution=res, num_workers=1)
    serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    fit_vpnls_grid(N, D, L, resolution=res, num_workers=num_workers)
    parallel = time.perf_counter() - t0

    speedup = serial / parallel

    def fmt(t):
        return f"{t * 1000:.0f}ms" if t < 1 else f"{t:.1f}s"

    print(f"{res:>10.5f} {grid_pts:>12,d} {fmt(serial):>10s} {fmt(parallel):>10s} {speedup:>7.1f}x")

    if parallel > 30:
        print("Stopping — parallel exceeded 30s")
        break

# Results from Apple M4 Pro (14 cores, 24 GB RAM), 128 data points:
#
# | Resolution | Grid Points |  Serial | w=10   | Speedup |
# |------------|-------------|---------|--------|---------|
# |    0.01000 |       9,801 |    12ms |  103ms |    0.1x |
# |    0.00631 |      24,336 |    28ms |   80ms |    0.3x |
# |    0.00398 |      61,009 |    72ms |   85ms |    0.8x |
# |    0.00251 |     152,881 |   180ms |   98ms |    1.8x |
# |    0.00158 |     383,161 |   442ms |  134ms |    3.3x |
# |    0.00100 |     962,361 |    1.1s |   0.2s |    4.8x |
# |    0.00063 |   2,414,916 |    2.8s |   0.5s |    5.9x |
# |    0.00040 |   6,061,444 |    7.1s |   1.1s |    6.6x |
# |    0.00025 |  15,225,604 |   17.9s |   2.6s |    6.8x |
# |    0.00016 |  38,241,856 |   45.7s |   6.6s |    7.0x |
# |    0.00010 |  96,059,601 | 116.5s |  16.1s |    7.2x |
#
# Crossover at ~150K grid points (resolution ≈ 0.0025).
# Speedup plateaus at ~7x due to process spawn + data serialization overhead.
# Default resolution=0.01 (~10K pts) is <15ms serial — no need for parallelism.
# Fine resolution=0.001 (~1M pts) benefits most: 1.1s → 0.2s.
