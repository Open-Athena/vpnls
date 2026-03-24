import pytest

from vpnls.types import IsoFlopExperiment, LossSurface

NOISELESS = IsoFlopExperiment(noise_std=0.0)

SYMMETRIC = LossSurface(alpha=0.31, beta=0.31, A=400, B=400, E=1.69)
CHINCHILLA = LossSurface(alpha=0.34, beta=0.28, A=406.4, B=410.7, E=1.69)
ASYMMETRIC = LossSurface(alpha=0.50, beta=0.20, A=200, B=800, E=1.50)

SURFACES = [
    pytest.param(SYMMETRIC, id="symmetric"),
    pytest.param(CHINCHILLA, id="chinchilla"),
    pytest.param(ASYMMETRIC, id="asymmetric"),
]
