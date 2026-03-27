"""Reproduce README usage examples and comparisons.

Usage:
    uv run --with datasets --with joblib --with /path/to/ml-scalefit python scripts/usage.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from datasets import load_dataset
from scalefit import ScalingLaw
from scalefit.optim import huber_loss as sf_huber_loss

from vpnls.api import bounds, fit_vpnls, huber

EXPERIMENTS = {
    "ml_scalefit__massivetext__chinchilla": "chinchilla",
    "llama_3__raw_loss": "llama3",
    "marin_202603__comma__llama_2": "marin/comma",
    "marin_202603__dclm__llama_2": "marin/dclm",
    "marin_202603__nemotron__llama_2": "marin/nemotron",
}

SHARED_BOUNDS = {
    "E": (0.0, 5.0),
    "A": (0.0, 10.0),
    "B": (0.0, 10.0),
    "alpha": (0.01, 1.0),
    "beta": (0.01, 1.0),
}

HUBER_DELTA = 1e-3


def load_experiment(df: pd.DataFrame, experiment: str):
    """Extract normalized (N in millions, D in billions) arrays."""
    data = df[df["experiment"] == experiment]
    return (
        data["params"].values.copy() / 1e6,
        data["tokens"].values.copy() / 1e9,
        data["loss"].values.copy(),
    )


def predict(params, N, D):
    return params["E"] + params["A"] / N ** params["alpha"] + params["B"] / D ** params["beta"]


def huber_objective(y_true, y_pred, delta):
    """Evaluate Huber loss using scalefit's implementation."""
    import jax.numpy as jnp

    return float(sf_huber_loss(jnp.array(y_true), jnp.array(y_pred), delta=delta))


def sf_model(params, inputs):
    return (
        params["E"]
        + params["A"] / inputs["N"] ** params["alpha"]
        + params["B"] / inputs["D"] ** params["beta"]
    )


# ── Quick-start example ─────────────────────────────────────────────────────


def quickstart(df: pd.DataFrame):
    """Minimal vpnls usage example."""
    N, D, L = load_experiment(df, "ml_scalefit__massivetext__chinchilla")
    r = fit_vpnls(N, D, L)

    print("Quick start: fit Chinchilla data with vpnls")
    print(f"  α={r.alpha:.4f}  β={r.beta:.4f}  E={r.E:.4f}  A={r.A:.4f}  B={r.B:.4f}")
    print()


# ── vpnls vs scalefit comparison ────────────────────────────────────────────


TIME_REPS = 11  # first run discarded as warmup


def compare(df: pd.DataFrame):
    """Compare vpnls and scalefit across all experiments."""
    rows = []

    for experiment, short in EXPERIMENTS.items():
        N, D, L = load_experiment(df, experiment)
        inputs = pd.DataFrame({"N": N, "D": D})
        targets = pd.Series(L)

        # vpnls — use first run for results, average rest for timing
        vp_times = []
        for i in range(TIME_REPS):
            t0 = time.perf_counter()
            vp = fit_vpnls(N, D, L, loss=huber(HUBER_DELTA), bounds=bounds(**SHARED_BOUNDS))
            vp_times.append(time.perf_counter() - t0)

        # scalefit
        sf_times = []
        for i in range(TIME_REPS):
            sf = ScalingLaw(
                model_fn=sf_model,
                bounds=SHARED_BOUNDS,
                loss="huber",
                loss_kwargs={"delta": HUBER_DELTA},
                seed=42,
                n_bootstraps=1,
            )
            t0 = time.perf_counter()
            sf.fit(inputs, targets)
            sf_times.append(time.perf_counter() - t0)

        # Evaluate both with same prediction function (first run results)
        vp_p = {"E": vp.E, "A": vp.A, "B": vp.B, "alpha": vp.alpha, "beta": vp.beta}
        sf_p = {k: float(v) for k, v in sf.optimal_params_.items()}
        vp_h = huber_objective(L, predict(vp_p, N, D), HUBER_DELTA)
        sf_h = huber_objective(L, predict(sf_p, N, D), HUBER_DELTA)

        rows.append((short, len(L), vp_h, sf_h, vp_times, sf_times))

    # Print summary table
    scale = 1e6
    huber_w = 8 + 2 + 8 + 2 + 10
    n_avg = TIME_REPS - 1

    def print_table(label, get_vp_t, get_sf_t):
        print(f"{'':>16s}  {'':>4s}  {'── Huber loss ':─<{huber_w}s}  ── {label} ")
        print(
            f"{'experiment':>16s}  {'n':>4s}  {'vpnls':>8s}  {'scalefit':>8s}  "
            f"{'Δ':>10s}  {'vpnls':>6s}  {'scalefit':>8s}  {'speedup':>7s}"
        )
        w = [16, 4, 8, 8, 10, 6, 8, 7]
        sep = "  ".join("─" * n for n in w)
        print(sep)
        for name, n, vp_h, sf_h, vp_ts, sf_ts in rows:
            vp_t, sf_t = get_vp_t(vp_ts), get_sf_t(sf_ts)
            print(
                f"{name:>16s}  {n:4d}  {vp_h * scale:8.1f}  {sf_h * scale:8.1f}  "
                f"{(vp_h - sf_h) * scale:+10.2f}  {vp_t:6.3f}  {sf_t:8.3f}  {sf_t / vp_t:6.1f}x"
            )

        def tot(i):
            return sum(r[i] for r in rows)

        tot_vp = sum(get_vp_t(r[4]) for r in rows)
        tot_sf = sum(get_sf_t(r[5]) for r in rows)
        print(sep)
        h_diff = (tot(2) - tot(3)) * scale
        print(
            f"{'total':>16s}  {tot(1):4d}  {tot(2) * scale:8.1f}  {tot(3) * scale:8.1f}  "
            f"{h_diff:+10.2f}  {tot_vp:6.3f}  {tot_sf:8.3f}  {tot_sf / tot_vp:6.1f}x"
        )

    print(f"vpnls vs scalefit (Huber δ={HUBER_DELTA}, loss ×10⁶)")
    print()
    print_table("Time: first run (s) ", lambda ts: ts[0], lambda ts: ts[0])
    print()
    print_table(
        f"Time: avg of {n_avg} runs (s) ", lambda ts: np.mean(ts[1:]), lambda ts: np.mean(ts[1:])
    )
    print()


def main():
    ds = load_dataset("open-athena/isoflop-experiments", split="train")
    df = ds.to_pandas()
    quickstart(df)
    compare(df)


if __name__ == "__main__":
    main()
