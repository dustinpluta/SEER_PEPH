# tests/models/test_survival_only.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numpyro.infer import MCMC, NUTS

from seer_peph.data.model_data import make_model_data
from seer_peph.data.prep import build_survival_long, build_treatment_long
from seer_peph.graphs import make_ring_lattice
from seer_peph.models.survival_only import model
from seer_peph.validation.simulate import simulate_joint


def _make_small_model_data(seed: int = 123):
    """
    Build a small but nondegenerate survival dataset by running the actual
    simulate -> prep -> model_data pipeline.
    """
    graph = make_ring_lattice(A=8, k=4)

    wide = simulate_joint(
        graph,
        n_per_area=30,
        seed=seed,
    )

    # Mimic load_and_encode() fields directly from simulated wide data.
    df = wide.copy()
    df["time_m"] = df["time"] / 30.4375
    df["treatment_time_m"] = df["treatment_time"] / 30.4375
    df["treatment_time_obs_m"] = df["treatment_time_obs"] / 30.4375
    df["sex_male"] = (df["sex"] == "M").astype(np.int8)
    df["stage_II"] = (df["stage"] == "II").astype(np.int8)
    df["stage_III"] = (df["stage"] == "III").astype(np.int8)

    # The simulator emits one ZIP per area in sorted order, so this reproduces
    # the prep.py area_id convention.
    area_map = {z: i for i, z in enumerate(sorted(df["zip"].unique()))}
    df["area_id"] = df["zip"].map(area_map).astype(np.int16)

    surv_long = build_survival_long(df)
    ttt_long = build_treatment_long(df)

    data = make_model_data(
        surv_long,
        ttt_long,
        graph,
        as_jax=True,
    )
    return data, surv_long, ttt_long, graph


def test_survival_only_mcmc_runs_and_returns_expected_sample_shapes():
    import jax.random as random

    data, surv_long, _, _ = _make_small_model_data(seed=101)

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=50,
        num_samples=50,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(0), data)

    samples = mcmc.get_samples()

    assert "alpha" in samples
    assert "beta" in samples
    assert "beta_td" in samples
    assert "delta_post" in samples

    K_surv = int(np.asarray(data["k_surv"]).max()) + 1
    P_surv = int(data["P_surv"])

    assert samples["alpha"].shape == (50, K_surv)
    assert samples["beta"].shape == (50, P_surv)
    assert samples["beta_td"].shape == (50,)
    assert samples["delta_post"].ndim == 2
    assert samples["delta_post"].shape[0] == 50

    # Deterministic delta_post should always include the acute reference interval.
    assert np.allclose(samples["delta_post"][:, 0], 0.0)


def test_survival_only_log_likelihood_path_is_non_degenerate():
    import jax.random as random

    data, surv_long, _, _ = _make_small_model_data(seed=202)

    # Ensure the test problem is nontrivial.
    assert int(np.asarray(data["y_surv"]).sum()) > 0
    assert int(np.asarray(data["treated_td"]).sum()) > 0
    assert len(surv_long) > 0

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=30,
        num_samples=30,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(1), data)

    samples = mcmc.get_samples()

    # Posterior draws should be finite.
    for name in ["alpha", "beta", "beta_td", "delta_post"]:
        arr = np.asarray(samples[name])
        assert np.isfinite(arr).all(), f"Non-finite samples in {name}"


def test_survival_only_accepts_minimal_custom_survival_design():
    import jax.random as random

    graph = make_ring_lattice(A=6, k=4)
    wide = simulate_joint(graph, n_per_area=20, seed=303)

    df = wide.copy()
    df["time_m"] = df["time"] / 30.4375
    df["treatment_time_m"] = df["treatment_time"] / 30.4375
    df["treatment_time_obs_m"] = df["treatment_time_obs"] / 30.4375
    df["sex_male"] = (df["sex"] == "M").astype(np.int8)
    df["stage_II"] = (df["stage"] == "II").astype(np.int8)
    df["stage_III"] = (df["stage"] == "III").astype(np.int8)
    area_map = {z: i for i, z in enumerate(sorted(df["zip"].unique()))}
    df["area_id"] = df["zip"].map(area_map).astype(np.int16)

    surv_long = build_survival_long(
        df,
        x_cols=["area_id", "age_per10_centered", "stage_III"],
    )
    ttt_long = build_treatment_long(df)

    data = make_model_data(
        surv_long,
        ttt_long,
        graph,
        surv_x_cols=["age_per10_centered", "stage_III"],
        as_jax=True,
    )

    assert data["P_surv"] == 2

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=25,
        num_samples=25,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(2), data)

    samples = mcmc.get_samples()
    assert samples["beta"].shape == (25, 2)


def test_survival_only_raises_on_bad_y_values():
    data, _, _, _ = _make_small_model_data(seed=404)
    bad = dict(data)
    y = np.asarray(bad["y_surv"]).copy()
    y[0] = 2
    bad["y_surv"] = y

    with pytest.raises(ValueError, match="y_surv must contain only 0/1"):
        model(bad)


def test_survival_only_raises_on_bad_treated_td_values():
    data, _, _, _ = _make_small_model_data(seed=405)
    bad = dict(data)
    treated = np.asarray(bad["treated_td"]).copy()
    treated[0] = 2
    bad["treated_td"] = treated

    with pytest.raises(ValueError, match="treated_td must contain only 0/1"):
        model(bad)


def test_survival_only_raises_when_untreated_row_has_nonnegative_k_post():
    data, _, _, _ = _make_small_model_data(seed=406)
    bad = dict(data)

    treated = np.asarray(bad["treated_td"])
    idx = np.where(treated == 0)[0]
    assert len(idx) > 0, "Need at least one untreated row for this test"

    k_post = np.asarray(bad["k_post"]).copy()
    k_post[idx[0]] = 0
    bad["k_post"] = k_post

    with pytest.raises(ValueError, match="treated_td == 0 must have k_post == -1"):
        model(bad)


def test_survival_only_raises_when_treated_row_has_negative_k_post():
    data, _, _, _ = _make_small_model_data(seed=407)
    bad = dict(data)

    treated = np.asarray(bad["treated_td"])
    idx = np.where(treated == 1)[0]
    assert len(idx) > 0, "Need at least one treated row for this test"

    k_post = np.asarray(bad["k_post"]).copy()
    k_post[idx[0]] = -1
    bad["k_post"] = k_post

    with pytest.raises(ValueError, match="treated_td == 1 must have k_post >= 0"):
        model(bad)


def test_survival_only_raises_on_shape_mismatch():
    data, _, _, _ = _make_small_model_data(seed=408)
    bad = dict(data)

    X = np.asarray(bad["X_surv"])
    bad["X_surv"] = X[:-1, :]  # wrong N

    with pytest.raises(ValueError, match="X_surv must have shape"):
        model(bad)