# tests/models/test_survival_spatial.py

from __future__ import annotations

import numpy as np
import pytest

from numpyro.infer import MCMC, NUTS

from seer_peph.data.model_data import make_model_data
from seer_peph.data.prep import build_survival_long, build_treatment_long
from seer_peph.graphs import make_ring_lattice
from seer_peph.models.survival_spatial import model
from seer_peph.validation.simulate import simulate_joint


def _make_small_model_data(seed: int = 123):
    """
    Build a small but nondegenerate dataset via the actual
    simulate -> prep -> model_data pipeline.
    """
    graph = make_ring_lattice(A=8, k=4)

    wide = simulate_joint(
        graph,
        n_per_area=30,
        seed=seed,
    )

    df = wide.copy()
    df["time_m"] = df["time"] / 30.4375
    df["treatment_time_m"] = df["treatment_time"] / 30.4375
    df["treatment_time_obs_m"] = df["treatment_time_obs"] / 30.4375
    df["sex_male"] = (df["sex"] == "M").astype(np.int8)
    df["stage_II"] = (df["stage"] == "II").astype(np.int8)
    df["stage_III"] = (df["stage"] == "III").astype(np.int8)

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


def test_survival_spatial_mcmc_runs_and_returns_expected_sample_shapes():
    import jax.random as random

    data, _, _, graph = _make_small_model_data(seed=101)

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
    assert "rho" in samples
    assert "tau" in samples
    assert "eps" in samples
    assert "s_free" in samples
    assert "s" in samples
    assert "u" in samples

    K_surv = int(np.asarray(data["k_surv"]).max()) + 1
    P_surv = int(data["P_surv"])
    A = graph.A

    assert samples["alpha"].shape == (50, K_surv)
    assert samples["beta"].shape == (50, P_surv)
    assert samples["beta_td"].shape == (50,)
    assert samples["delta_post"].ndim == 2
    assert samples["delta_post"].shape[0] == 50

    assert samples["rho"].shape == (50,)
    assert samples["tau"].shape == (50,)
    assert samples["eps"].shape == (50, A)
    assert samples["s_free"].shape == (50, A - 1)
    assert samples["s"].shape == (50, A)
    assert samples["u"].shape == (50, A)

    assert np.allclose(samples["delta_post"][:, 0], 0.0)


def test_survival_spatial_spatial_parameters_have_expected_domain():
    import jax.random as random

    data, _, _, graph = _make_small_model_data(seed=202)

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

    rho = np.asarray(samples["rho"])
    tau = np.asarray(samples["tau"])
    s = np.asarray(samples["s"])
    u = np.asarray(samples["u"])

    assert np.all(np.isfinite(rho))
    assert np.all(np.isfinite(tau))
    assert np.all(np.isfinite(s))
    assert np.all(np.isfinite(u))

    assert np.all((rho > 0.0) & (rho < 1.0))
    assert np.all(tau > 0.0)

    assert s.shape[1] == graph.A
    assert u.shape[1] == graph.A


def test_survival_spatial_sum_to_zero_constraint_holds_for_s():
    import jax.random as random

    data, _, _, _ = _make_small_model_data(seed=303)

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
    s = np.asarray(samples["s"])

    # Allow small numerical tolerance
    assert np.allclose(s.sum(axis=1), 0.0, atol=1e-6)


def test_survival_spatial_log_likelihood_path_is_non_degenerate():
    import jax.random as random

    data, surv_long, _, _ = _make_small_model_data(seed=404)

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
    mcmc.run(random.PRNGKey(3), data)

    samples = mcmc.get_samples()
    for name in ["alpha", "beta", "beta_td", "delta_post", "rho", "tau", "s", "u"]:
        arr = np.asarray(samples[name])
        assert np.isfinite(arr).all(), f"Non-finite samples in {name}"


def test_survival_spatial_accepts_minimal_custom_survival_design():
    import jax.random as random

    graph = make_ring_lattice(A=6, k=4)
    wide = simulate_joint(graph, n_per_area=20, seed=505)

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
    mcmc.run(random.PRNGKey(4), data)

    samples = mcmc.get_samples()
    assert samples["beta"].shape == (25, 2)
    assert samples["u"].shape == (25, graph.A)


def test_survival_spatial_raises_on_bad_y_values():
    data, _, _, _ = _make_small_model_data(seed=606)
    bad = dict(data)
    y = np.asarray(bad["y_surv"]).copy()
    y[0] = 2
    bad["y_surv"] = y

    with pytest.raises(ValueError, match="y_surv must contain only 0/1 values"):
        model(bad)


def test_survival_spatial_raises_on_bad_treated_td_values():
    data, _, _, _ = _make_small_model_data(seed=707)
    bad = dict(data)
    treated = np.asarray(bad["treated_td"]).copy()
    treated[0] = 2
    bad["treated_td"] = treated

    with pytest.raises(ValueError, match="treated_td must contain only 0/1 values"):
        model(bad)


def test_survival_spatial_raises_when_untreated_row_has_nonnegative_k_post():
    data, _, _, _ = _make_small_model_data(seed=808)
    bad = dict(data)

    treated = np.asarray(bad["treated_td"])
    idx = np.where(treated == 0)[0]
    assert len(idx) > 0, "Need at least one untreated row for this test"

    k_post = np.asarray(bad["k_post"]).copy()
    k_post[idx[0]] = 0
    bad["k_post"] = k_post

    with pytest.raises(ValueError, match="Rows with treated_td == 0 must have k_post == -1"):
        model(bad)


def test_survival_spatial_raises_when_treated_row_has_negative_k_post():
    data, _, _, _ = _make_small_model_data(seed=909)
    bad = dict(data)

    treated = np.asarray(bad["treated_td"])
    idx = np.where(treated == 1)[0]
    assert len(idx) > 0, "Need at least one treated row for this test"

    k_post = np.asarray(bad["k_post"]).copy()
    k_post[idx[0]] = -1
    bad["k_post"] = k_post

    with pytest.raises(ValueError, match="Rows with treated_td == 1 must have k_post >= 0"):
        model(bad)


def test_survival_spatial_raises_on_shape_mismatch():
    data, _, _, _ = _make_small_model_data(seed=1001)
    bad = dict(data)

    X = np.asarray(bad["X_surv"])
    bad["X_surv"] = X[:-1, :]

    with pytest.raises(ValueError, match="X_surv must have shape"):
        model(bad)


def test_survival_spatial_raises_when_area_id_out_of_range():
    data, _, _, _ = _make_small_model_data(seed=1101)
    bad = dict(data)

    area = np.asarray(bad["area_id_surv"]).copy()
    area[0] = int(data["A"])
    bad["area_id_surv"] = area

    with pytest.raises(ValueError, match="area_id_surv must be < A"):
        model(bad)


def test_survival_spatial_raises_when_graph_edges_are_invalid():
    data, _, _, _ = _make_small_model_data(seed=1201)
    bad = dict(data)

    node1 = np.asarray(bad["node1"]).copy()
    node1[0] = int(data["A"])
    bad["node1"] = node1

    with pytest.raises(ValueError, match="node1 and node2 entries must be < A"):
        model(bad)


def test_survival_spatial_raises_when_scaling_factor_is_nonpositive():
    data, _, _, _ = _make_small_model_data(seed=1301)
    bad = dict(data)
    bad["scaling_factor"] = np.array(0.0)

    with pytest.raises(ValueError, match="scaling_factor must be finite and > 0"):
        model(bad)