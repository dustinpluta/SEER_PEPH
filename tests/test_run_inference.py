# tests/test_run_inference.py

from __future__ import annotations

import numpy as np
import pytest
from numpyro.infer import init_to_median

from seer_peph.data.model_data import make_model_data
from seer_peph.data.prep import build_survival_long, build_treatment_long
from seer_peph.graphs import make_ring_lattice
from seer_peph.inference.run import (
    InferenceConfig,
    InferenceResult,
    print_summary,
    run_mcmc,
    summarise_samples,
)
from seer_peph.models.survival_only import model as survival_only_model
from seer_peph.validation.simulate import simulate_joint


def _build_model_data(seed: int = 123):
    graph = make_ring_lattice(A=8, k=4)

    wide = simulate_joint(
        graph,
        n_per_area=25,
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
    return data


def test_run_mcmc_returns_inference_result_with_expected_fields():
    import jax.random as random

    data = _build_model_data(seed=101)
    config = InferenceConfig(
        num_chains=1,
        num_warmup=30,
        num_samples=30,
        progress_bar=False,
    )

    result = run_mcmc(
        survival_only_model,
        data,
        rng_key=random.PRNGKey(0),
        config=config,
    )

    assert isinstance(result, InferenceResult)
    assert result.config == config
    assert "alpha" in result.samples
    assert "beta" in result.samples
    assert "beta_td" in result.samples
    assert "delta_post" in result.samples

    assert isinstance(result.summary, dict)
    assert "alpha" in result.summary
    assert "beta" in result.summary
    assert "beta_td" in result.summary


def test_run_mcmc_sample_shapes_match_config_and_model_data():
    import jax.random as random

    data = _build_model_data(seed=102)
    config = InferenceConfig(
        num_chains=1,
        num_warmup=25,
        num_samples=25,
        progress_bar=False,
    )

    result = run_mcmc(
        survival_only_model,
        data,
        rng_key=random.PRNGKey(1),
        config=config,
    )

    samples = result.samples
    K_surv = int(np.asarray(data["k_surv"]).max()) + 1
    P_surv = int(data["P_surv"])

    assert samples["alpha"].shape == (config.num_samples, K_surv)
    assert samples["beta"].shape == (config.num_samples, P_surv)
    assert samples["beta_td"].shape == (config.num_samples,)
    assert samples["delta_post"].shape[0] == config.num_samples

    # Acute reference interval is fixed at 0 in survival_only.py
    assert np.allclose(np.asarray(samples["delta_post"])[:, 0], 0.0)


def test_run_mcmc_with_init_strategy_runs():
    import jax.random as random

    data = _build_model_data(seed=103)
    config = InferenceConfig(
        num_chains=1,
        num_warmup=20,
        num_samples=20,
        progress_bar=False,
    )

    result = run_mcmc(
        survival_only_model,
        data,
        rng_key=random.PRNGKey(2),
        config=config,
        init_strategy=init_to_median(),
    )

    assert "beta_td" in result.samples
    assert np.isfinite(np.asarray(result.samples["beta_td"])).all()


def test_summarise_samples_returns_expected_structure():
    samples = {
        "beta_td": np.array([0.1, 0.2, 0.3, 0.4]),
        "beta": np.array(
            [
                [1.0, 2.0],
                [1.5, 2.5],
                [2.0, 3.0],
                [2.5, 3.5],
            ]
        ),
    }

    out = summarise_samples(samples)

    assert "beta_td" in out
    assert "beta[0]" in out
    assert "beta[1]" in out

    for key in ["mean", "sd", "median", "q05", "q95"]:
        assert key in out["beta_td"]
        assert key in out["beta[0]"]

    assert out["beta_td"]["mean"] == pytest.approx(0.25)
    assert out["beta[0]"]["median"] == pytest.approx(1.75)
    assert out["beta[1]"]["median"] == pytest.approx(2.75)


def test_summarise_samples_handles_scalar_like_1d_samples():
    samples = {
        "alpha0": np.array([1.0, 2.0, 3.0]),
    }

    out = summarise_samples(samples)

    assert list(out.keys()) == ["alpha0"]
    assert out["alpha0"]["mean"] == pytest.approx(2.0)
    assert out["alpha0"]["median"] == pytest.approx(2.0)


def test_print_summary_runs_without_error(capsys):
    summary_dict = {
        "beta_td": {
            "mean": 0.10,
            "sd": 0.02,
            "median": 0.11,
            "q05": 0.07,
            "q95": 0.13,
        },
        "beta[0]": {
            "mean": -0.20,
            "sd": 0.05,
            "median": -0.19,
            "q05": -0.28,
            "q95": -0.12,
        },
    }

    print_summary(summary_dict)
    captured = capsys.readouterr()

    assert "parameter" in captured.out
    assert "beta_td" in captured.out
    assert "beta[0]" in captured.out


def test_print_summary_handles_empty_summary(capsys):
    print_summary({})
    captured = capsys.readouterr()
    assert "(empty summary)" in captured.out


def test_inference_config_validation_rejects_bad_values():
    import jax.random as random

    data = _build_model_data(seed=104)

    bad_configs = [
        InferenceConfig(num_chains=0),
        InferenceConfig(num_warmup=0),
        InferenceConfig(num_samples=0),
        InferenceConfig(target_accept_prob=0.0),
        InferenceConfig(target_accept_prob=1.0),
        InferenceConfig(max_tree_depth=0),
    ]

    for cfg in bad_configs:
        with pytest.raises(ValueError):
            run_mcmc(
                survival_only_model,
                data,
                rng_key=random.PRNGKey(3),
                config=cfg,
            )


def test_run_mcmc_rejects_noncallable_model():
    import jax.random as random

    data = _build_model_data(seed=105)

    with pytest.raises(TypeError, match="model must be callable"):
        run_mcmc(
            "not_a_model",
            data,
            rng_key=random.PRNGKey(4),
        )


def test_run_mcmc_rejects_nonmapping_data():
    import jax.random as random

    with pytest.raises(TypeError, match="data must be a mapping"):
        run_mcmc(
            survival_only_model,
            ["not", "a", "mapping"],
            rng_key=random.PRNGKey(5),
        )


def test_run_mcmc_rejects_empty_data():
    import jax.random as random

    with pytest.raises(ValueError, match="data must not be empty"):
        run_mcmc(
            survival_only_model,
            {},
            rng_key=random.PRNGKey(6),
        )


def test_run_mcmc_collects_extra_fields():
    import jax.random as random

    data = _build_model_data(seed=106)
    config = InferenceConfig(
        num_chains=1,
        num_warmup=20,
        num_samples=20,
        progress_bar=False,
    )

    result = run_mcmc(
        survival_only_model,
        data,
        rng_key=random.PRNGKey(7),
        config=config,
        extra_fields=("diverging",),
    )

    extra = result.mcmc.get_extra_fields()
    assert "diverging" in extra
    assert len(np.asarray(extra["diverging"])) == config.num_samples