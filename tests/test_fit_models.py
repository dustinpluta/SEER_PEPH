# tests/test_fit_models.py

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from seer_peph.inference.run import InferenceConfig, InferenceResult
import seer_peph.fitting.fit_models as fm


class _DummyMCMC:
    pass


def _fake_model_data() -> dict[str, Any]:
    return {
        "node1": np.array([0, 1, 2], dtype=int),
        "node2": np.array([1, 2, 0], dtype=int),
        "A": 3,
        "N_surv": 11,
        "N_ttt": 7,
        "P_surv": 5,
        "P_ttt": 6,
        "surv_x_cols": (
            "age_per10_centered",
            "cci",
            "tumor_size_log",
            "stage_II",
            "stage_III",
        ),
        "ttt_x_cols": (
            "age_per10_centered",
            "cci",
            "ses",
            "sex_male",
            "stage_II",
            "stage_III",
        ),
    }


def _fake_inference_result() -> InferenceResult:
    return InferenceResult(
        mcmc=_DummyMCMC(),
        samples={
            "beta": np.array(
                [
                    [0.10, 0.20, 0.30, 0.40, 0.50],
                    [0.11, 0.21, 0.31, 0.41, 0.51],
                ]
            ),
            "rho": np.array([0.70, 0.72]),
        },
        summary={"dummy": {"mean": 1.0}},
        config=InferenceConfig(
            num_chains=1,
            num_warmup=10,
            num_samples=10,
            target_accept_prob=0.9,
            dense_mass=False,
            max_tree_depth=5,
            progress_bar=False,
        ),
    )


def _fake_scalar_summary() -> dict[str, dict[str, float]]:
    return {
        "beta[0]": {"mean": 0.105, "sd": 0.01, "median": 0.105, "q05": 0.10, "q95": 0.11},
        "rho": {"mean": 0.71, "sd": 0.01, "median": 0.71, "q05": 0.70, "q95": 0.72},
    }


@pytest.fixture
def surv_long() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 1, 2], "k": [0, 1, 0], "event": [0, 1, 1]})


@pytest.fixture
def ttt_long() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2], "k": [0, 1], "event": [1, 0]})


def test_fit_survival_model_uses_supplied_data_without_rebuilding(monkeypatch) -> None:
    fake_data = _fake_model_data()
    fake_result = _fake_inference_result()
    fake_scalar = _fake_scalar_summary()

    captured: dict[str, Any] = {}

    def _boom(*args, **kwargs):
        raise AssertionError("make_model_data should not be called when data is supplied directly")

    def _fake_run_mcmc(model, data, *, rng_key, config, extra_fields, init_strategy=None):
        captured["model"] = model
        captured["data"] = data
        captured["config"] = config
        captured["extra_fields"] = extra_fields
        return fake_result

    monkeypatch.setattr(fm, "make_model_data", _boom)
    monkeypatch.setattr(fm, "run_mcmc", _fake_run_mcmc)
    monkeypatch.setattr(fm, "summarise_samples", lambda samples: fake_scalar)

    infer_cfg = InferenceConfig(
        num_chains=2,
        num_warmup=20,
        num_samples=30,
        target_accept_prob=0.95,
        dense_mass=False,
        max_tree_depth=7,
        progress_bar=False,
    )

    fit = fm.fit_survival_model(
        data=fake_data,
        rng_seed=777,
        inference_config=infer_cfg,
        surv_breaks=[0.0, 3.0, 6.0],
        post_ttt_breaks=[0.0, 3.0, 6.0, 12.0],
        extra_metadata={"analysis_name": "unit_test"},
    )

    assert fit.model_name == "survival_spatial_delta_only"
    assert fit.inference_result is fake_result
    assert fit.samples == fake_result.samples
    assert fit.summary == fake_result.summary
    assert fit.scalar_summary == fake_scalar
    assert fit.data == fake_data

    assert fit.metadata.surv_x_cols == fake_data["surv_x_cols"]
    assert fit.metadata.ttt_x_cols == fake_data["ttt_x_cols"]
    assert fit.metadata.graph_A == 3
    assert fit.metadata.graph_n_edges == 3
    assert fit.metadata.n_surv == 11
    assert fit.metadata.n_ttt == 7
    assert fit.metadata.p_surv == 5
    assert fit.metadata.p_ttt == 6
    assert fit.metadata.rng_seed == 777
    assert fit.metadata.surv_breaks == (0.0, 3.0, 6.0)
    assert fit.metadata.post_ttt_breaks == (0.0, 3.0, 6.0, 12.0)

    assert fit.extra["analysis_name"] == "unit_test"

    assert captured["model"] is fm.survival_spatial_delta_only_model
    assert captured["data"] == fake_data
    assert captured["config"] is infer_cfg
    assert captured["extra_fields"] == ("diverging",)


def test_fit_treatment_model_builds_model_data_from_raw_inputs(
    monkeypatch,
    surv_long: pd.DataFrame,
    ttt_long: pd.DataFrame,
) -> None:
    fake_data = _fake_model_data()
    fake_result = _fake_inference_result()

    captured: dict[str, Any] = {}

    def _fake_make_model_data(
        surv_long_arg,
        ttt_long_arg,
        graph_arg,
        *,
        surv_x_cols=None,
        ttt_x_cols=None,
        as_jax=True,
    ):
        captured["surv_long"] = surv_long_arg
        captured["ttt_long"] = ttt_long_arg
        captured["graph"] = graph_arg
        captured["surv_x_cols"] = surv_x_cols
        captured["ttt_x_cols"] = ttt_x_cols
        captured["as_jax"] = as_jax
        return fake_data

    def _fake_run_mcmc(model, data, *, rng_key, config, extra_fields, init_strategy=None):
        captured["model"] = model
        captured["data"] = data
        captured["config"] = config
        return fake_result

    monkeypatch.setattr(fm, "make_model_data", _fake_make_model_data)
    monkeypatch.setattr(fm, "run_mcmc", _fake_run_mcmc)
    monkeypatch.setattr(fm, "summarise_samples", lambda samples: _fake_scalar_summary())

    graph = object()

    fit = fm.fit_treatment_model(
        surv_long=surv_long,
        ttt_long=ttt_long,
        graph=graph,
        ttt_x_cols=[
            "age_per10_centered",
            "cci",
            "ses",
            "sex_male",
            "stage_II",
            "stage_III",
        ],
        rng_seed=222,
        as_jax=False,
    )

    assert fit.model_name == "treatment_spatial_pe"
    assert captured["surv_long"] is surv_long
    assert captured["ttt_long"] is ttt_long
    assert captured["graph"] is graph
    assert captured["as_jax"] is False
    assert captured["ttt_x_cols"] == [
        "age_per10_centered",
        "cci",
        "ses",
        "sex_male",
        "stage_II",
        "stage_III",
    ]
    assert captured["model"] is fm.treatment_spatial_pe_model
    assert captured["data"] == fake_data
    assert fit.metadata.rng_seed == 222


def test_fit_joint_model_uses_joint_model_callable(monkeypatch) -> None:
    fake_data = _fake_model_data()
    fake_result = _fake_inference_result()
    captured: dict[str, Any] = {}

    def _fake_run_mcmc(model, data, *, rng_key, config, extra_fields, init_strategy=None):
        captured["model"] = model
        captured["data"] = data
        return fake_result

    monkeypatch.setattr(fm, "run_mcmc", _fake_run_mcmc)
    monkeypatch.setattr(fm, "summarise_samples", lambda samples: _fake_scalar_summary())

    fit = fm.fit_joint_model(
        data=fake_data,
        rng_seed=999,
    )

    assert fit.model_name == "joint_spatial_treatment_survival"
    assert captured["model"] is fm.joint_spatial_treatment_survival_model
    assert captured["data"] == fake_data
    assert fit.metadata.rng_seed == 999


def test_fit_wrappers_raise_if_both_data_and_raw_inputs_are_supplied(
    surv_long: pd.DataFrame,
    ttt_long: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Pass either `data` or"):
        fm.fit_survival_model(
            data=_fake_model_data(),
            surv_long=surv_long,
            ttt_long=ttt_long,
            graph=object(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"surv_long": None, "ttt_long": pd.DataFrame(), "graph": object()},
        {"surv_long": pd.DataFrame(), "ttt_long": None, "graph": object()},
        {"surv_long": pd.DataFrame(), "ttt_long": pd.DataFrame(), "graph": None},
    ],
)
def test_fit_wrappers_raise_if_raw_inputs_are_incomplete(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="all required"):
        fm.fit_joint_model(**kwargs)


def test_metadata_breaks_are_optional_and_coerced_to_tuples(monkeypatch) -> None:
    fake_data = _fake_model_data()
    fake_result = _fake_inference_result()

    monkeypatch.setattr(fm, "run_mcmc", lambda *args, **kwargs: fake_result)
    monkeypatch.setattr(fm, "summarise_samples", lambda samples: _fake_scalar_summary())

    fit = fm.fit_survival_model(
        data=fake_data,
        surv_breaks=np.array([0, 1, 2], dtype=float),
        ttt_breaks=[0, 3, 6],
        post_ttt_breaks=None,
    )

    assert fit.metadata.surv_breaks == (0.0, 1.0, 2.0)
    assert fit.metadata.ttt_breaks == (0.0, 3.0, 6.0)
    assert fit.metadata.post_ttt_breaks is None