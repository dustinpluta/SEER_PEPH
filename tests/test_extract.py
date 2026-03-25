from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seer_peph.fitting.extract import (
    extract_joint_coupling,
    extract_spatial_fields,
    extract_survival_effects,
    extract_treatment_effects,
)
from seer_peph.fitting.fit_models import (
    FitMetadata,
    JointFit,
    SurvivalFit,
    TreatmentFit,
)
from seer_peph.inference.run import InferenceConfig, InferenceResult


class _DummyMCMC:
    pass


def _dummy_inference_result(samples: dict, summary: dict | None = None) -> InferenceResult:
    return InferenceResult(
        mcmc=_DummyMCMC(),
        samples=samples,
        summary=summary if summary is not None else {"ok": {"mean": 1.0}},
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


def _survival_fit() -> SurvivalFit:
    samples = {
        "beta": np.array(
            [
                [0.10, 0.20, 0.30],
                [0.11, 0.21, 0.31],
                [0.09, 0.19, 0.29],
            ]
        ),
        "alpha": np.array(
            [
                [-4.0, -3.0],
                [-4.1, -3.1],
                [-3.9, -2.9],
            ]
        ),
        "delta_post": np.array(
            [
                [-0.10, -0.20],
                [-0.11, -0.19],
                [-0.09, -0.21],
            ]
        ),
        "u": np.array(
            [
                [0.5, -0.2, 0.1],
                [0.4, -0.1, 0.2],
                [0.6, -0.3, 0.0],
            ]
        ),
    }

    scalar_summary = {
        "beta[0]": {"mean": 0.10, "sd": 0.01, "median": 0.10, "q05": 0.09, "q95": 0.11},
        "beta[1]": {"mean": 0.20, "sd": 0.01, "median": 0.20, "q05": 0.19, "q95": 0.21},
        "beta[2]": {"mean": 0.30, "sd": 0.01, "median": 0.30, "q05": 0.29, "q95": 0.31},
        "alpha[0]": {"mean": -4.0, "sd": 0.1, "median": -4.0, "q05": -4.1, "q95": -3.9},
        "alpha[1]": {"mean": -3.0, "sd": 0.1, "median": -3.0, "q05": -3.1, "q95": -2.9},
        "delta_post[0]": {"mean": -0.10, "sd": 0.01, "median": -0.10, "q05": -0.11, "q95": -0.09},
        "delta_post[1]": {"mean": -0.20, "sd": 0.01, "median": -0.20, "q05": -0.21, "q95": -0.19},
        "u[0]": {"mean": 0.50, "sd": 0.10, "median": 0.50, "q05": 0.40, "q95": 0.60},
        "u[1]": {"mean": -0.20, "sd": 0.10, "median": -0.20, "q05": -0.30, "q95": -0.10},
        "u[2]": {"mean": 0.10, "sd": 0.10, "median": 0.10, "q05": 0.00, "q95": 0.20},
        "rho": {"mean": 0.70, "sd": 0.02, "median": 0.70, "q05": 0.68, "q95": 0.72},
        "tau": {"mean": 0.40, "sd": 0.05, "median": 0.40, "q05": 0.35, "q95": 0.45},
    }

    return SurvivalFit(
        model_name="survival_spatial_delta_only",
        inference_result=_dummy_inference_result(samples),
        samples=samples,
        summary={"ok": {"mean": 1.0}},
        scalar_summary=scalar_summary,
        data={},
        metadata=FitMetadata(
            surv_x_cols=("age", "cci", "tumor_size_log"),
            graph_A=3,
            n_surv=12,
            p_surv=3,
            rng_seed=1,
        ),
        extra={},
    )


def _treatment_fit() -> TreatmentFit:
    samples = {
        "theta": np.array(
            [
                [0.10, 0.20, -0.10, 0.05],
                [0.12, 0.18, -0.08, 0.03],
            ]
        ),
        "gamma": np.array(
            [
                [-2.0, -2.5, -3.0],
                [-2.1, -2.4, -3.1],
            ]
        ),
        "u": np.array(
            [
                [0.2, -0.1],
                [0.1, -0.2],
            ]
        ),
    }

    scalar_summary = {
        "theta[0]": {"mean": 0.11, "sd": 0.01, "median": 0.11, "q05": 0.10, "q95": 0.12},
        "theta[1]": {"mean": 0.19, "sd": 0.01, "median": 0.19, "q05": 0.18, "q95": 0.20},
        "theta[2]": {"mean": -0.09, "sd": 0.01, "median": -0.09, "q05": -0.10, "q95": -0.08},
        "theta[3]": {"mean": 0.04, "sd": 0.01, "median": 0.04, "q05": 0.03, "q95": 0.05},
        "gamma[0]": {"mean": -2.05, "sd": 0.05, "median": -2.05, "q05": -2.10, "q95": -2.00},
        "gamma[1]": {"mean": -2.45, "sd": 0.05, "median": -2.45, "q05": -2.50, "q95": -2.40},
        "gamma[2]": {"mean": -3.05, "sd": 0.05, "median": -3.05, "q05": -3.10, "q95": -3.00},
        "u[0]": {"mean": 0.15, "sd": 0.05, "median": 0.15, "q05": 0.10, "q95": 0.20},
        "u[1]": {"mean": -0.15, "sd": 0.05, "median": -0.15, "q05": -0.20, "q95": -0.10},
        "rho": {"mean": 0.60, "sd": 0.03, "median": 0.60, "q05": 0.57, "q95": 0.63},
        "tau": {"mean": 0.30, "sd": 0.04, "median": 0.30, "q05": 0.26, "q95": 0.34},
    }

    return TreatmentFit(
        model_name="treatment_spatial_pe",
        inference_result=_dummy_inference_result(samples),
        samples=samples,
        summary={"ok": {"mean": 1.0}},
        scalar_summary=scalar_summary,
        data={},
        metadata=FitMetadata(
            ttt_x_cols=("age", "cci", "ses", "sex_male"),
            graph_A=2,
            n_ttt=8,
            p_ttt=4,
            rng_seed=2,
        ),
        extra={},
    )


def _joint_fit() -> JointFit:
    samples = {
        "beta": np.array([[0.1, 0.2], [0.2, 0.3]]),
        "alpha": np.array([[-4.0, -3.0], [-4.1, -3.1]]),
        "delta_post": np.array([[-0.1, -0.2], [-0.2, -0.3]]),
        "theta": np.array([[0.4, 0.5, 0.6], [0.3, 0.4, 0.5]]),
        "gamma": np.array([[-2.0, -2.5], [-2.1, -2.4]]),
        "u_surv": np.array([[0.5, 0.0, -0.5], [0.4, 0.1, -0.4]]),
        "u_ttt": np.array([[0.3, 0.1, -0.2], [0.2, 0.2, -0.1]]),
        "u_ttt_ind": np.array([[0.1, 0.0, -0.1], [0.2, -0.1, -0.2]]),
        "s_surv": np.array([[1.0, 0.0, -1.0], [0.9, 0.1, -1.0]]),
        "s_ttt": np.array([[0.6, 0.0, -0.6], [0.5, 0.1, -0.6]]),
        "rho_u_cross": np.array([0.45, 0.55]),
    }

    scalar_summary = {
        "beta[0]": {"mean": 0.15, "sd": 0.05, "median": 0.15, "q05": 0.10, "q95": 0.20},
        "beta[1]": {"mean": 0.25, "sd": 0.05, "median": 0.25, "q05": 0.20, "q95": 0.30},
        "alpha[0]": {"mean": -4.05, "sd": 0.05, "median": -4.05, "q05": -4.10, "q95": -4.00},
        "alpha[1]": {"mean": -3.05, "sd": 0.05, "median": -3.05, "q05": -3.10, "q95": -3.00},
        "delta_post[0]": {"mean": -0.15, "sd": 0.05, "median": -0.15, "q05": -0.20, "q95": -0.10},
        "delta_post[1]": {"mean": -0.25, "sd": 0.05, "median": -0.25, "q05": -0.30, "q95": -0.20},
        "theta[0]": {"mean": 0.35, "sd": 0.05, "median": 0.35, "q05": 0.30, "q95": 0.40},
        "theta[1]": {"mean": 0.45, "sd": 0.05, "median": 0.45, "q05": 0.40, "q95": 0.50},
        "theta[2]": {"mean": 0.55, "sd": 0.05, "median": 0.55, "q05": 0.50, "q95": 0.60},
        "gamma[0]": {"mean": -2.05, "sd": 0.05, "median": -2.05, "q05": -2.10, "q95": -2.00},
        "gamma[1]": {"mean": -2.45, "sd": 0.05, "median": -2.45, "q05": -2.50, "q95": -2.40},
        "u_surv[0]": {"mean": 0.45, "sd": 0.05, "median": 0.45, "q05": 0.40, "q95": 0.50},
        "u_surv[1]": {"mean": 0.05, "sd": 0.05, "median": 0.05, "q05": 0.00, "q95": 0.10},
        "u_surv[2]": {"mean": -0.45, "sd": 0.05, "median": -0.45, "q05": -0.50, "q95": -0.40},
        "u_ttt[0]": {"mean": 0.25, "sd": 0.05, "median": 0.25, "q05": 0.20, "q95": 0.30},
        "u_ttt[1]": {"mean": 0.15, "sd": 0.05, "median": 0.15, "q05": 0.10, "q95": 0.20},
        "u_ttt[2]": {"mean": -0.15, "sd": 0.05, "median": -0.15, "q05": -0.20, "q95": -0.10},
        "u_ttt_ind[0]": {"mean": 0.15, "sd": 0.05, "median": 0.15, "q05": 0.10, "q95": 0.20},
        "u_ttt_ind[1]": {"mean": -0.05, "sd": 0.05, "median": -0.05, "q05": -0.10, "q95": 0.00},
        "u_ttt_ind[2]": {"mean": -0.15, "sd": 0.05, "median": -0.15, "q05": -0.20, "q95": -0.10},
        "s_surv[0]": {"mean": 0.95, "sd": 0.05, "median": 0.95, "q05": 0.90, "q95": 1.00},
        "s_surv[1]": {"mean": 0.05, "sd": 0.05, "median": 0.05, "q05": 0.00, "q95": 0.10},
        "s_surv[2]": {"mean": -1.00, "sd": 0.05, "median": -1.00, "q05": -1.05, "q95": -0.95},
        "s_ttt[0]": {"mean": 0.55, "sd": 0.05, "median": 0.55, "q05": 0.50, "q95": 0.60},
        "s_ttt[1]": {"mean": 0.05, "sd": 0.05, "median": 0.05, "q05": 0.00, "q95": 0.10},
        "s_ttt[2]": {"mean": -0.60, "sd": 0.05, "median": -0.60, "q05": -0.65, "q95": -0.55},
        "rho_surv": {"mean": 0.80, "sd": 0.03, "median": 0.80, "q05": 0.77, "q95": 0.83},
        "tau_surv": {"mean": 0.50, "sd": 0.04, "median": 0.50, "q05": 0.46, "q95": 0.54},
        "rho_ttt": {"mean": 0.70, "sd": 0.03, "median": 0.70, "q05": 0.67, "q95": 0.73},
        "tau_ttt": {"mean": 0.30, "sd": 0.04, "median": 0.30, "q05": 0.26, "q95": 0.34},
        "rho_u_cross": {"mean": 0.50, "sd": 0.05, "median": 0.50, "q05": 0.45, "q95": 0.55},
    }

    return JointFit(
        model_name="joint_spatial_treatment_survival",
        inference_result=_dummy_inference_result(samples),
        samples=samples,
        summary={"ok": {"mean": 1.0}},
        scalar_summary=scalar_summary,
        data={},
        metadata=FitMetadata(
            surv_x_cols=("age", "tumor_size_log"),
            ttt_x_cols=("cci", "ses", "sex_male"),
            graph_A=3,
            n_surv=12,
            n_ttt=8,
            p_surv=2,
            p_ttt=3,
            rng_seed=3,
        ),
        extra={},
    )


def test_extract_survival_effects_summary_tables():
    fit = _survival_fit()
    out = extract_survival_effects(fit, include_draws=False)

    assert set(out.keys()) == {"beta", "alpha", "delta_post"}

    beta_df = out["beta"]
    assert list(beta_df["label"]) == ["age", "cci", "tumor_size_log"]
    assert list(beta_df["parameter"]) == ["beta[0]", "beta[1]", "beta[2]"]
    assert beta_df["param_type"].iloc[0] == "survival_beta"

    alpha_df = out["alpha"]
    assert list(alpha_df["label"]) == ["survival_interval_0", "survival_interval_1"]

    delta_df = out["delta_post"]
    assert list(delta_df["label"]) == ["post_treatment_interval_0", "post_treatment_interval_1"]


def test_extract_survival_effects_with_draws():
    fit = _survival_fit()
    out = extract_survival_effects(fit, include_draws=True)

    assert "beta_draws" in out
    assert "alpha_draws" in out
    assert "delta_post_draws" in out

    beta_draws = out["beta_draws"]
    assert set(beta_draws.columns) == {"draw", "parameter", "index", "label", "value"}
    assert beta_draws.shape[0] == 3 * 3
    assert sorted(beta_draws["parameter"].unique().tolist()) == ["beta[0]", "beta[1]", "beta[2]"]


def test_extract_treatment_effects_summary_tables():
    fit = _treatment_fit()
    out = extract_treatment_effects(fit, include_draws=False)

    assert set(out.keys()) == {"theta", "gamma"}

    theta_df = out["theta"]
    assert list(theta_df["label"]) == ["age", "cci", "ses", "sex_male"]
    assert theta_df["param_type"].iloc[0] == "treatment_theta"

    gamma_df = out["gamma"]
    assert list(gamma_df["label"]) == [
        "treatment_interval_0",
        "treatment_interval_1",
        "treatment_interval_2",
    ]


def test_extract_spatial_fields_for_survival_fit():
    fit = _survival_fit()
    out = extract_spatial_fields(fit, include_draws=True)

    assert set(out.keys()) == {"field", "field_draws", "hyperparameters"}

    field_df = out["field"]
    assert set(field_df.columns) == {
        "parameter",
        "field",
        "field_type",
        "area_id",
        "mean",
        "sd",
        "median",
        "q05",
        "q95",
    }
    assert list(field_df["area_id"]) == [0, 1, 2]
    assert field_df["field"].iloc[0] == "u"
    assert field_df["field_type"].iloc[0] == "survival_field"

    hyper_df = out["hyperparameters"]
    assert hyper_df["parameter"].tolist() == ["rho", "tau"]
    assert hyper_df["param_group"].iloc[0] == "survival_spatial_hyper"


def test_extract_spatial_fields_for_treatment_fit():
    fit = _treatment_fit()
    out = extract_spatial_fields(fit, include_draws=False)

    assert set(out.keys()) == {"field", "hyperparameters"}
    assert out["field"]["field_type"].iloc[0] == "treatment_field"
    assert out["hyperparameters"]["parameter"].tolist() == ["rho", "tau"]


def test_extract_spatial_fields_for_joint_fit():
    fit = _joint_fit()
    out = extract_spatial_fields(fit, include_draws=True)

    expected = {
        "u_surv",
        "u_surv_draws",
        "u_ttt",
        "u_ttt_draws",
        "u_ttt_ind",
        "u_ttt_ind_draws",
        "s_surv",
        "s_surv_draws",
        "s_ttt",
        "s_ttt_draws",
        "hyperparameters",
    }
    assert set(out.keys()) == expected

    assert out["u_surv"]["field_type"].iloc[0] == "joint_survival_field"
    assert out["u_ttt"]["field_type"].iloc[0] == "joint_treatment_field"
    assert out["u_ttt_ind"]["field_type"].iloc[0] == "joint_treatment_independent_field"
    assert out["s_surv"]["field_type"].iloc[0] == "joint_survival_structured"
    assert out["s_ttt"]["field_type"].iloc[0] == "joint_treatment_structured"

    hyper_df = out["hyperparameters"]
    assert hyper_df["parameter"].tolist() == ["rho_surv", "tau_surv", "rho_ttt", "tau_ttt"]


def test_extract_joint_coupling_summary_and_draws():
    fit = _joint_fit()
    out = extract_joint_coupling(fit, include_draws=True)

    assert set(out.keys()) == {"coupling", "coupling_draws", "field_correlations"}

    coupling_df = out["coupling"]
    assert coupling_df["parameter"].tolist() == ["rho_u_cross"]
    assert coupling_df["param_group"].tolist() == ["joint_coupling"]

    coupling_draws = out["coupling_draws"]
    assert coupling_draws["parameter"].unique().tolist() == ["rho_u_cross"]
    assert coupling_draws["value"].tolist() == [0.45, 0.55]

    field_corr = out["field_correlations"]
    assert "corr_u_surv_u_ttt_draw_mean" in field_corr["metric"].tolist()
    assert field_corr.shape[0] == 5


def test_extract_joint_coupling_requires_joint_fit():
    fit = _survival_fit()
    with pytest.raises(TypeError, match="JointFit"):
        extract_joint_coupling(fit)


def test_extract_joint_coupling_raises_on_mismatched_shapes():
    fit = _joint_fit()
    fit.samples["u_ttt"] = np.array([[0.1, 0.2], [0.3, 0.4]])

    with pytest.raises(ValueError, match="matching shapes"):
        extract_joint_coupling(fit)


def test_extract_survival_effects_works_with_missing_optional_blocks():
    fit = _survival_fit()
    del fit.samples["delta_post"]

    out = extract_survival_effects(fit, include_draws=False)
    assert set(out.keys()) == {"beta", "alpha"}


def test_extract_treatment_effects_works_with_missing_optional_blocks():
    fit = _treatment_fit()
    del fit.samples["gamma"]

    out = extract_treatment_effects(fit, include_draws=False)
    assert set(out.keys()) == {"theta"}