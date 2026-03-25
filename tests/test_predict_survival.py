from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seer_peph.fitting.fit_models import FitMetadata, SurvivalFit
from seer_peph.inference.run import InferenceConfig, InferenceResult
from seer_peph.predict.survival import (
    _compute_survival_interval_hazards_one_draw,
    _post_treatment_interval_index,
    _rmst_from_survival_curve,
    _survival_curve_from_piecewise_hazards,
    predict_counterfactual_survival_draws,
    predict_counterfactual_survival_summary,
    predict_rmst,
    predict_survival_at_times,
)


def _dummy_survival_fit() -> SurvivalFit:
    samples = {
        "alpha": np.array(
            [
                [-2.0, -2.0, -2.0],
                [-2.1, -2.1, -2.1],
            ],
            dtype=float,
        ),
        "beta": np.array(
            [
                [0.5, 0.2],
                [0.6, 0.1],
            ],
            dtype=float,
        ),
        "delta_post": np.array(
            [
                [-0.4, -0.2],
                [-0.5, -0.3],
            ],
            dtype=float,
        ),
        "u": np.array(
            [
                [0.0, 0.3],
                [0.1, 0.2],
            ],
            dtype=float,
        ),
    }

    summary = {
        "alpha[0]": {"mean": -2.05},
        "beta[0]": {"mean": 0.55},
        "delta_post[0]": {"mean": -0.45},
    }

    scalar_summary = {
        "alpha[0]": {"mean": -2.05, "sd": 0.05, "median": -2.05, "q05": -2.10, "q95": -2.00},
        "alpha[1]": {"mean": -2.05, "sd": 0.05, "median": -2.05, "q05": -2.10, "q95": -2.00},
        "alpha[2]": {"mean": -2.05, "sd": 0.05, "median": -2.05, "q05": -2.10, "q95": -2.00},
        "beta[0]": {"mean": 0.55, "sd": 0.05, "median": 0.55, "q05": 0.50, "q95": 0.60},
        "beta[1]": {"mean": 0.15, "sd": 0.05, "median": 0.15, "q05": 0.10, "q95": 0.20},
        "delta_post[0]": {"mean": -0.45, "sd": 0.05, "median": -0.45, "q05": -0.50, "q95": -0.40},
        "delta_post[1]": {"mean": -0.25, "sd": 0.05, "median": -0.25, "q05": -0.30, "q95": -0.20},
        "u[0]": {"mean": 0.05, "sd": 0.05, "median": 0.05, "q05": 0.00, "q95": 0.10},
        "u[1]": {"mean": 0.25, "sd": 0.05, "median": 0.25, "q05": 0.20, "q95": 0.30},
    }

    inference_result = InferenceResult(
        mcmc=None,
        samples=samples,
        summary=summary,
        config=InferenceConfig(
            num_chains=1,
            num_warmup=10,
            num_samples=20,
            target_accept_prob=0.9,
            dense_mass=False,
            max_tree_depth=5,
            progress_bar=False,
        ),
    )

    return SurvivalFit(
        model_name="survival_spatial_delta_only",
        inference_result=inference_result,
        samples=samples,
        summary=summary,
        scalar_summary=scalar_summary,
        data={},
        metadata=FitMetadata(
            surv_x_cols=("x1", "x2"),
            surv_breaks=(0.0, 1.0, 2.0, 3.0),
            post_ttt_breaks=(0.0, 1.0, 2.0),
            graph_A=2,
            graph_n_edges=1,
            n_surv=10,
            p_surv=2,
            rng_seed=123,
        ),
        extra={},
    )


def test_post_treatment_interval_index_basic_behavior() -> None:
    breaks = np.array([0.0, 1.0, 2.0], dtype=float)

    assert _post_treatment_interval_index(elapsed_m=0.0, post_treatment_breaks=breaks) == 0
    assert _post_treatment_interval_index(elapsed_m=0.5, post_treatment_breaks=breaks) == 0
    assert _post_treatment_interval_index(elapsed_m=1.0, post_treatment_breaks=breaks) == 1
    assert _post_treatment_interval_index(elapsed_m=1.5, post_treatment_breaks=breaks) == 1
    assert _post_treatment_interval_index(elapsed_m=10.0, post_treatment_breaks=breaks) == 1


def test_post_treatment_interval_index_rejects_negative_elapsed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _post_treatment_interval_index(
            elapsed_m=-0.1,
            post_treatment_breaks=np.array([0.0, 1.0, 2.0], dtype=float),
        )


def test_compute_hazards_no_treatment_matches_baseline() -> None:
    alpha_draw = np.array([-2.0, -2.0, -2.0], dtype=float)
    beta_draw = np.array([0.5, 0.2], dtype=float)
    delta_draw = np.array([-0.4, -0.2], dtype=float)
    u_draw = np.array([0.0, 0.3], dtype=float)
    x = np.array([1.0, 0.0], dtype=float)

    hazards = _compute_survival_interval_hazards_one_draw(
        alpha_draw=alpha_draw,
        beta_draw=beta_draw,
        delta_post_draw=delta_draw,
        u_draw=u_draw,
        x=x,
        area_id=0,
        surv_breaks=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        post_treatment_breaks=np.array([0.0, 1.0, 2.0], dtype=float),
        treatment_time_m=None,
    )

    expected_eta = -2.0 + 0.5
    expected_hazard = np.exp(expected_eta)
    np.testing.assert_allclose(hazards, np.repeat(expected_hazard, 3))


def test_compute_hazards_with_treatment_changes_later_intervals() -> None:
    alpha_draw = np.array([-2.0, -2.0, -2.0], dtype=float)
    beta_draw = np.array([0.5, 0.0], dtype=float)
    delta_draw = np.array([-0.4, -0.2], dtype=float)
    u_draw = np.array([0.0, 0.0], dtype=float)
    x = np.array([1.0, 0.0], dtype=float)

    hazards = _compute_survival_interval_hazards_one_draw(
        alpha_draw=alpha_draw,
        beta_draw=beta_draw,
        delta_post_draw=delta_draw,
        u_draw=u_draw,
        x=x,
        area_id=0,
        surv_breaks=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        post_treatment_breaks=np.array([0.0, 1.0, 2.0], dtype=float),
        treatment_time_m=1.5,
    )

    base_hazard = np.exp(-2.0 + 0.5)
    post0_hazard = np.exp(-2.0 + 0.5 - 0.4)

    np.testing.assert_allclose(hazards[0], base_hazard)
    np.testing.assert_allclose(hazards[1], post0_hazard)
    np.testing.assert_allclose(hazards[2], post0_hazard)


def test_survival_curve_from_piecewise_hazards_matches_closed_form_single_interval() -> None:
    hazards = np.array([2.0], dtype=float)
    breaks = np.array([0.0, 1.0], dtype=float)
    eval_times = np.array([0.0, 0.25, 0.5, 1.0], dtype=float)

    surv = _survival_curve_from_piecewise_hazards(
        hazards=hazards,
        breaks=breaks,
        eval_times=eval_times,
    )

    expected = np.exp(-2.0 * eval_times)
    expected[0] = 1.0
    np.testing.assert_allclose(surv, expected)


def test_rmst_from_survival_curve_constant_one_is_horizon() -> None:
    times = np.linspace(0.0, 5.0, 11)
    surv = np.ones_like(times)

    rmst = _rmst_from_survival_curve(times=times, survival=surv)
    assert np.isclose(rmst, 5.0)


def test_predict_counterfactual_survival_draws_returns_expected_columns() -> None:
    fit = _dummy_survival_fit()

    out = predict_counterfactual_survival_draws(
        fit,
        x=[1.0, 0.0],
        area_id=0,
        surv_breaks=[0.0, 1.0, 2.0, 3.0],
        post_treatment_breaks=[0.0, 1.0, 2.0],
        eval_times=[0.0, 1.0, 2.0],
        treatment_time_m=None,
    )

    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {"draw", "time_m", "survival", "treatment_time_m", "area_id"}
    assert out.shape[0] == 2 * 3
    assert out["draw"].nunique() == 2
    assert out["time_m"].nunique() == 3
    assert out["area_id"].eq(0).all()
    assert out["survival"].between(0.0, 1.0).all()
    assert out.loc[out["time_m"] == 0.0, "survival"].eq(1.0).all()


def test_predict_counterfactual_survival_draws_subset_draw_indices() -> None:
    fit = _dummy_survival_fit()

    out = predict_counterfactual_survival_draws(
        fit,
        x=[1.0, 0.0],
        area_id=0,
        surv_breaks=[0.0, 1.0, 2.0, 3.0],
        post_treatment_breaks=[0.0, 1.0, 2.0],
        eval_times=[0.0, 1.0],
        treatment_time_m=None,
        draw_indices=[1],
    )

    assert out["draw"].unique().tolist() == [1]
    assert out.shape[0] == 2


def test_earlier_treatment_improves_survival_when_delta_is_negative() -> None:
    fit = _dummy_survival_fit()

    no_treatment = predict_counterfactual_survival_summary(
        fit,
        x=[1.0, 0.0],
        area_id=0,
        surv_breaks=[0.0, 1.0, 2.0, 3.0],
        post_treatment_breaks=[0.0, 1.0, 2.0],
        eval_times=[3.0],
        treatment_time_m=None,
    )

    early_treatment = predict_counterfactual_survival_summary(
        fit,
        x=[1.0, 0.0],
        area_id=0,
        surv_breaks=[0.0, 1.0, 2.0, 3.0],
        post_treatment_breaks=[0.0, 1.0, 2.0],
        eval_times=[3.0],
        treatment_time_m=0.5,
    )

    assert early_treatment["mean_survival"].iloc[0] > no_treatment["mean_survival"].iloc[0]


def test_predict_counterfactual_survival_summary_returns_expected_columns() -> None:
    fit = _dummy_survival_fit()

    out = predict_counterfactual_survival_summary(
        fit,
        x=[1.0, 0.0],
        area_id=1,
        surv_breaks=[0.0, 1.0, 2.0, 3.0],
        post_treatment_breaks=[0.0, 1.0, 2.0],
        eval_times=[0.0, 1.0, 2.0],
        treatment_time_m=1.0,
    )

    assert set(out.columns) == {
        "time_m",
        "treatment_time_m",
        "area_id",
        "mean_survival",
        "median_survival",
        "q05_survival",
        "q95_survival",
    }
    assert out["area_id"].eq(1).all()
    assert out["treatment_time_m"].eq(1.0).all()
    assert out["mean_survival"].between(0.0, 1.0).all()


def test_predict_survival_at_times_returns_multiple_scenarios() -> None:
    fit = _dummy_survival_fit()

    out = predict_survival_at_times(
        fit,
        x=[1.0, 0.0],
        area_id=0,
        surv_breaks=[0.0, 1.0, 2.0, 3.0],
        post_treatment_breaks=[0.0, 1.0, 2.0],
        times=[1.0, 2.0, 3.0],
        treatment_times_m=[None, 0.5, 1.5],
    )

    assert out["time_m"].nunique() == 3
    # None becomes NaN in the returned table
    assert out.shape[0] == 3 * 3
    assert out["area_id"].eq(0).all()
    assert out["mean_survival"].between(0.0, 1.0).all()


def test_predict_rmst_returns_expected_columns_and_ordering() -> None:
    fit = _dummy_survival_fit()

    out = predict_rmst(
        fit,
        x=[1.0, 0.0],
        area_id=0,
        surv_breaks=[0.0, 1.0, 2.0, 3.0],
        post_treatment_breaks=[0.0, 1.0, 2.0],
        horizon_m=3.0,
        treatment_times_m=[None, 0.5, 1.5],
        grid_size=100,
    )

    assert set(out.columns) == {
        "treatment_time_m",
        "mean_rmst",
        "median_rmst",
        "q05_rmst",
        "q95_rmst",
        "horizon_m",
        "area_id",
    }
    assert out.shape[0] == 3
    assert out["area_id"].eq(0).all()
    assert out["horizon_m"].eq(3.0).all()
    assert (out["mean_rmst"] >= 0.0).all()
    assert (out["mean_rmst"] <= 3.0).all()


def test_predict_rmst_earlier_treatment_improves_rmst_when_delta_is_negative() -> None:
    fit = _dummy_survival_fit()

    out = predict_rmst(
        fit,
        x=[1.0, 0.0],
        area_id=0,
        surv_breaks=[0.0, 1.0, 2.0, 3.0],
        post_treatment_breaks=[0.0, 1.0, 2.0],
        horizon_m=3.0,
        treatment_times_m=[None, 0.5],
        grid_size=200,
    )

    no_treat_rmst = out.loc[out["treatment_time_m"].isna(), "mean_rmst"].iloc[0]
    early_rmst = out.loc[out["treatment_time_m"] == 0.5, "mean_rmst"].iloc[0]

    assert early_rmst > no_treat_rmst


def test_predict_counterfactual_survival_draws_rejects_bad_x_length() -> None:
    fit = _dummy_survival_fit()

    with pytest.raises(ValueError, match="Length of x"):
        predict_counterfactual_survival_draws(
            fit,
            x=[1.0],  # should have length 2
            area_id=0,
            surv_breaks=[0.0, 1.0, 2.0, 3.0],
            post_treatment_breaks=[0.0, 1.0, 2.0],
            eval_times=[0.0, 1.0],
            treatment_time_m=None,
        )


def test_predict_counterfactual_survival_draws_rejects_bad_area_id() -> None:
    fit = _dummy_survival_fit()

    with pytest.raises(ValueError, match="area_id"):
        predict_counterfactual_survival_draws(
            fit,
            x=[1.0, 0.0],
            area_id=99,
            surv_breaks=[0.0, 1.0, 2.0, 3.0],
            post_treatment_breaks=[0.0, 1.0, 2.0],
            eval_times=[0.0, 1.0],
            treatment_time_m=None,
        )


def test_predict_counterfactual_survival_draws_rejects_mismatched_surv_breaks() -> None:
    fit = _dummy_survival_fit()

    with pytest.raises(ValueError, match="alpha intervals"):
        predict_counterfactual_survival_draws(
            fit,
            x=[1.0, 0.0],
            area_id=0,
            surv_breaks=[0.0, 1.0, 2.0],  # only 2 intervals, but alpha has 3
            post_treatment_breaks=[0.0, 1.0, 2.0],
            eval_times=[0.0, 1.0],
            treatment_time_m=None,
        )


def test_predict_counterfactual_survival_draws_rejects_mismatched_post_breaks() -> None:
    fit = _dummy_survival_fit()

    with pytest.raises(ValueError, match="delta_post intervals"):
        predict_counterfactual_survival_draws(
            fit,
            x=[1.0, 0.0],
            area_id=0,
            surv_breaks=[0.0, 1.0, 2.0, 3.0],
            post_treatment_breaks=[0.0, 1.0, 2.0, 3.0],  # 3 intervals, delta_post has 2
            eval_times=[0.0, 1.0],
            treatment_time_m=None,
        )


def test_rmst_helper_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="equal length"):
        _rmst_from_survival_curve(
            times=np.array([0.0, 1.0, 2.0]),
            survival=np.array([1.0, 0.8]),
        )