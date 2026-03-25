from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seer_peph.fitting.fit_models import FitMetadata, SurvivalFit
from seer_peph.inference.run import InferenceConfig, InferenceResult
from seer_peph.predict.survival_contrasts import (
    predict_rmst_contrast_draws,
    predict_rmst_contrast_summary,
    predict_rmst_scenarios,
    predict_survival_contrast_draws,
    predict_survival_contrast_summary,
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


@pytest.fixture
def fit() -> SurvivalFit:
    return _dummy_survival_fit()


@pytest.fixture
def pred_kwargs() -> dict:
    return {
        "x": [1.0, 0.0],
        "area_id": 0,
        "surv_breaks": [0.0, 1.0, 2.0, 3.0],
        "post_treatment_breaks": [0.0, 1.0, 2.0],
    }


def test_predict_survival_contrast_draws_columns_and_shape(fit: SurvivalFit, pred_kwargs: dict) -> None:
    out = predict_survival_contrast_draws(
        fit,
        **pred_kwargs,
        eval_times=[0.0, 1.0, 2.0, 3.0],
        treatment_time_m_a=0.5,
        treatment_time_m_b=None,
    )

    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {
        "draw",
        "time_m",
        "area_id",
        "treatment_time_m_a",
        "treatment_time_m_b",
        "survival_a",
        "survival_b",
        "survival_diff",
        "survival_ratio",
    }
    assert out.shape[0] == 2 * 4
    assert out["draw"].nunique() == 2
    assert out["time_m"].nunique() == 4
    assert out["area_id"].eq(0).all()


def test_predict_survival_contrast_draws_no_treatment_is_nan_scenario_label(
    fit: SurvivalFit,
    pred_kwargs: dict,
) -> None:
    out = predict_survival_contrast_draws(
        fit,
        **pred_kwargs,
        eval_times=[1.0, 2.0],
        treatment_time_m_a=0.5,
        treatment_time_m_b=None,
    )

    assert out["treatment_time_m_a"].eq(0.5).all()
    assert out["treatment_time_m_b"].isna().all()


def test_predict_survival_contrast_draws_diff_and_ratio_are_computed(
    fit: SurvivalFit,
    pred_kwargs: dict,
) -> None:
    out = predict_survival_contrast_draws(
        fit,
        **pred_kwargs,
        eval_times=[1.0, 2.0],
        treatment_time_m_a=0.5,
        treatment_time_m_b=None,
    )

    np.testing.assert_allclose(
        out["survival_diff"].to_numpy(),
        out["survival_a"].to_numpy() - out["survival_b"].to_numpy(),
    )

    ratio_expected = np.where(
        out["survival_b"].to_numpy() > 0.0,
        out["survival_a"].to_numpy() / out["survival_b"].to_numpy(),
        np.nan,
    )
    np.testing.assert_allclose(out["survival_ratio"].to_numpy(), ratio_expected)


def test_earlier_treatment_improves_survival_contrast_when_delta_is_negative(
    fit: SurvivalFit,
    pred_kwargs: dict,
) -> None:
    out = predict_survival_contrast_summary(
        fit,
        **pred_kwargs,
        eval_times=[3.0],
        treatment_time_m_a=0.5,
        treatment_time_m_b=None,
    )

    assert out.shape[0] == 1
    assert out["mean_survival_diff"].iloc[0] > 0.0
    assert out["mean_survival_ratio"].iloc[0] > 1.0


def test_predict_survival_contrast_summary_columns(
    fit: SurvivalFit,
    pred_kwargs: dict,
) -> None:
    out = predict_survival_contrast_summary(
        fit,
        **pred_kwargs,
        eval_times=[0.0, 1.0, 2.0, 3.0],
        treatment_time_m_a=0.5,
        treatment_time_m_b=1.5,
    )

    assert set(out.columns) == {
        "time_m",
        "area_id",
        "treatment_time_m_a",
        "treatment_time_m_b",
        "mean_survival_a",
        "median_survival_a",
        "q05_survival_a",
        "q95_survival_a",
        "mean_survival_b",
        "median_survival_b",
        "q05_survival_b",
        "q95_survival_b",
        "mean_survival_diff",
        "median_survival_diff",
        "q05_survival_diff",
        "q95_survival_diff",
        "mean_survival_ratio",
        "median_survival_ratio",
        "q05_survival_ratio",
        "q95_survival_ratio",
    }
    assert out["time_m"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert out["area_id"].eq(0).all()


def test_predict_rmst_contrast_draws_columns_and_shape(fit: SurvivalFit, pred_kwargs: dict) -> None:
    out = predict_rmst_contrast_draws(
        fit,
        **pred_kwargs,
        horizon_m=3.0,
        treatment_time_m_a=0.5,
        treatment_time_m_b=None,
        grid_size=100,
    )

    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {
        "draw",
        "area_id",
        "horizon_m",
        "treatment_time_m_a",
        "treatment_time_m_b",
        "rmst_a",
        "rmst_b",
        "rmst_diff",
        "rmst_ratio",
    }
    assert out.shape[0] == 2
    assert out["area_id"].eq(0).all()
    assert out["horizon_m"].eq(3.0).all()


def test_predict_rmst_contrast_draws_diff_and_ratio_are_computed(
    fit: SurvivalFit,
    pred_kwargs: dict,
) -> None:
    out = predict_rmst_contrast_draws(
        fit,
        **pred_kwargs,
        horizon_m=3.0,
        treatment_time_m_a=0.5,
        treatment_time_m_b=None,
        grid_size=100,
    )

    np.testing.assert_allclose(
        out["rmst_diff"].to_numpy(),
        out["rmst_a"].to_numpy() - out["rmst_b"].to_numpy(),
    )

    ratio_expected = np.where(
        out["rmst_b"].to_numpy() > 0.0,
        out["rmst_a"].to_numpy() / out["rmst_b"].to_numpy(),
        np.nan,
    )
    np.testing.assert_allclose(out["rmst_ratio"].to_numpy(), ratio_expected)


def test_earlier_treatment_improves_rmst_contrast_when_delta_is_negative(
    fit: SurvivalFit,
    pred_kwargs: dict,
) -> None:
    out = predict_rmst_contrast_summary(
        fit,
        **pred_kwargs,
        horizon_m=3.0,
        treatment_time_m_a=0.5,
        treatment_time_m_b=None,
        grid_size=150,
    )

    assert out.shape[0] == 1
    assert out["mean_rmst_diff"].iloc[0] > 0.0
    assert out["mean_rmst_ratio"].iloc[0] > 1.0


def test_predict_rmst_contrast_summary_columns(
    fit: SurvivalFit,
    pred_kwargs: dict,
) -> None:
    out = predict_rmst_contrast_summary(
        fit,
        **pred_kwargs,
        horizon_m=3.0,
        treatment_time_m_a=0.5,
        treatment_time_m_b=1.5,
        grid_size=120,
    )

    assert set(out.columns) == {
        "area_id",
        "horizon_m",
        "treatment_time_m_a",
        "treatment_time_m_b",
        "mean_rmst_a",
        "median_rmst_a",
        "q05_rmst_a",
        "q95_rmst_a",
        "mean_rmst_b",
        "median_rmst_b",
        "q05_rmst_b",
        "q95_rmst_b",
        "mean_rmst_diff",
        "median_rmst_diff",
        "q05_rmst_diff",
        "q95_rmst_diff",
        "mean_rmst_ratio",
        "median_rmst_ratio",
        "q05_rmst_ratio",
        "q95_rmst_ratio",
    }
    assert out["area_id"].eq(0).all()
    assert out["horizon_m"].eq(3.0).all()


def test_predict_rmst_scenarios_passthrough(fit: SurvivalFit, pred_kwargs: dict) -> None:
    out = predict_rmst_scenarios(
        fit,
        **pred_kwargs,
        horizon_m=3.0,
        treatment_times_m=[None, 0.5, 1.5],
        grid_size=100,
    )

    assert isinstance(out, pd.DataFrame)
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


def test_predict_survival_contrast_draws_subset_draw_indices(
    fit: SurvivalFit,
    pred_kwargs: dict,
) -> None:
    out = predict_survival_contrast_draws(
        fit,
        **pred_kwargs,
        eval_times=[1.0, 2.0],
        treatment_time_m_a=0.5,
        treatment_time_m_b=None,
        draw_indices=[1],
    )

    assert out["draw"].unique().tolist() == [1]
    assert out.shape[0] == 2


def test_predict_rmst_contrast_draws_subset_draw_indices(
    fit: SurvivalFit,
    pred_kwargs: dict,
) -> None:
    out = predict_rmst_contrast_draws(
        fit,
        **pred_kwargs,
        horizon_m=3.0,
        treatment_time_m_a=0.5,
        treatment_time_m_b=None,
        grid_size=100,
        draw_indices=[0],
    )

    assert out["draw"].unique().tolist() == [0]
    assert out.shape[0] == 1