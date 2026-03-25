from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from seer_peph.fitting.fit_models import SurvivalFit
from seer_peph.predict.survival import (
    predict_counterfactual_survival_draws,
    predict_rmst,
)


def predict_survival_contrast_draws(
    fit: SurvivalFit,
    *,
    x: Sequence[float],
    area_id: int,
    surv_breaks: Sequence[float],
    post_treatment_breaks: Sequence[float],
    eval_times: Sequence[float],
    treatment_time_m_a: float | None,
    treatment_time_m_b: float | None,
    draw_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Draw-level contrast in survival between two treatment-time scenarios.

    Returns
    -------
    pd.DataFrame
        One row per draw-time combination with survival under scenario A,
        survival under scenario B, and their difference and ratio.
    """
    draws_a = predict_counterfactual_survival_draws(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_treatment_breaks,
        eval_times=eval_times,
        treatment_time_m=treatment_time_m_a,
        draw_indices=draw_indices,
    ).rename(columns={"survival": "survival_a"})

    draws_b = predict_counterfactual_survival_draws(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_treatment_breaks,
        eval_times=eval_times,
        treatment_time_m=treatment_time_m_b,
        draw_indices=draw_indices,
    ).rename(columns={"survival": "survival_b"})

    merge_cols = ["draw", "time_m", "area_id"]
    merged = draws_a.drop(columns=["treatment_time_m"]).merge(
        draws_b.drop(columns=["treatment_time_m"]),
        on=merge_cols,
        how="inner",
        validate="one_to_one",
    )

    merged["treatment_time_m_a"] = (
        np.nan if treatment_time_m_a is None else float(treatment_time_m_a)
    )
    merged["treatment_time_m_b"] = (
        np.nan if treatment_time_m_b is None else float(treatment_time_m_b)
    )
    merged["survival_diff"] = merged["survival_a"] - merged["survival_b"]
    merged["survival_ratio"] = np.where(
        merged["survival_b"] > 0.0,
        merged["survival_a"] / merged["survival_b"],
        np.nan,
    )

    return merged[
        [
            "draw",
            "time_m",
            "area_id",
            "treatment_time_m_a",
            "treatment_time_m_b",
            "survival_a",
            "survival_b",
            "survival_diff",
            "survival_ratio",
        ]
    ].sort_values(["draw", "time_m"]).reset_index(drop=True)


def predict_survival_contrast_summary(
    fit: SurvivalFit,
    *,
    x: Sequence[float],
    area_id: int,
    surv_breaks: Sequence[float],
    post_treatment_breaks: Sequence[float],
    eval_times: Sequence[float],
    treatment_time_m_a: float | None,
    treatment_time_m_b: float | None,
    draw_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Posterior summary of survival contrasts between two treatment-time scenarios.
    """
    draws = predict_survival_contrast_draws(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_treatment_breaks,
        eval_times=eval_times,
        treatment_time_m_a=treatment_time_m_a,
        treatment_time_m_b=treatment_time_m_b,
        draw_indices=draw_indices,
    )

    grouped = draws.groupby(
        ["time_m", "area_id", "treatment_time_m_a", "treatment_time_m_b"],
        as_index=False,
        dropna=False,
    )

    out = grouped.agg(
        mean_survival_a=("survival_a", "mean"),
        median_survival_a=("survival_a", "median"),
        q05_survival_a=("survival_a", lambda z: float(np.quantile(z, 0.05))),
        q95_survival_a=("survival_a", lambda z: float(np.quantile(z, 0.95))),
        mean_survival_b=("survival_b", "mean"),
        median_survival_b=("survival_b", "median"),
        q05_survival_b=("survival_b", lambda z: float(np.quantile(z, 0.05))),
        q95_survival_b=("survival_b", lambda z: float(np.quantile(z, 0.95))),
        mean_survival_diff=("survival_diff", "mean"),
        median_survival_diff=("survival_diff", "median"),
        q05_survival_diff=("survival_diff", lambda z: float(np.quantile(z, 0.05))),
        q95_survival_diff=("survival_diff", lambda z: float(np.quantile(z, 0.95))),
        mean_survival_ratio=("survival_ratio", "mean"),
        median_survival_ratio=("survival_ratio", "median"),
        q05_survival_ratio=("survival_ratio", lambda z: float(np.quantile(z, 0.05))),
        q95_survival_ratio=("survival_ratio", lambda z: float(np.quantile(z, 0.95))),
    )

    return out.sort_values("time_m").reset_index(drop=True)


def predict_rmst_contrast_draws(
    fit: SurvivalFit,
    *,
    x: Sequence[float],
    area_id: int,
    surv_breaks: Sequence[float],
    post_treatment_breaks: Sequence[float],
    horizon_m: float,
    treatment_time_m_a: float | None,
    treatment_time_m_b: float | None,
    grid_size: int = 400,
    draw_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Draw-level RMST contrast between two treatment-time scenarios.
    """
    rmst_a = _predict_rmst_draws(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_treatment_breaks,
        horizon_m=horizon_m,
        treatment_time_m=treatment_time_m_a,
        grid_size=grid_size,
        draw_indices=draw_indices,
    ).rename(columns={"rmst": "rmst_a"})

    rmst_b = _predict_rmst_draws(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_treatment_breaks,
        horizon_m=horizon_m,
        treatment_time_m=treatment_time_m_b,
        grid_size=grid_size,
        draw_indices=draw_indices,
    ).rename(columns={"rmst": "rmst_b"})

    merged = rmst_a.drop(columns=["treatment_time_m"]).merge(
        rmst_b.drop(columns=["treatment_time_m"]),
        on=["draw", "area_id", "horizon_m"],
        how="inner",
        validate="one_to_one",
    )

    merged["treatment_time_m_a"] = (
        np.nan if treatment_time_m_a is None else float(treatment_time_m_a)
    )
    merged["treatment_time_m_b"] = (
        np.nan if treatment_time_m_b is None else float(treatment_time_m_b)
    )
    merged["rmst_diff"] = merged["rmst_a"] - merged["rmst_b"]
    merged["rmst_ratio"] = np.where(
        merged["rmst_b"] > 0.0,
        merged["rmst_a"] / merged["rmst_b"],
        np.nan,
    )

    return merged[
        [
            "draw",
            "area_id",
            "horizon_m",
            "treatment_time_m_a",
            "treatment_time_m_b",
            "rmst_a",
            "rmst_b",
            "rmst_diff",
            "rmst_ratio",
        ]
    ].sort_values("draw").reset_index(drop=True)


def predict_rmst_contrast_summary(
    fit: SurvivalFit,
    *,
    x: Sequence[float],
    area_id: int,
    surv_breaks: Sequence[float],
    post_treatment_breaks: Sequence[float],
    horizon_m: float,
    treatment_time_m_a: float | None,
    treatment_time_m_b: float | None,
    grid_size: int = 400,
    draw_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Posterior summary of RMST contrasts between two treatment-time scenarios.
    """
    draws = predict_rmst_contrast_draws(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_treatment_breaks,
        horizon_m=horizon_m,
        treatment_time_m_a=treatment_time_m_a,
        treatment_time_m_b=treatment_time_m_b,
        grid_size=grid_size,
        draw_indices=draw_indices,
    )

    out = (
        draws.groupby(
            ["area_id", "horizon_m", "treatment_time_m_a", "treatment_time_m_b"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_rmst_a=("rmst_a", "mean"),
            median_rmst_a=("rmst_a", "median"),
            q05_rmst_a=("rmst_a", lambda z: float(np.quantile(z, 0.05))),
            q95_rmst_a=("rmst_a", lambda z: float(np.quantile(z, 0.95))),
            mean_rmst_b=("rmst_b", "mean"),
            median_rmst_b=("rmst_b", "median"),
            q05_rmst_b=("rmst_b", lambda z: float(np.quantile(z, 0.05))),
            q95_rmst_b=("rmst_b", lambda z: float(np.quantile(z, 0.95))),
            mean_rmst_diff=("rmst_diff", "mean"),
            median_rmst_diff=("rmst_diff", "median"),
            q05_rmst_diff=("rmst_diff", lambda z: float(np.quantile(z, 0.05))),
            q95_rmst_diff=("rmst_diff", lambda z: float(np.quantile(z, 0.95))),
            mean_rmst_ratio=("rmst_ratio", "mean"),
            median_rmst_ratio=("rmst_ratio", "median"),
            q05_rmst_ratio=("rmst_ratio", lambda z: float(np.quantile(z, 0.05))),
            q95_rmst_ratio=("rmst_ratio", lambda z: float(np.quantile(z, 0.95))),
        )
        .reset_index(drop=True)
    )

    return out


def predict_rmst_scenarios(
    fit: SurvivalFit,
    *,
    x: Sequence[float],
    area_id: int,
    surv_breaks: Sequence[float],
    post_treatment_breaks: Sequence[float],
    horizon_m: float,
    treatment_times_m: Sequence[float | None],
    grid_size: int = 400,
    draw_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper around predict_rmst(...) for multiple scenarios.

    This is mostly a thin pass-through, but it is useful as a standardized
    scenario table for later artifact writing.
    """
    return predict_rmst(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_treatment_breaks,
        horizon_m=horizon_m,
        treatment_times_m=treatment_times_m,
        grid_size=grid_size,
        draw_indices=draw_indices,
    )


def _predict_rmst_draws(
    fit: SurvivalFit,
    *,
    x: Sequence[float],
    area_id: int,
    surv_breaks: Sequence[float],
    post_treatment_breaks: Sequence[float],
    horizon_m: float,
    treatment_time_m: float | None,
    grid_size: int,
    draw_indices: Sequence[int] | None,
) -> pd.DataFrame:
    """
    Internal draw-level RMST helper for a single treatment-time scenario.
    """
    from seer_peph.predict.survival import predict_counterfactual_survival_draws, _rmst_from_survival_curve

    eval_times = np.linspace(0.0, float(horizon_m), int(grid_size))

    draws_df = predict_counterfactual_survival_draws(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_treatment_breaks,
        eval_times=eval_times,
        treatment_time_m=treatment_time_m,
        draw_indices=draw_indices,
    )

    rmst_by_draw = (
        draws_df.groupby("draw", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "rmst": _rmst_from_survival_curve(
                        times=g["time_m"].to_numpy(dtype=float),
                        survival=g["survival"].to_numpy(dtype=float),
                    )
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    rmst_by_draw["treatment_time_m"] = (
        np.nan if treatment_time_m is None else float(treatment_time_m)
    )
    rmst_by_draw["area_id"] = int(area_id)
    rmst_by_draw["horizon_m"] = float(horizon_m)

    return rmst_by_draw