from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from seer_peph.fitting.fit_models import SurvivalFit


@dataclass(frozen=True)
class SurvivalPredictionSpec:
    """
    Specification for counterfactual survival prediction.

    Parameters
    ----------
    surv_breaks
        Survival interval breakpoints in months.
    post_treatment_breaks
        Post-treatment interval breakpoints in months.
    eval_times
        Times in months at which survival should be returned.
    treatment_time_m
        Proposed treatment initiation time in months.
        Use None for a no-treatment scenario.
    """
    surv_breaks: tuple[float, ...]
    post_treatment_breaks: tuple[float, ...]
    eval_times: tuple[float, ...]
    treatment_time_m: float | None


def predict_counterfactual_survival_draws(
    fit: SurvivalFit,
    *,
    x: Sequence[float],
    area_id: int,
    surv_breaks: Sequence[float],
    post_treatment_breaks: Sequence[float],
    eval_times: Sequence[float],
    treatment_time_m: float | None,
    draw_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Return posterior-draw survival probabilities under a specified treatment-time scenario.

    Parameters
    ----------
    fit
        Fitted standalone survival model.
    x
        Survival covariate vector in the same order as `fit.metadata.surv_x_cols`.
    area_id
        Spatial area index.
    surv_breaks
        Survival interval breakpoints in months.
    post_treatment_breaks
        Post-treatment interval breakpoints in months since treatment.
    eval_times
        Times in months at which survival should be evaluated.
    treatment_time_m
        Proposed treatment initiation time in months. Use None for no treatment.
    draw_indices
        Optional subset of posterior draws to use.

    Returns
    -------
    pd.DataFrame
        Tidy table with one row per draw-time combination.
    """
    samples = fit.samples
    alpha = np.asarray(samples["alpha"], dtype=float)
    beta = np.asarray(samples["beta"], dtype=float)
    delta_post = np.asarray(samples["delta_post"], dtype=float)
    u = np.asarray(samples["u"], dtype=float)

    _validate_survival_prediction_inputs(
        alpha=alpha,
        beta=beta,
        delta_post=delta_post,
        u=u,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_treatment_breaks,
        eval_times=eval_times,
    )

    x_arr = np.asarray(x, dtype=float)
    surv_breaks_arr = np.asarray(surv_breaks, dtype=float)
    post_breaks_arr = np.asarray(post_treatment_breaks, dtype=float)
    eval_times_arr = np.asarray(eval_times, dtype=float)

    if draw_indices is None:
        draw_idx = np.arange(alpha.shape[0], dtype=int)
    else:
        draw_idx = np.asarray(draw_indices, dtype=int)

    rows: list[pd.DataFrame] = []

    for d in draw_idx:
        hazards = _compute_survival_interval_hazards_one_draw(
            alpha_draw=alpha[d],
            beta_draw=beta[d],
            delta_post_draw=delta_post[d],
            u_draw=u[d],
            x=x_arr,
            area_id=area_id,
            surv_breaks=surv_breaks_arr,
            post_treatment_breaks=post_breaks_arr,
            treatment_time_m=treatment_time_m,
        )

        surv_vals = _survival_curve_from_piecewise_hazards(
            hazards=hazards,
            breaks=surv_breaks_arr,
            eval_times=eval_times_arr,
        )

        rows.append(
            pd.DataFrame(
                {
                    "draw": int(d),
                    "time_m": eval_times_arr,
                    "survival": surv_vals,
                    "treatment_time_m": np.nan if treatment_time_m is None else float(treatment_time_m),
                    "area_id": int(area_id),
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def predict_counterfactual_survival_summary(
    fit: SurvivalFit,
    *,
    x: Sequence[float],
    area_id: int,
    surv_breaks: Sequence[float],
    post_treatment_breaks: Sequence[float],
    eval_times: Sequence[float],
    treatment_time_m: float | None,
    draw_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Summarize posterior counterfactual survival probabilities over draws.
    """
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

    out = (
        draws_df.groupby(
            ["time_m", "treatment_time_m", "area_id"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_survival=("survival", "mean"),
            median_survival=("survival", "median"),
            q05_survival=("survival", lambda z: float(np.quantile(z, 0.05))),
            q95_survival=("survival", lambda z: float(np.quantile(z, 0.95))),
        )
        .sort_values("time_m")
        .reset_index(drop=True)
    )
    return out


def predict_survival_at_times(
    fit: SurvivalFit,
    *,
    x: Sequence[float],
    area_id: int,
    surv_breaks: Sequence[float],
    post_treatment_breaks: Sequence[float],
    times: Sequence[float],
    treatment_times_m: Sequence[float | None],
    draw_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Compare survival at specified times across multiple treatment-time scenarios.
    """
    pieces: list[pd.DataFrame] = []

    for ttx in treatment_times_m:
        df = predict_counterfactual_survival_summary(
            fit,
            x=x,
            area_id=area_id,
            surv_breaks=surv_breaks,
            post_treatment_breaks=post_treatment_breaks,
            eval_times=times,
            treatment_time_m=ttx,
            draw_indices=draw_indices,
        )
        pieces.append(df)

    return pd.concat(pieces, ignore_index=True)


def predict_rmst(
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
    Compute RMST up to a fixed horizon for multiple treatment-time scenarios.
    """
    eval_times = np.linspace(0.0, float(horizon_m), int(grid_size))

    rows: list[pd.DataFrame] = []
    for ttx in treatment_times_m:
        draws_df = predict_counterfactual_survival_draws(
            fit,
            x=x,
            area_id=area_id,
            surv_breaks=surv_breaks,
            post_treatment_breaks=post_treatment_breaks,
            eval_times=eval_times,
            treatment_time_m=ttx,
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

        rmst_by_draw["treatment_time_m"] = np.nan if ttx is None else float(ttx)
        rows.append(rmst_by_draw)

    all_rmst = pd.concat(rows, ignore_index=True)

    out = (
        all_rmst.groupby("treatment_time_m", as_index=False, dropna=False)
        .agg(
            mean_rmst=("rmst", "mean"),
            median_rmst=("rmst", "median"),
            q05_rmst=("rmst", lambda z: float(np.quantile(z, 0.05))),
            q95_rmst=("rmst", lambda z: float(np.quantile(z, 0.95))),
        )
        .sort_values("treatment_time_m", na_position="first")
        .reset_index(drop=True)
    )
    out["horizon_m"] = float(horizon_m)
    out["area_id"] = int(area_id)
    return out


def _compute_survival_interval_hazards_one_draw(
    *,
    alpha_draw: np.ndarray,
    beta_draw: np.ndarray,
    delta_post_draw: np.ndarray,
    u_draw: np.ndarray,
    x: np.ndarray,
    area_id: int,
    surv_breaks: np.ndarray,
    post_treatment_breaks: np.ndarray,
    treatment_time_m: float | None,
) -> np.ndarray:
    """
    Compute interval-specific hazards on the survival break grid for one posterior draw.
    """
    K_surv = len(surv_breaks) - 1
    hazards = np.zeros(K_surv, dtype=float)

    eta_baseline = float(np.dot(x, beta_draw) + u_draw[area_id])

    for k in range(K_surv):
        start = float(surv_breaks[k])
        stop = float(surv_breaks[k + 1])

        eta = float(alpha_draw[k] + eta_baseline)

        if treatment_time_m is not None and stop > treatment_time_m:
            time_since_treatment = max(start - treatment_time_m, 0.0)
            post_idx = _post_treatment_interval_index(
                elapsed_m=time_since_treatment,
                post_treatment_breaks=post_treatment_breaks,
            )
            eta += float(delta_post_draw[post_idx])

        hazards[k] = np.exp(eta)

    return hazards


def _survival_curve_from_piecewise_hazards(
    *,
    hazards: np.ndarray,
    breaks: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    """
    Evaluate S(t) on an arbitrary time grid given piecewise-constant hazards.
    """
    out = np.empty(eval_times.shape[0], dtype=float)

    for i, t in enumerate(eval_times):
        if t <= 0.0:
            out[i] = 1.0
            continue

        cum_hazard = 0.0
        for k in range(len(hazards)):
            start = float(breaks[k])
            stop = float(breaks[k + 1])

            if t <= start:
                break

            dt = min(t, stop) - start
            if dt > 0.0:
                cum_hazard += hazards[k] * dt

            if t <= stop:
                break

        out[i] = np.exp(-cum_hazard)

    return out


def _rmst_from_survival_curve(
    *,
    times: np.ndarray,
    survival: np.ndarray,
) -> float:
    """
    Numerical integral of survival curve over time.
    """
    if times.ndim != 1 or survival.ndim != 1 or times.shape != survival.shape:
        raise ValueError("times and survival must be 1-D arrays of equal length.")
    return float(np.trapz(survival, times))


def _post_treatment_interval_index(
    *,
    elapsed_m: float,
    post_treatment_breaks: np.ndarray,
) -> int:
    """
    Map elapsed time since treatment to the corresponding post-treatment interval index.
    """
    if elapsed_m < 0.0:
        raise ValueError("elapsed_m must be non-negative.")

    idx = int(np.searchsorted(post_treatment_breaks, elapsed_m, side="right") - 1)
    idx = max(idx, 0)
    idx = min(idx, len(post_treatment_breaks) - 2)
    return idx


def _validate_survival_prediction_inputs(
    *,
    alpha: np.ndarray,
    beta: np.ndarray,
    delta_post: np.ndarray,
    u: np.ndarray,
    x: Sequence[float],
    area_id: int,
    surv_breaks: Sequence[float],
    post_treatment_breaks: Sequence[float],
    eval_times: Sequence[float],
) -> None:
    x_arr = np.asarray(x, dtype=float)
    surv_breaks_arr = np.asarray(surv_breaks, dtype=float)
    post_breaks_arr = np.asarray(post_treatment_breaks, dtype=float)
    eval_times_arr = np.asarray(eval_times, dtype=float)

    if alpha.ndim != 2:
        raise ValueError("alpha samples must be a 2-D array.")
    if beta.ndim != 2:
        raise ValueError("beta samples must be a 2-D array.")
    if delta_post.ndim != 2:
        raise ValueError("delta_post samples must be a 2-D array.")
    if u.ndim != 2:
        raise ValueError("u samples must be a 2-D array.")

    n_draws = alpha.shape[0]
    if beta.shape[0] != n_draws or delta_post.shape[0] != n_draws or u.shape[0] != n_draws:
        raise ValueError("All posterior sample blocks must have the same number of draws.")

    if x_arr.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if x_arr.shape[0] != beta.shape[1]:
        raise ValueError("Length of x must match number of survival covariates.")

    if not (0 <= area_id < u.shape[1]):
        raise ValueError("area_id is out of range for the fitted spatial field.")

    if surv_breaks_arr.ndim != 1 or len(surv_breaks_arr) < 2:
        raise ValueError("surv_breaks must be a one-dimensional sequence of length >= 2.")
    if np.any(np.diff(surv_breaks_arr) <= 0.0):
        raise ValueError("surv_breaks must be strictly increasing.")
    if alpha.shape[1] != len(surv_breaks_arr) - 1:
        raise ValueError("Number of alpha intervals does not match surv_breaks.")

    if post_breaks_arr.ndim != 1 or len(post_breaks_arr) < 2:
        raise ValueError("post_treatment_breaks must be a one-dimensional sequence of length >= 2.")
    if np.any(np.diff(post_breaks_arr) <= 0.0):
        raise ValueError("post_treatment_breaks must be strictly increasing.")
    if delta_post.shape[1] != len(post_breaks_arr) - 1:
        raise ValueError("Number of delta_post intervals does not match post_treatment_breaks.")

    if eval_times_arr.ndim != 1 or len(eval_times_arr) == 0:
        raise ValueError("eval_times must be a nonempty one-dimensional sequence.")
    if np.any(eval_times_arr < 0.0):
        raise ValueError("eval_times must be non-negative.")
    if np.any(np.diff(np.sort(eval_times_arr)) < 0.0):
        raise ValueError("eval_times must be sortable.")