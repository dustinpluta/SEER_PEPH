from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from seer_peph.fitting.fit_models import SurvivalFit


def survival_ppc_interval_counts(
    fit: SurvivalFit,
    surv_long: pd.DataFrame,
    *,
    draw_indices: Sequence[int] | None = None,
    sample_posterior_predictive: bool = True,
    random_seed: int = 123,
) -> pd.DataFrame:
    """
    Posterior predictive check for survival event counts by survival interval.

    Returns one row per interval with:
    - observed event count
    - observed exposure
    - posterior predictive mean/median/quantiles of the interval event count
    - posterior predictive mean event rate per unit exposure

    Parameters
    ----------
    fit
        Fitted standalone survival model.
    surv_long
        Survival long-format dataframe.
    draw_indices
        Optional subset of posterior draws to use.
    sample_posterior_predictive
        If True, simulate posterior predictive counts using Poisson draws.
        If False, use posterior expected counts only.
    random_seed
        Seed used for posterior predictive simulation.

    Returns
    -------
    pd.DataFrame
    """
    df = _prepare_surv_long(fit, surv_long)
    draw_matrix = _rowwise_expected_counts(
        fit,
        df,
        draw_indices=draw_indices,
    )

    agg = _aggregate_draw_matrix(
        group_values=df["k"].to_numpy(dtype=int),
        draw_matrix=draw_matrix,
        observed_events=df["event"].to_numpy(dtype=int),
        exposure=df["exposure"].to_numpy(dtype=float),
        group_name="k",
        sample_posterior_predictive=sample_posterior_predictive,
        random_seed=random_seed,
    )

    agg["pp_mean_rate"] = agg["pp_mean_events"] / agg["observed_exposure"]
    agg["observed_rate"] = agg["observed_events"] / agg["observed_exposure"]
    return agg.sort_values("k").reset_index(drop=True)


def survival_ppc_area_counts(
    fit: SurvivalFit,
    surv_long: pd.DataFrame,
    *,
    draw_indices: Sequence[int] | None = None,
    sample_posterior_predictive: bool = True,
    random_seed: int = 123,
) -> pd.DataFrame:
    """
    Posterior predictive check for survival event counts by area.
    """
    df = _prepare_surv_long(fit, surv_long)
    draw_matrix = _rowwise_expected_counts(
        fit,
        df,
        draw_indices=draw_indices,
    )

    agg = _aggregate_draw_matrix(
        group_values=df["area_id"].to_numpy(dtype=int),
        draw_matrix=draw_matrix,
        observed_events=df["event"].to_numpy(dtype=int),
        exposure=df["exposure"].to_numpy(dtype=float),
        group_name="area_id",
        sample_posterior_predictive=sample_posterior_predictive,
        random_seed=random_seed,
    )

    agg["pp_mean_rate"] = agg["pp_mean_events"] / agg["observed_exposure"]
    agg["observed_rate"] = agg["observed_events"] / agg["observed_exposure"]
    return agg.sort_values("area_id").reset_index(drop=True)


def survival_ppc_interval_by_treatment_counts(
    fit: SurvivalFit,
    surv_long: pd.DataFrame,
    *,
    draw_indices: Sequence[int] | None = None,
    sample_posterior_predictive: bool = True,
    random_seed: int = 123,
) -> pd.DataFrame:
    """
    Posterior predictive check for survival event counts by (interval, treated_td).
    """
    df = _prepare_surv_long(fit, surv_long)
    draw_matrix = _rowwise_expected_counts(
        fit,
        df,
        draw_indices=draw_indices,
    )

    key = (
        df["k"].astype(str)
        + "|"
        + df["treated_td"].astype(str)
    ).to_numpy(dtype=object)

    agg = _aggregate_draw_matrix(
        group_values=key,
        draw_matrix=draw_matrix,
        observed_events=df["event"].to_numpy(dtype=int),
        exposure=df["exposure"].to_numpy(dtype=float),
        group_name="interval_treated_key",
        sample_posterior_predictive=sample_posterior_predictive,
        random_seed=random_seed,
    )

    split_key = agg["interval_treated_key"].str.split("|", expand=True)
    agg["k"] = split_key[0].astype(int)
    agg["treated_td"] = split_key[1].astype(int)
    agg = agg.drop(columns=["interval_treated_key"])

    agg["pp_mean_rate"] = agg["pp_mean_events"] / agg["observed_exposure"]
    agg["observed_rate"] = agg["observed_events"] / agg["observed_exposure"]
    return agg.sort_values(["k", "treated_td"]).reset_index(drop=True)

def survival_ppc_row_expectations(
    fit: SurvivalFit,
    surv_long: pd.DataFrame,
    *,
    draw_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Return row-level posterior expected event counts for each survival long row.

    This is useful for custom downstream diagnostics beyond the provided
    interval/area summaries.
    """
    df = _prepare_surv_long(fit, surv_long)
    draw_matrix = _rowwise_expected_counts(
        fit,
        df,
        draw_indices=draw_indices,
    )

    out = df[["id", "k", "t0", "t1", "exposure", "event", "area_id", "treated_td", "k_post"]].copy()
    out["pp_mean_mu"] = draw_matrix.mean(axis=0)
    out["pp_median_mu"] = np.median(draw_matrix, axis=0)
    out["pp_q05_mu"] = np.quantile(draw_matrix, 0.05, axis=0)
    out["pp_q95_mu"] = np.quantile(draw_matrix, 0.95, axis=0)
    return out


def _prepare_surv_long(fit: SurvivalFit, surv_long: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "k", "exposure", "event", "area_id", "treated_td", "k_post"}
    x_cols = tuple(fit.metadata.surv_x_cols or ())
    required |= set(x_cols)

    missing = [c for c in required if c not in surv_long.columns]
    if missing:
        raise ValueError(f"surv_long is missing required columns: {missing}")

    df = surv_long.copy()

    df["k"] = df["k"].astype(int)
    df["event"] = df["event"].astype(int)
    df["area_id"] = df["area_id"].astype(int)
    df["treated_td"] = df["treated_td"].astype(int)
    df["k_post"] = df["k_post"].astype(int)
    df["exposure"] = df["exposure"].astype(float)

    return df


def _rowwise_expected_counts(
    fit: SurvivalFit,
    surv_long: pd.DataFrame,
    *,
    draw_indices: Sequence[int] | None,
) -> np.ndarray:
    """
    Return posterior expected event counts mu_{d,r} for each draw d and row r.

    Shape: (n_draws_used, n_rows)
    """
    samples = fit.samples

    alpha = np.asarray(samples["alpha"], dtype=float)
    beta = np.asarray(samples["beta"], dtype=float)
    delta_post = np.asarray(samples["delta_post"], dtype=float)
    u = np.asarray(samples["u"], dtype=float)

    if draw_indices is None:
        draw_idx = np.arange(alpha.shape[0], dtype=int)
    else:
        draw_idx = np.asarray(draw_indices, dtype=int)

    alpha = alpha[draw_idx]
    beta = beta[draw_idx]
    delta_post = delta_post[draw_idx]
    u = u[draw_idx]

    x_cols = list(fit.metadata.surv_x_cols or ())
    X = surv_long[x_cols].to_numpy(dtype=float)  # (R, P)
    k = surv_long["k"].to_numpy(dtype=int)  # (R,)
    area_id = surv_long["area_id"].to_numpy(dtype=int)  # (R,)
    treated_td = surv_long["treated_td"].to_numpy(dtype=float)  # (R,)
    k_post = surv_long["k_post"].to_numpy(dtype=int)  # (R,)
    exposure = surv_long["exposure"].to_numpy(dtype=float)  # (R,)

    _validate_ppc_shapes(
        alpha=alpha,
        beta=beta,
        delta_post=delta_post,
        u=u,
        X=X,
        k=k,
        area_id=area_id,
        k_post=k_post,
    )

    # Baseline linear predictor in (D, R) orientation:
    #   alpha[:, k]          -> (D, R)
    #   beta @ X.T           -> (D, R)
    #   u[:, area_id]        -> (D, R)
    eta = (
        alpha[:, k]
        + beta @ X.T
        + u[:, area_id]
    )

    # Add post-treatment effect only for treated rows.
    treated_mask = treated_td > 0.0
    if treated_mask.any():
        treated_cols = np.where(treated_mask)[0]
        eta[:, treated_cols] += delta_post[:, k_post[treated_cols]]

    mu = exposure[None, :] * np.exp(eta)
    return mu


def _aggregate_draw_matrix(
    *,
    group_values: np.ndarray,
    draw_matrix: np.ndarray,
    observed_events: np.ndarray,
    exposure: np.ndarray,
    group_name: str,
    sample_posterior_predictive: bool,
    random_seed: int,
) -> pd.DataFrame:
    """
    Aggregate row-level expected counts into grouped posterior predictive summaries.
    """
    if draw_matrix.ndim != 2:
        raise ValueError("draw_matrix must be 2-dimensional.")

    groups = pd.Series(group_values, name=group_name)
    unique_groups = list(pd.unique(groups))

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []

    for g in unique_groups:
        idx = np.where(groups.to_numpy() == g)[0]

        mu_g = draw_matrix[:, idx].sum(axis=1)
        if sample_posterior_predictive:
            yrep_g = rng.poisson(mu_g)
            draw_counts = yrep_g
        else:
            draw_counts = mu_g

        obs_g = int(observed_events[idx].sum())
        exp_g = float(exposure[idx].sum())

        rows.append(
            {
                group_name: g,
                "observed_events": obs_g,
                "observed_exposure": exp_g,
                "pp_mean_events": float(draw_counts.mean()),
                "pp_median_events": float(np.median(draw_counts)),
                "pp_q05_events": float(np.quantile(draw_counts, 0.05)),
                "pp_q95_events": float(np.quantile(draw_counts, 0.95)),
                "pp_mean_expected_events": float(mu_g.mean()),
                "pp_median_expected_events": float(np.median(mu_g)),
            }
        )

    return pd.DataFrame(rows)


def _validate_ppc_shapes(
    *,
    alpha: np.ndarray,
    beta: np.ndarray,
    delta_post: np.ndarray,
    u: np.ndarray,
    X: np.ndarray,
    k: np.ndarray,
    area_id: np.ndarray,
    k_post: np.ndarray,
) -> None:
    if alpha.ndim != 2:
        raise ValueError("alpha must be 2-dimensional.")
    if beta.ndim != 2:
        raise ValueError("beta must be 2-dimensional.")
    if delta_post.ndim != 2:
        raise ValueError("delta_post must be 2-dimensional.")
    if u.ndim != 2:
        raise ValueError("u must be 2-dimensional.")
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional.")

    n_draws = alpha.shape[0]
    if beta.shape[0] != n_draws or delta_post.shape[0] != n_draws or u.shape[0] != n_draws:
        raise ValueError("All sample blocks must have the same number of draws.")

    if X.shape[1] != beta.shape[1]:
        raise ValueError("X column count does not match beta dimension.")

    if np.any(k < 0) or np.any(k >= alpha.shape[1]):
        raise ValueError("k contains out-of-range interval indices.")

    if np.any(area_id < 0) or np.any(area_id >= u.shape[1]):
        raise ValueError("area_id contains out-of-range spatial indices.")

    valid_k_post = k_post[k_post >= 0]
    if valid_k_post.size > 0 and (np.any(valid_k_post >= delta_post.shape[1])):
        raise ValueError("k_post contains out-of-range post-treatment interval indices.")