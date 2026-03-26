from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from seer_peph.fitting.fit_models import TreatmentFit


def treatment_ppc_interval_counts(
    fit: TreatmentFit,
    ttt_long: pd.DataFrame,
    *,
    draw_indices: Sequence[int] | None = None,
    sample_posterior_predictive: bool = True,
    random_seed: int = 123,
) -> pd.DataFrame:
    """
    Posterior predictive check for treatment event counts by treatment interval.

    Returns one row per interval with:
    - observed event count
    - observed exposure
    - posterior predictive mean/median/quantiles of the interval event count
    - posterior predictive mean event rate per unit exposure
    """
    df = _prepare_ttt_long(fit, ttt_long)
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


def treatment_ppc_area_counts(
    fit: TreatmentFit,
    ttt_long: pd.DataFrame,
    *,
    draw_indices: Sequence[int] | None = None,
    sample_posterior_predictive: bool = True,
    random_seed: int = 123,
) -> pd.DataFrame:
    """
    Posterior predictive check for treatment event counts by area.
    """
    df = _prepare_ttt_long(fit, ttt_long)
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


def treatment_ppc_row_expectations(
    fit: TreatmentFit,
    ttt_long: pd.DataFrame,
    *,
    draw_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Return row-level posterior expected event counts for each treatment long row.
    """
    df = _prepare_ttt_long(fit, ttt_long)
    draw_matrix = _rowwise_expected_counts(
        fit,
        df,
        draw_indices=draw_indices,
    )

    out = df[["id", "k", "t0", "t1", "exposure", "event", "area_id"]].copy()
    out["pp_mean_mu"] = draw_matrix.mean(axis=0)
    out["pp_median_mu"] = np.median(draw_matrix, axis=0)
    out["pp_q05_mu"] = np.quantile(draw_matrix, 0.05, axis=0)
    out["pp_q95_mu"] = np.quantile(draw_matrix, 0.95, axis=0)
    return out


def _prepare_ttt_long(fit: TreatmentFit, ttt_long: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "k", "exposure", "event", "area_id"}
    x_cols = tuple(fit.metadata.ttt_x_cols or ())
    required |= set(x_cols)

    missing = [c for c in required if c not in ttt_long.columns]
    if missing:
        raise ValueError(f"ttt_long is missing required columns: {missing}")

    df = ttt_long.copy()

    df["k"] = df["k"].astype(int)
    df["event"] = df["event"].astype(int)
    df["area_id"] = df["area_id"].astype(int)
    df["exposure"] = df["exposure"].astype(float)

    return df


def _rowwise_expected_counts(
    fit: TreatmentFit,
    ttt_long: pd.DataFrame,
    *,
    draw_indices: Sequence[int] | None,
) -> np.ndarray:
    """
    Return posterior expected event counts mu_{d,r} for each draw d and row r.

    Shape: (n_draws_used, n_rows)
    """
    samples = fit.samples

    theta = np.asarray(samples["theta"], dtype=float)
    gamma = np.asarray(samples["gamma"], dtype=float)
    u = np.asarray(samples["u"], dtype=float)

    if draw_indices is None:
        draw_idx = np.arange(theta.shape[0], dtype=int)
    else:
        draw_idx = np.asarray(draw_indices, dtype=int)

    theta = theta[draw_idx]
    gamma = gamma[draw_idx]
    u = u[draw_idx]

    x_cols = list(fit.metadata.ttt_x_cols or ())
    X = ttt_long[x_cols].to_numpy(dtype=float)  # (R, P)
    k = ttt_long["k"].to_numpy(dtype=int)  # (R,)
    area_id = ttt_long["area_id"].to_numpy(dtype=int)  # (R,)
    exposure = ttt_long["exposure"].to_numpy(dtype=float)  # (R,)

    _validate_ppc_shapes(
        theta=theta,
        gamma=gamma,
        u=u,
        X=X,
        k=k,
        area_id=area_id,
    )

    # Linear predictor in (D, R) orientation:
    #   gamma[:, k] -> (D, R)
    #   theta @ X.T -> (D, R)
    #   u[:, area_id] -> (D, R)
    eta = (
        gamma[:, k]
        + theta @ X.T
        + u[:, area_id]
    )

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
        idx = np.flatnonzero(groups.to_numpy(object) == g)

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
    theta: np.ndarray,
    gamma: np.ndarray,
    u: np.ndarray,
    X: np.ndarray,
    k: np.ndarray,
    area_id: np.ndarray,
) -> None:
    if theta.ndim != 2:
        raise ValueError("theta must be 2-dimensional.")
    if gamma.ndim != 2:
        raise ValueError("gamma must be 2-dimensional.")
    if u.ndim != 2:
        raise ValueError("u must be 2-dimensional.")
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional.")

    n_draws = theta.shape[0]
    if gamma.shape[0] != n_draws or u.shape[0] != n_draws:
        raise ValueError("All sample blocks must have the same number of draws.")

    if X.shape[1] != theta.shape[1]:
        raise ValueError("X column count does not match theta dimension.")

    if np.any(k < 0) or np.any(k >= gamma.shape[1]):
        raise ValueError("k contains out-of-range interval indices.")

    if np.any(area_id < 0) or np.any(area_id >= u.shape[1]):
        raise ValueError("area_id contains out-of-range spatial indices.")