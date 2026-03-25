from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from seer_peph.fitting.fit_models import JointFit, SurvivalFit, TreatmentFit


def extract_survival_effects(
    fit: SurvivalFit | JointFit,
    *,
    include_draws: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Extract tidy survival-side parameter summaries from a fitted survival or joint model.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys:
        - "beta": baseline survival covariate effects
        - "alpha": survival baseline interval effects
        - "delta_post": post-treatment survival effects

        If `include_draws=True`, also includes:
        - "beta_draws"
        - "alpha_draws"
        - "delta_post_draws"
    """
    samples = fit.samples
    scalar_summary = fit.scalar_summary
    surv_x_cols = tuple(fit.metadata.surv_x_cols)

    out: dict[str, pd.DataFrame] = {}

    if "beta" in samples:
        beta_draws = np.asarray(samples["beta"])
        out["beta"] = _extract_vector_param_summary(
            scalar_summary=scalar_summary,
            base_name="beta",
            labels=surv_x_cols if surv_x_cols else _default_labels("beta", beta_draws.shape[1]),
            param_type="survival_beta",
        )
        if include_draws:
            out["beta_draws"] = _extract_vector_draws(
                draws=beta_draws,
                base_name="beta",
                labels=surv_x_cols if surv_x_cols else _default_labels("beta", beta_draws.shape[1]),
            )

    if "alpha" in samples:
        alpha_draws = np.asarray(samples["alpha"])
        out["alpha"] = _extract_vector_param_summary(
            scalar_summary=scalar_summary,
            base_name="alpha",
            labels=_default_interval_labels("survival_interval", alpha_draws.shape[1]),
            param_type="survival_alpha",
        )
        if include_draws:
            out["alpha_draws"] = _extract_vector_draws(
                draws=alpha_draws,
                base_name="alpha",
                labels=_default_interval_labels("survival_interval", alpha_draws.shape[1]),
            )

    if "delta_post" in samples:
        delta_draws = np.asarray(samples["delta_post"])
        out["delta_post"] = _extract_vector_param_summary(
            scalar_summary=scalar_summary,
            base_name="delta_post",
            labels=_default_interval_labels("post_treatment_interval", delta_draws.shape[1]),
            param_type="survival_delta_post",
        )
        if include_draws:
            out["delta_post_draws"] = _extract_vector_draws(
                draws=delta_draws,
                base_name="delta_post",
                labels=_default_interval_labels("post_treatment_interval", delta_draws.shape[1]),
            )

    return out


def extract_treatment_effects(
    fit: TreatmentFit | JointFit,
    *,
    include_draws: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Extract tidy treatment-side parameter summaries from a fitted treatment or joint model.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys:
        - "theta": baseline treatment covariate effects
        - "gamma": treatment baseline interval effects

        If `include_draws=True`, also includes:
        - "theta_draws"
        - "gamma_draws"
    """
    samples = fit.samples
    scalar_summary = fit.scalar_summary
    ttt_x_cols = tuple(fit.metadata.ttt_x_cols)

    out: dict[str, pd.DataFrame] = {}

    if "theta" in samples:
        theta_draws = np.asarray(samples["theta"])
        out["theta"] = _extract_vector_param_summary(
            scalar_summary=scalar_summary,
            base_name="theta",
            labels=ttt_x_cols if ttt_x_cols else _default_labels("theta", theta_draws.shape[1]),
            param_type="treatment_theta",
        )
        if include_draws:
            out["theta_draws"] = _extract_vector_draws(
                draws=theta_draws,
                base_name="theta",
                labels=ttt_x_cols if ttt_x_cols else _default_labels("theta", theta_draws.shape[1]),
            )

    if "gamma" in samples:
        gamma_draws = np.asarray(samples["gamma"])
        out["gamma"] = _extract_vector_param_summary(
            scalar_summary=scalar_summary,
            base_name="gamma",
            labels=_default_interval_labels("treatment_interval", gamma_draws.shape[1]),
            param_type="treatment_gamma",
        )
        if include_draws:
            out["gamma_draws"] = _extract_vector_draws(
                draws=gamma_draws,
                base_name="gamma",
                labels=_default_interval_labels("treatment_interval", gamma_draws.shape[1]),
            )

    return out


def extract_spatial_fields(
    fit: SurvivalFit | TreatmentFit | JointFit,
    *,
    include_draws: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Extract spatial latent fields and hyperparameters.

    Returns
    -------
    dict[str, pd.DataFrame]
        For survival-only or treatment-only fits:
        - "field": area-level latent field summary
        - "hyperparameters": spatial hyperparameter summary

        For joint fits:
        - "u_surv": survival area-level latent field summary
        - "u_ttt": treatment area-level latent field summary
        - "u_ttt_ind": independent treatment field summary (if present)
        - "s_surv": structured survival component summary (if present)
        - "s_ttt": structured treatment component summary (if present)
        - "hyperparameters": spatial hyperparameter summary

        If `include_draws=True`, corresponding `*_draws` tables are added.
    """
    samples = fit.samples
    scalar_summary = fit.scalar_summary
    out: dict[str, pd.DataFrame] = {}

    if isinstance(fit, SurvivalFit):
        if "u" in samples:
            u_draws = np.asarray(samples["u"])
            out["field"] = _extract_area_param_summary(
                scalar_summary=scalar_summary,
                base_name="u",
                n_areas=u_draws.shape[1],
                field_name="u",
                field_type="survival_field",
            )
            if include_draws:
                out["field_draws"] = _extract_area_draws(
                    draws=u_draws,
                    base_name="u",
                    field_name="u",
                )

        out["hyperparameters"] = _extract_scalar_rows(
            scalar_summary=scalar_summary,
            names=["rho", "tau"],
            param_group="survival_spatial_hyper",
        )
        return out

    if isinstance(fit, TreatmentFit):
        if "u" in samples:
            u_draws = np.asarray(samples["u"])
            out["field"] = _extract_area_param_summary(
                scalar_summary=scalar_summary,
                base_name="u",
                n_areas=u_draws.shape[1],
                field_name="u",
                field_type="treatment_field",
            )
            if include_draws:
                out["field_draws"] = _extract_area_draws(
                    draws=u_draws,
                    base_name="u",
                    field_name="u",
                )

        out["hyperparameters"] = _extract_scalar_rows(
            scalar_summary=scalar_summary,
            names=["rho", "tau"],
            param_group="treatment_spatial_hyper",
        )
        return out

    # Joint fit
    joint_field_specs = [
        ("u_surv", "joint_survival_field"),
        ("u_ttt", "joint_treatment_field"),
        ("u_ttt_ind", "joint_treatment_independent_field"),
        ("s_surv", "joint_survival_structured"),
        ("s_ttt", "joint_treatment_structured"),
    ]

    for base_name, field_type in joint_field_specs:
        if base_name in samples:
            draws = np.asarray(samples[base_name])
            out[base_name] = _extract_area_param_summary(
                scalar_summary=scalar_summary,
                base_name=base_name,
                n_areas=draws.shape[1],
                field_name=base_name,
                field_type=field_type,
            )
            if include_draws:
                out[f"{base_name}_draws"] = _extract_area_draws(
                    draws=draws,
                    base_name=base_name,
                    field_name=base_name,
                )

    out["hyperparameters"] = _extract_scalar_rows(
        scalar_summary=scalar_summary,
        names=["rho_surv", "tau_surv", "rho_ttt", "tau_ttt"],
        param_group="joint_spatial_hyper",
    )
    return out


def extract_joint_coupling(
    fit: JointFit,
    *,
    include_draws: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Extract joint-model coupling summaries.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys:
        - "coupling": scalar coupling parameter summary
        - "field_correlations": posterior draw correlations between spatial fields

        If `include_draws=True`, also includes:
        - "coupling_draws"
    """
    if not isinstance(fit, JointFit):
        raise TypeError("extract_joint_coupling requires a JointFit object.")

    out: dict[str, pd.DataFrame] = {}

    out["coupling"] = _extract_scalar_rows(
        scalar_summary=fit.scalar_summary,
        names=["rho_u_cross"],
        param_group="joint_coupling",
    )

    if include_draws and "rho_u_cross" in fit.samples:
        rho_draws = np.asarray(fit.samples["rho_u_cross"]).reshape(-1)
        out["coupling_draws"] = pd.DataFrame(
            {
                "draw": np.arange(rho_draws.shape[0], dtype=int),
                "parameter": "rho_u_cross",
                "value": rho_draws,
            }
        )

    if "u_surv" in fit.samples and "u_ttt" in fit.samples:
        u_surv_draws = np.asarray(fit.samples["u_surv"])
        u_ttt_draws = np.asarray(fit.samples["u_ttt"])

        if u_surv_draws.shape != u_ttt_draws.shape:
            raise ValueError("u_surv and u_ttt draws must have matching shapes.")

        corrs = np.array(
            [_safe_corr(u_surv_draws[i, :], u_ttt_draws[i, :]) for i in range(u_surv_draws.shape[0])],
            dtype=float,
        )

        out["field_correlations"] = pd.DataFrame(
            {
                "metric": [
                    "corr_u_surv_u_ttt_draw_mean",
                    "corr_u_surv_u_ttt_draw_sd",
                    "corr_u_surv_u_ttt_draw_median",
                    "corr_u_surv_u_ttt_draw_q05",
                    "corr_u_surv_u_ttt_draw_q95",
                ],
                "value": [
                    float(np.nanmean(corrs)),
                    float(np.nanstd(corrs, ddof=1)) if np.sum(np.isfinite(corrs)) > 1 else np.nan,
                    float(np.nanmedian(corrs)),
                    float(np.nanquantile(corrs, 0.05)),
                    float(np.nanquantile(corrs, 0.95)),
                ],
            }
        )

    return out


def _extract_vector_param_summary(
    *,
    scalar_summary: Mapping[str, Mapping[str, float]],
    base_name: str,
    labels: tuple[str, ...],
    param_type: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for j, label in enumerate(labels):
        key = f"{base_name}[{j}]"
        if key not in scalar_summary:
            continue
        stats = scalar_summary[key]
        rows.append(
            {
                "parameter": key,
                "index": j,
                "label": label,
                "param_type": param_type,
                "mean": _get_stat(stats, "mean"),
                "sd": _get_stat(stats, "sd"),
                "median": _get_stat(stats, "median"),
                "q05": _get_stat(stats, "q05"),
                "q95": _get_stat(stats, "q95"),
            }
        )
    return pd.DataFrame(rows)


def _extract_vector_draws(
    *,
    draws: np.ndarray,
    base_name: str,
    labels: tuple[str, ...],
) -> pd.DataFrame:
    if draws.ndim != 2:
        raise ValueError(f"{base_name} draws must be 2-D with shape (n_draws, n_params).")

    rows: list[pd.DataFrame] = []
    for j, label in enumerate(labels):
        rows.append(
            pd.DataFrame(
                {
                    "draw": np.arange(draws.shape[0], dtype=int),
                    "parameter": f"{base_name}[{j}]",
                    "index": j,
                    "label": label,
                    "value": draws[:, j],
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["draw", "parameter", "index", "label", "value"]
    )


def _extract_area_param_summary(
    *,
    scalar_summary: Mapping[str, Mapping[str, float]],
    base_name: str,
    n_areas: int,
    field_name: str,
    field_type: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for a in range(n_areas):
        key = f"{base_name}[{a}]"
        if key not in scalar_summary:
            continue
        stats = scalar_summary[key]
        rows.append(
            {
                "parameter": key,
                "field": field_name,
                "field_type": field_type,
                "area_id": a,
                "mean": _get_stat(stats, "mean"),
                "sd": _get_stat(stats, "sd"),
                "median": _get_stat(stats, "median"),
                "q05": _get_stat(stats, "q05"),
                "q95": _get_stat(stats, "q95"),
            }
        )
    return pd.DataFrame(rows)


def _extract_area_draws(
    *,
    draws: np.ndarray,
    base_name: str,
    field_name: str,
) -> pd.DataFrame:
    if draws.ndim != 2:
        raise ValueError(f"{base_name} draws must be 2-D with shape (n_draws, n_areas).")

    rows: list[pd.DataFrame] = []
    for a in range(draws.shape[1]):
        rows.append(
            pd.DataFrame(
                {
                    "draw": np.arange(draws.shape[0], dtype=int),
                    "parameter": f"{base_name}[{a}]",
                    "field": field_name,
                    "area_id": a,
                    "value": draws[:, a],
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["draw", "parameter", "field", "area_id", "value"]
    )


def _extract_scalar_rows(
    *,
    scalar_summary: Mapping[str, Mapping[str, float]],
    names: list[str],
    param_group: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in names:
        if name not in scalar_summary:
            continue
        stats = scalar_summary[name]
        rows.append(
            {
                "parameter": name,
                "param_group": param_group,
                "mean": _get_stat(stats, "mean"),
                "sd": _get_stat(stats, "sd"),
                "median": _get_stat(stats, "median"),
                "q05": _get_stat(stats, "q05"),
                "q95": _get_stat(stats, "q95"),
            }
        )
    return pd.DataFrame(rows)


def _get_stat(stats: Mapping[str, float], key: str) -> float:
    if key not in stats:
        return np.nan
    val = stats[key]
    return float(val) if val is not None else np.nan


def _default_labels(prefix: str, n: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{j}" for j in range(n))


def _default_interval_labels(prefix: str, n: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{j}" for j in range(n))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])