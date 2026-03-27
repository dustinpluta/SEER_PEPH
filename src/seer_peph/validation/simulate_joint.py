from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from seer_peph.data.prep import DAYS_PER_MONTH
from seer_peph.validation.joint_results import JointSimulationResult
from seer_peph.validation.joint_scenarios import JointSimulationScenario


def simulate_joint_scenario(
    scenario: JointSimulationScenario,
    *,
    seed: int,
) -> JointSimulationResult:
    rng = np.random.default_rng(seed)

    neighbors = _build_neighbors_for_scenario(scenario)

    u_surv_true, u_ttt_true, u_ttt_ind_true = _sample_joint_spatial_fields(
        neighbors=neighbors,
        rng=rng,
        phi_surv=float(scenario.phi_surv),
        phi_ttt=float(scenario.phi_ttt),
        sigma_surv=float(scenario.sigma_surv),
        sigma_ttt=float(scenario.sigma_ttt),
        rho_u=float(scenario.rho_u),
        sigma_ttt_ind=(
            float(scenario.sigma_ttt_ind)
            if scenario.sigma_ttt_ind is not None
            else float(scenario.sigma_ttt)
        ),
    )

    area_shift = (
        rng.normal(0.0, 0.35, size=scenario.n_areas)
        if scenario.include_area_level_covariate_shift
        else np.zeros(scenario.n_areas, dtype=float)
    )
    area_zip = np.arange(30000, 30000 + scenario.n_areas, dtype=int)

    rows: list[dict[str, Any]] = []
    next_id = 1

    for area_id in range(scenario.n_areas):
        for _ in range(scenario.n_per_area):
            covs = _sample_subject_covariates(
                rng=rng,
                scenario=scenario,
                area_shift=float(area_shift[area_id]),
            )

            treatment_time_m, treatment_event = _simulate_treatment_time(
                rng=rng,
                scenario=scenario,
                x_ttt=covs["x_ttt"],
                u_ttt=float(u_ttt_true[area_id]),
            )

            survival_time_m, survival_event = _simulate_survival_time(
                rng=rng,
                scenario=scenario,
                x_surv=covs["x_surv"],
                u_surv=float(u_surv_true[area_id]),
                treatment_time_m=(treatment_time_m if treatment_event == 1 else None),
            )

            censor_time_m = _sample_censor_time(rng=rng, scenario=scenario)

            observed_survival_time_m = min(
                survival_time_m,
                censor_time_m,
                float(scenario.admin_censor_months),
            )
            observed_survival_event = int(
                (survival_event == 1)
                and (survival_time_m <= censor_time_m)
                and (survival_time_m <= float(scenario.admin_censor_months))
            )

            treatment_time_obs_m = min(
                treatment_time_m,
                survival_time_m,
                censor_time_m,
                float(scenario.admin_censor_months),
            )
            observed_treatment_event = int(
                (treatment_event == 1)
                and (treatment_time_m <= survival_time_m)
                and (treatment_time_m <= censor_time_m)
                and (treatment_time_m <= float(scenario.admin_censor_months))
            )

            row = {
                "id": next_id,
                "zip": int(area_zip[area_id]),
                "area_id": int(area_id),
                "time": float(observed_survival_time_m * DAYS_PER_MONTH),
                "event": int(observed_survival_event),
                "treatment_time": float(treatment_time_m * DAYS_PER_MONTH),
                "treatment_time_obs": float(treatment_time_obs_m * DAYS_PER_MONTH),
                "treatment_event": int(observed_treatment_event),
                "time_true_m": float(survival_time_m),
                "event_true": int(survival_event),
                "treatment_time_true_m": float(treatment_time_m),
                "treatment_event_true": int(treatment_event),
                "censor_time_true_m": float(censor_time_m),
                "u_surv_true": float(u_surv_true[area_id]),
                "u_ttt_true": float(u_ttt_true[area_id]),
                "u_ttt_ind_true": float(u_ttt_ind_true[area_id]),
                "phi_surv_true": float(scenario.phi_surv),
                "phi_ttt_true": float(scenario.phi_ttt),
                "sigma_surv_true": float(scenario.sigma_surv),
                "sigma_ttt_true": float(scenario.sigma_ttt),
                "rho_u_true": float(scenario.rho_u),
            }
            row.update(covs["row_covariates"])
            row.update(_truth_columns_from_scenario(scenario))
            rows.append(row)
            next_id += 1

    wide = (
        pd.DataFrame.from_records(rows)
        .sort_values(["area_id", "id"])
        .reset_index(drop=True)
    )

    parameter_truth = _build_parameter_truth(scenario)
    area_truth = pd.DataFrame(
        {
            "area_id": np.arange(scenario.n_areas, dtype=int),
            "u_surv_true": u_surv_true,
            "u_ttt_true": u_ttt_true,
            "u_ttt_ind_true": u_ttt_ind_true,
        }
    )
    support_diagnostics = _build_support_diagnostics(wide, scenario)

    metadata = {
        "scenario_name": scenario.name,
        "seed": int(seed),
        "graph_name": scenario.graph_name,
        "n_subjects": int(len(wide)),
        "n_areas": int(scenario.n_areas),
        "n_per_area": int(scenario.n_per_area),
        "observed_survival_events": int(wide["event"].sum()),
        "observed_treatment_events": int(wide["treatment_event"].sum()),
        "true_survival_events": int(wide["event_true"].sum()),
        "true_treatment_events": int(wide["treatment_event_true"].sum()),
    }

    return JointSimulationResult(
        scenario=scenario,
        seed=int(seed),
        wide=wide,
        parameter_truth=parameter_truth,
        area_truth=area_truth,
        support_diagnostics=support_diagnostics,
        metadata=metadata,
    )


def _sample_joint_spatial_fields(
    *,
    neighbors: list[list[int]],
    rng: np.random.Generator,
    phi_surv: float,
    phi_ttt: float,
    sigma_surv: float,
    sigma_ttt: float,
    rho_u: float,
    sigma_ttt_ind: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample model-matched joint spatial fields.

    We generate a structured spatial component by smoothing iid normals over the
    graph, combine that with an iid component via a BYM2-like mixture using
    phi, standardize each latent field, then induce cross-process dependence
    through:
        u_ttt = rho_u * u_surv + sqrt(1-rho_u^2) * u_ttt_ind
    """
    s_surv = _sample_structured_field(neighbors=neighbors, rng=rng)
    s_ttt_ind = _sample_structured_field(neighbors=neighbors, rng=rng)

    z_surv = rng.normal(size=len(neighbors))
    z_ttt_ind = rng.normal(size=len(neighbors))

    g_surv = np.sqrt(phi_surv) * s_surv + np.sqrt(1.0 - phi_surv) * _standardize(z_surv)
    g_ttt_ind = np.sqrt(phi_ttt) * s_ttt_ind + np.sqrt(1.0 - phi_ttt) * _standardize(z_ttt_ind)

    u_surv = sigma_surv * _standardize(g_surv)
    u_ttt_ind = sigma_ttt_ind * _standardize(g_ttt_ind)
    u_ttt = (
        rho_u * _standardize(u_surv)
        + np.sqrt(max(0.0, 1.0 - rho_u**2)) * _standardize(u_ttt_ind)
    )
    u_ttt = sigma_ttt * _standardize(u_ttt)

    return u_surv, u_ttt, u_ttt_ind


def _sample_structured_field(
    *,
    neighbors: list[list[int]],
    rng: np.random.Generator,
    n_iter: int = 8,
) -> np.ndarray:
    """
    Lightweight graph-smoothing generator for a standardized structured field.
    """
    x = rng.normal(size=len(neighbors))
    for _ in range(n_iter):
        x_new = np.empty_like(x)
        for i, nbrs in enumerate(neighbors):
            if len(nbrs) == 0:
                x_new[i] = x[i]
            else:
                x_new[i] = 0.60 * x[i] + 0.40 * np.mean(x[nbrs])
        x = x_new
    return _standardize(x)


def _sample_subject_covariates(
    *,
    rng: np.random.Generator,
    scenario: JointSimulationScenario,
    area_shift: float,
) -> dict[str, Any]:
    age_raw = rng.normal(loc=scenario.age_mean, scale=scenario.age_sd)
    age_per10_centered = (age_raw - scenario.age_mean) / 10.0

    cci = int(min(rng.poisson(lam=scenario.cci_lambda), 6))
    tumor_size_log = float(
        rng.normal(
            loc=scenario.tumor_size_log_mean + 0.10 * area_shift,
            scale=scenario.tumor_size_log_sd,
        )
    )
    ses = float(rng.normal(loc=scenario.ses_mean + area_shift, scale=scenario.ses_sd))
    sex_male = int(rng.binomial(1, scenario.prob_male))

    p1 = max(0.0, 1.0 - scenario.prob_stage_ii - scenario.prob_stage_iii)
    stage = rng.choice(
        ["I", "II", "III"],
        p=[p1, scenario.prob_stage_ii, scenario.prob_stage_iii],
    )
    stage_ii = int(stage == "II")
    stage_iii = int(stage == "III")

    x_surv = {
        "age_per10_centered": float(age_per10_centered),
        "cci": float(cci),
        "tumor_size_log": float(tumor_size_log),
        "stage_II": float(stage_ii),
        "stage_III": float(stage_iii),
    }
    x_ttt = {
        "age_per10_centered": float(age_per10_centered),
        "cci": float(cci),
        "tumor_size_log": float(tumor_size_log),
        "ses": float(ses),
        "sex_male": float(sex_male),
        "stage_II": float(stage_ii),
        "stage_III": float(stage_iii),
    }

    row_covariates = {
        "age_raw": float(age_raw),
        "age_per10_centered": float(age_per10_centered),
        "cci": int(cci),
        "tumor_size_log": float(tumor_size_log),
        "ses": float(ses),
        "sex": "M" if sex_male == 1 else "F",
        "sex_male": int(sex_male),
        "stage": str(stage),
        "stage_II": int(stage_ii),
        "stage_III": int(stage_iii),
    }

    return {
        "x_surv": x_surv,
        "x_ttt": x_ttt,
        "row_covariates": row_covariates,
    }


def _simulate_treatment_time(
    *,
    rng: np.random.Generator,
    scenario: JointSimulationScenario,
    x_ttt: dict[str, float],
    u_ttt: float,
) -> tuple[float, int]:
    eta_x = _linpred(x_ttt, scenario.theta_ttt)
    log_rates = np.asarray(scenario.gamma_ttt, dtype=float) + eta_x + u_ttt
    return _simulate_piecewise_time(
        rng=rng,
        breaks=np.asarray(scenario.ttt_breaks, dtype=float),
        log_rates=log_rates,
        max_time=float(scenario.admin_censor_months),
    )


def _simulate_survival_time(
    *,
    rng: np.random.Generator,
    scenario: JointSimulationScenario,
    x_surv: dict[str, float],
    u_surv: float,
    treatment_time_m: float | None,
) -> tuple[float, int]:
    surv_breaks = np.asarray(scenario.surv_breaks, dtype=float)
    post_breaks = np.asarray(scenario.post_ttt_breaks, dtype=float)
    alpha = np.asarray(scenario.alpha_surv, dtype=float)
    delta = np.asarray(scenario.delta_post, dtype=float)

    eta_x = _linpred(x_surv, scenario.beta_surv)

    t = 0.0
    max_time = float(scenario.admin_censor_months)

    while t < max_time:
        k_surv = int(np.searchsorted(surv_breaks, t, side="right") - 1)
        if k_surv >= len(alpha):
            return max_time, 0

        current_interval_end = float(surv_breaks[k_surv + 1])

        if treatment_time_m is not None and t < treatment_time_m < current_interval_end:
            segment_end = treatment_time_m
        else:
            segment_end = current_interval_end

        if treatment_time_m is not None and t >= treatment_time_m:
            time_since = t - treatment_time_m
            k_post = int(np.searchsorted(post_breaks, time_since, side="right") - 1)
            k_post = max(0, min(k_post, len(delta) - 1))
            post_effect = float(delta[k_post])
        else:
            post_effect = 0.0

        log_rate = float(alpha[k_surv] + eta_x + u_surv + post_effect)
        rate = float(np.exp(log_rate))

        wait = float(rng.exponential(scale=1.0 / max(rate, 1e-12)))
        if t + wait < min(segment_end, max_time):
            return t + wait, 1

        t = min(segment_end, max_time)

    return max_time, 0


def _simulate_piecewise_time(
    *,
    rng: np.random.Generator,
    breaks: np.ndarray,
    log_rates: np.ndarray,
    max_time: float,
) -> tuple[float, int]:
    t = 0.0
    n_intervals = len(breaks) - 1

    for k in range(n_intervals):
        start = float(breaks[k])
        end = float(min(breaks[k + 1], max_time))
        if end <= start:
            continue
        if t < start:
            t = start
        if t >= max_time:
            break

        rate = float(np.exp(log_rates[k]))
        wait = float(rng.exponential(scale=1.0 / max(rate, 1e-12)))
        if t + wait < end:
            return t + wait, 1
        t = end

    return float(max_time), 0


def _sample_censor_time(
    *,
    rng: np.random.Generator,
    scenario: JointSimulationScenario,
) -> float:
    admin = float(scenario.admin_censor_months)

    draws = [admin]

    if scenario.censor_rate > 0.0:
        draws.append(float(rng.exponential(scale=1.0 / scenario.censor_rate)))

    if scenario.censor_uniform_max is not None:
        draws.append(float(rng.uniform(0.0, scenario.censor_uniform_max)))

    return float(min(draws))


def _build_parameter_truth(scenario: JointSimulationScenario) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for j, val in enumerate(scenario.alpha_surv):
        rows.append(
            {
                "parameter": "alpha_surv",
                "index": int(j),
                "label": f"alpha[{j}]",
                "group": "survival_baseline",
                "truth": float(val),
            }
        )

    for j, val in enumerate(scenario.gamma_ttt):
        rows.append(
            {
                "parameter": "gamma_ttt",
                "index": int(j),
                "label": f"gamma[{j}]",
                "group": "treatment_baseline",
                "truth": float(val),
            }
        )

    rows.append(
        {
            "parameter": "delta_post_intercept",
            "index": None,
            "label": "delta_post_intercept",
            "group": "post_treatment_effect_linear",
            "truth": float(scenario.delta_post_intercept),
        }
    )
    rows.append(
        {
            "parameter": "delta_post_slope",
            "index": None,
            "label": "delta_post_slope",
            "group": "post_treatment_effect_linear",
            "truth": float(scenario.delta_post_slope),
        }
    )

    for j, val in enumerate(scenario.delta_post):
        rows.append(
            {
                "parameter": "delta_post",
                "index": int(j),
                "label": f"delta_post[{j}]",
                "group": "post_treatment_effect",
                "truth": float(val),
            }
        )

    for name, val in scenario.beta_surv.items():
        rows.append(
            {
                "parameter": f"beta_surv_{name}",
                "index": None,
                "label": name,
                "group": "survival_beta",
                "truth": float(val),
            }
        )

    for name, val in scenario.theta_ttt.items():
        rows.append(
            {
                "parameter": f"theta_ttt_{name}",
                "index": None,
                "label": name,
                "group": "treatment_theta",
                "truth": float(val),
            }
        )

    for name, val in [
        ("phi_surv", scenario.phi_surv),
        ("phi_ttt", scenario.phi_ttt),
        ("sigma_surv", scenario.sigma_surv),
        ("sigma_ttt", scenario.sigma_ttt),
        ("rho_u", scenario.rho_u),
    ]:
        rows.append(
            {
                "parameter": name,
                "index": None,
                "label": name,
                "group": "spatial_hyperparameter",
                "truth": float(val),
            }
        )

    return pd.DataFrame(rows)


def _build_neighbors_for_scenario(
    scenario: JointSimulationScenario,
) -> list[list[int]]:
    if scenario.graph_name == "ring_lattice":
        k = int(scenario.graph_kwargs.get("k", 4))
        return _ring_lattice_neighbors(n_areas=int(scenario.n_areas), k=k)

    raise ValueError(f"Unsupported graph_name: {scenario.graph_name}")


def _ring_lattice_neighbors(
    *,
    n_areas: int,
    k: int,
) -> list[list[int]]:
    """
    Build an undirected ring-lattice neighbor list with k total neighbors per node.

    Requires k to be a positive even integer less than n_areas.
    """
    if n_areas <= 2:
        raise ValueError("n_areas must be greater than 2 for a ring lattice.")
    if k <= 0 or k % 2 != 0:
        raise ValueError("Ring-lattice k must be a positive even integer.")
    if k >= n_areas:
        raise ValueError("Ring-lattice k must be smaller than n_areas.")

    half = k // 2
    neighbors: list[list[int]] = []

    for i in range(n_areas):
        nbrs = set()
        for h in range(1, half + 1):
            nbrs.add((i - h) % n_areas)
            nbrs.add((i + h) % n_areas)
        neighbors.append(sorted(nbrs))

    return neighbors


def _build_support_diagnostics(
    wide: pd.DataFrame,
    scenario: JointSimulationScenario,
) -> dict[str, pd.DataFrame]:
    surv_breaks = np.asarray(scenario.surv_breaks, dtype=float)
    ttt_breaks = np.asarray(scenario.ttt_breaks, dtype=float)
    post_breaks = np.asarray(scenario.post_ttt_breaks, dtype=float)

    surv_obs_m = wide["time"] / DAYS_PER_MONTH
    ttt_obs_m = wide["treatment_time_obs"] / DAYS_PER_MONTH

    surv_counts = _bin_counts(surv_obs_m.to_numpy(dtype=float), surv_breaks, "k")
    ttt_obs_counts = _bin_counts(ttt_obs_m.to_numpy(dtype=float), ttt_breaks, "k")
    treated_only = wide.loc[wide["treatment_event_true"] == 1, "treatment_time_true_m"]
    ttt_true_counts = _bin_counts(treated_only.to_numpy(dtype=float), ttt_breaks, "k")

    post_exposure = _post_treatment_support_table(
        wide=wide,
        post_breaks=post_breaks,
    )

    return {
        "survival_interval_support": surv_counts,
        "treatment_interval_support_observed": ttt_obs_counts,
        "treatment_interval_support_true": ttt_true_counts,
        "post_treatment_interval_support": post_exposure,
    }


def _bin_counts(values: np.ndarray, breaks: np.ndarray, index_name: str) -> pd.DataFrame:
    idx = np.searchsorted(breaks, values, side="right") - 1
    idx = np.clip(idx, 0, len(breaks) - 2)
    counts = pd.Series(idx).value_counts().sort_index()
    out = pd.DataFrame(
        {
            index_name: np.arange(len(breaks) - 1, dtype=int),
            "interval_start": breaks[:-1],
            "interval_end": breaks[1:],
            "count": 0,
        }
    )
    out.loc[counts.index, "count"] = counts.to_numpy(dtype=int)
    return out


def _post_treatment_support_table(
    *,
    wide: pd.DataFrame,
    post_breaks: np.ndarray,
) -> pd.DataFrame:
    treated = wide.loc[wide["treatment_event_true"] == 1].copy()
    if treated.empty:
        return pd.DataFrame(
            {
                "k_post": np.arange(len(post_breaks) - 1, dtype=int),
                "interval_start": post_breaks[:-1],
                "interval_end": post_breaks[1:],
                "subject_count": 0,
            }
        )

    time_since = treated["time_true_m"] - treated["treatment_time_true_m"]
    time_since = time_since.clip(lower=0.0).to_numpy(dtype=float)
    idx = np.searchsorted(post_breaks, time_since, side="right") - 1
    idx = np.clip(idx, 0, len(post_breaks) - 2)
    counts = pd.Series(idx).value_counts().sort_index()

    out = pd.DataFrame(
        {
            "k_post": np.arange(len(post_breaks) - 1, dtype=int),
            "interval_start": post_breaks[:-1],
            "interval_end": post_breaks[1:],
            "subject_count": 0,
        }
    )
    out.loc[counts.index, "subject_count"] = counts.to_numpy(dtype=int)
    return out


def _truth_columns_from_scenario(scenario: JointSimulationScenario) -> dict[str, float]:
    out: dict[str, float] = {}

    for j, val in enumerate(scenario.alpha_surv):
        out[f"alpha_surv_{j}_true"] = float(val)
    for j, val in enumerate(scenario.gamma_ttt):
        out[f"gamma_ttt_{j}_true"] = float(val)

    out["delta_post_intercept_true"] = float(scenario.delta_post_intercept)
    out["delta_post_slope_true"] = float(scenario.delta_post_slope)

    for j, val in enumerate(scenario.post_index_scaled):
        out[f"post_index_scaled_{j}_true"] = float(val)
    for j, val in enumerate(scenario.delta_post):
        out[f"delta_post_{j}_true"] = float(val)

    for name, val in scenario.beta_surv.items():
        out[f"beta_surv_{name}_true"] = float(val)
    for name, val in scenario.theta_ttt.items():
        out[f"theta_ttt_{name}_true"] = float(val)

    return out


def _linpred(x: dict[str, float], beta: dict[str, float]) -> float:
    total = 0.0
    for name, coef in beta.items():
        total += float(x.get(name, 0.0)) * float(coef)
    return float(total)


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(x.std(ddof=0))
    if sd <= 0.0:
        return np.zeros_like(x)
    return (x - float(x.mean())) / sd