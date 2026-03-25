from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from seer_peph.data.prep import DAYS_PER_MONTH, POST_TTT_BREAKS, SURV_BREAKS, TTT_BREAKS
from seer_peph.graphs import SpatialGraph


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_SURV_BETAS: dict[str, float] = {
    "age_per10_centered": 0.18,
    "cci": 0.22,
    "tumor_size_log": 0.30,
    "stage_II": 0.30,
    "stage_III": 0.75,
}

DEFAULT_TTT_BETAS: dict[str, float] = {
    "age_per10_centered": -0.08,
    "cci": -0.10,
    "ses": 0.25,
    "sex_male": -0.08,
    "stage_II": 0.10,
    "stage_III": 0.22,
}

# Per-month baseline hazards on the log scale.
DEFAULT_ALPHA_SURV: np.ndarray = np.log(
    np.array(
        [0.012, 0.014, 0.016, 0.018, 0.017, 0.016, 0.015, 0.014, 0.013, 0.012, 0.011],
        dtype=float,
    )
)

DEFAULT_GAMMA_TTT: np.ndarray = np.log(
    np.array(
        [0.12, 0.11, 0.10, 0.09, 0.075, 0.060, 0.050, 0.035, 0.025, 0.018],
        dtype=float,
    )
)

# delta_post[0] is the reference acute post-treatment interval.
DEFAULT_DELTA_POST: np.ndarray = np.array([0.00, 0.12, 0.20, 0.28, 0.35], dtype=float)


# -----------------------------------------------------------------------------
# Parameter bundle
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationParams:
    """Fully resolved parameter bundle used by `simulate_joint()`."""

    alpha_surv: np.ndarray
    gamma_ttt: np.ndarray
    beta_surv: dict[str, float]
    theta_ttt: dict[str, float]
    delta_post: np.ndarray
    beta_td: float
    rho_u: float
    phi_surv: float
    phi_ttt: float
    sigma_surv: float
    sigma_ttt: float
    admin_censor_months: float


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def simulate_joint(
    graph: SpatialGraph,
    *,
    n_per_area: int | Sequence[int] = 500,
    rho_u: float = 0.85,
    phi_surv: float = 0.80,
    phi_ttt: float = 0.75,
    sigma_surv: float = 1.0,
    sigma_ttt: float = 0.3,
    beta_td: float = -0.25,
    alpha_surv: Sequence[float] | None = None,
    gamma_ttt: Sequence[float] | None = None,
    beta_surv: Mapping[str, float] | None = None,
    theta_ttt: Mapping[str, float] | None = None,
    delta_post: Sequence[float] | None = None,
    admin_censor_months: float = 60.0,
    zip_start: int = 30000,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate wide-format joint treatment / survival data.

    Output times are in days, matching the current wide-data contract
    expected by `data/prep.py`.

    Parameters
    ----------
    graph
        Spatial graph defining area adjacency.
    n_per_area
        Either one integer (same size in every area) or a length-A sequence
        of area-specific sample sizes.
    rho_u
        Cross-process frailty correlation.
    phi_surv, phi_ttt
        BYM2 spatial mixing parameters in [0, 1].
    sigma_surv, sigma_ttt
        Total SD of the survival and treatment frailty fields.
    beta_td
        Treatment indicator effect on the survival log-hazard.
    alpha_surv
        Length len(SURV_BREAKS)-1 vector of survival log-baseline hazards.
    gamma_ttt
        Length len(TTT_BREAKS)-1 vector of treatment log-baseline hazards.
    beta_surv
        Survival fixed-effect coefficients with keys:
            age_per10_centered, cci, tumor_size_log, stage_II, stage_III
    theta_ttt
        Treatment fixed-effect coefficients with keys:
            age_per10_centered, cci, ses, sex_male, stage_II, stage_III
    delta_post
        Length len(POST_TTT_BREAKS)-1 vector of post-treatment adjustments.
        Convention: delta_post[0] is the acute reference interval.
    admin_censor_months
        Administrative follow-up limit.
    zip_start
        First synthetic ZIP code.
    seed
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Wide-format synthetic dataset with validation `*_true` columns.
    """
    params = _coerce_params(
        rho_u=rho_u,
        phi_surv=phi_surv,
        phi_ttt=phi_ttt,
        sigma_surv=sigma_surv,
        sigma_ttt=sigma_ttt,
        beta_td=beta_td,
        alpha_surv=alpha_surv,
        gamma_ttt=gamma_ttt,
        beta_surv=beta_surv,
        theta_ttt=theta_ttt,
        delta_post=delta_post,
        admin_censor_months=admin_censor_months,
    )

    rng = np.random.default_rng(seed)
    area_sizes = _expand_n_per_area(n_per_area=n_per_area, A=graph.A)
    zip_codes = np.arange(zip_start, zip_start + graph.A, dtype=int)

    u_surv, u_ttt = _sample_correlated_bym2_fields(
        graph=graph,
        rng=rng,
        rho_u=params.rho_u,
        phi_surv=params.phi_surv,
        phi_ttt=params.phi_ttt,
        sigma_surv=params.sigma_surv,
        sigma_ttt=params.sigma_ttt,
    )
    ses_area = rng.normal(loc=0.0, scale=1.0, size=graph.A)

    records: list[dict[str, float | int | str]] = []
    next_id = 1

    for area_id, n_area in enumerate(area_sizes):
        for _ in range(n_area):
            rec = _simulate_subject(
                subject_id=next_id,
                area_id=area_id,
                zip_code=int(zip_codes[area_id]),
                u_surv=float(u_surv[area_id]),
                u_ttt=float(u_ttt[area_id]),
                ses_area=float(ses_area[area_id]),
                params=params,
                rng=rng,
            )
            records.append(rec)
            next_id += 1

    out = pd.DataFrame.from_records(records)
    out = out.sort_values(["zip", "id"]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Subject-level simulation
# -----------------------------------------------------------------------------

def _simulate_subject(
    *,
    subject_id: int,
    area_id: int,
    zip_code: int,
    u_surv: float,
    u_ttt: float,
    ses_area: float,
    params: SimulationParams,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    age_per10_centered = float(rng.normal(0.0, 1.0))
    cci = int(np.minimum(rng.poisson(1.2), 6))
    tumor_size_log = float(rng.normal(3.25, 0.35))
    sex_male = int(rng.binomial(1, 0.52))
    sex = "M" if sex_male == 1 else "F"

    stage = str(rng.choice(["I", "II", "III"], p=[0.34, 0.36, 0.30]))
    stage_II = int(stage == "II")
    stage_III = int(stage == "III")

    x_surv = {
        "age_per10_centered": age_per10_centered,
        "cci": float(cci),
        "tumor_size_log": tumor_size_log,
        "stage_II": float(stage_II),
        "stage_III": float(stage_III),
    }
    x_ttt = {
        "age_per10_centered": age_per10_centered,
        "cci": float(cci),
        "ses": ses_area,
        "sex_male": float(sex_male),
        "stage_II": float(stage_II),
        "stage_III": float(stage_III),
    }

    eta_surv_x = _linpred(x_surv, params.beta_surv)
    eta_ttt_x = _linpred(x_ttt, params.theta_ttt)

    ttt_log_rates = params.gamma_ttt + eta_ttt_x + u_ttt
    treatment_time_true_m, treatment_event_true = _simulate_piecewise_time(
        rng=rng,
        breaks=np.asarray(TTT_BREAKS, dtype=float),
        log_rates=ttt_log_rates,
        max_time=params.admin_censor_months,
    )

    survival_time_true_m, survival_event_true = _simulate_survival_time(
        rng=rng,
        alpha_surv=params.alpha_surv,
        eta_surv_x=eta_surv_x,
        u_surv=u_surv,
        beta_td=params.beta_td,
        delta_post=params.delta_post,
        treatment_time_m=treatment_time_true_m if treatment_event_true == 1 else None,
        admin_censor_months=params.admin_censor_months,
    )

    censor_time_m = params.admin_censor_months
    observed_survival_time_m = min(survival_time_true_m, censor_time_m)
    event = int(survival_event_true == 1 and survival_time_true_m <= censor_time_m)

    treatment_time_obs_m = min(treatment_time_true_m, observed_survival_time_m, censor_time_m)
    treatment_event = int(
        treatment_event_true == 1
        and treatment_time_true_m <= observed_survival_time_m
        and treatment_time_true_m <= censor_time_m
    )

    rec: dict[str, float | int | str] = {
        # Core wide-data contract
        "id": int(subject_id),
        "zip": int(zip_code),
        "age_per10_centered": age_per10_centered,
        "cci": int(cci),
        "tumor_size_log": tumor_size_log,
        "ses": float(ses_area),
        "sex": sex,
        "stage": stage,
        "treatment_time": _months_to_days_or_nan(
            treatment_time_true_m if treatment_event_true == 1 else np.nan
        ),
        "treatment_time_obs": float(treatment_time_obs_m * DAYS_PER_MONTH),
        "treatment_event": int(treatment_event),
        "time": float(observed_survival_time_m * DAYS_PER_MONTH),
        "event": int(event),
        # Compatibility / validation columns
        "eta_base_survival_true": float(eta_surv_x),
        "u_true": float(u_surv),
        "eta_spatial_survival_true": float(u_surv),
        "gamma_treated_true": float(params.beta_td),
        "tau_true": float(params.sigma_surv),
        "rho_true": float(params.rho_u),
        "sigma_treatment_true": float(params.sigma_ttt),
        "treatment_intercept_true": float(params.gamma_ttt[0]),
        "treatment_spatial_mode_true": "bym2",
        "u_treatment_true": float(u_ttt),
        "treatment_tau_true": float(params.sigma_ttt),
        "treatment_rho_true": float(params.rho_u),
        "treatment_time_true": _months_to_days_or_nan(
            treatment_time_true_m if treatment_event_true == 1 else np.nan
        ),
        "survival_time_true": float(survival_time_true_m * DAYS_PER_MONTH),
        "censor_time": float(censor_time_m * DAYS_PER_MONTH),
        "treated_observed": int(treatment_event),
        # Explicit true columns for later tests / diagnostics
        "area_id_true": int(area_id),
        "eta_surv_x_true": float(eta_surv_x),
        "eta_ttt_x_true": float(eta_ttt_x),
        "u_surv_true": float(u_surv),
        "u_ttt_true": float(u_ttt),
        "phi_surv_true": float(params.phi_surv),
        "phi_ttt_true": float(params.phi_ttt),
        "sigma_surv_true": float(params.sigma_surv),
        "sigma_ttt_true": float(params.sigma_ttt),
        "beta_td_true": float(params.beta_td),
    }

    for j, val in enumerate(params.delta_post):
        rec[f"delta_post_{j}_true"] = float(val)
    for name, val in params.beta_surv.items():
        rec[f"beta_surv_{name}_true"] = float(val)
    for j, val in enumerate(params.gamma_ttt):
        rec[f"gamma_ttt_{j}_true"] = float(val)
    for name, val in params.theta_ttt.items():
        rec[f"theta_ttt_{name}_true"] = float(val)

    return rec


def _simulate_survival_time(
    *,
    rng: np.random.Generator,
    alpha_surv: np.ndarray,
    eta_surv_x: float,
    u_surv: float,
    beta_td: float,
    delta_post: np.ndarray,
    treatment_time_m: float | None,
    admin_censor_months: float,
) -> tuple[float, int]:
    """
    Simulate survival time under a piecewise survival baseline with
    treatment-induced hazard switching.
    """
    global_breaks = np.asarray(SURV_BREAKS, dtype=float)
    extra_cuts: list[float] = []

    if treatment_time_m is not None and treatment_time_m < admin_censor_months:
        extra_cuts.append(float(treatment_time_m))
        for pb in POST_TTT_BREAKS[1:]:
            extra_cuts.append(float(treatment_time_m + pb))

    event_breaks = _merge_breaks(
        global_breaks=global_breaks,
        extra_cuts=extra_cuts,
        max_time=admin_censor_months,
    )

    log_rates = []
    for t0 in event_breaks[:-1]:
        k = _interval_index(global_breaks, t0)
        log_h = float(alpha_surv[k] + eta_surv_x + u_surv)
        if treatment_time_m is not None and t0 >= treatment_time_m:
            k_post = _post_treatment_index(t0=t0, treatment_time_m=treatment_time_m)
            log_h += beta_td + float(delta_post[k_post])
        log_rates.append(log_h)

    return _simulate_piecewise_time(
        rng=rng,
        breaks=event_breaks,
        log_rates=np.asarray(log_rates, dtype=float),
        max_time=admin_censor_months,
    )


# -----------------------------------------------------------------------------
# Spatial field generation
# -----------------------------------------------------------------------------

def _sample_correlated_bym2_fields(
    *,
    graph: SpatialGraph,
    rng: np.random.Generator,
    rho_u: float,
    phi_surv: float,
    phi_ttt: float,
    sigma_surv: float,
    sigma_ttt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample correlated BYM2 frailty fields using the guide's Cholesky form:

        u_surv ~ BYM2(sigma_surv, phi_surv)
        u_ttt  = rho_u * u_surv + sqrt(1-rho_u^2) * u_ttt_ind

    where u_ttt_ind is an independent BYM2 field.
    """
    u_surv = _sample_bym2_field(
        graph=graph,
        rng=rng,
        phi=phi_surv,
        sigma=sigma_surv,
    )
    u_ttt_ind = _sample_bym2_field(
        graph=graph,
        rng=rng,
        phi=phi_ttt,
        sigma=sigma_ttt,
    )
    rho = float(np.clip(rho_u, -0.999999, 0.999999))
    u_ttt = rho * u_surv + np.sqrt(1.0 - rho * rho) * u_ttt_ind
    return u_surv, u_ttt


def _sample_bym2_field(
    *,
    graph: SpatialGraph,
    rng: np.random.Generator,
    phi: float,
    sigma: float,
) -> np.ndarray:
    s_scaled = _sample_scaled_icar(graph=graph, rng=rng)
    eps = rng.normal(loc=0.0, scale=1.0, size=graph.A)
    u = sigma * (np.sqrt(phi) * s_scaled + np.sqrt(1.0 - phi) * eps)
    return np.asarray(u, dtype=float)


def _sample_scaled_icar(
    *,
    graph: SpatialGraph,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw one scaled ICAR field with approximate unit marginal variance.
    """
    W = np.asarray(graph.adjacency, dtype=float)
    D = np.diag(W.sum(axis=1))
    Q = D - W

    evals, evecs = np.linalg.eigh(Q)
    keep = evals > 1e-10
    if int(keep.sum()) != graph.A - 1:
        raise RuntimeError("Connected graph Laplacian should have rank A-1.")

    z = rng.normal(loc=0.0, scale=1.0, size=int(keep.sum()))
    s = evecs[:, keep] @ (z / np.sqrt(evals[keep]))
    s = np.asarray(s, dtype=float)
    s -= s.mean()
    s /= np.sqrt(graph.scaling_factor)
    return s


# -----------------------------------------------------------------------------
# Piecewise-exponential helpers
# -----------------------------------------------------------------------------

def _simulate_piecewise_time(
    *,
    rng: np.random.Generator,
    breaks: np.ndarray,
    log_rates: np.ndarray,
    max_time: float,
) -> tuple[float, int]:
    """
    Simulate one event time from a piecewise exponential hazard.

    Returns
    -------
    time, event
        If event == 1, `time` is the event time.
        If event == 0, `time` is the administrative censoring time.
    """
    if breaks.ndim != 1 or len(breaks) != len(log_rates) + 1:
        raise ValueError("breaks must have length len(log_rates) + 1")
    if breaks[0] != 0.0 or np.any(np.diff(breaks) <= 0):
        raise ValueError("breaks must be strictly increasing and start at 0")

    t = 0.0
    for j, log_h in enumerate(log_rates):
        start = float(breaks[j])
        stop = float(min(breaks[j + 1], max_time))
        if stop <= start:
            continue

        width = stop - start
        h = float(np.exp(log_h))
        wait = rng.exponential(scale=1.0 / h)

        if wait < width:
            return start + wait, 1

        t = stop
        if np.isclose(t, max_time) or t >= max_time:
            return max_time, 0

    return min(t, max_time), 0


def _merge_breaks(
    *,
    global_breaks: np.ndarray,
    extra_cuts: Sequence[float],
    max_time: float,
) -> np.ndarray:
    valid = [float(c) for c in extra_cuts if 0.0 < float(c) < max_time]
    merged = np.unique(
        np.concatenate(
            [
                global_breaks,
                np.asarray(valid, dtype=float),
                np.array([max_time], dtype=float),
            ]
        )
    )
    merged.sort()
    return merged[merged <= max_time]


def _interval_index(breaks: np.ndarray, t: float) -> int:
    return int(np.searchsorted(breaks, t, side="right") - 1)


def _post_treatment_index(*, t0: float, treatment_time_m: float) -> int:
    time_since = t0 - treatment_time_m
    post_breaks = np.asarray(POST_TTT_BREAKS, dtype=float)
    k = int(np.searchsorted(post_breaks, time_since, side="right") - 1)
    return max(0, min(k, len(post_breaks) - 2))


# -----------------------------------------------------------------------------
# Validation / parameter helpers
# -----------------------------------------------------------------------------

def _coerce_params(
    *,
    rho_u: float,
    phi_surv: float,
    phi_ttt: float,
    sigma_surv: float,
    sigma_ttt: float,
    beta_td: float,
    alpha_surv: Sequence[float] | None,
    gamma_ttt: Sequence[float] | None,
    beta_surv: Mapping[str, float] | None,
    theta_ttt: Mapping[str, float] | None,
    delta_post: Sequence[float] | None,
    admin_censor_months: float,
) -> SimulationParams:
    if not (-0.999999 < rho_u < 0.999999):
        raise ValueError("rho_u must lie strictly inside (-1, 1)")
    if not (0.0 <= phi_surv <= 1.0):
        raise ValueError("phi_surv must lie in [0, 1]")
    if not (0.0 <= phi_ttt <= 1.0):
        raise ValueError("phi_ttt must lie in [0, 1]")
    if sigma_surv <= 0.0:
        raise ValueError("sigma_surv must be > 0")
    if sigma_ttt <= 0.0:
        raise ValueError("sigma_ttt must be > 0")
    if admin_censor_months <= 0.0:
        raise ValueError("admin_censor_months must be > 0")

    alpha = np.asarray(
        DEFAULT_ALPHA_SURV if alpha_surv is None else alpha_surv,
        dtype=float,
    )
    gamma = np.asarray(
        DEFAULT_GAMMA_TTT if gamma_ttt is None else gamma_ttt,
        dtype=float,
    )
    dpost = np.asarray(
        DEFAULT_DELTA_POST if delta_post is None else delta_post,
        dtype=float,
    )

    if alpha.shape != (len(SURV_BREAKS) - 1,):
        raise ValueError(f"alpha_surv must have length {len(SURV_BREAKS) - 1}")
    if gamma.shape != (len(TTT_BREAKS) - 1,):
        raise ValueError(f"gamma_ttt must have length {len(TTT_BREAKS) - 1}")
    if dpost.shape != (len(POST_TTT_BREAKS) - 1,):
        raise ValueError(f"delta_post must have length {len(POST_TTT_BREAKS) - 1}")

    beta = dict(DEFAULT_SURV_BETAS if beta_surv is None else beta_surv)
    theta = dict(DEFAULT_TTT_BETAS if theta_ttt is None else theta_ttt)

    beta_required = {
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "stage_II",
        "stage_III",
    }
    theta_required = {
        "age_per10_centered",
        "cci",
        "ses",
        "sex_male",
        "stage_II",
        "stage_III",
    }

    if set(beta) != beta_required:
        raise ValueError(f"beta_surv keys must be exactly {sorted(beta_required)}")
    if set(theta) != theta_required:
        raise ValueError(f"theta_ttt keys must be exactly {sorted(theta_required)}")

    return SimulationParams(
        alpha_surv=alpha,
        gamma_ttt=gamma,
        beta_surv={k: float(v) for k, v in beta.items()},
        theta_ttt={k: float(v) for k, v in theta.items()},
        delta_post=dpost,
        beta_td=float(beta_td),
        rho_u=float(rho_u),
        phi_surv=float(phi_surv),
        phi_ttt=float(phi_ttt),
        sigma_surv=float(sigma_surv),
        sigma_ttt=float(sigma_ttt),
        admin_censor_months=float(admin_censor_months),
    )


def _expand_n_per_area(*, n_per_area: int | Sequence[int], A: int) -> np.ndarray:
    if isinstance(n_per_area, (int, np.integer)):
        if int(n_per_area) <= 0:
            raise ValueError("n_per_area must be positive")
        return np.repeat(int(n_per_area), A).astype(int)

    vals = np.asarray(list(n_per_area), dtype=int)
    if vals.shape != (A,):
        raise ValueError(f"Sequence n_per_area must have length {A}")
    if np.any(vals <= 0):
        raise ValueError("All area-specific sample sizes must be positive")
    return vals


def _linpred(x: Mapping[str, float], beta: Mapping[str, float]) -> float:
    return float(sum(float(x[k]) * float(beta[k]) for k in beta))


def _months_to_days_or_nan(value_m: float) -> float:
    if np.isnan(value_m):
        return float("nan")
    return float(value_m * DAYS_PER_MONTH)