from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class JointSimulationScenario:
    """
    Structured specification for a model-matched joint treatment-survival
    simulation scenario.

    This object is intended to be the single source of truth for a validation
    scenario. A simulator should consume one of these objects plus a seed and
    produce:
      - simulated wide data
      - a structured truth bundle
      - scenario metadata / support diagnostics

    Conventions
    -----------
    - Time scale for the DGP parameters is months.
    - Interval vectors are interpreted on the same grid conventions used in the
      fitted package:
        * surv_breaks        -> baseline survival PE grid
        * ttt_breaks         -> baseline treatment-time PE grid
        * post_ttt_breaks    -> post-treatment survival effect grid
    - Vector lengths should satisfy:
        len(alpha_surv)  == len(surv_breaks) - 1
        len(gamma_ttt)   == len(ttt_breaks) - 1
        len(delta_post)  == len(post_ttt_breaks) - 1
    - Covariate coefficient dictionaries are keyed by model-ready covariate
      column names used downstream in fitting.
    - The post-treatment effect is parameterized as:
          delta_post[j] = delta_post_intercept + delta_post_slope * z_j
      where z_j is the centered/scaled post-treatment interval index.
    """

    # ------------------------------------------------------------------
    # Scenario identity / bookkeeping
    # ------------------------------------------------------------------
    name: str
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Population / graph structure
    # ------------------------------------------------------------------
    n_areas: int = 20
    n_per_area: int = 100
    graph_name: str = "ring_lattice"
    graph_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"k": 4})

    # ------------------------------------------------------------------
    # Piecewise-exponential interval grids (months)
    # ------------------------------------------------------------------
    surv_breaks: tuple[float, ...] = (
        0.0,
        1.0,
        2.0,
        3.0,
        6.0,
        9.0,
        12.0,
        18.0,
        24.0,
        36.0,
        48.0,
        60.0,
    )
    ttt_breaks: tuple[float, ...] = (
        0.0,
        1.0,
        2.0,
        3.0,
        6.0,
        9.0,
        12.0,
        18.0,
        24.0,
        36.0,
        60.0,
    )
    post_ttt_breaks: tuple[float, ...] = (
        0.0,
        3.0,
        6.0,
        12.0,
        24.0,
        60.0,
    )

    # ------------------------------------------------------------------
    # Baseline hazard pieces
    # ------------------------------------------------------------------
    # Survival baseline log-hazard by survival interval.
    alpha_surv: tuple[float, ...] = (
        -4.60,
        -4.35,
        -4.20,
        -4.05,
        -3.90,
        -3.82,
        -3.75,
        -3.68,
        -3.60,
        -3.52,
        -3.45,
    )

    # Treatment baseline log-hazard by treatment interval.
    gamma_ttt: tuple[float, ...] = (
        -2.40,
        -2.15,
        -1.95,
        -1.80,
        -1.95,
        -2.10,
        -2.25,
        -2.45,
        -2.70,
        -3.05,
    )

    # ------------------------------------------------------------------
    # Post-treatment survival effect: linear-trend parameterization
    # ------------------------------------------------------------------
    # For the default 5-bin post_ttt grid, these values closely reproduce the
    # earlier baseline vector (-0.35, -0.28, -0.20, -0.12, -0.06).
    delta_post_intercept: float = -0.202
    delta_post_slope: float = 0.104

    # ------------------------------------------------------------------
    # Fixed effects
    # ------------------------------------------------------------------
    beta_surv: Mapping[str, float] = field(
        default_factory=lambda: {
            "age_per10_centered": 0.10,
            "cci": 0.18,
            "tumor_size_log": 0.25,
            "stage_II": 0.35,
            "stage_III": 0.75,
        }
    )

    theta_ttt: Mapping[str, float] = field(
        default_factory=lambda: {
            "age_per10_centered": 0.08,
            "cci": -0.12,
            "tumor_size_log": 0.10,
            "ses": 0.20,
            "sex_male": -0.05,
            "stage_II": 0.10,
            "stage_III": 0.22,
        }
    )

    # ------------------------------------------------------------------
    # Spatial latent-process parameters
    # ------------------------------------------------------------------
    # Spatial dependence / BYM2 mixing parameters for the two processes.
    phi_surv: float = 0.85
    phi_ttt: float = 0.85

    # Spatial scale parameters.
    sigma_surv: float = 0.35
    sigma_ttt: float = 0.30

    # Cross-process correlation between survival and treatment frailties:
    #   u_ttt = rho_u * u_surv + sqrt(1-rho_u^2) * u_ttt_ind
    rho_u: float = 0.50

    # Optional explicit scale for the independent treatment component before
    # combination. If None, use sigma_ttt consistently in the simulator.
    sigma_ttt_ind: float | None = None

    # ------------------------------------------------------------------
    # Censoring / follow-up regime
    # ------------------------------------------------------------------
    admin_censor_months: float = 60.0

    # Independent censoring rate on the month scale. A simulator may use this
    # as the rate for Exp(censor_rate) censoring, capped at admin censor.
    censor_rate: float = 0.01

    # Optional additional uniform censoring support (months). If provided, a
    # simulator may draw censoring from Uniform(0, censor_uniform_max) or blend
    # it with exponential censoring depending on implementation.
    censor_uniform_max: float | None = None

    # ------------------------------------------------------------------
    # Covariate-generation controls
    # ------------------------------------------------------------------
    # These are DGP-side controls. The simulator can use them to generate
    # model-ready covariates before applying beta/theta.
    age_mean: float = 70.0
    age_sd: float = 8.0
    cci_lambda: float = 1.2
    tumor_size_log_mean: float = 1.2
    tumor_size_log_sd: float = 0.45
    ses_mean: float = 0.0
    ses_sd: float = 1.0
    prob_male: float = 0.50
    prob_stage_ii: float = 0.35
    prob_stage_iii: float = 0.25

    # ------------------------------------------------------------------
    # Optional DGP feature flags
    # ------------------------------------------------------------------
    # Keep these simple in the model-matched phase; they can later support
    # robustness / misspecification studies.
    include_subject_level_noise: bool = True
    include_area_level_covariate_shift: bool = False

    # ------------------------------------------------------------------
    # Miscellaneous scenario metadata
    # ------------------------------------------------------------------
    notes: str = ""

    def __post_init__(self) -> None:
        self._validate_name()
        self._validate_sizes()
        self._validate_breaks()
        self._validate_interval_vectors()
        self._validate_parameter_ranges()
        self._validate_covariate_coefficients()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def n_subjects(self) -> int:
        return int(self.n_areas * self.n_per_area)

    @property
    def n_surv_intervals(self) -> int:
        return len(self.surv_breaks) - 1

    @property
    def n_ttt_intervals(self) -> int:
        return len(self.ttt_breaks) - 1

    @property
    def n_post_ttt_intervals(self) -> int:
        return len(self.post_ttt_breaks) - 1

    @property
    def post_index_scaled(self) -> tuple[float, ...]:
        k_post = self.n_post_ttt_intervals
        if k_post <= 1:
            return (0.0,)
        idx = np.arange(k_post, dtype=float)
        idx = idx - idx.mean()
        idx = idx / idx.std(ddof=0)
        return tuple(float(x) for x in idx)

    @property
    def delta_post(self) -> tuple[float, ...]:
        z = np.asarray(self.post_index_scaled, dtype=float)
        vals = self.delta_post_intercept + self.delta_post_slope * z
        return tuple(float(v) for v in vals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "n_areas": self.n_areas,
            "n_per_area": self.n_per_area,
            "graph_name": self.graph_name,
            "graph_kwargs": dict(self.graph_kwargs),
            "surv_breaks": list(self.surv_breaks),
            "ttt_breaks": list(self.ttt_breaks),
            "post_ttt_breaks": list(self.post_ttt_breaks),
            "alpha_surv": list(self.alpha_surv),
            "gamma_ttt": list(self.gamma_ttt),
            "delta_post_intercept": self.delta_post_intercept,
            "delta_post_slope": self.delta_post_slope,
            "post_index_scaled": list(self.post_index_scaled),
            "delta_post": list(self.delta_post),
            "beta_surv": dict(self.beta_surv),
            "theta_ttt": dict(self.theta_ttt),
            "phi_surv": self.phi_surv,
            "phi_ttt": self.phi_ttt,
            "sigma_surv": self.sigma_surv,
            "sigma_ttt": self.sigma_ttt,
            "rho_u": self.rho_u,
            "sigma_ttt_ind": self.sigma_ttt_ind,
            "admin_censor_months": self.admin_censor_months,
            "censor_rate": self.censor_rate,
            "censor_uniform_max": self.censor_uniform_max,
            "age_mean": self.age_mean,
            "age_sd": self.age_sd,
            "cci_lambda": self.cci_lambda,
            "tumor_size_log_mean": self.tumor_size_log_mean,
            "tumor_size_log_sd": self.tumor_size_log_sd,
            "ses_mean": self.ses_mean,
            "ses_sd": self.ses_sd,
            "prob_male": self.prob_male,
            "prob_stage_ii": self.prob_stage_ii,
            "prob_stage_iii": self.prob_stage_iii,
            "include_subject_level_noise": self.include_subject_level_noise,
            "include_area_level_covariate_shift": self.include_area_level_covariate_shift,
            "notes": self.notes,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_name(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Scenario name must be non-empty.")

    def _validate_sizes(self) -> None:
        if self.n_areas <= 0:
            raise ValueError("n_areas must be positive.")
        if self.n_per_area <= 0:
            raise ValueError("n_per_area must be positive.")
        if self.admin_censor_months <= 0.0:
            raise ValueError("admin_censor_months must be positive.")

    def _validate_breaks(self) -> None:
        self._validate_break_vector(self.surv_breaks, "surv_breaks")
        self._validate_break_vector(self.ttt_breaks, "ttt_breaks")
        self._validate_break_vector(self.post_ttt_breaks, "post_ttt_breaks")

    @staticmethod
    def _validate_break_vector(breaks: Sequence[float], name: str) -> None:
        if len(breaks) < 2:
            raise ValueError(f"{name} must contain at least two values.")
        if float(breaks[0]) != 0.0:
            raise ValueError(f"{name} must start at 0.0.")
        for i in range(len(breaks) - 1):
            if not float(breaks[i]) < float(breaks[i + 1]):
                raise ValueError(f"{name} must be strictly increasing.")

    def _validate_interval_vectors(self) -> None:
        if len(self.alpha_surv) != self.n_surv_intervals:
            raise ValueError("len(alpha_surv) must equal len(surv_breaks) - 1.")
        if len(self.gamma_ttt) != self.n_ttt_intervals:
            raise ValueError("len(gamma_ttt) must equal len(ttt_breaks) - 1.")
        if self.n_post_ttt_intervals <= 0:
            raise ValueError("post_ttt_breaks must define at least one interval.")
        if len(self.delta_post) != self.n_post_ttt_intervals:
            raise ValueError("len(delta_post) must equal len(post_ttt_breaks) - 1.")

    def _validate_parameter_ranges(self) -> None:
        for value, name in [
            (self.phi_surv, "phi_surv"),
            (self.phi_ttt, "phi_ttt"),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")

        if not -1.0 <= self.rho_u <= 1.0:
            raise ValueError("rho_u must lie in [-1, 1].")

        for value, name in [
            (self.sigma_surv, "sigma_surv"),
            (self.sigma_ttt, "sigma_ttt"),
            (self.censor_rate, "censor_rate"),
            (self.age_sd, "age_sd"),
            (self.cci_lambda, "cci_lambda"),
            (self.tumor_size_log_sd, "tumor_size_log_sd"),
            (self.ses_sd, "ses_sd"),
        ]:
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")

        if self.sigma_ttt_ind is not None and self.sigma_ttt_ind < 0.0:
            raise ValueError("sigma_ttt_ind must be nonnegative when provided.")

        for value, name in [
            (self.prob_male, "prob_male"),
            (self.prob_stage_ii, "prob_stage_ii"),
            (self.prob_stage_iii, "prob_stage_iii"),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")

        if self.prob_stage_ii + self.prob_stage_iii > 1.0:
            raise ValueError("prob_stage_ii + prob_stage_iii must be <= 1.")

        if self.censor_uniform_max is not None and self.censor_uniform_max <= 0.0:
            raise ValueError("censor_uniform_max must be positive when provided.")

    def _validate_covariate_coefficients(self) -> None:
        if len(self.beta_surv) == 0:
            raise ValueError("beta_surv must not be empty.")
        if len(self.theta_ttt) == 0:
            raise ValueError("theta_ttt must not be empty.")

        for mapping, name in [
            (self.beta_surv, "beta_surv"),
            (self.theta_ttt, "theta_ttt"),
        ]:
            for key, value in mapping.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(f"{name} keys must be non-empty strings.")
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{name}[{key!r}] must be numeric.")


def baseline_joint_scenario(
    *,
    name: str = "J00",
    description: str = "Baseline model-matched joint validation scenario.",
    tags: Sequence[str] = ("baseline", "model_matched"),
) -> JointSimulationScenario:
    """
    Canonical baseline scenario for joint-model validation.

    This is intended as the reference scenario for early development and smoke
    validation before constructing a broader scenario grid.
    """
    return JointSimulationScenario(
        name=name,
        description=description,
        tags=tuple(tags),
    )


def low_spatial_signal_scenario(
    *,
    name: str = "J01",
    description: str = "Low spatial signal, moderate coupling.",
    tags: Sequence[str] = ("low_spatial", "model_matched"),
) -> JointSimulationScenario:
    """
    Reduced spatial signal in both processes.
    """
    return JointSimulationScenario(
        name=name,
        description=description,
        tags=tuple(tags),
        sigma_surv=0.15,
        sigma_ttt=0.12,
        rho_u=0.50,
    )


def high_spatial_signal_scenario(
    *,
    name: str = "J02",
    description: str = "High spatial signal, moderate coupling.",
    tags: Sequence[str] = ("high_spatial", "model_matched"),
) -> JointSimulationScenario:
    """
    Elevated spatial signal in both processes.
    """
    return JointSimulationScenario(
        name=name,
        description=description,
        tags=tuple(tags),
        sigma_surv=0.55,
        sigma_ttt=0.50,
        rho_u=0.50,
    )


def low_coupling_scenario(
    *,
    name: str = "J03",
    description: str = "Moderate spatial signal, low cross-process coupling.",
    tags: Sequence[str] = ("low_coupling", "model_matched"),
) -> JointSimulationScenario:
    """
    Weak correlation between survival and treatment spatial frailties.
    """
    return JointSimulationScenario(
        name=name,
        description=description,
        tags=tuple(tags),
        rho_u=0.15,
    )


def high_coupling_scenario(
    *,
    name: str = "J04",
    description: str = "Moderate spatial signal, high cross-process coupling.",
    tags: Sequence[str] = ("high_coupling", "model_matched"),
) -> JointSimulationScenario:
    """
    Strong correlation between survival and treatment spatial frailties.
    """
    return JointSimulationScenario(
        name=name,
        description=description,
        tags=tuple(tags),
        rho_u=0.85,
    )


def weak_post_treatment_effect_scenario(
    *,
    name: str = "J05",
    description: str = "Moderate spatial signal, weak post-treatment survival effect.",
    tags: Sequence[str] = ("weak_delta", "model_matched"),
) -> JointSimulationScenario:
    """
    Attenuated post-treatment effect to test detectability in lower-signal
    treatment-history settings.

    This approximates the earlier weak vector
      (-0.15, -0.10, -0.07, -0.04, -0.02)
    using a model-matched linear trend.
    """
    return JointSimulationScenario(
        name=name,
        description=description,
        tags=tuple(tags),
        delta_post_intercept=-0.076,
        delta_post_slope=0.044,
    )


def stronger_censoring_scenario(
    *,
    name: str = "J06",
    description: str = "Moderate spatial signal with stronger independent censoring.",
    tags: Sequence[str] = ("stronger_censoring", "model_matched"),
) -> JointSimulationScenario:
    """
    Higher censoring rate while preserving the same model-matched structure.
    """
    return JointSimulationScenario(
        name=name,
        description=description,
        tags=tuple(tags),
        censor_rate=0.03,
    )


def default_joint_validation_scenarios() -> list[JointSimulationScenario]:
    """
    Default compact grid of model-matched scenarios for the first formal joint
    validation study.

    The grid is intentionally small but informative:
      - baseline
      - low / high spatial signal
      - low / high cross-process coupling
      - weak post-treatment effect
      - stronger censoring
    """
    return [
        baseline_joint_scenario(),
        low_spatial_signal_scenario(),
        high_spatial_signal_scenario(),
        low_coupling_scenario(),
        high_coupling_scenario(),
        weak_post_treatment_effect_scenario(),
        stronger_censoring_scenario(),
    ]