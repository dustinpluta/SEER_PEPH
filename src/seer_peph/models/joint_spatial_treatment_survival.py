from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def model(data: Mapping[str, Any]) -> None:
    """
    Joint spatial treatment-survival model.

    Survival component
    ------------------
    For survival long-format row r,

        y_surv_r ~ Poisson(exposure_surv_r * lambda_surv_r)

        log(lambda_surv_r) =
            alpha[k_surv_r]
            + x_surv_r^T beta
            + u_surv[area_id_surv_r]
            + treated_td_r * delta_post[k_post_r]

    where delta_post[j] is the full post-treatment log-hazard shift
    in post-treatment interval j.

    Treatment component
    -------------------
    For treatment long-format row q,

        y_ttt_q ~ Poisson(exposure_ttt_q * lambda_ttt_q)

        log(lambda_ttt_q) =
            gamma[k_ttt_q]
            + x_ttt_q^T theta
            + u_ttt[area_id_ttt_q]

    Cross-process spatial coupling
    ------------------------------
    Let u_ttt_ind be an independent treatment BYM2 field. Then

        u_ttt = rho_u_cross * u_surv
                + sqrt(1 - rho_u_cross^2) * u_ttt_ind

    Notes
    -----
    - The survival side uses a delta-only post-treatment parameterization.
    - Here delta_post is constrained to follow an intercept + linear trend
      over the post-treatment interval index.
    - The treatment side is a standalone spatial piecewise-exponential model.
    - The joint model couples the two processes only through correlated
      area-level frailties.
    """

    # ------------------------------------------------------------------
    # Survival inputs
    # ------------------------------------------------------------------
    y_surv = jnp.asarray(data["y_surv"])
    log_exposure_surv = jnp.asarray(data["log_exposure_surv"])
    k_surv = jnp.asarray(data["k_surv"])
    k_post = jnp.asarray(data["k_post"])
    treated_td = jnp.asarray(data["treated_td"])
    area_id_surv = jnp.asarray(data["area_id_surv"])
    X_surv = jnp.asarray(data["X_surv"])

    # ------------------------------------------------------------------
    # Treatment inputs
    # ------------------------------------------------------------------
    y_ttt = jnp.asarray(data["y_ttt"])
    log_exposure_ttt = jnp.asarray(data["log_exposure_ttt"])
    k_ttt = jnp.asarray(data["k_ttt"])
    area_id_ttt = jnp.asarray(data["area_id_ttt"])
    X_ttt = jnp.asarray(data["X_ttt"])

    # ------------------------------------------------------------------
    # Graph inputs
    # ------------------------------------------------------------------
    node1 = jnp.asarray(data["node1"])
    node2 = jnp.asarray(data["node2"])
    scaling_factor = jnp.asarray(data["scaling_factor"])
    A = int(data["A"])

    _validate_inputs(
        y_surv=y_surv,
        log_exposure_surv=log_exposure_surv,
        k_surv=k_surv,
        k_post=k_post,
        treated_td=treated_td,
        area_id_surv=area_id_surv,
        X_surv=X_surv,
        y_ttt=y_ttt,
        log_exposure_ttt=log_exposure_ttt,
        k_ttt=k_ttt,
        area_id_ttt=area_id_ttt,
        X_ttt=X_ttt,
        node1=node1,
        node2=node2,
        scaling_factor=scaling_factor,
        A=A,
        P_surv=data.get("P_surv"),
        P_ttt=data.get("P_ttt"),
    )

    N_surv, P_surv = X_surv.shape
    N_ttt, P_ttt = X_ttt.shape
    K_surv = int(np.asarray(k_surv).max()) + 1
    K_ttt = int(np.asarray(k_ttt).max()) + 1

    treated_np = np.asarray(treated_td)
    k_post_np = np.asarray(k_post)
    if np.any(treated_np == 1):
        K_post = int(k_post_np[treated_np == 1].max()) + 1
    else:
        K_post = 1

    # ------------------------------------------------------------------
    # Fixed effects: survival
    # ------------------------------------------------------------------
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 2.0).expand([K_surv]))
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([P_surv]))

    # Intercept + linear trend over centered/scaled post-treatment interval index.
    delta_post_intercept = numpyro.sample(
        "delta_post_intercept",
        dist.Normal(0.0, 0.5),
    )
    delta_post_slope = numpyro.sample(
        "delta_post_slope",
        dist.Normal(0.0, 0.5),
    )

    if K_post > 1:
        post_index = jnp.arange(K_post, dtype=X_surv.dtype)
        post_index_centered = post_index - jnp.mean(post_index)
        post_index_scaled = post_index_centered / jnp.std(post_index)
    else:
        post_index_scaled = jnp.zeros((1,), dtype=X_surv.dtype)

    delta_post = numpyro.deterministic(
        "delta_post",
        delta_post_intercept + delta_post_slope * post_index_scaled,
    )

    # ------------------------------------------------------------------
    # Fixed effects: treatment
    # ------------------------------------------------------------------
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 2.0).expand([K_ttt]))
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([P_ttt]))

    # ------------------------------------------------------------------
    # Survival BYM2 field
    # ------------------------------------------------------------------
    rho_surv = numpyro.sample("rho_surv", dist.Beta(0.5, 0.5))
    tau_surv = numpyro.sample("tau_surv", dist.HalfNormal(1.0))
    eps_surv = numpyro.sample("eps_surv", dist.Normal(0.0, 1.0).expand([A]))

    s_surv = _sample_sum_to_zero_field(name="s_surv_free", A=A)
    diff_surv = s_surv[node1] - s_surv[node2]
    numpyro.factor("icar_prior_surv", -0.5 * jnp.sum(diff_surv * diff_surv))

    s_surv_scaled = s_surv / jnp.sqrt(scaling_factor)
    u_surv = tau_surv * (
        jnp.sqrt(rho_surv) * s_surv_scaled + jnp.sqrt(1.0 - rho_surv) * eps_surv
    )

    # ------------------------------------------------------------------
    # Independent treatment BYM2 field
    # ------------------------------------------------------------------
    rho_ttt = numpyro.sample("rho_ttt", dist.Beta(0.5, 0.5))
    tau_ttt = numpyro.sample("tau_ttt", dist.HalfNormal(1.0))
    eps_ttt = numpyro.sample("eps_ttt", dist.Normal(0.0, 1.0).expand([A]))

    s_ttt = _sample_sum_to_zero_field(name="s_ttt_free", A=A)
    diff_ttt = s_ttt[node1] - s_ttt[node2]
    numpyro.factor("icar_prior_ttt", -0.5 * jnp.sum(diff_ttt * diff_ttt))

    s_ttt_scaled = s_ttt / jnp.sqrt(scaling_factor)
    u_ttt_ind = tau_ttt * (
        jnp.sqrt(rho_ttt) * s_ttt_scaled + jnp.sqrt(1.0 - rho_ttt) * eps_ttt
    )

    # ------------------------------------------------------------------
    # Cross-process frailty coupling
    # ------------------------------------------------------------------
    rho_u_cross_raw = numpyro.sample("rho_u_cross_raw", dist.Beta(2.0, 2.0))
    rho_u_cross = numpyro.deterministic("rho_u_cross", 2.0 * rho_u_cross_raw - 1.0)

    u_ttt = rho_u_cross * u_surv + jnp.sqrt(1.0 - rho_u_cross**2) * u_ttt_ind

    numpyro.deterministic("s_surv", s_surv)
    numpyro.deterministic("s_ttt", s_ttt)
    numpyro.deterministic("u_surv", u_surv)
    numpyro.deterministic("u_ttt_ind", u_ttt_ind)
    numpyro.deterministic("u_ttt", u_ttt)

    # ------------------------------------------------------------------
    # Survival likelihood
    # ------------------------------------------------------------------
    k_post_safe = jnp.maximum(k_post, 0)
    eta_surv = (
        alpha[k_surv]
        + jnp.sum(X_surv * beta[None, :], axis=1)
        + u_surv[area_id_surv]
        + treated_td * delta_post[k_post_safe]
    )
    mu_surv = jnp.exp(log_exposure_surv + eta_surv)

    with numpyro.plate("obs_surv", N_surv):
        numpyro.sample("y_surv_obs", dist.Poisson(mu_surv), obs=y_surv)

    # ------------------------------------------------------------------
    # Treatment likelihood
    # ------------------------------------------------------------------
    eta_ttt = (
        gamma[k_ttt]
        + jnp.sum(X_ttt * theta[None, :], axis=1)
        + u_ttt[area_id_ttt]
    )
    mu_ttt = jnp.exp(log_exposure_ttt + eta_ttt)

    with numpyro.plate("obs_ttt", N_ttt):
        numpyro.sample("y_ttt_obs", dist.Poisson(mu_ttt), obs=y_ttt)


def _sample_sum_to_zero_field(*, name: str, A: int) -> jnp.ndarray:
    if A > 1:
        s_free = numpyro.sample(name, dist.Normal(0.0, 1.0).expand([A - 1]))
        s_last = -jnp.sum(s_free, keepdims=True)
        return jnp.concatenate([s_free, s_last], axis=0)
    return jnp.array([0.0])


def _validate_inputs(
    *,
    y_surv,
    log_exposure_surv,
    k_surv,
    k_post,
    treated_td,
    area_id_surv,
    X_surv,
    y_ttt,
    log_exposure_ttt,
    k_ttt,
    area_id_ttt,
    X_ttt,
    node1,
    node2,
    scaling_factor,
    A: int,
    P_surv: Any,
    P_ttt: Any,
) -> None:
    y_surv_np = np.asarray(y_surv)
    log_exp_surv_np = np.asarray(log_exposure_surv)
    k_surv_np = np.asarray(k_surv)
    k_post_np = np.asarray(k_post)
    treated_np = np.asarray(treated_td)
    area_surv_np = np.asarray(area_id_surv)
    X_surv_np = np.asarray(X_surv)

    y_ttt_np = np.asarray(y_ttt)
    log_exp_ttt_np = np.asarray(log_exposure_ttt)
    k_ttt_np = np.asarray(k_ttt)
    area_ttt_np = np.asarray(area_id_ttt)
    X_ttt_np = np.asarray(X_ttt)

    node1_np = np.asarray(node1)
    node2_np = np.asarray(node2)
    sf = float(np.asarray(scaling_factor))

    # Survival checks
    if y_surv_np.ndim != 1:
        raise ValueError("y_surv must be 1-D")
    if log_exp_surv_np.shape != y_surv_np.shape:
        raise ValueError("log_exposure_surv must have same shape as y_surv")
    if k_surv_np.shape != y_surv_np.shape:
        raise ValueError("k_surv must have same shape as y_surv")
    if k_post_np.shape != y_surv_np.shape:
        raise ValueError("k_post must have same shape as y_surv")
    if treated_np.shape != y_surv_np.shape:
        raise ValueError("treated_td must have same shape as y_surv")
    if area_surv_np.shape != y_surv_np.shape:
        raise ValueError("area_id_surv must have same shape as y_surv")
    if X_surv_np.ndim != 2 or X_surv_np.shape[0] != y_surv_np.shape[0]:
        raise ValueError("X_surv must be 2-D with one row per survival observation")
    if P_surv is not None and X_surv_np.shape[1] != int(P_surv):
        raise ValueError("X_surv second dimension does not match P_surv")
    if np.any(y_surv_np < 0):
        raise ValueError("y_surv must be non-negative")
    if not np.isfinite(log_exp_surv_np).all():
        raise ValueError("log_exposure_surv contains non-finite values")
    if np.any(k_surv_np < 0):
        raise ValueError("k_surv must be non-negative")
    if not np.isin(treated_np, [0, 1]).all():
        raise ValueError("treated_td must contain only 0/1")
    untreated_mask = treated_np == 0
    treated_mask = treated_np == 1
    if np.any(k_post_np[untreated_mask] != -1):
        raise ValueError("Untreated survival rows must have k_post == -1")
    if np.any(k_post_np[treated_mask] < 0):
        raise ValueError("Treated survival rows must have non-negative k_post")
    if np.any(area_surv_np < 0) or np.any(area_surv_np >= A):
        raise ValueError("area_id_surv out of range")

    # Treatment checks
    if y_ttt_np.ndim != 1:
        raise ValueError("y_ttt must be 1-D")
    if log_exp_ttt_np.shape != y_ttt_np.shape:
        raise ValueError("log_exposure_ttt must have same shape as y_ttt")
    if k_ttt_np.shape != y_ttt_np.shape:
        raise ValueError("k_ttt must have same shape as y_ttt")
    if area_ttt_np.shape != y_ttt_np.shape:
        raise ValueError("area_id_ttt must have same shape as y_ttt")
    if X_ttt_np.ndim != 2 or X_ttt_np.shape[0] != y_ttt_np.shape[0]:
        raise ValueError("X_ttt must be 2-D with one row per treatment observation")
    if P_ttt is not None and X_ttt_np.shape[1] != int(P_ttt):
        raise ValueError("X_ttt second dimension does not match P_ttt")
    if np.any(y_ttt_np < 0):
        raise ValueError("y_ttt must be non-negative")
    if not np.isfinite(log_exp_ttt_np).all():
        raise ValueError("log_exposure_ttt contains non-finite values")
    if np.any(k_ttt_np < 0):
        raise ValueError("k_ttt must be non-negative")
    if np.any(area_ttt_np < 0) or np.any(area_ttt_np >= A):
        raise ValueError("area_id_ttt out of range")

    # Graph checks
    if node1_np.shape != node2_np.shape:
        raise ValueError("node1 and node2 must have same shape")
    if node1_np.ndim != 1:
        raise ValueError("node1 and node2 must be 1-D")
    if np.any(node1_np < 0) or np.any(node1_np >= A):
        raise ValueError("node1 out of range")
    if np.any(node2_np < 0) or np.any(node2_np >= A):
        raise ValueError("node2 out of range")
    if not np.isfinite(sf) or sf <= 0.0:
        raise ValueError("scaling_factor must be finite and > 0")