from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def model(data: Mapping[str, Any]) -> None:
    """
    Spatial piecewise-exponential treatment-time model.

    Long-row likelihood
    -------------------
    For treatment long-format row r,

        y_r ~ Poisson(exposure_r * lambda_r)

        log(lambda_r) =
            gamma[k_r]
            + x_r^T theta
            + u[area_id_r]

    where
    -----
    gamma[k] : treatment interval log-baseline hazard
    theta    : treatment fixed effects
    u[a]     : BYM2 spatial frailty

    Notes
    -----
    - This model targets the treatment-event process only.
    - Death censors treatment upstream in `build_treatment_long(...)`.
    - The model consumes only the treatment-side keys from `make_model_data(...)`.
    """
    y = jnp.asarray(data["y_ttt"])
    log_exposure = jnp.asarray(data["log_exposure_ttt"])
    k_ttt = jnp.asarray(data["k_ttt"])
    area_id = jnp.asarray(data["area_id_ttt"])
    X = jnp.asarray(data["X_ttt"])

    node1 = jnp.asarray(data["node1"])
    node2 = jnp.asarray(data["node2"])
    scaling_factor = jnp.asarray(data["scaling_factor"])
    A = int(data["A"])

    _validate_inputs(
        y=y,
        log_exposure=log_exposure,
        k_ttt=k_ttt,
        area_id=area_id,
        X=X,
        node1=node1,
        node2=node2,
        scaling_factor=scaling_factor,
        A=A,
        P_ttt=data.get("P_ttt"),
    )

    N, P = X.shape
    K_ttt = int(np.asarray(k_ttt).max()) + 1

    gamma = numpyro.sample("gamma", dist.Normal(0.0, 2.0).expand([K_ttt]))
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([P]))

    rho = numpyro.sample("rho", dist.Beta(0.5, 0.5))
    tau = numpyro.sample("tau", dist.HalfNormal(1.0))

    eps = numpyro.sample("eps", dist.Normal(0.0, 1.0).expand([A]))

    if A > 1:
        s_free = numpyro.sample("s_free", dist.Normal(0.0, 1.0).expand([A - 1]))
        s_last = -jnp.sum(s_free, keepdims=True)
        s = jnp.concatenate([s_free, s_last], axis=0)
    else:
        s = jnp.array([0.0])

    diff = s[node1] - s[node2]
    icar_quad = jnp.sum(diff * diff)
    numpyro.factor("icar_prior", -0.5 * icar_quad)

    s_scaled = s / jnp.sqrt(scaling_factor)
    u = tau * (jnp.sqrt(rho) * s_scaled + jnp.sqrt(1.0 - rho) * eps)

    numpyro.deterministic("s", s)
    numpyro.deterministic("u", u)

    eta = gamma[k_ttt] + jnp.sum(X * theta[None, :], axis=1) + u[area_id]
    mu = jnp.exp(log_exposure + eta)

    with numpyro.plate("obs_ttt", N):
        numpyro.sample("y_obs", dist.Poisson(mu), obs=y)


def _validate_inputs(
    *,
    y,
    log_exposure,
    k_ttt,
    area_id,
    X,
    node1,
    node2,
    scaling_factor,
    A: int,
    P_ttt: Any,
) -> None:
    y_np = np.asarray(y)
    log_exp_np = np.asarray(log_exposure)
    k_np = np.asarray(k_ttt)
    area_np = np.asarray(area_id)
    X_np = np.asarray(X)
    node1_np = np.asarray(node1)
    node2_np = np.asarray(node2)
    sf = float(np.asarray(scaling_factor))

    if y_np.ndim != 1:
        raise ValueError("y_ttt must be 1-D")
    if log_exp_np.shape != y_np.shape:
        raise ValueError("log_exposure_ttt must have same shape as y_ttt")
    if k_np.shape != y_np.shape:
        raise ValueError("k_ttt must have same shape as y_ttt")
    if area_np.shape != y_np.shape:
        raise ValueError("area_id_ttt must have same shape as y_ttt")
    if X_np.ndim != 2 or X_np.shape[0] != y_np.shape[0]:
        raise ValueError("X_ttt must be 2-D with one row per observation")
    if P_ttt is not None and X_np.shape[1] != int(P_ttt):
        raise ValueError("X_ttt second dimension does not match P_ttt")
    if np.any(y_np < 0):
        raise ValueError("y_ttt must be non-negative")
    if not np.isfinite(log_exp_np).all():
        raise ValueError("log_exposure_ttt contains non-finite values")
    if np.any(k_np < 0):
        raise ValueError("k_ttt must be non-negative")
    if np.any(area_np < 0) or np.any(area_np >= A):
        raise ValueError("area_id_ttt out of range")
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