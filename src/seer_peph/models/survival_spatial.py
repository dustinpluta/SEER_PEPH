"""
seer_peph/models/survival_spatial.py
====================================

Phase 2 survival model:
Piecewise-exponential survival with

- interval-specific log-baseline hazards alpha[k]
- survival fixed effects beta
- treatment indicator effect beta_td
- piecewise post-treatment deviations delta_post
- BYM2 spatial frailty u[a]

This extends `survival_only.py` by adding a spatial survival frailty and
consumes the survival block plus graph block from `make_model_data(...)`.

Required data keys
------------------
Survival block:
    y_surv
    log_exposure_surv
    k_surv
    k_post
    treated_td
    area_id_surv
    X_surv

Graph block:
    node1
    node2
    scaling_factor
    A

Model
-----
For survival long row r:

    y_r ~ Poisson(exposure_r * lambda_r)

    log lambda_r
        = alpha[k_surv[r]]
        + X_surv[r] @ beta
        + treated_td[r] * beta_td
        + treated_td[r] * delta_post[k_post[r]]
        + u[area_id_surv[r]]

BYM2 prior
----------
Let s be a scaled ICAR field satisfying sum(s)=0.

    eps[a] ~ Normal(0, 1)
    s_free[j] ~ Normal(0, 1), j = 1, ..., A-1
    s[A] = -sum_{j=1}^{A-1} s_free[j]

    log p(s) ∝ -0.5 / scaling_factor * sum_{(i,j) in edges} (s[i] - s[j])^2

    rho ~ Beta(0.5, 0.5)
    tau ~ HalfNormal(1.0)

    u[a] = tau * ( sqrt(rho) * s[a] + sqrt(1-rho) * eps[a] )

Notes
-----
- `s` is stored as an unnormalized ICAR field with a scaling correction in
  the edge penalty. This matches the graph scaling-factor convention already
  used elsewhere in the project.
- The acute post-treatment interval is fixed at 0 for identifiability, so
  `beta_td` is the acute treatment effect and later `delta_post` entries are
  deviations from that reference.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def model(data: Mapping[str, Any]) -> None:
    y = jnp.asarray(data["y_surv"])
    log_exposure = jnp.asarray(data["log_exposure_surv"])
    k_surv = jnp.asarray(data["k_surv"])
    k_post = jnp.asarray(data["k_post"])
    treated_td = jnp.asarray(data["treated_td"])
    area_id = jnp.asarray(data["area_id_surv"])
    X = jnp.asarray(data["X_surv"])

    node1 = jnp.asarray(data["node1"])
    node2 = jnp.asarray(data["node2"])
    scaling_factor = jnp.asarray(data["scaling_factor"])
    A = int(data["A"])

    _validate_inputs(
        y=y,
        log_exposure=log_exposure,
        k_surv=k_surv,
        k_post=k_post,
        treated_td=treated_td,
        area_id=area_id,
        X=X,
        node1=node1,
        node2=node2,
        scaling_factor=scaling_factor,
        A=A,
        P_surv=data.get("P_surv"),
    )

    N, P = X.shape
    K_surv = int(np.asarray(k_surv).max()) + 1

    treated_np = np.asarray(treated_td)
    k_post_np = np.asarray(k_post)
    if np.any(treated_np == 1):
        K_post = int(k_post_np[treated_np == 1].max()) + 1
    else:
        K_post = 1

    alpha = numpyro.sample("alpha", dist.Normal(0.0, 2.0).expand([K_surv]))
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([P]))
    beta_td = numpyro.sample("beta_td", dist.Normal(0.0, 1.0))

    if K_post > 1:
        delta_free = numpyro.sample(
            "delta_post_free",
            dist.Normal(0.0, 0.5).expand([K_post - 1]),
        )
        delta_post = jnp.concatenate([jnp.array([0.0]), delta_free], axis=0)
    else:
        delta_post = jnp.array([0.0])

    numpyro.deterministic("delta_post", delta_post)

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
    icar_quad = jnp.sum(diff * diff) / scaling_factor
    numpyro.factor("icar_prior", -0.5 * icar_quad)

    u = tau * (jnp.sqrt(rho) * s + jnp.sqrt(1.0 - rho) * eps)

    numpyro.deterministic("s", s)
    numpyro.deterministic("u", u)

    k_post_safe = jnp.clip(k_post, a_min=0)
    eta = (
        alpha[k_surv]
        + jnp.matmul(X, beta)
        + treated_td * beta_td
        + treated_td * delta_post[k_post_safe]
        + u[area_id]
    )

    log_mu = log_exposure + eta
    numpyro.deterministic("eta_surv", eta)
    numpyro.deterministic("log_mu_surv", log_mu)

    with numpyro.plate("survival_rows", N):
        numpyro.sample("obs_surv", dist.Poisson(rate=jnp.exp(log_mu)), obs=y)


def _validate_inputs(
    *,
    y: jnp.ndarray,
    log_exposure: jnp.ndarray,
    k_surv: jnp.ndarray,
    k_post: jnp.ndarray,
    treated_td: jnp.ndarray,
    area_id: jnp.ndarray,
    X: jnp.ndarray,
    node1: jnp.ndarray,
    node2: jnp.ndarray,
    scaling_factor: jnp.ndarray,
    A: int,
    P_surv: int | None,
) -> None:
    y_np = np.asarray(y)
    log_exposure_np = np.asarray(log_exposure)
    k_surv_np = np.asarray(k_surv)
    k_post_np = np.asarray(k_post)
    treated_td_np = np.asarray(treated_td)
    area_id_np = np.asarray(area_id)
    X_np = np.asarray(X)
    node1_np = np.asarray(node1)
    node2_np = np.asarray(node2)
    scaling_np = float(np.asarray(scaling_factor))

    if y_np.ndim != 1:
        raise ValueError(f"y_surv must be 1-D, got shape {y_np.shape}")
    N = y_np.shape[0]

    for name, arr in [
        ("log_exposure_surv", log_exposure_np),
        ("k_surv", k_surv_np),
        ("k_post", k_post_np),
        ("treated_td", treated_td_np),
        ("area_id_surv", area_id_np),
    ]:
        if arr.ndim != 1 or arr.shape[0] != N:
            raise ValueError(f"{name} must have shape ({N},), got {arr.shape}")

    if X_np.ndim != 2 or X_np.shape[0] != N:
        raise ValueError(f"X_surv must have shape ({N}, P), got {X_np.shape}")

    if P_surv is not None and X_np.shape[1] != int(P_surv):
        raise ValueError(
            f"X_surv second dimension {X_np.shape[1]} does not match P_surv={P_surv}"
        )

    if not np.all(np.isin(y_np, [0, 1])):
        raise ValueError("y_surv must contain only 0/1 values")
    if not np.all(np.isfinite(log_exposure_np)):
        raise ValueError("log_exposure_surv must be finite")
    if not np.all(k_surv_np >= 0):
        raise ValueError("k_surv must be non-negative")
    if not np.all(np.isin(treated_td_np, [0, 1])):
        raise ValueError("treated_td must contain only 0/1 values")

    untreated_bad = (treated_td_np == 0) & (k_post_np != -1)
    if np.any(untreated_bad):
        raise ValueError("Rows with treated_td == 0 must have k_post == -1")

    treated_bad = (treated_td_np == 1) & (k_post_np < 0)
    if np.any(treated_bad):
        raise ValueError("Rows with treated_td == 1 must have k_post >= 0")

    if not np.all(area_id_np >= 0):
        raise ValueError("area_id_surv must be non-negative")
    if not np.all(area_id_np < A):
        raise ValueError("area_id_surv must be < A")

    if node1_np.ndim != 1 or node2_np.ndim != 1 or node1_np.shape != node2_np.shape:
        raise ValueError("node1 and node2 must be 1-D arrays of the same length")
    if not np.all(node1_np >= 0) or not np.all(node2_np >= 0):
        raise ValueError("node1 and node2 must be non-negative")
    if not np.all(node1_np < A) or not np.all(node2_np < A):
        raise ValueError("node1 and node2 entries must be < A")
    if scaling_np <= 0.0 or not np.isfinite(scaling_np):
        raise ValueError("scaling_factor must be finite and > 0")