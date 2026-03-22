"""
seer_peph/models/survival_only.py
================================

Phase 1 / Competitor Model M1:
Piecewise-exponential survival model with

- interval-specific log-baseline hazards alpha[k]
- survival fixed effects beta
- treatment indicator effect beta_td
- piecewise post-treatment deviations delta_post

and NO frailties.

The model consumes the survival block of the `make_model_data()` contract:

    y_surv
    log_exposure_surv
    k_surv
    k_post
    treated_td
    X_surv

No treatment-process arrays are used in this model. No graph quantities are
used, even though they may be present in the data dict.

Model equation
--------------
For survival long row r:

    y_r ~ Poisson(exposure_r * lambda_r)

    log lambda_r
        = alpha[k_surv[r]]
        + X_surv[r] @ beta
        + treated_td[r] * beta_td
        + treated_td[r] * delta_eff[k_post[r]]

where delta_eff is a length-K_post vector of post-treatment deviations with
the acute interval (k_post = 0) fixed at 0 for identifiability:

    delta_eff = [0, delta_free_1, ..., delta_free_{K_post-1}]

Thus:
- beta_td is the acute post-treatment log-hazard ratio
- later post-treatment intervals modify that acute effect by delta_eff[k]

Priors
------
alpha_k         ~ Normal(0, 2)
beta_j          ~ Normal(0, 1)
beta_td         ~ Normal(0, 1)
delta_free_j    ~ Normal(0, 0.5)
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def model(data: Mapping[str, Any]) -> None:
    """
    NumPyro model for Phase 1 survival-only inference.

    Parameters
    ----------
    data
        Dictionary returned by `seer_peph.data.model_data.make_model_data(...)`.

    Required keys
    -------------
    y_surv : int array, shape (N,)
    log_exposure_surv : float array, shape (N,)
    k_surv : int array, shape (N,)
    k_post : int array, shape (N,)
    treated_td : int array, shape (N,)
    X_surv : float array, shape (N, P)

    Optional metadata
    -----------------
    P_surv : int
        Used only for consistency checking when present.

    Notes
    -----
    - `k_post == -1` is expected for untreated rows.
    - Untreated rows contribute no post-treatment term because the model
      multiplies the post-treatment component by `treated_td`.
    - For treated rows, `k_post` must be in {0, ..., K_post-1}.
    """
    y = jnp.asarray(data["y_surv"])
    log_exposure = jnp.asarray(data["log_exposure_surv"])
    k_surv = jnp.asarray(data["k_surv"])
    k_post = jnp.asarray(data["k_post"])
    treated_td = jnp.asarray(data["treated_td"])
    X = jnp.asarray(data["X_surv"])

    _validate_inputs(
        y=y,
        log_exposure=log_exposure,
        k_surv=k_surv,
        k_post=k_post,
        treated_td=treated_td,
        X=X,
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

    k_post_safe = jnp.clip(k_post, a_min=0)
    eta = (
        alpha[k_surv]
        + jnp.matmul(X, beta)
        + treated_td * beta_td
        + treated_td * delta_post[k_post_safe]
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
    X: jnp.ndarray,
    P_surv: int | None,
) -> None:
    """
    Runtime validation for the survival-only model.

    Only Python/NumPy checks are used here so validation remains compatible
    with JAX tracing inside NUTS.
    """
    y_np = np.asarray(y)
    log_exposure_np = np.asarray(log_exposure)
    k_surv_np = np.asarray(k_surv)
    k_post_np = np.asarray(k_post)
    treated_td_np = np.asarray(treated_td)
    X_np = np.asarray(X)

    if y_np.ndim != 1:
        raise ValueError(f"y_surv must be 1-D, got shape {y_np.shape}")
    N = y_np.shape[0]

    for name, arr in [
        ("log_exposure_surv", log_exposure_np),
        ("k_surv", k_surv_np),
        ("k_post", k_post_np),
        ("treated_td", treated_td_np),
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
        raise ValueError("y_surv must contain only 0/1 values in the expanded PE data")

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