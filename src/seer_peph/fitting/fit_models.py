from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import jax.random as random
import numpy as np
import pandas as pd

from seer_peph.data.model_data import make_model_data
from seer_peph.graphs import SpatialGraph
from seer_peph.inference.run import InferenceConfig, InferenceResult, run_mcmc, summarise_samples
from seer_peph.models.joint_spatial_treatment_survival import (
    model as joint_spatial_treatment_survival_model,
)
from seer_peph.models.survival_spatial_delta_only import (
    model as survival_spatial_delta_only_model,
)
from seer_peph.models.treatment_spatial_pe import (
    model as treatment_spatial_pe_model,
)


@dataclass(frozen=True)
class FitMetadata:
    """
    Lightweight metadata carried alongside fitted model objects.

    Parameters
    ----------
    surv_x_cols
        Ordered survival design columns used to construct X_surv.
    ttt_x_cols
        Ordered treatment design columns used to construct X_ttt.
    surv_breaks
        Optional survival interval break grid used upstream in prep.
    ttt_breaks
        Optional treatment interval break grid used upstream in prep.
    post_ttt_breaks
        Optional post-treatment interval break grid used upstream in prep.
    graph_A
        Number of spatial areas in the graph.
    graph_n_edges
        Number of undirected graph edges represented by node1/node2.
    n_surv
        Number of survival long-format rows.
    n_ttt
        Number of treatment long-format rows.
    p_surv
        Number of survival fixed-effect covariates.
    p_ttt
        Number of treatment fixed-effect covariates.
    rng_seed
        Integer seed used to generate the JAX PRNG key.
    """
    surv_x_cols: tuple[str, ...] = ()
    ttt_x_cols: tuple[str, ...] = ()
    surv_breaks: tuple[float, ...] | None = None
    ttt_breaks: tuple[float, ...] | None = None
    post_ttt_breaks: tuple[float, ...] | None = None
    graph_A: int | None = None
    graph_n_edges: int | None = None
    n_surv: int | None = None
    n_ttt: int | None = None
    p_surv: int | None = None
    p_ttt: int | None = None
    rng_seed: int | None = None


@dataclass
class SurvivalFit:
    """
    Fitted standalone survival model object.
    """
    model_name: str
    inference_result: InferenceResult
    samples: dict[str, Any]
    summary: dict[str, Any]
    scalar_summary: dict[str, dict[str, float]]
    data: dict[str, Any]
    metadata: FitMetadata
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TreatmentFit:
    """
    Fitted standalone treatment-time model object.
    """
    model_name: str
    inference_result: InferenceResult
    samples: dict[str, Any]
    summary: dict[str, Any]
    scalar_summary: dict[str, dict[str, float]]
    data: dict[str, Any]
    metadata: FitMetadata
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class JointFit:
    """
    Fitted joint treatment-survival model object.
    """
    model_name: str
    inference_result: InferenceResult
    samples: dict[str, Any]
    summary: dict[str, Any]
    scalar_summary: dict[str, dict[str, float]]
    data: dict[str, Any]
    metadata: FitMetadata
    extra: dict[str, Any] = field(default_factory=dict)


def fit_survival_model(
    *,
    surv_long: pd.DataFrame | None = None,
    ttt_long: pd.DataFrame | None = None,
    graph: SpatialGraph | None = None,
    data: Mapping[str, Any] | None = None,
    surv_x_cols: Sequence[str] | None = None,
    ttt_x_cols: Sequence[str] | None = None,
    surv_breaks: Sequence[float] | None = None,
    ttt_breaks: Sequence[float] | None = None,
    post_ttt_breaks: Sequence[float] | None = None,
    rng_seed: int = 123,
    inference_config: InferenceConfig | None = None,
    as_jax: bool = True,
    extra_fields: tuple[str, ...] = ("diverging",),
    extra_metadata: Mapping[str, Any] | None = None,
) -> SurvivalFit:
    """
    Fit the standalone delta-only spatial survival model.

    Either pass a prebuilt `data` dictionary from `make_model_data(...)`, or
    pass `surv_long`, `ttt_long`, and `graph` so the wrapper can build it.

    Notes
    -----
    The current model-data contract requires both survival and treatment long
    tables even for survival-only fitting, because `make_model_data(...)`
    validates and returns the full shared array contract. :contentReference[oaicite:2]{index=2}
    """
    model_data = _resolve_model_data(
        surv_long=surv_long,
        ttt_long=ttt_long,
        graph=graph,
        data=data,
        surv_x_cols=surv_x_cols,
        ttt_x_cols=ttt_x_cols,
        as_jax=as_jax,
    )

    result = run_mcmc(
        survival_spatial_delta_only_model,
        model_data,
        rng_key=random.PRNGKey(rng_seed),
        config=inference_config,
        extra_fields=extra_fields,
    )

    scalar_summary = summarise_samples(result.samples)
    metadata = _build_fit_metadata(
        data=model_data,
        surv_breaks=surv_breaks,
        ttt_breaks=ttt_breaks,
        post_ttt_breaks=post_ttt_breaks,
        rng_seed=rng_seed,
    )

    return SurvivalFit(
        model_name="survival_spatial_delta_only",
        inference_result=result,
        samples=result.samples,
        summary=result.summary,
        scalar_summary=scalar_summary,
        data=dict(model_data),
        metadata=metadata,
        extra=dict(extra_metadata or {}),
    )


def fit_treatment_model(
    *,
    surv_long: pd.DataFrame | None = None,
    ttt_long: pd.DataFrame | None = None,
    graph: SpatialGraph | None = None,
    data: Mapping[str, Any] | None = None,
    surv_x_cols: Sequence[str] | None = None,
    ttt_x_cols: Sequence[str] | None = None,
    surv_breaks: Sequence[float] | None = None,
    ttt_breaks: Sequence[float] | None = None,
    post_ttt_breaks: Sequence[float] | None = None,
    rng_seed: int = 123,
    inference_config: InferenceConfig | None = None,
    as_jax: bool = True,
    extra_fields: tuple[str, ...] = ("diverging",),
    extra_metadata: Mapping[str, Any] | None = None,
) -> TreatmentFit:
    """
    Fit the standalone spatial treatment-time PE model.
    """
    model_data = _resolve_model_data(
        surv_long=surv_long,
        ttt_long=ttt_long,
        graph=graph,
        data=data,
        surv_x_cols=surv_x_cols,
        ttt_x_cols=ttt_x_cols,
        as_jax=as_jax,
    )

    result = run_mcmc(
        treatment_spatial_pe_model,
        model_data,
        rng_key=random.PRNGKey(rng_seed),
        config=inference_config,
        extra_fields=extra_fields,
    )

    scalar_summary = summarise_samples(result.samples)
    metadata = _build_fit_metadata(
        data=model_data,
        surv_breaks=surv_breaks,
        ttt_breaks=ttt_breaks,
        post_ttt_breaks=post_ttt_breaks,
        rng_seed=rng_seed,
    )

    return TreatmentFit(
        model_name="treatment_spatial_pe",
        inference_result=result,
        samples=result.samples,
        summary=result.summary,
        scalar_summary=scalar_summary,
        data=dict(model_data),
        metadata=metadata,
        extra=dict(extra_metadata or {}),
    )


def fit_joint_model(
    *,
    surv_long: pd.DataFrame | None = None,
    ttt_long: pd.DataFrame | None = None,
    graph: SpatialGraph | None = None,
    data: Mapping[str, Any] | None = None,
    surv_x_cols: Sequence[str] | None = None,
    ttt_x_cols: Sequence[str] | None = None,
    surv_breaks: Sequence[float] | None = None,
    ttt_breaks: Sequence[float] | None = None,
    post_ttt_breaks: Sequence[float] | None = None,
    rng_seed: int = 123,
    inference_config: InferenceConfig | None = None,
    as_jax: bool = True,
    extra_fields: tuple[str, ...] = ("diverging",),
    extra_metadata: Mapping[str, Any] | None = None,
) -> JointFit:
    """
    Fit the current joint spatial treatment-survival model.
    """
    model_data = _resolve_model_data(
        surv_long=surv_long,
        ttt_long=ttt_long,
        graph=graph,
        data=data,
        surv_x_cols=surv_x_cols,
        ttt_x_cols=ttt_x_cols,
        as_jax=as_jax,
    )

    result = run_mcmc(
        joint_spatial_treatment_survival_model,
        model_data,
        rng_key=random.PRNGKey(rng_seed),
        config=inference_config,
        extra_fields=extra_fields,
    )

    scalar_summary = summarise_samples(result.samples)
    metadata = _build_fit_metadata(
        data=model_data,
        surv_breaks=surv_breaks,
        ttt_breaks=ttt_breaks,
        post_ttt_breaks=post_ttt_breaks,
        rng_seed=rng_seed,
    )

    return JointFit(
        model_name="joint_spatial_treatment_survival",
        inference_result=result,
        samples=result.samples,
        summary=result.summary,
        scalar_summary=scalar_summary,
        data=dict(model_data),
        metadata=metadata,
        extra=dict(extra_metadata or {}),
    )


def _resolve_model_data(
    *,
    surv_long: pd.DataFrame | None,
    ttt_long: pd.DataFrame | None,
    graph: SpatialGraph | None,
    data: Mapping[str, Any] | None,
    surv_x_cols: Sequence[str] | None,
    ttt_x_cols: Sequence[str] | None,
    as_jax: bool,
) -> dict[str, Any]:
    if data is not None:
        if any(x is not None for x in (surv_long, ttt_long, graph)):
            raise ValueError(
                "Pass either `data` or (`surv_long`, `ttt_long`, `graph`), not both."
            )
        return dict(data)

    if surv_long is None or ttt_long is None or graph is None:
        raise ValueError(
            "When `data` is not provided, `surv_long`, `ttt_long`, and `graph` are all required."
        )

    return make_model_data(
        surv_long,
        ttt_long,
        graph,
        surv_x_cols=surv_x_cols,
        ttt_x_cols=ttt_x_cols,
        as_jax=as_jax,
    )


def _build_fit_metadata(
    *,
    data: Mapping[str, Any],
    surv_breaks: Sequence[float] | None,
    ttt_breaks: Sequence[float] | None,
    post_ttt_breaks: Sequence[float] | None,
    rng_seed: int,
) -> FitMetadata:
    node1 = np.asarray(data["node1"])
    surv_x_cols = tuple(str(x) for x in data.get("surv_x_cols", ()))
    ttt_x_cols = tuple(str(x) for x in data.get("ttt_x_cols", ()))

    return FitMetadata(
        surv_x_cols=surv_x_cols,
        ttt_x_cols=ttt_x_cols,
        surv_breaks=_as_optional_tuple(surv_breaks),
        ttt_breaks=_as_optional_tuple(ttt_breaks),
        post_ttt_breaks=_as_optional_tuple(post_ttt_breaks),
        graph_A=int(data.get("A")) if "A" in data else None,
        graph_n_edges=int(node1.shape[0]),
        n_surv=int(data.get("N_surv")) if "N_surv" in data else None,
        n_ttt=int(data.get("N_ttt")) if "N_ttt" in data else None,
        p_surv=int(data.get("P_surv")) if "P_surv" in data else None,
        p_ttt=int(data.get("P_ttt")) if "P_ttt" in data else None,
        rng_seed=int(rng_seed),
    )


def _as_optional_tuple(x: Sequence[float] | None) -> tuple[float, ...] | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Break grids must be one-dimensional sequences.")
    return tuple(float(v) for v in arr)