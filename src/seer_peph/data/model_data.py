"""
seer_peph/data/model_data.py
============================

Convert long-format DataFrames from `seer_peph.data.prep` into the array
contract consumed by downstream NumPyro models.

Design goals
------------
1. Treat the long-format DataFrames as the single source of truth for column
   names and row semantics.
2. Validate aggressively: shapes, dtypes, ranges, missing values, and graph /
   area-index consistency.
3. Return arrays in a model-friendly contract, using JAX arrays when JAX is
   available (and requested), otherwise NumPy arrays.
4. Keep the interface simple and explicit so future model files can rely on a
   stable data contract.

Public API
----------
    make_model_data(surv_long, ttt_long, graph, ...)
    get_default_surv_x_cols(surv_long)
    get_default_ttt_x_cols(ttt_long)

Returned dict keys
------------------
Survival:
    y_surv              int32[N_surv]
    log_exposure_surv   float32[N_surv]
    k_surv              int32[N_surv]
    k_post              int32[N_surv]
    treated_td          int32[N_surv]
    area_id_surv        int32[N_surv]
    X_surv              float32[N_surv, P_surv]

Treatment:
    y_ttt               int32[N_ttt]
    log_exposure_ttt    float32[N_ttt]
    k_ttt               int32[N_ttt]
    area_id_ttt         int32[N_ttt]
    X_ttt               float32[N_ttt, P_ttt]

Graph constants:
    node1               int32[n_edges]
    node2               int32[n_edges]
    scaling_factor      float32 scalar
    A                   int scalar

Metadata:
    surv_x_cols         tuple[str, ...]
    ttt_x_cols          tuple[str, ...]
    N_surv, N_ttt       int
    P_surv, P_ttt       int
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..graphs import SpatialGraph


# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------

SURV_REQUIRED_COLS: tuple[str, ...] = (
    "id",
    "k",
    "exposure",
    "event",
    "treated_td",
    "k_post",
    "area_id",
)

TTT_REQUIRED_COLS: tuple[str, ...] = (
    "id",
    "k",
    "exposure",
    "event",
    "area_id",
)

SURV_DEFAULT_X_COLS: tuple[str, ...] = (
    "age_per10_centered",
    "cci",
    "tumor_size_log",
    "stage_II",
    "stage_III",
)

TTT_DEFAULT_X_COLS: tuple[str, ...] = (
    "age_per10_centered",
    "cci",
    "ses",
    "sex_male",
    "stage_II",
    "stage_III",
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_default_surv_x_cols(surv_long: pd.DataFrame) -> list[str]:
    """
    Return the default survival design columns, filtered to those present.

    This is useful if the caller built survival long data with a custom,
    reduced covariate set and wants a convenient default that still respects
    the actual DataFrame columns.
    """
    return [c for c in SURV_DEFAULT_X_COLS if c in surv_long.columns]


def get_default_ttt_x_cols(ttt_long: pd.DataFrame) -> list[str]:
    """
    Return the default treatment design columns, filtered to those present.

    This is useful if the caller built treatment long data with a custom,
    reduced covariate set and wants a convenient default that still respects
    the actual DataFrame columns.
    """
    return [c for c in TTT_DEFAULT_X_COLS if c in ttt_long.columns]


def make_model_data(
    surv_long: pd.DataFrame,
    ttt_long: pd.DataFrame,
    graph: SpatialGraph,
    *,
    surv_x_cols: Sequence[str] | None = None,
    ttt_x_cols: Sequence[str] | None = None,
    as_jax: bool = True,
    float_dtype: Any = np.float32,
    int_dtype: Any = np.int32,
) -> dict[str, Any]:
    """
    Build the array contract consumed by downstream NumPyro models.

    Parameters
    ----------
    surv_long
        Survival long-format DataFrame from `seer_peph.data.prep.build_survival_long`.
    ttt_long
        Treatment long-format DataFrame from `seer_peph.data.prep.build_treatment_long`.
    graph
        Spatial graph containing edge list and BYM2 scaling information.
    surv_x_cols
        Design columns for the survival fixed effects. If None, use the
        default survival covariate set, filtered to columns actually present.
    ttt_x_cols
        Design columns for the treatment fixed effects. If None, use the
        default treatment covariate set, filtered to columns actually present.
    as_jax
        If True, return JAX arrays when JAX is installed; otherwise fall back
        to NumPy arrays.
    float_dtype
        Floating dtype for design matrices and log exposures.
    int_dtype
        Integer dtype for count/index arrays.

    Returns
    -------
    dict
        Model-ready arrays plus graph constants and metadata.

    Notes
    -----
    The returned dictionary intentionally contains both arrays and small
    metadata fields (`surv_x_cols`, `ttt_x_cols`, sizes). Models should rely
    only on the array keys they need.
    """
    _validate_graph(graph)
    _validate_surv_long(surv_long, graph)
    _validate_ttt_long(ttt_long, graph)

    resolved_surv_x_cols = _resolve_x_cols(
        df=surv_long,
        x_cols=surv_x_cols,
        default_x_cols=get_default_surv_x_cols(surv_long),
        label="survival",
    )
    resolved_ttt_x_cols = _resolve_x_cols(
        df=ttt_long,
        x_cols=ttt_x_cols,
        default_x_cols=get_default_ttt_x_cols(ttt_long),
        label="treatment",
    )

    xp = _get_array_namespace(as_jax=as_jax)

    y_surv = _to_array(surv_long["event"], xp=xp, dtype=int_dtype)
    log_exposure_surv = _to_array(np.log(surv_long["exposure"].to_numpy(dtype=float)), xp=xp, dtype=float_dtype)
    k_surv = _to_array(surv_long["k"], xp=xp, dtype=int_dtype)
    k_post = _to_array(surv_long["k_post"], xp=xp, dtype=int_dtype)
    treated_td = _to_array(surv_long["treated_td"], xp=xp, dtype=int_dtype)
    area_id_surv = _to_array(surv_long["area_id"], xp=xp, dtype=int_dtype)
    X_surv = _matrix_from_cols(surv_long, resolved_surv_x_cols, xp=xp, dtype=float_dtype)

    y_ttt = _to_array(ttt_long["event"], xp=xp, dtype=int_dtype)
    log_exposure_ttt = _to_array(np.log(ttt_long["exposure"].to_numpy(dtype=float)), xp=xp, dtype=float_dtype)
    k_ttt = _to_array(ttt_long["k"], xp=xp, dtype=int_dtype)
    area_id_ttt = _to_array(ttt_long["area_id"], xp=xp, dtype=int_dtype)
    X_ttt = _matrix_from_cols(ttt_long, resolved_ttt_x_cols, xp=xp, dtype=float_dtype)

    node1 = xp.asarray(graph.node1, dtype=int_dtype)
    node2 = xp.asarray(graph.node2, dtype=int_dtype)
    scaling_factor = xp.asarray(graph.scaling_factor, dtype=float_dtype)
    A = int(graph.A)

    out: dict[str, Any] = {
        # Survival
        "y_surv": y_surv,
        "log_exposure_surv": log_exposure_surv,
        "k_surv": k_surv,
        "k_post": k_post,
        "treated_td": treated_td,
        "area_id_surv": area_id_surv,
        "X_surv": X_surv,
        # Treatment
        "y_ttt": y_ttt,
        "log_exposure_ttt": log_exposure_ttt,
        "k_ttt": k_ttt,
        "area_id_ttt": area_id_ttt,
        "X_ttt": X_ttt,
        # Graph
        "node1": node1,
        "node2": node2,
        "scaling_factor": scaling_factor,
        "A": A,
        # Metadata
        "surv_x_cols": tuple(resolved_surv_x_cols),
        "ttt_x_cols": tuple(resolved_ttt_x_cols),
        "N_surv": int(len(surv_long)),
        "N_ttt": int(len(ttt_long)),
        "P_surv": int(len(resolved_surv_x_cols)),
        "P_ttt": int(len(resolved_ttt_x_cols)),
    }

    _validate_output_shapes(out)

    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_graph(graph: SpatialGraph) -> None:
    if not isinstance(graph, SpatialGraph):
        raise TypeError("graph must be a SpatialGraph instance")
    if graph.A <= 1:
        raise ValueError("graph.A must be > 1")
    if graph.n_edges <= 0:
        raise ValueError("graph must contain at least one edge")
    if graph.node1.shape != graph.node2.shape:
        raise ValueError("graph.node1 and graph.node2 must have the same shape")
    if graph.node1.ndim != 1:
        raise ValueError("graph.node1 and graph.node2 must be 1-D arrays")
    if np.any(graph.node1 < 0) or np.any(graph.node2 < 0):
        raise ValueError("graph edge indices must be non-negative")
    if np.any(graph.node1 >= graph.A) or np.any(graph.node2 >= graph.A):
        raise ValueError("graph edge indices exceed graph.A - 1")
    if not np.isfinite(graph.scaling_factor) or graph.scaling_factor <= 0:
        raise ValueError("graph.scaling_factor must be finite and > 0")


def _validate_surv_long(df: pd.DataFrame, graph: SpatialGraph) -> None:
    _require_columns(df, SURV_REQUIRED_COLS, label="surv_long")
    _check_no_missing(df, SURV_REQUIRED_COLS, label="surv_long")
    _check_positive_exposure(df, label="surv_long")
    _check_binary(df, "event", label="surv_long")
    _check_binary(df, "treated_td", label="surv_long")
    _check_integerish_nonnegative(df, "k", label="surv_long")
    _check_integerish(df, "k_post", label="surv_long")
    _check_area_ids(df, graph, label="surv_long")

    if (df["k_post"] < -1).any():
        raise ValueError("surv_long.k_post must be >= -1")

    treated = df["treated_td"].to_numpy(dtype=int)
    k_post = df["k_post"].to_numpy(dtype=int)

    if np.any((treated == 0) & (k_post != -1)):
        raise ValueError("surv_long rows with treated_td == 0 must have k_post == -1")
    if np.any((treated == 1) & (k_post < 0)):
        raise ValueError("surv_long rows with treated_td == 1 must have k_post >= 0")


def _validate_ttt_long(df: pd.DataFrame, graph: SpatialGraph) -> None:
    _require_columns(df, TTT_REQUIRED_COLS, label="ttt_long")
    _check_no_missing(df, TTT_REQUIRED_COLS, label="ttt_long")
    _check_positive_exposure(df, label="ttt_long")
    _check_binary(df, "event", label="ttt_long")
    _check_integerish_nonnegative(df, "k", label="ttt_long")
    _check_area_ids(df, graph, label="ttt_long")


def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _check_no_missing(df: pd.DataFrame, cols: Sequence[str], *, label: str) -> None:
    missing_cols = [c for c in cols if df[c].isna().any()]
    if missing_cols:
        raise ValueError(f"{label} contains missing values in required columns: {missing_cols}")


def _check_positive_exposure(df: pd.DataFrame, *, label: str) -> None:
    if (df["exposure"] <= 0).any():
        raise ValueError(f"{label}.exposure must be strictly positive")
    if not np.isfinite(df["exposure"]).all():
        raise ValueError(f"{label}.exposure must be finite")


def _check_binary(df: pd.DataFrame, col: str, *, label: str) -> None:
    vals = set(pd.Series(df[col]).astype(int).unique())
    if not vals.issubset({0, 1}):
        raise ValueError(f"{label}.{col} must contain only 0/1; found {sorted(vals)}")


def _check_integerish_nonnegative(df: pd.DataFrame, col: str, *, label: str) -> None:
    _check_integerish(df, col, label=label)
    if (pd.Series(df[col]).astype(int) < 0).any():
        raise ValueError(f"{label}.{col} must be >= 0")


def _check_integerish(df: pd.DataFrame, col: str, *, label: str) -> None:
    vals = pd.Series(df[col]).to_numpy()
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"{label}.{col} must be finite")
    if not np.all(np.equal(vals, np.round(vals))):
        raise ValueError(f"{label}.{col} must contain integer values")


def _check_area_ids(df: pd.DataFrame, graph: SpatialGraph, *, label: str) -> None:
    area = pd.Series(df["area_id"]).astype(int).to_numpy()
    if np.any(area < 0):
        raise ValueError(f"{label}.area_id must be non-negative")
    if np.any(area >= graph.A):
        raise ValueError(f"{label}.area_id contains values >= graph.A")
    used = np.unique(area)
    if used.size == 0:
        raise ValueError(f"{label} must contain at least one row")


def _resolve_x_cols(
    *,
    df: pd.DataFrame,
    x_cols: Sequence[str] | None,
    default_x_cols: Sequence[str],
    label: str,
) -> list[str]:
    cols = list(default_x_cols if x_cols is None else x_cols)
    if len(cols) == 0:
        raise ValueError(f"{label} x_cols must contain at least one column")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} x_cols contain missing columns: {missing}")

    bad = [c for c in cols if df[c].isna().any()]
    if bad:
        raise ValueError(f"{label} x_cols contain missing values: {bad}")

    return cols


# ---------------------------------------------------------------------------
# Array conversion
# ---------------------------------------------------------------------------

def _get_array_namespace(*, as_jax: bool):
    if not as_jax:
        return np
    try:
        import jax.numpy as jnp
        return jnp
    except ImportError:
        return np


def _to_array(x: pd.Series | np.ndarray, *, xp, dtype: Any):
    return xp.asarray(np.asarray(x), dtype=dtype)


def _matrix_from_cols(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    xp,
    dtype: Any,
):
    arr = df.loc[:, list(cols)].to_numpy(dtype=float, copy=True)
    if arr.ndim != 2:
        raise RuntimeError("Design matrix construction failed: result is not 2-D")
    if not np.isfinite(arr).all():
        raise ValueError("Design matrix contains non-finite values")
    return xp.asarray(arr, dtype=dtype)


def _validate_output_shapes(data: Mapping[str, Any]) -> None:
    N_surv = data["N_surv"]
    N_ttt = data["N_ttt"]
    P_surv = data["P_surv"]
    P_ttt = data["P_ttt"]

    _assert_len(data["y_surv"], N_surv, "y_surv")
    _assert_len(data["log_exposure_surv"], N_surv, "log_exposure_surv")
    _assert_len(data["k_surv"], N_surv, "k_surv")
    _assert_len(data["k_post"], N_surv, "k_post")
    _assert_len(data["treated_td"], N_surv, "treated_td")
    _assert_len(data["area_id_surv"], N_surv, "area_id_surv")
    _assert_shape_2d(data["X_surv"], (N_surv, P_surv), "X_surv")

    _assert_len(data["y_ttt"], N_ttt, "y_ttt")
    _assert_len(data["log_exposure_ttt"], N_ttt, "log_exposure_ttt")
    _assert_len(data["k_ttt"], N_ttt, "k_ttt")
    _assert_len(data["area_id_ttt"], N_ttt, "area_id_ttt")
    _assert_shape_2d(data["X_ttt"], (N_ttt, P_ttt), "X_ttt")

    if int(np.asarray(data["node1"]).shape[0]) != int(np.asarray(data["node2"]).shape[0]):
        raise RuntimeError("node1 and node2 must have the same length")


def _assert_len(x: Any, n: int, name: str) -> None:
    shape = tuple(np.asarray(x).shape)
    if len(shape) != 1 or shape[0] != n:
        raise RuntimeError(f"{name} has shape {shape}, expected ({n},)")


def _assert_shape_2d(x: Any, shape_expected: tuple[int, int], name: str) -> None:
    shape = tuple(np.asarray(x).shape)
    if shape != shape_expected:
        raise RuntimeError(f"{name} has shape {shape}, expected {shape_expected}")