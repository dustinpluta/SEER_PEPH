"""
data/prep.py
============
Data preparation pipeline for the Bayesian spatial joint treatment–survival
model (colorectal cancer, SEER-Medicare Georgia cohort).

Produces two long-format Poisson datasets:

  surv_long  — survival sub-model
      One row per (subject, survival-interval, post-treatment-interval) cell.
      Poisson likelihood: event ~ Poisson(exposure * lambda_surv)

  ttt_long   — treatment sub-model (semi-competing risks)
      One row per (subject, treatment-interval) cell.
      Poisson likelihood: event ~ Poisson(exposure * lambda_ttt)

Time unit throughout: months (days / DAYS_PER_MONTH).

Public API
----------
    load_and_encode(path)       -> pd.DataFrame   wide data, encoded
    build_survival_long(df)     -> pd.DataFrame   survival long format
    build_treatment_long(df)    -> pd.DataFrame   treatment long format
    build_area_map(df)          -> pd.DataFrame   area_id <-> zip mapping
    main(path)                  -> dict           all artefacts

Usage
-----
    from data.prep import main
    data = main("path/to/joint_ttt_survival_dataset.csv")
    surv_long = data["surv_long"]
    ttt_long  = data["ttt_long"]

Command line:
    python -m data.prep path/to/joint_ttt_survival_dataset.csv
"""
from __future__ import annotations

import sys
from math import isclose
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ── constants ──────────────────────────────────────────────────────────────────

DAYS_PER_MONTH: float = 365.25 / 12  # 30.4375 days per month

# Survival grid: monthly resolution for year 1, quarterly/semi-annual thereafter.
DEFAULT_SURV_BREAKS: list[float] = [0, 1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 60]

# Treatment grid: concentrated in the first year post-diagnosis.
DEFAULT_TTT_BREAKS: list[float] = [0, 1, 2, 3, 6, 9, 12, 18, 24, 36, 60]

# Post-treatment piecewise effect grid (months since treatment onset).
# Implements delta^post_k: acute (0-3), early (3-6), mid (6-12),
# late (12-24), long-term (24-60).
DEFAULT_POST_TTT_BREAKS: list[float] = [0, 3, 6, 12, 24, 60]

# Temporary backward-compatible aliases.
SURV_BREAKS = DEFAULT_SURV_BREAKS
TTT_BREAKS = DEFAULT_TTT_BREAKS
POST_TTT_BREAKS = DEFAULT_POST_TTT_BREAKS

# Default covariates carried into every survival long row (area_id must be first).
SURV_X_COLS: list[str] = [
    "area_id",
    "age_per10_centered",
    "cci",
    "tumor_size_log",
    "ses",
    "sex_male",
    "stage_II",
    "stage_III",
]

# Default covariates carried into every treatment long row.
TTT_X_COLS: list[str] = [
    "area_id",
    "age_per10_centered",
    "cci",
    "tumor_size_log",
    "ses",
    "sex_male",
    "stage_II",
    "stage_III",
]


# ── private interval helpers ───────────────────────────────────────────────────

def _resolve_breaks(
    breaks: Optional[Sequence[float]],
    *,
    default: Sequence[float],
    name: str,
    require_zero_start: bool = True,
) -> List[float]:
    arr = np.asarray(default if breaks is None else breaks, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two values")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if require_zero_start and not np.isclose(arr[0], 0.0):
        raise ValueError(f"{name} must start at 0")

    return [float(v) for v in arr]


def _interval_index(breaks: np.ndarray, t: float) -> int:
    """
    Return k such that t in [breaks[k], breaks[k+1]) (left-closed, right-open).

    Uses searchsorted(side="right") so that t exactly equal to a break is
    assigned to the interval starting at that break.
    """
    return int(np.searchsorted(breaks, t, side="right") - 1)


def _event_interval_index(breaks: np.ndarray, t: float, *, atol: float = 1e-12) -> int:
    """
    Return the PE interval index that should carry the event at time t.

    Convention
    ----------
    - t in (breaks[k], breaks[k+1])  ->  k  (interior, standard case)
    - t == breaks[j] for interior j  ->  j-1  (event closes the preceding cell)
    - t <= 0 or t == breaks[-1]      ->  -1  (invalid; caller should skip/censor)
    """
    if t <= 0.0:
        return -1
    if isclose(t, float(breaks[-1]), abs_tol=atol):
        return -1

    match = np.where(np.isclose(breaks, t, atol=atol, rtol=0.0))[0]
    if match.size > 0:
        j = int(match[0])
        if j == 0 or j >= len(breaks) - 1:
            return -1
        return j - 1

    return _interval_index(breaks, t)


def _coerce_cut_times(value: object) -> List[float]:
    """
    Normalise a row-level cut-time field to list[float].

    Accepts scalar, list-like, or missing (-> []). Strings are not parsed.
    """
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        out: List[float] = []
        for v in value:
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            out.append(float(v))
        return out
    if np.isscalar(value):
        return [float(value)]
    raise ValueError("cut_times values must be numeric, list-like of numerics, or missing")


def _merged_breaks(
    *,
    global_breaks: np.ndarray,
    t_obs: float,
    extra_cuts: List[float],
) -> np.ndarray:
    """
    Merge global PE breaks with subject-specific cut times.

    Only cut times strictly inside (0, t_obs) are kept; duplicates are removed.
    """
    valid = [c for c in extra_cuts if 0.0 < c < t_obs]
    merged = np.unique(np.concatenate([global_breaks, np.asarray(valid, dtype=float)]))
    merged.sort()
    return merged


def _post_ttt_k(
    *,
    t0: float,
    treatment_time: Optional[float],
    post_breaks: np.ndarray,
) -> int:
    """
    Piecewise post-treatment interval index for a long row starting at t0.

    Returns -1 for pre-treatment rows or when treatment_time is None.
    Returns k in [0, K_post-1] based on time-since-treatment at t0, clamped
    to the last interval if time-since exceeds the final break.
    """
    if treatment_time is None or t0 < treatment_time:
        return -1
    time_since = t0 - treatment_time
    k = int(np.searchsorted(post_breaks, time_since, side="right") - 1)
    return max(0, min(k, len(post_breaks) - 2))


def _resolve_x_cols(
    *,
    df: pd.DataFrame,
    x_cols: Optional[Sequence[str]],
    default_x_cols: Sequence[str],
    label: str,
) -> List[str]:
    """
    Resolve and validate the covariate columns to propagate into long format.

    Rules
    -----
    - If x_cols is None, use the module default.
    - area_id must be included exactly once and will be forced to the front.
    - All requested columns must exist in df.
    """
    cols = list(default_x_cols if x_cols is None else x_cols)
    if len(cols) == 0:
        raise ValueError(f"{label} x_cols must contain at least one column")

    if "area_id" not in cols:
        cols = ["area_id"] + cols
    else:
        cols = ["area_id"] + [c for c in cols if c != "area_id"]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} x_cols contain missing columns: {missing}")

    return cols


# ── private expansion core ─────────────────────────────────────────────────────

def _expand(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    event_col: str,
    x_cols: List[str],
    breaks: Sequence[float],
    cut_times_col: Optional[str] = None,
    treatment_time_col: Optional[str] = None,
    post_ttt_breaks: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Core piecewise-exponential long-format expansion.

    When treatment_time_col is supplied:
      - treatment_time is inserted as a subject-level cut point so each row
        lies entirely within pre- or post-treatment person-time.
      - treated_td (0/1) is added to every row.

    When post_ttt_breaks is additionally supplied:
      - interior post-treatment boundaries are also inserted as cut points so
        each row lies within a single (treatment-status, post-ttt-interval)
        cell.
      - k_post (0-based index; -1 for pre-treatment) is added to every row.

    Zero-exposure rows are never emitted. Events exactly at tmax are
    administratively censored. Records with t_obs == 0 are dropped.
    """
    b = np.asarray(breaks, dtype=float)
    if b.ndim != 1 or b.size < 2 or b[0] != 0 or np.any(np.diff(b) <= 0):
        raise ValueError("breaks must be strictly increasing, start at 0, and have length >= 2")

    post_b: Optional[np.ndarray] = None
    if post_ttt_breaks is not None:
        if treatment_time_col is None:
            raise ValueError("post_ttt_breaks requires treatment_time_col")
        post_b = np.asarray(post_ttt_breaks, dtype=float)
        if post_b[0] != 0 or np.any(np.diff(post_b) <= 0):
            raise ValueError("post_ttt_breaks must be strictly increasing and start at 0")

    required = [id_col, time_col, event_col] + list(x_cols)
    if cut_times_col:
        required.append(cut_times_col)
    if treatment_time_col and treatment_time_col not in required:
        required.append(treatment_time_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    tmax = float(b[-1])
    K = b.size - 1
    out_rows: List[dict] = []

    for row in df.itertuples(index=False):
        rid = getattr(row, id_col)
        t = float(getattr(row, time_col))
        e = int(getattr(row, event_col))

        if t < 0:
            raise ValueError(f"Negative time for id={rid}: {t}")
        if e not in (0, 1):
            raise ValueError(f"event must be 0/1 for id={rid}, got {e}")

        t_obs = min(t, tmax)
        e_obs = 0 if (isclose(t_obs, tmax) or e == 0 or t > tmax) else 1

        if t_obs == 0:
            continue

        k_event: Optional[int] = None
        if e_obs == 1:
            k_event = _event_interval_index(b, t_obs)
            if not (0 <= k_event < K):
                raise RuntimeError(f"Event interval out of range for id={rid}, t={t_obs}")

        extra_cuts: List[float] = []
        if cut_times_col:
            extra_cuts = _coerce_cut_times(getattr(row, cut_times_col))

        ttt_val: Optional[float] = None
        if treatment_time_col:
            raw = getattr(row, treatment_time_col)
            if raw is not None:
                try:
                    if not pd.isna(raw):
                        ttt_val = float(raw)
                        if ttt_val < 0:
                            raise ValueError(f"Negative treatment time for id={rid}: {ttt_val}")
                except (TypeError, ValueError):
                    pass

        if ttt_val is not None:
            extra_cuts.append(ttt_val)
            if post_b is not None:
                for pb in post_b[1:]:
                    extra_cuts.append(ttt_val + float(pb))

        row_breaks = _merged_breaks(global_breaks=b, t_obs=t_obs, extra_cuts=extra_cuts)

        for j in range(len(row_breaks) - 1):
            t0 = float(row_breaks[j])
            t1g = float(row_breaks[j + 1])

            if t_obs <= t0:
                break

            k = _interval_index(b, t0)
            if not (0 <= k < K):
                continue

            if e_obs == 1 and k_event is not None and k > k_event:
                break

            t1 = min(t1g, t_obs)
            exposure = t1 - t0
            if exposure <= 0:
                continue

            is_event = (
                e_obs == 1
                and k_event is not None
                and k == k_event
                and isclose(t1, t_obs)
            )

            rec: dict = {
                "id": rid,
                "k": k,
                "t0": t0,
                "t1": t1,
                "exposure": exposure,
                "event": int(is_event),
            }
            for c in x_cols:
                rec[c] = getattr(row, c)

            if treatment_time_col is not None:
                rec["treated_td"] = int(ttt_val is not None and t0 >= ttt_val)
                if post_b is not None:
                    rec["k_post"] = _post_ttt_k(
                        t0=t0,
                        treatment_time=ttt_val,
                        post_breaks=post_b,
                    )

            out_rows.append(rec)

            if is_event:
                break

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out["k"] = out["k"].astype(int)
        out["event"] = out["event"].astype(int)
        if treatment_time_col is not None:
            out["treated_td"] = out["treated_td"].astype(int)
            if post_b is not None:
                out["k_post"] = out["k_post"].astype(int)
    return out


# ── public API ─────────────────────────────────────────────────────────────────

def load_and_encode(path: str | Path) -> pd.DataFrame:
    """
    Load wide-format SEER-like CSV and produce a modelling-ready dataframe.

    Transformations
    ---------------
    - Time variables converted from days to months (_m suffix).
    - Dummy coding: sex_male, stage_II, stage_III (reference: F, stage I).
    - Contiguous 0-based area_id assigned from sorted zip codes.

    Returns
    -------
    pd.DataFrame with all original columns plus:
        time_m, treatment_time_m, treatment_time_obs_m,
        sex_male, stage_II, stage_III, area_id.
    """
    df = pd.read_csv(path)

    df["time_m"] = df["time"] / DAYS_PER_MONTH
    df["treatment_time_m"] = df["treatment_time"] / DAYS_PER_MONTH
    df["treatment_time_obs_m"] = df["treatment_time_obs"] / DAYS_PER_MONTH

    df["sex_male"] = (df["sex"] == "M").astype(np.int8)
    df["stage_II"] = (df["stage"] == "II").astype(np.int8)
    df["stage_III"] = (df["stage"] == "III").astype(np.int8)

    sorted_zips = sorted(df["zip"].unique())
    df["area_id"] = df["zip"].map({z: i for i, z in enumerate(sorted_zips)}).astype(np.int16)

    return df


def build_survival_long(
    df: pd.DataFrame,
    *,
    x_cols: Optional[Sequence[str]] = None,
    surv_breaks: Optional[Sequence[float]] = None,
    post_ttt_breaks: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Expand survival data to long-format Poisson rows.

    Parameters
    ----------
    df
        Encoded wide-format dataframe from load_and_encode().
    x_cols
        Optional covariate columns to propagate into every survival long row.
        If None, uses SURV_X_COLS. area_id is required and will be forced to
        the first position if omitted or misplaced.
    surv_breaks
        Survival interval breakpoints in months. If None, uses
        DEFAULT_SURV_BREAKS.
    post_ttt_breaks
        Post-treatment interval breakpoints in months since treatment onset.
        If None, uses DEFAULT_POST_TTT_BREAKS.

    Output columns
    --------------
    id, k, t0, t1, exposure, event,
    treated_td, k_post,
    <x_cols...>

    k      : survival interval index  -> indexes alpha[k] in the NumPyro model
    k_post : post-treatment interval  -> indexes delta_post[k_post]; -1 for pre-treatment
    """
    resolved_x_cols = _resolve_x_cols(
        df=df,
        x_cols=x_cols,
        default_x_cols=SURV_X_COLS,
        label="survival",
    )
    resolved_surv_breaks = _resolve_breaks(
        surv_breaks,
        default=DEFAULT_SURV_BREAKS,
        name="surv_breaks",
    )
    resolved_post_ttt_breaks = _resolve_breaks(
        post_ttt_breaks,
        default=DEFAULT_POST_TTT_BREAKS,
        name="post_ttt_breaks",
    )
    return _expand(
        df,
        id_col="id",
        time_col="time_m",
        event_col="event",
        x_cols=resolved_x_cols,
        breaks=resolved_surv_breaks,
        treatment_time_col="treatment_time_m",
        post_ttt_breaks=resolved_post_ttt_breaks,
    )


def build_treatment_long(
    df: pd.DataFrame,
    *,
    x_cols: Optional[Sequence[str]] = None,
    ttt_breaks: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Expand treatment timing data to long-format Poisson rows.

    Semi-competing risks semantics
    ------------------------------
    Death censors treatment. The composite observed time
        treatment_time_obs_m = min(treatment_time, death_time, admin_censor)
    is used as the follow-up time, with treatment_event as the event indicator.
    Dependence between treatment and survival is captured entirely through the
    shared spatial frailties in the Bayesian model.

    Parameters
    ----------
    df
        Encoded wide-format dataframe from load_and_encode().
    x_cols
        Optional covariate columns to propagate into every treatment long row.
        If None, uses TTT_X_COLS. area_id is required and will be forced to
        the first position if omitted or misplaced.
    ttt_breaks
        Treatment interval breakpoints in months. If None, uses
        DEFAULT_TTT_BREAKS.

    Output columns
    --------------
    id, k, t0, t1, exposure, event,
    <x_cols...>

    k : treatment interval index -> indexes gamma[k] in the NumPyro model
    """
    resolved_x_cols = _resolve_x_cols(
        df=df,
        x_cols=x_cols,
        default_x_cols=TTT_X_COLS,
        label="treatment",
    )
    resolved_ttt_breaks = _resolve_breaks(
        ttt_breaks,
        default=DEFAULT_TTT_BREAKS,
        name="ttt_breaks",
    )
    return _expand(
        df,
        id_col="id",
        time_col="treatment_time_obs_m",
        event_col="treatment_event",
        x_cols=resolved_x_cols,
        breaks=resolved_ttt_breaks,
    )


def build_area_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a dataframe mapping area_id (0-based) to zip code.

    For real SEER-Medicare data, replace with a Georgia county adjacency
    matrix built from census shapefiles.
    """
    return (
        df[["area_id", "zip"]]
        .drop_duplicates()
        .sort_values("area_id")
        .reset_index(drop=True)
    )


# ── validation summary ─────────────────────────────────────────────────────────

def _check_exposure(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    obs_time_col: str,
    tmax: float,
    label: str,
) -> None:
    actual = long_df.groupby("id")["exposure"].sum()
    expected = (
        wide_df.set_index("id")[obs_time_col]
        .reindex(actual.index)
        .clip(upper=tmax)
    )
    max_err = (actual - expected).abs().max()
    flag = "✓" if max_err <= 1e-8 else "⚠"
    print(f"  {flag}  {label}: max exposure discrepancy = {max_err:.2e}")


def summarize(
    df: pd.DataFrame,
    surv_long: pd.DataFrame,
    ttt_long: pd.DataFrame,
    *,
    surv_breaks: Optional[Sequence[float]] = None,
    ttt_breaks: Optional[Sequence[float]] = None,
    post_ttt_breaks: Optional[Sequence[float]] = None,
) -> None:
    """Print a structured validation summary comparing wide and long formats."""
    resolved_surv_breaks = _resolve_breaks(
        surv_breaks,
        default=DEFAULT_SURV_BREAKS,
        name="surv_breaks",
    )
    resolved_ttt_breaks = _resolve_breaks(
        ttt_breaks,
        default=DEFAULT_TTT_BREAKS,
        name="ttt_breaks",
    )
    resolved_post_ttt_breaks = _resolve_breaks(
        post_ttt_breaks,
        default=DEFAULT_POST_TTT_BREAKS,
        name="post_ttt_breaks",
    )

    sep = "─" * 62
    K_surv = len(resolved_surv_breaks) - 1
    K_ttt = len(resolved_ttt_breaks) - 1

    print(sep)
    print("  WIDE DATA SUMMARY")
    print(sep)
    print(f"    Subjects           : {len(df):,}")
    print(f"    Areas              : {df['area_id'].nunique()}")
    n_se = df["event"].sum()
    n_te = df["treatment_event"].sum()
    print(f"    Survival events    : {n_se:,}  ({100 * df['event'].mean():.1f}%)")
    print(f"    Treatment events   : {n_te:,}  ({100 * df['treatment_event'].mean():.1f}%)")
    print(f"    Median surv time   : {df['time_m'].median():.1f} months")
    print(
        f"    Median ttt time    : "
        f"{df.loc[df.treatment_event == 1, 'treatment_time_obs_m'].median():.1f}"
        f" months (treated only)"
    )

    print()
    print(sep)
    print("  SURVIVAL LONG FORMAT")
    print(sep)
    n_se_long = surv_long["event"].sum()
    print(f"    Rows               : {len(surv_long):,}")
    print(
        f"    Events             : {n_se_long:,}  "
        f"{'✓' if n_se_long == n_se else '⚠'}  (wide: {n_se:,})"
    )
    print(f"    Total exposure     : {surv_long['exposure'].sum():,.1f} person-months")
    print(f"    Intervals active   : {surv_long['k'].nunique()} / {K_surv}")

    print()
    print("    Events and rate by survival interval:")
    grp = surv_long.groupby("k")[["event", "exposure"]].sum()
    grp.columns = ["events", "exposure_pm"]
    grp["rate/100pm"] = (100 * grp["events"] / grp["exposure_pm"]).round(3)
    grp.insert(
        0,
        "interval(mo)",
        [f"[{resolved_surv_breaks[k]:.0f},{resolved_surv_breaks[k + 1]:.0f})" for k in grp.index],
    )
    print(grp.to_string())

    print()
    print("    Post-treatment intervals (treated person-time only):")
    treated_rows = surv_long[surv_long["treated_td"] == 1]
    if len(treated_rows) > 0:
        kpg = treated_rows.groupby("k_post")[["event", "exposure"]].sum()
        kpg.columns = ["events", "exposure_pm"]
        kpg["rate/100pm"] = (100 * kpg["events"] / kpg["exposure_pm"]).round(3)
        kpg.insert(
            0,
            "post-ttt interval",
            [
                f"[{resolved_post_ttt_breaks[k]:.0f},{resolved_post_ttt_breaks[k + 1]:.0f})mo"
                for k in kpg.index
            ],
        )
        print(kpg.to_string())

    n_tr = (surv_long["treated_td"] == 1).sum()
    print(f"\n    Treated rows       : {n_tr:,}  ({100 * n_tr / len(surv_long):.1f}%)")
    print(
        f"    Untreated rows     : {len(surv_long) - n_tr:,}  "
        f"({100 * (len(surv_long) - n_tr) / len(surv_long):.1f}%)"
    )

    print()
    print(sep)
    print("  TREATMENT LONG FORMAT")
    print(sep)
    n_te_long = ttt_long["event"].sum()
    print(f"    Rows               : {len(ttt_long):,}")
    print(
        f"    Events             : {n_te_long:,}  "
        f"{'✓' if n_te_long == n_te else '⚠'}  (wide: {n_te:,})"
    )
    print(f"    Total exposure     : {ttt_long['exposure'].sum():,.1f} person-months")
    print(f"    Intervals active   : {ttt_long['k'].nunique()} / {K_ttt}")

    print()
    print("    Events and rate by treatment interval:")
    grp_t = ttt_long.groupby("k")[["event", "exposure"]].sum()
    grp_t.columns = ["events", "exposure_pm"]
    grp_t["rate/100pm"] = (100 * grp_t["events"] / grp_t["exposure_pm"]).round(3)
    grp_t.insert(
        0,
        "interval(mo)",
        [f"[{resolved_ttt_breaks[k]:.0f},{resolved_ttt_breaks[k + 1]:.0f})" for k in grp_t.index],
    )
    print(grp_t.to_string())

    print()
    print(sep)
    print("  EXPOSURE CONSISTENCY CHECKS")
    print(sep)
    _check_exposure(surv_long, df, "time_m", float(resolved_surv_breaks[-1]), "survival ")
    _check_exposure(ttt_long, df, "treatment_time_obs_m", float(resolved_ttt_breaks[-1]), "treatment")
    print(sep)


# ── main entry point ───────────────────────────────────────────────────────────

def main(
    path: str | Path = "joint_ttt_survival_dataset.csv",
    *,
    surv_breaks: Optional[Sequence[float]] = None,
    ttt_breaks: Optional[Sequence[float]] = None,
    post_ttt_breaks: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """
    Run the full data preparation pipeline.

    Returns
    -------
    dict with keys:
        "wide"      : pd.DataFrame  — encoded wide-format data
        "surv_long" : pd.DataFrame  — survival long-format Poisson rows
        "ttt_long"  : pd.DataFrame  — treatment long-format Poisson rows
        "area_map"  : pd.DataFrame  — area_id <-> zip mapping
        "grids"     : dict          — surv_breaks, ttt_breaks, post_ttt_breaks
    """
    resolved_surv_breaks = _resolve_breaks(
        surv_breaks,
        default=DEFAULT_SURV_BREAKS,
        name="surv_breaks",
    )
    resolved_ttt_breaks = _resolve_breaks(
        ttt_breaks,
        default=DEFAULT_TTT_BREAKS,
        name="ttt_breaks",
    )
    resolved_post_ttt_breaks = _resolve_breaks(
        post_ttt_breaks,
        default=DEFAULT_POST_TTT_BREAKS,
        name="post_ttt_breaks",
    )

    print(f"\nLoading data from: {path}")
    df = load_and_encode(path)
    print(f"  {len(df):,} subjects, {df['area_id'].nunique()} areas")

    print("\nExpanding survival long format ...")
    surv_long = build_survival_long(
        df,
        surv_breaks=resolved_surv_breaks,
        post_ttt_breaks=resolved_post_ttt_breaks,
    )

    print("Expanding treatment long format ...")
    ttt_long = build_treatment_long(
        df,
        ttt_breaks=resolved_ttt_breaks,
    )

    print()
    summarize(
        df,
        surv_long,
        ttt_long,
        surv_breaks=resolved_surv_breaks,
        ttt_breaks=resolved_ttt_breaks,
        post_ttt_breaks=resolved_post_ttt_breaks,
    )

    return {
        "wide": df,
        "surv_long": surv_long,
        "ttt_long": ttt_long,
        "area_map": build_area_map(df),
        "grids": {
            "surv_breaks": resolved_surv_breaks,
            "ttt_breaks": resolved_ttt_breaks,
            "post_ttt_breaks": resolved_post_ttt_breaks,
        },
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "joint_ttt_survival_dataset.csv"
    result = main(path)
    surv = result["surv_long"]
    ttt = result["ttt_long"]
    print(f"\nSurv long: {surv.shape}  cols: {list(surv.columns)}")
    print(f"TTT  long: {ttt.shape}  cols: {list(ttt.columns)}")
    print("\nSample survival rows:")
    print(surv.head(8).to_string(index=False))
    print("\nSample treatment rows:")
    print(ttt.head(5).to_string(index=False))