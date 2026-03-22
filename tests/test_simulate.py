# tests/test_simulate.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seer_peph.data.prep import (
    SURV_BREAKS,
    TTT_BREAKS,
    build_survival_long,
    build_treatment_long,
)
from seer_peph.graphs import make_ring_lattice
from seer_peph.validation.simulate import simulate_joint


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def test_simulate_joint_returns_expected_wide_schema_and_basic_types():
    graph = make_ring_lattice(A=12, k=4)
    df = simulate_joint(graph, n_per_area=25, seed=123)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12 * 25

    required_cols = {
        "id",
        "zip",
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "ses",
        "sex",
        "stage",
        "treatment_time",
        "treatment_time_obs",
        "treatment_event",
        "time",
        "event",
        "area_id_true",
        "eta_surv_x_true",
        "eta_ttt_x_true",
        "u_surv_true",
        "u_ttt_true",
        "phi_surv_true",
        "phi_ttt_true",
        "sigma_surv_true",
        "sigma_ttt_true",
        "beta_td_true",
        "survival_time_true",
        "censor_time",
    }
    assert required_cols.issubset(df.columns)

    assert df["id"].is_unique
    assert set(df["sex"].unique()).issubset({"M", "F"})
    assert set(df["stage"].unique()).issubset({"I", "II", "III"})
    assert set(df["event"].unique()).issubset({0, 1})
    assert set(df["treatment_event"].unique()).issubset({0, 1})

    assert (df["time"] >= 0).all()
    assert (df["treatment_time_obs"] >= 0).all()
    assert (df["survival_time_true"] >= 0).all()
    assert (df["censor_time"] > 0).all()

    ttt_nonmissing = df["treatment_time"].notna()
    assert (df.loc[ttt_nonmissing, "treatment_time"] >= 0).all()

    assert (df["treatment_time_obs"] <= df["time"] + 1e-8).all()

    treated = df["treatment_event"] == 1
    assert df.loc[treated, "treatment_time"].notna().all()
    assert np.allclose(
        df.loc[treated, "treatment_time"],
        df.loc[treated, "treatment_time_obs"],
    )

    untreated = df["treatment_event"] == 0
    assert df.loc[untreated, "treatment_time_obs"].notna().all()

    by_area = df.groupby("area_id_true")[["u_surv_true", "u_ttt_true"]].nunique()
    assert (by_area["u_surv_true"] == 1).all()
    assert (by_area["u_ttt_true"] == 1).all()


def test_simulate_joint_round_trips_through_prep_long_format():
    graph = make_ring_lattice(A=10, k=4)
    df = simulate_joint(graph, n_per_area=20, seed=321)

    df2 = df.copy()
    df2["time_m"] = df2["time"] / 30.4375
    df2["treatment_time_m"] = df2["treatment_time"] / 30.4375
    df2["treatment_time_obs_m"] = df2["treatment_time_obs"] / 30.4375
    df2["sex_male"] = (df2["sex"] == "M").astype(np.int8)
    df2["stage_II"] = (df2["stage"] == "II").astype(np.int8)
    df2["stage_III"] = (df2["stage"] == "III").astype(np.int8)

    area_map = {z: i for i, z in enumerate(sorted(df2["zip"].unique()))}
    df2["area_id"] = df2["zip"].map(area_map).astype(np.int16)

    surv_long = build_survival_long(df2)
    ttt_long = build_treatment_long(df2)

    assert len(surv_long) > 0
    assert len(ttt_long) > 0

    assert int(surv_long["event"].sum()) == int(df2["event"].sum())
    assert int(ttt_long["event"].sum()) == int(df2["treatment_event"].sum())

    surv_exposure = surv_long.groupby("id")["exposure"].sum().sort_index()
    surv_expected = (
        df2.set_index("id")["time_m"]
        .clip(upper=float(SURV_BREAKS[-1]))
        .sort_index()
    )
    assert np.allclose(surv_exposure.values, surv_expected.values)

    ttt_exposure = ttt_long.groupby("id")["exposure"].sum().sort_index()
    ttt_expected = (
        df2.set_index("id")["treatment_time_obs_m"]
        .clip(upper=float(TTT_BREAKS[-1]))
        .sort_index()
    )
    assert np.allclose(ttt_exposure.values, ttt_expected.values)


def test_simulate_joint_fixed_effects_are_encoded_in_true_linear_predictors():
    graph = make_ring_lattice(A=20, k=4)
    df = simulate_joint(graph, n_per_area=200, seed=2026)

    # Continuous covariates should align with the corresponding true LPs
    assert _corr(df["ses"], df["eta_ttt_x_true"]) > 0.20
    assert _corr(df["cci"], df["eta_surv_x_true"]) > 0.20
    assert _corr(df["tumor_size_log"], df["eta_surv_x_true"]) > 0.20

    # Stage effects are better checked by group means than by raw correlation
    ttt_stage_means = df.groupby("stage")["eta_ttt_x_true"].mean()
    surv_stage_means = df.groupby("stage")["eta_surv_x_true"].mean()

    assert ttt_stage_means["II"] > ttt_stage_means["I"]
    assert ttt_stage_means["III"] > ttt_stage_means["II"]

    assert surv_stage_means["II"] > surv_stage_means["I"]
    assert surv_stage_means["III"] > surv_stage_means["II"]


def test_simulate_joint_observed_outcomes_track_latent_risk_scores():
    graph = make_ring_lattice(A=24, k=4)
    df = simulate_joint(graph, n_per_area=180, seed=404)

    # Higher treatment LP should generally mean more observed treatment
    corr_ttt_lp_treated = _corr(df["eta_ttt_x_true"], df["treatment_event"])
    assert corr_ttt_lp_treated > 0.08

    # Higher survival LP should generally mean more death and shorter time
    corr_surv_lp_event = _corr(df["eta_surv_x_true"], df["event"])
    corr_surv_lp_time = _corr(df["eta_surv_x_true"], df["time"])

    assert corr_surv_lp_event > 0.08
    assert corr_surv_lp_time < -0.08


def test_simulate_joint_area_level_survival_signal_tracks_true_survival_frailty():
    graph = make_ring_lattice(A=24, k=4)
    df = simulate_joint(
        graph,
        n_per_area=150,
        sigma_surv=1.25,
        phi_surv=0.85,
        seed=11,
    )

    area = (
        df.groupby("area_id_true")
        .agg(
            u_surv_true=("u_surv_true", "first"),
            surv_event_rate=("event", "mean"),
            mean_time=("time", "mean"),
        )
        .reset_index(drop=True)
    )

    corr_u_event = _corr(area["u_surv_true"], area["surv_event_rate"])
    corr_u_time = _corr(area["u_surv_true"], area["mean_time"])

    assert corr_u_event > 0.30
    assert corr_u_time < -0.20


def test_simulate_joint_area_level_treatment_signal_tracks_true_treatment_frailty():
    graph = make_ring_lattice(A=24, k=4)
    df = simulate_joint(
        graph,
        n_per_area=150,
        sigma_ttt=0.80,
        phi_ttt=0.85,
        seed=12,
    )

    area = (
        df.groupby("area_id_true")
        .agg(
            u_ttt_true=("u_ttt_true", "first"),
            treated_rate=("treatment_event", "mean"),
            mean_ttt_obs=("treatment_time_obs", "mean"),
        )
        .reset_index(drop=True)
    )

    corr_u_treated = _corr(area["u_ttt_true"], area["treated_rate"])
    corr_u_ttt_time = _corr(area["u_ttt_true"], area["mean_ttt_obs"])

    assert corr_u_treated > 0.30
    assert corr_u_ttt_time < -0.20


def test_simulate_joint_cross_process_correlation_is_present_at_area_level():
    graph = make_ring_lattice(A=40, k=4)
    df = simulate_joint(
        graph,
        n_per_area=80,
        rho_u=0.90,
        phi_surv=0.80,
        phi_ttt=0.75,
        seed=999,
    )

    area = (
        df.groupby("area_id_true")
        .agg(
            u_surv_true=("u_surv_true", "first"),
            u_ttt_true=("u_ttt_true", "first"),
        )
        .reset_index(drop=True)
    )

    corr_u = _corr(area["u_surv_true"], area["u_ttt_true"])
    assert corr_u > 0.65


def test_simulate_joint_spatial_smoothing_creates_neighbor_similarity():
    graph = make_ring_lattice(A=30, k=4)
    df = simulate_joint(
        graph,
        n_per_area=60,
        phi_surv=0.95,
        sigma_surv=1.0,
        seed=444,
    )

    area = (
        df.groupby("area_id_true")
        .agg(u_surv_true=("u_surv_true", "first"))
        .sort_index()
    )
    u = area["u_surv_true"].to_numpy()

    edge_diffs = np.abs(u[graph.node1] - u[graph.node2])

    W = graph.adjacency
    nonedge_diffs = []
    for i in range(graph.A):
        for j in range(i + 1, graph.A):
            if W[i, j] == 0:
                nonedge_diffs.append(abs(u[i] - u[j]))
    nonedge_diffs = np.asarray(nonedge_diffs, dtype=float)

    assert edge_diffs.mean() < nonedge_diffs.mean()


def test_simulate_joint_respects_area_sample_size_vector():
    graph = make_ring_lattice(A=6, k=2)
    n_per_area = [5, 6, 7, 8, 9, 10]

    df = simulate_joint(graph, n_per_area=n_per_area, seed=101)

    counts = df.groupby("area_id_true").size().sort_index().to_numpy()
    assert np.array_equal(counts, np.asarray(n_per_area))