from __future__ import annotations

import numpy as np
import pytest

from seer_peph.data.prep import build_survival_long, build_treatment_long
from seer_peph.graphs import make_ring_lattice
from seer_peph.inference.run import InferenceConfig
from seer_peph.validation.simulate import simulate_joint
from seer_peph.fitting.fit_models import (
    fit_joint_model,
    fit_survival_model,
    fit_treatment_model,
)


def _encode_like_prep(wide):
    df = wide.copy()
    df["time_m"] = df["time"] / 30.4375
    df["treatment_time_m"] = df["treatment_time"] / 30.4375
    df["treatment_time_obs_m"] = df["treatment_time_obs"] / 30.4375
    df["sex_male"] = (df["sex"] == "M").astype(np.int8)
    df["stage_II"] = (df["stage"] == "II").astype(np.int8)
    df["stage_III"] = (df["stage"] == "III").astype(np.int8)
    area_map = {z: i for i, z in enumerate(sorted(df["zip"].unique()))}
    df["area_id"] = df["zip"].map(area_map).astype(np.int16)
    return df


@pytest.fixture(scope="module")
def small_sim_data():
    graph = make_ring_lattice(A=6, k=2)

    wide = simulate_joint(
        graph,
        n_per_area=15,
        rho_u=0.4,
        phi_surv=0.8,
        phi_ttt=0.8,
        sigma_surv=0.35,
        sigma_ttt=0.25,
        seed=2026,
    )
    df = _encode_like_prep(wide)

    surv_long = build_survival_long(df)
    ttt_long = build_treatment_long(df)

    infer_cfg = InferenceConfig(
        num_chains=1,
        num_warmup=25,
        num_samples=25,
        target_accept_prob=0.9,
        dense_mass=False,
        max_tree_depth=6,
        progress_bar=False,
    )

    return {
        "graph": graph,
        "wide": wide,
        "df": df,
        "surv_long": surv_long,
        "ttt_long": ttt_long,
        "infer_cfg": infer_cfg,
    }


@pytest.mark.integration
def test_fit_survival_model_smoke(small_sim_data):
    fit = fit_survival_model(
        surv_long=small_sim_data["surv_long"],
        ttt_long=small_sim_data["ttt_long"],
        graph=small_sim_data["graph"],
        surv_x_cols=[
            "age_per10_centered",
            "cci",
            "tumor_size_log",
            "stage_II",
            "stage_III",
        ],
        rng_seed=101,
        inference_config=small_sim_data["infer_cfg"],
        extra_fields=("diverging",),
    )

    assert fit.model_name == "survival_spatial_delta_only"
    assert fit.metadata.graph_A == small_sim_data["graph"].A
    assert fit.metadata.n_surv == len(small_sim_data["surv_long"])
    assert fit.metadata.n_ttt == len(small_sim_data["ttt_long"])
    assert fit.metadata.p_surv == 5
    assert fit.metadata.surv_x_cols == (
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "stage_II",
        "stage_III",
    )

    assert "alpha" in fit.samples
    assert "beta" in fit.samples
    assert "delta_post" in fit.samples
    assert "u" in fit.samples
    assert "rho" in fit.scalar_summary
    assert "tau" in fit.scalar_summary

    assert np.asarray(fit.samples["beta"]).ndim == 2
    assert np.asarray(fit.samples["beta"]).shape[1] == 5


@pytest.mark.integration
def test_fit_treatment_model_smoke(small_sim_data):
    fit = fit_treatment_model(
        surv_long=small_sim_data["surv_long"],
        ttt_long=small_sim_data["ttt_long"],
        graph=small_sim_data["graph"],
        ttt_x_cols=[
            "age_per10_centered",
            "cci",
            "ses",
            "sex_male",
            "stage_II",
            "stage_III",
        ],
        rng_seed=102,
        inference_config=small_sim_data["infer_cfg"],
        extra_fields=("diverging",),
    )

    assert fit.model_name == "treatment_spatial_pe"
    assert fit.metadata.graph_A == small_sim_data["graph"].A
    assert fit.metadata.n_ttt == len(small_sim_data["ttt_long"])
    assert fit.metadata.p_ttt == 6
    assert fit.metadata.ttt_x_cols == (
        "age_per10_centered",
        "cci",
        "ses",
        "sex_male",
        "stage_II",
        "stage_III",
    )

    assert "gamma" in fit.samples
    assert "theta" in fit.samples
    assert "u" in fit.samples
    assert "rho" in fit.scalar_summary
    assert "tau" in fit.scalar_summary

    assert np.asarray(fit.samples["theta"]).ndim == 2
    assert np.asarray(fit.samples["theta"]).shape[1] == 6


@pytest.mark.integration
def test_fit_joint_model_smoke(small_sim_data):
    fit = fit_joint_model(
        surv_long=small_sim_data["surv_long"],
        ttt_long=small_sim_data["ttt_long"],
        graph=small_sim_data["graph"],
        surv_x_cols=[
            "age_per10_centered",
            "cci",
            "tumor_size_log",
            "stage_II",
            "stage_III",
        ],
        ttt_x_cols=[
            "age_per10_centered",
            "cci",
            "ses",
            "sex_male",
            "stage_II",
            "stage_III",
        ],
        rng_seed=103,
        inference_config=small_sim_data["infer_cfg"],
        extra_fields=("diverging",),
    )

    assert fit.model_name == "joint_spatial_treatment_survival"
    assert fit.metadata.graph_A == small_sim_data["graph"].A
    assert fit.metadata.n_surv == len(small_sim_data["surv_long"])
    assert fit.metadata.n_ttt == len(small_sim_data["ttt_long"])
    assert fit.metadata.p_surv == 5
    assert fit.metadata.p_ttt == 6

    assert "alpha" in fit.samples
    assert "beta" in fit.samples
    assert "delta_post" in fit.samples
    assert "gamma" in fit.samples
    assert "theta" in fit.samples
    assert "u_surv" in fit.samples
    assert "u_ttt" in fit.samples
    assert "rho_u_cross" in fit.scalar_summary

    assert np.asarray(fit.samples["beta"]).ndim == 2
    assert np.asarray(fit.samples["beta"]).shape[1] == 5
    assert np.asarray(fit.samples["theta"]).ndim == 2
    assert np.asarray(fit.samples["theta"]).shape[1] == 6
    assert np.asarray(fit.samples["u_surv"]).shape[1] == small_sim_data["graph"].A
    assert np.asarray(fit.samples["u_ttt"]).shape[1] == small_sim_data["graph"].A