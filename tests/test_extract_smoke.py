from __future__ import annotations

import numpy as np
import pytest

from seer_peph.data.prep import build_survival_long, build_treatment_long
from seer_peph.fitting.extract import (
    extract_joint_coupling,
    extract_spatial_fields,
    extract_survival_effects,
    extract_treatment_effects,
)
from seer_peph.fitting.fit_models import (
    fit_joint_model,
    fit_survival_model,
    fit_treatment_model,
)
from seer_peph.graphs import make_ring_lattice
from seer_peph.inference.run import InferenceConfig
from seer_peph.validation.simulate import simulate_joint


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
def test_extract_survival_effects_smoke(small_sim_data):
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

    out = extract_survival_effects(fit, include_draws=True)

    assert set(out.keys()) == {
        "beta",
        "alpha",
        "delta_post",
        "beta_draws",
        "alpha_draws",
        "delta_post_draws",
    }

    beta_df = out["beta"]
    assert not beta_df.empty
    assert set(beta_df.columns) == {
        "parameter",
        "index",
        "label",
        "param_type",
        "mean",
        "sd",
        "median",
        "q05",
        "q95",
    }
    assert beta_df["label"].tolist() == [
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "stage_II",
        "stage_III",
    ]

    assert not out["alpha"].empty
    assert not out["delta_post"].empty

    beta_draws = out["beta_draws"]
    assert not beta_draws.empty
    assert set(beta_draws.columns) == {"draw", "parameter", "index", "label", "value"}
    assert beta_draws["parameter"].str.startswith("beta[").all()


@pytest.mark.integration
def test_extract_treatment_effects_smoke(small_sim_data):
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

    out = extract_treatment_effects(fit, include_draws=True)

    assert set(out.keys()) == {"theta", "gamma", "theta_draws", "gamma_draws"}

    theta_df = out["theta"]
    assert not theta_df.empty
    assert set(theta_df.columns) == {
        "parameter",
        "index",
        "label",
        "param_type",
        "mean",
        "sd",
        "median",
        "q05",
        "q95",
    }
    assert theta_df["label"].tolist() == [
        "age_per10_centered",
        "cci",
        "ses",
        "sex_male",
        "stage_II",
        "stage_III",
    ]

    gamma_df = out["gamma"]
    assert not gamma_df.empty
    assert gamma_df["parameter"].str.startswith("gamma[").all()

    theta_draws = out["theta_draws"]
    assert not theta_draws.empty
    assert set(theta_draws.columns) == {"draw", "parameter", "index", "label", "value"}


@pytest.mark.integration
def test_extract_spatial_fields_survival_smoke(small_sim_data):
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
        rng_seed=103,
        inference_config=small_sim_data["infer_cfg"],
        extra_fields=("diverging",),
    )

    out = extract_spatial_fields(fit, include_draws=True)

    assert set(out.keys()) == {"field", "field_draws", "hyperparameters"}

    field_df = out["field"]
    assert not field_df.empty
    assert set(field_df.columns) == {
        "parameter",
        "field",
        "field_type",
        "area_id",
        "mean",
        "sd",
        "median",
        "q05",
        "q95",
    }
    assert field_df["field"].eq("u").all()
    assert field_df["field_type"].eq("survival_field").all()
    assert field_df["area_id"].nunique() == small_sim_data["graph"].A

    hyper_df = out["hyperparameters"]
    assert not hyper_df.empty
    assert hyper_df["parameter"].tolist() == ["rho", "tau"]

    field_draws = out["field_draws"]
    assert not field_draws.empty
    assert set(field_draws.columns) == {"draw", "parameter", "field", "area_id", "value"}


@pytest.mark.integration
def test_extract_spatial_fields_joint_smoke(small_sim_data):
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
        rng_seed=104,
        inference_config=small_sim_data["infer_cfg"],
        extra_fields=("diverging",),
    )

    out = extract_spatial_fields(fit, include_draws=False)

    expected = {"u_surv", "u_ttt", "u_ttt_ind", "s_surv", "s_ttt", "hyperparameters"}
    assert set(out.keys()) == expected

    assert not out["u_surv"].empty
    assert not out["u_ttt"].empty
    assert not out["hyperparameters"].empty

    assert out["u_surv"]["field_type"].eq("joint_survival_field").all()
    assert out["u_ttt"]["field_type"].eq("joint_treatment_field").all()

    hyper_df = out["hyperparameters"]
    assert hyper_df["parameter"].tolist() == ["rho_surv", "tau_surv", "rho_ttt", "tau_ttt"]


@pytest.mark.integration
def test_extract_joint_coupling_smoke(small_sim_data):
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
        rng_seed=105,
        inference_config=small_sim_data["infer_cfg"],
        extra_fields=("diverging",),
    )

    out = extract_joint_coupling(fit, include_draws=True)

    assert set(out.keys()) == {"coupling", "coupling_draws", "field_correlations"}

    coupling_df = out["coupling"]
    assert not coupling_df.empty
    assert coupling_df["parameter"].tolist() == ["rho_u_cross"]
    assert coupling_df["param_group"].tolist() == ["joint_coupling"]

    coupling_draws = out["coupling_draws"]
    assert not coupling_draws.empty
    assert set(coupling_draws.columns) == {"draw", "parameter", "value"}
    assert coupling_draws["parameter"].eq("rho_u_cross").all()

    field_corr = out["field_correlations"]
    assert not field_corr.empty
    assert set(field_corr.columns) == {"metric", "value"}
    assert "corr_u_surv_u_ttt_draw_mean" in field_corr["metric"].tolist()