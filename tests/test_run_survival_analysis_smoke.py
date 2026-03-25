from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from seer_peph.graphs import make_ring_lattice
from seer_peph.validation.simulate import simulate_joint
from scripts.run_survival_analysis import (
    SurvivalAnalysisConfig,
    run_survival_analysis,
)


def _encode_like_prep(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()
    df["time_m"] = df["time"] / 30.4375
    df["treatment_time_m"] = df["treatment_time"] / 30.4375
    df["treatment_time_obs_m"] = df["treatment_time_obs"] / 30.4375
    df["sex_male"] = (df["sex"] == "M").astype("int8")
    df["stage_II"] = (df["stage"] == "II").astype("int8")
    df["stage_III"] = (df["stage"] == "III").astype("int8")

    area_map = {z: i for i, z in enumerate(sorted(df["zip"].unique()))}
    df["area_id"] = df["zip"].map(area_map).astype("int16")
    return df


@pytest.mark.integration
def test_run_survival_analysis_smoke(tmp_path: Path) -> None:
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
    wide = _encode_like_prep(wide)

    input_path = tmp_path / "simulated_wide.csv"
    wide.to_csv(input_path, index=False)

    out_dir = tmp_path / "survival_analysis_artifacts"

    cfg = SurvivalAnalysisConfig(
        input_path=str(input_path),
        out_dir=str(out_dir),
        graph_mode="from_area_id_ring",
        graph_A=6,
        graph_k=2,
        rng_seed=101,
        inference={
            "num_chains": 1,
            "num_warmup": 25,
            "num_samples": 25,
            "target_accept_prob": 0.9,
            "dense_mass": False,
            "max_tree_depth": 6,
            "progress_bar": False,
        },
    )

    returned_out_dir = run_survival_analysis(cfg)

    assert returned_out_dir == out_dir
    assert out_dir.exists()

    expected_files = [
        "surv_long.csv",
        "ttt_long.csv",
        "survival_beta_summary.csv",
        "survival_alpha_summary.csv",
        "survival_delta_post_summary.csv",
        "survival_spatial_field_summary.csv",
        "survival_spatial_hyperparameter_summary.csv",
        "predicted_survival_scenarios.csv",
        "predicted_rmst_scenarios.csv",
        "predicted_survival_contrasts.csv",
        "predicted_rmst_contrasts.csv",
        "prediction_profile.json",
        "analysis_config.json",
        "run_manifest.json",
    ]

    for name in expected_files:
        path = out_dir / name
        assert path.exists(), f"Missing expected artifact: {name}"
        assert path.stat().st_size > 0, f"Artifact is empty: {name}"

    fit_dir = out_dir / "fit_bundle"
    assert fit_dir.exists()
    assert fit_dir.is_dir()

    expected_fit_bundle_files = [
        "manifest.json",
        "samples.npz",
        "data_arrays.npz",
        "summary.json",
        "scalar_summary.json",
    ]
    for name in expected_fit_bundle_files:
        path = fit_dir / name
        assert path.exists(), f"Missing fit bundle artifact: {name}"
        assert path.stat().st_size > 0, f"Fit bundle artifact is empty: {name}"

    surv_long = pd.read_csv(out_dir / "surv_long.csv")
    ttt_long = pd.read_csv(out_dir / "ttt_long.csv")
    beta_summary = pd.read_csv(out_dir / "survival_beta_summary.csv")
    alpha_summary = pd.read_csv(out_dir / "survival_alpha_summary.csv")
    delta_summary = pd.read_csv(out_dir / "survival_delta_post_summary.csv")
    spatial_field = pd.read_csv(out_dir / "survival_spatial_field_summary.csv")
    spatial_hyper = pd.read_csv(out_dir / "survival_spatial_hyperparameter_summary.csv")
    surv_scenarios = pd.read_csv(out_dir / "predicted_survival_scenarios.csv")
    rmst_scenarios = pd.read_csv(out_dir / "predicted_rmst_scenarios.csv")
    surv_contrasts = pd.read_csv(out_dir / "predicted_survival_contrasts.csv")
    rmst_contrasts = pd.read_csv(out_dir / "predicted_rmst_contrasts.csv")

    assert not surv_long.empty
    assert not ttt_long.empty
    assert not beta_summary.empty
    assert not alpha_summary.empty
    assert not delta_summary.empty
    assert not spatial_field.empty
    assert not spatial_hyper.empty
    assert not surv_scenarios.empty
    assert not rmst_scenarios.empty
    assert not surv_contrasts.empty
    assert not rmst_contrasts.empty

    assert {"parameter", "label", "mean"}.issubset(beta_summary.columns)
    assert {"parameter", "label", "mean"}.issubset(alpha_summary.columns)
    assert {"parameter", "label", "mean"}.issubset(delta_summary.columns)
    assert {"parameter", "field", "area_id", "mean"}.issubset(spatial_field.columns)
    assert {"parameter", "param_group", "mean"}.issubset(spatial_hyper.columns)

    assert {
        "time_m",
        "treatment_time_m",
        "area_id",
        "mean_survival",
    }.issubset(surv_scenarios.columns)

    assert {
        "treatment_time_m",
        "mean_rmst",
        "horizon_m",
        "area_id",
    }.issubset(rmst_scenarios.columns)

    assert {
        "time_m",
        "treatment_time_m_a",
        "treatment_time_m_b",
        "mean_survival_diff",
    }.issubset(surv_contrasts.columns)

    assert {
        "treatment_time_m_a",
        "treatment_time_m_b",
        "mean_rmst_diff",
        "horizon_m",
    }.issubset(rmst_contrasts.columns)

    assert surv_scenarios["mean_survival"].between(0.0, 1.0).all()
    assert (rmst_scenarios["mean_rmst"] >= 0.0).all()
    assert (rmst_scenarios["mean_rmst"] <= rmst_scenarios["horizon_m"]).all()