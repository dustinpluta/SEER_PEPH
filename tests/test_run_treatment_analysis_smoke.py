from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from seer_peph.analysis.treatment_analysis import (
    DerivedColumnConfig,
    InputColumnConfig,
    TreatmentAnalysisConfig,
    run_treatment_analysis,
)
from seer_peph.data.prep import (
    DEFAULT_POST_TTT_BREAKS,
    DEFAULT_SURV_BREAKS,
    DEFAULT_TTT_BREAKS,
)
from seer_peph.graphs import make_ring_lattice
from seer_peph.validation.simulate import simulate_joint


def _rename_to_custom_schema(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()

    rename_map = {
        "id": "patient_id",
        "time": "os_days",
        "event": "os_event",
        "treatment_time": "tx_start_days",
        "treatment_time_obs": "tx_start_obs_days",
        "treatment_event": "tx_event",
        "zip": "county_zip",
        "sex": "sex_raw",
        "stage": "stage_raw",
    }
    df = df.rename(columns=rename_map)

    df["os_months"] = df["os_days"] / 30.4375
    df["tx_start_months"] = df["tx_start_days"] / 30.4375
    df["tx_start_obs_months"] = df["tx_start_obs_days"] / 30.4375
    df["is_male"] = (df["sex_raw"] == "M").astype("int8")
    df["stage_two"] = (df["stage_raw"] == "II").astype("int8")
    df["stage_three"] = (df["stage_raw"] == "III").astype("int8")

    area_map = {z: i for i, z in enumerate(sorted(df["county_zip"].unique()))}
    df["county_area"] = df["county_zip"].map(area_map).astype("int16")

    return df


@pytest.mark.integration
def test_run_treatment_analysis_smoke_with_configured_schema(tmp_path: Path) -> None:
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
    wide = _rename_to_custom_schema(wide)

    input_path = tmp_path / "simulated_wide_custom_schema.csv"
    wide.to_csv(input_path, index=False)

    out_dir = tmp_path / "treatment_analysis_artifacts"

    cfg = TreatmentAnalysisConfig(
        input_path=str(input_path),
        out_dir=str(out_dir),
        graph_mode="from_area_id_ring",
        graph_A=6,
        graph_k=2,
        input_columns=InputColumnConfig(
            id_col="patient_id",
            time_days_col="os_days",
            event_col="os_event",
            treatment_time_days_col="tx_start_days",
            treatment_time_obs_days_col="tx_start_obs_days",
            treatment_event_col="tx_event",
            zip_col="county_zip",
            sex_col="sex_raw",
            stage_col="stage_raw",
        ),
        derived_columns=DerivedColumnConfig(
            time_m_col="os_months",
            treatment_time_m_col="tx_start_months",
            treatment_time_obs_m_col="tx_start_obs_months",
            area_id_col="county_area",
            sex_male_col="is_male",
            stage_ii_col="stage_two",
            stage_iii_col="stage_three",
        ),
        surv_breaks=tuple(DEFAULT_SURV_BREAKS),
        ttt_breaks=tuple(DEFAULT_TTT_BREAKS),
        post_ttt_breaks=tuple(DEFAULT_POST_TTT_BREAKS),
        ttt_x_cols=(
            "age_per10_centered",
            "cci",
            "tumor_size_log",
            "ses",
            "is_male",
            "stage_two",
            "stage_three",
        ),
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

    returned_out_dir = run_treatment_analysis(cfg)

    assert returned_out_dir == out_dir
    assert out_dir.exists()

    expected_files = [
        "surv_long.csv",
        "ttt_long.csv",
        "treatment_theta_summary.csv",
        "treatment_gamma_summary.csv",
        "treatment_spatial_field_summary.csv",
        "treatment_spatial_hyperparameter_summary.csv",
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
    theta_summary = pd.read_csv(out_dir / "treatment_theta_summary.csv")
    gamma_summary = pd.read_csv(out_dir / "treatment_gamma_summary.csv")
    spatial_field = pd.read_csv(out_dir / "treatment_spatial_field_summary.csv")
    spatial_hyper = pd.read_csv(out_dir / "treatment_spatial_hyperparameter_summary.csv")

    assert not surv_long.empty
    assert not ttt_long.empty
    assert not theta_summary.empty
    assert not gamma_summary.empty
    assert not spatial_field.empty
    assert not spatial_hyper.empty

    # Canonical internal long-format schema should be present.
    assert {"id", "k", "t0", "t1", "exposure", "event", "area_id"}.issubset(surv_long.columns)
    assert {"id", "k", "t0", "t1", "exposure", "event", "area_id"}.issubset(ttt_long.columns)

    # Configured treatment covariates should be preserved into long format.
    assert {
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "ses",
        "is_male",
        "stage_two",
        "stage_three",
    }.issubset(ttt_long.columns)

    assert {"parameter", "label", "mean"}.issubset(theta_summary.columns)
    assert {"parameter", "label", "mean"}.issubset(gamma_summary.columns)
    assert {"parameter", "field", "area_id", "mean"}.issubset(spatial_field.columns)
    assert {"parameter", "param_group", "mean"}.issubset(spatial_hyper.columns)

    analysis_cfg = pd.read_json(out_dir / "analysis_config.json", typ="series")
    run_manifest = pd.read_json(out_dir / "run_manifest.json", typ="series")

    assert "input_columns" in analysis_cfg.index
    assert "derived_columns" in analysis_cfg.index
    assert "surv_breaks" in analysis_cfg.index
    assert "ttt_breaks" in analysis_cfg.index
    assert "post_ttt_breaks" in analysis_cfg.index
    assert "ttt_x_cols" in analysis_cfg.index

    assert "input_columns" in run_manifest.index
    assert "derived_columns" in run_manifest.index
    assert "surv_breaks" in run_manifest.index
    assert "ttt_breaks" in run_manifest.index
    assert "post_ttt_breaks" in run_manifest.index
    assert "ttt_x_cols" in run_manifest.index