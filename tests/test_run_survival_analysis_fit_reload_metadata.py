from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from seer_peph.analysis.survival_analysis import (
    DerivedColumnConfig,
    InputColumnConfig,
    SurvivalAnalysisConfig,
    run_survival_analysis,
)
from seer_peph.fitting.io import load_survival_fit
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
def test_run_survival_analysis_fit_reload_metadata_propagation(tmp_path: Path) -> None:
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

    out_dir = tmp_path / "survival_analysis_reload_metadata_artifacts"

    custom_surv_breaks = (0.0, 2.0, 4.0, 8.0, 16.0, 60.0)
    custom_ttt_breaks = (0.0, 1.0, 3.0, 6.0, 12.0, 60.0)
    custom_post_ttt_breaks = (0.0, 2.0, 6.0, 18.0, 60.0)
    custom_surv_x_cols = (
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "stage_two",
        "stage_three",
    )

    cfg = SurvivalAnalysisConfig(
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
        surv_breaks=custom_surv_breaks,
        ttt_breaks=custom_ttt_breaks,
        post_ttt_breaks=custom_post_ttt_breaks,
        surv_x_cols=custom_surv_x_cols,
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

    fit = load_survival_fit(out_dir / "fit_bundle")

    # Core fit metadata should match the config exactly.
    assert fit.metadata.surv_breaks == custom_surv_breaks
    assert fit.metadata.ttt_breaks == custom_ttt_breaks
    assert fit.metadata.post_ttt_breaks == custom_post_ttt_breaks
    assert fit.metadata.surv_x_cols == custom_surv_x_cols

    # Basic dimensions should be populated.
    assert fit.metadata.graph_A == 6
    assert fit.metadata.graph_n_edges is not None
    assert fit.metadata.graph_n_edges > 0
    assert fit.metadata.n_surv is not None
    assert fit.metadata.n_surv > 0
    assert fit.metadata.n_ttt is not None
    assert fit.metadata.n_ttt > 0
    assert fit.metadata.p_surv == len(custom_surv_x_cols)
    assert fit.metadata.rng_seed == 101

    # Saved data bundle should also reflect the propagated metadata.
    assert tuple(fit.data["surv_x_cols"]) == custom_surv_x_cols
    assert int(fit.data["P_surv"]) == len(custom_surv_x_cols)
    assert int(fit.data["A"]) == 6

    # Inference config should round-trip too.
    assert fit.inference_result.config.num_chains == 1
    assert fit.inference_result.config.num_warmup == 25
    assert fit.inference_result.config.num_samples == 25
    assert fit.inference_result.config.target_accept_prob == 0.9
    assert fit.inference_result.config.dense_mass is False
    assert fit.inference_result.config.max_tree_depth == 6
    assert fit.inference_result.config.progress_bar is False

    # The live MCMC object is not expected to round-trip.
    assert fit.inference_result.mcmc is None

    # Sanity check that posterior samples exist after reload.
    assert "alpha" in fit.samples
    assert "beta" in fit.samples
    assert "delta_post" in fit.samples
    assert "u" in fit.samples

    # Summary objects should be present and nonempty.
    assert fit.summary
    assert fit.scalar_summary

    # Runner-level config artifacts should agree too.
    analysis_cfg = pd.read_json(out_dir / "analysis_config.json", typ="series")
    run_manifest = pd.read_json(out_dir / "run_manifest.json", typ="series")

    assert tuple(analysis_cfg["surv_breaks"]) == custom_surv_breaks
    assert tuple(analysis_cfg["ttt_breaks"]) == custom_ttt_breaks
    assert tuple(analysis_cfg["post_ttt_breaks"]) == custom_post_ttt_breaks
    assert tuple(analysis_cfg["surv_x_cols"]) == custom_surv_x_cols

    assert tuple(run_manifest["surv_breaks"]) == custom_surv_breaks
    assert tuple(run_manifest["ttt_breaks"]) == custom_ttt_breaks
    assert tuple(run_manifest["post_ttt_breaks"]) == custom_post_ttt_breaks
    assert tuple(run_manifest["surv_x_cols"]) == custom_surv_x_cols