from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

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
def test_run_treatment_analysis_cli_smoke_nondefault_schema(tmp_path: Path) -> None:
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

    out_dir = tmp_path / "treatment_analysis_cli_artifacts"
    config_path = tmp_path / "treatment_analysis_config.json"

    config = {
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "graph_mode": "from_area_id_ring",
        "graph_A": 6,
        "graph_k": 2,
        "input_columns": {
            "id_col": "patient_id",
            "time_days_col": "os_days",
            "event_col": "os_event",
            "treatment_time_days_col": "tx_start_days",
            "treatment_time_obs_days_col": "tx_start_obs_days",
            "treatment_event_col": "tx_event",
            "zip_col": "county_zip",
            "sex_col": "sex_raw",
            "stage_col": "stage_raw"
        },
        "derived_columns": {
            "time_m_col": "os_months",
            "treatment_time_m_col": "tx_start_months",
            "treatment_time_obs_m_col": "tx_start_obs_months",
            "area_id_col": "county_area",
            "sex_male_col": "is_male",
            "stage_ii_col": "stage_two",
            "stage_iii_col": "stage_three"
        },
        "ttt_x_cols": [
            "age_per10_centered",
            "cci",
            "tumor_size_log",
            "ses",
            "is_male",
            "stage_two",
            "stage_three"
        ],
        "surv_breaks": [0.0, 2.0, 4.0, 8.0, 16.0, 60.0],
        "ttt_breaks": [0.0, 1.0, 3.0, 6.0, 12.0, 60.0],
        "post_ttt_breaks": [0.0, 2.0, 6.0, 18.0, 60.0],
        "rng_seed": 101,
        "inference": {
            "num_chains": 1,
            "num_warmup": 25,
            "num_samples": 25,
            "target_accept_prob": 0.9,
            "dense_mass": False,
            "max_tree_depth": 6,
            "progress_bar": False
        },
        "ppc": {
            "enabled": True,
            "draw_indices": None,
            "sample_posterior_predictive": True,
            "random_seed": 123
        }
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_treatment_analysis.py"
    assert script_path.exists(), f"CLI script not found: {script_path}"

    result = subprocess.run(
        [sys.executable, str(script_path), str(config_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise AssertionError(
            "CLI run failed.\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    assert "Treatment analysis completed." in result.stdout
    assert out_dir.exists()

    expected_files = [
        "surv_long.csv",
        "ttt_long.csv",
        "treatment_theta_summary.csv",
        "treatment_gamma_summary.csv",
        "treatment_spatial_field_summary.csv",
        "treatment_spatial_hyperparameter_summary.csv",
        "treatment_ppc_interval_counts.csv",
        "treatment_ppc_area_counts.csv",
        "analysis_config.json",
        "run_manifest.json",
    ]
    for name in expected_files:
        path = out_dir / name
        assert path.exists(), f"Missing expected artifact: {name}"
        assert path.stat().st_size > 0, f"Artifact is empty: {name}"

    fit_dir = out_dir / "fit_bundle"
    assert fit_dir.exists()
    for name in ["manifest.json", "samples.npz", "data_arrays.npz", "summary.json", "scalar_summary.json"]:
        path = fit_dir / name
        assert path.exists(), f"Missing fit bundle artifact: {name}"
        assert path.stat().st_size > 0, f"Fit bundle artifact is empty: {name}"

    surv_long = pd.read_csv(out_dir / "surv_long.csv")
    ttt_long = pd.read_csv(out_dir / "ttt_long.csv")
    theta_summary = pd.read_csv(out_dir / "treatment_theta_summary.csv")
    gamma_summary = pd.read_csv(out_dir / "treatment_gamma_summary.csv")
    ppc_interval = pd.read_csv(out_dir / "treatment_ppc_interval_counts.csv")
    ppc_area = pd.read_csv(out_dir / "treatment_ppc_area_counts.csv")

    assert not surv_long.empty
    assert not ttt_long.empty
    assert not theta_summary.empty
    assert not gamma_summary.empty
    assert not ppc_interval.empty
    assert not ppc_area.empty

    input_df = pd.read_csv(input_path)

    assert {
        "patient_id",
        "os_days",
        "os_event",
        "tx_start_days",
        "tx_start_obs_days",
        "tx_event",
        "county_zip",
        "sex_raw",
        "stage_raw",
        "os_months",
        "tx_start_months",
        "tx_start_obs_months",
        "county_area",
    }.issubset(input_df.columns)

    assert {"id", "k", "t0", "t1", "exposure", "event", "area_id"}.issubset(surv_long.columns)
    assert {"id", "k", "t0", "t1", "exposure", "event", "area_id"}.issubset(ttt_long.columns)

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

    assert {
        "k",
        "observed_events",
        "observed_exposure",
        "pp_mean_events",
        "pp_q05_events",
        "pp_q95_events",
        "observed_rate",
        "pp_mean_rate",
    }.issubset(ppc_interval.columns)

    assert {
        "area_id",
        "observed_events",
        "observed_exposure",
        "pp_mean_events",
        "pp_q05_events",
        "pp_q95_events",
        "observed_rate",
        "pp_mean_rate",
    }.issubset(ppc_area.columns)

    with (out_dir / "analysis_config.json").open("r", encoding="utf-8") as f:
        analysis_cfg = json.load(f)
    with (out_dir / "run_manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert analysis_cfg["input_columns"]["id_col"] == "patient_id"
    assert analysis_cfg["derived_columns"]["area_id_col"] == "county_area"
    assert tuple(analysis_cfg["ttt_breaks"]) == (0.0, 1.0, 3.0, 6.0, 12.0, 60.0)
    assert analysis_cfg["ppc"]["enabled"] is True

    assert tuple(manifest["ttt_x_cols"]) == (
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "ses",
        "is_male",
        "stage_two",
        "stage_three",
    )
    assert manifest["ppc_enabled"] is True