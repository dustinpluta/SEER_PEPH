from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from seer_peph.analysis.survival_analysis import (
    SurvivalAnalysisConfig,
    run_survival_analysis,
)
from seer_peph.fitting.io import load_survival_fit
from seer_peph.graphs import make_ring_lattice
from seer_peph.validation.simulate import simulate_joint


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
def test_run_survival_analysis_breaks_pass_through(tmp_path: Path) -> None:
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

    out_dir = tmp_path / "survival_analysis_breaks_artifacts"

    custom_surv_breaks = (0.0, 2.0, 4.0, 8.0, 16.0, 60.0)
    custom_ttt_breaks = (0.0, 1.0, 3.0, 6.0, 12.0, 60.0)
    custom_post_ttt_breaks = (0.0, 2.0, 6.0, 18.0, 60.0)

    cfg = SurvivalAnalysisConfig(
        input_path=str(input_path),
        out_dir=str(out_dir),
        graph_mode="from_area_id_ring",
        graph_A=6,
        graph_k=2,
        surv_breaks=custom_surv_breaks,
        ttt_breaks=custom_ttt_breaks,
        post_ttt_breaks=custom_post_ttt_breaks,
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
        prediction={
            # dataclass accepts SurvivalPredictionConfig, so keep defaults by not overriding here
        },  # type: ignore[arg-type]
    )
    # Replace the invalid dict with the default dataclass by reconstructing cleanly.
    cfg = SurvivalAnalysisConfig(
        input_path=str(input_path),
        out_dir=str(out_dir),
        graph_mode="from_area_id_ring",
        graph_A=6,
        graph_k=2,
        surv_breaks=custom_surv_breaks,
        ttt_breaks=custom_ttt_breaks,
        post_ttt_breaks=custom_post_ttt_breaks,
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

    # Manifest and analysis config should record the exact custom grids.
    with (out_dir / "run_manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    with (out_dir / "analysis_config.json").open("r", encoding="utf-8") as f:
        analysis_cfg = json.load(f)
    with (out_dir / "prediction_profile.json").open("r", encoding="utf-8") as f:
        pred_profile = json.load(f)

    assert tuple(manifest["surv_breaks"]) == custom_surv_breaks
    assert tuple(manifest["ttt_breaks"]) == custom_ttt_breaks
    assert tuple(manifest["post_ttt_breaks"]) == custom_post_ttt_breaks

    assert tuple(analysis_cfg["surv_breaks"]) == custom_surv_breaks
    assert tuple(analysis_cfg["ttt_breaks"]) == custom_ttt_breaks
    assert tuple(analysis_cfg["post_ttt_breaks"]) == custom_post_ttt_breaks

    assert tuple(pred_profile["surv_breaks"]) == custom_surv_breaks
    assert tuple(pred_profile["post_ttt_breaks"]) == custom_post_ttt_breaks

    # Reloaded fit bundle should carry the exact same metadata.
    fit = load_survival_fit(out_dir / "fit_bundle")
    assert fit.metadata.surv_breaks == custom_surv_breaks
    assert fit.metadata.ttt_breaks == custom_ttt_breaks
    assert fit.metadata.post_ttt_breaks == custom_post_ttt_breaks

    # Long-format outputs should reflect the custom grids by their max interval index.
    surv_long = pd.read_csv(out_dir / "surv_long.csv")
    ttt_long = pd.read_csv(out_dir / "ttt_long.csv")

    assert not surv_long.empty
    assert not ttt_long.empty

    # Number of intervals = len(breaks) - 1, so max k must be <= len(breaks)-2.
    assert int(surv_long["k"].max()) <= len(custom_surv_breaks) - 2
    assert int(ttt_long["k"].max()) <= len(custom_ttt_breaks) - 2

    # If treated rows exist, post-treatment interval index should respect custom post grid.
    if "k_post" in surv_long.columns:
        treated_rows = surv_long.loc[surv_long["k_post"] >= 0]
        if not treated_rows.empty:
            assert int(treated_rows["k_post"].max()) <= len(custom_post_ttt_breaks) - 2

    # Prediction artifacts should exist and be nonempty under the custom grid.
    surv_scenarios = pd.read_csv(out_dir / "predicted_survival_scenarios.csv")
    rmst_scenarios = pd.read_csv(out_dir / "predicted_rmst_scenarios.csv")
    surv_contrasts = pd.read_csv(out_dir / "predicted_survival_contrasts.csv")
    rmst_contrasts = pd.read_csv(out_dir / "predicted_rmst_contrasts.csv")

    assert not surv_scenarios.empty
    assert not rmst_scenarios.empty
    assert not surv_contrasts.empty
    assert not rmst_contrasts.empty

    assert surv_scenarios["mean_survival"].between(0.0, 1.0).all()
    assert (rmst_scenarios["mean_rmst"] >= 0.0).all()
    assert (rmst_scenarios["mean_rmst"] <= rmst_scenarios["horizon_m"]).all()