from __future__ import annotations

import numpy as np
import pytest

from seer_peph.data.prep import (
    POST_TTT_BREAKS,
    SURV_BREAKS,
    build_survival_long,
    build_treatment_long,
)
from seer_peph.fitting.fit_models import fit_survival_model
from seer_peph.fitting.io import load_survival_fit, save_survival_fit
from seer_peph.graphs import make_ring_lattice
from seer_peph.inference.run import InferenceConfig
from seer_peph.predict.survival import (
    predict_counterfactual_survival_draws,
    predict_counterfactual_survival_summary,
    predict_rmst,
    predict_survival_at_times,
)
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
def small_survival_fit(tmp_path_factory):
    graph = make_ring_lattice(A=6, k=2)

    surv_breaks = list(SURV_BREAKS)
    post_ttt_breaks = list(POST_TTT_BREAKS)

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

    fit = fit_survival_model(
        surv_long=surv_long,
        ttt_long=ttt_long,
        graph=graph,
        surv_x_cols=[
            "age_per10_centered",
            "cci",
            "tumor_size_log",
            "stage_II",
            "stage_III",
        ],
        surv_breaks=surv_breaks,
        post_ttt_breaks=post_ttt_breaks,
        rng_seed=101,
        inference_config=infer_cfg,
        extra_fields=("diverging",),
    )

    save_dir = tmp_path_factory.mktemp("survival_fit_bundle") / "fit"
    save_survival_fit(fit, save_dir)
    reloaded = load_survival_fit(save_dir)

    return {
        "fit": fit,
        "reloaded_fit": reloaded,
        "df": df,
        "graph": graph,
        "surv_breaks": surv_breaks,
        "post_ttt_breaks": post_ttt_breaks,
    }


def _prediction_inputs(df, surv_breaks, post_ttt_breaks):
    first = df.iloc[0]
    x = [
        float(first["age_per10_centered"]),
        float(first["cci"]),
        float(first["tumor_size_log"]),
        float(first["stage_II"]),
        float(first["stage_III"]),
    ]
    area_id = int(first["area_id"])
    eval_times = [0.0, 6.0, 12.0, 24.0, 36.0, 60.0]
    return x, area_id, surv_breaks, post_ttt_breaks, eval_times


@pytest.mark.integration
def test_predict_counterfactual_survival_draws_smoke(small_survival_fit):
    fit = small_survival_fit["fit"]
    x, area_id, surv_breaks, post_breaks, eval_times = _prediction_inputs(
        small_survival_fit["df"],
        small_survival_fit["surv_breaks"],
        small_survival_fit["post_ttt_breaks"],
    )

    out = predict_counterfactual_survival_draws(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_breaks,
        eval_times=eval_times,
        treatment_time_m=None,
    )

    assert not out.empty
    assert set(out.columns) == {"draw", "time_m", "survival", "treatment_time_m", "area_id"}
    assert out["draw"].nunique() > 0
    assert out["time_m"].nunique() == len(eval_times)
    assert out["survival"].between(0.0, 1.0).all()
    assert out.loc[out["time_m"] == 0.0, "survival"].eq(1.0).all()


@pytest.mark.integration
def test_predict_counterfactual_survival_summary_smoke(small_survival_fit):
    fit = small_survival_fit["fit"]
    x, area_id, surv_breaks, post_breaks, eval_times = _prediction_inputs(
        small_survival_fit["df"],
        small_survival_fit["surv_breaks"],
        small_survival_fit["post_ttt_breaks"],
    )

    out = predict_counterfactual_survival_summary(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_breaks,
        eval_times=eval_times,
        treatment_time_m=3.0,
    )

    assert not out.empty
    assert set(out.columns) == {
        "time_m",
        "treatment_time_m",
        "area_id",
        "mean_survival",
        "median_survival",
        "q05_survival",
        "q95_survival",
    }
    assert out["time_m"].tolist() == eval_times
    assert out["area_id"].eq(area_id).all()
    assert out["treatment_time_m"].eq(3.0).all()
    assert out["mean_survival"].between(0.0, 1.0).all()


@pytest.mark.integration
def test_predict_survival_at_times_smoke(small_survival_fit):
    fit = small_survival_fit["fit"]
    x, area_id, surv_breaks, post_breaks, _ = _prediction_inputs(
        small_survival_fit["df"],
        small_survival_fit["surv_breaks"],
        small_survival_fit["post_ttt_breaks"],
    )

    out = predict_survival_at_times(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_breaks,
        times=[12.0, 24.0, 36.0, 60.0],
        treatment_times_m=[None, 3.0, 6.0],
    )

    assert not out.empty
    assert out["time_m"].nunique() == 4
    assert out["area_id"].eq(area_id).all()
    assert out["treatment_time_m"].isna().sum() > 0
    assert (out["treatment_time_m"] == 3.0).sum() > 0
    assert (out["treatment_time_m"] == 6.0).sum() > 0
    assert out["mean_survival"].between(0.0, 1.0).all()


@pytest.mark.integration
def test_predict_rmst_smoke(small_survival_fit):
    fit = small_survival_fit["fit"]
    x, area_id, surv_breaks, post_breaks, _ = _prediction_inputs(
        small_survival_fit["df"],
        small_survival_fit["surv_breaks"],
        small_survival_fit["post_ttt_breaks"],
    )

    out = predict_rmst(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_breaks,
        horizon_m=60.0,
        treatment_times_m=[None, 3.0, 6.0],
        grid_size=200,
    )

    assert not out.empty
    assert set(out.columns) == {
        "treatment_time_m",
        "mean_rmst",
        "median_rmst",
        "q05_rmst",
        "q95_rmst",
        "horizon_m",
        "area_id",
    }
    assert out.shape[0] == 3
    assert out["area_id"].eq(area_id).all()
    assert out["horizon_m"].eq(60.0).all()
    assert (out["mean_rmst"] >= 0.0).all()
    assert (out["mean_rmst"] <= 60.0).all()


@pytest.mark.integration
def test_prediction_on_reloaded_fit_smoke(small_survival_fit):
    fit = small_survival_fit["reloaded_fit"]
    x, area_id, surv_breaks, post_breaks, eval_times = _prediction_inputs(
        small_survival_fit["df"],
        small_survival_fit["surv_breaks"],
        small_survival_fit["post_ttt_breaks"],
    )

    surv_out = predict_counterfactual_survival_summary(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_breaks,
        eval_times=eval_times,
        treatment_time_m=3.0,
    )
    rmst_out = predict_rmst(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=surv_breaks,
        post_treatment_breaks=post_breaks,
        horizon_m=60.0,
        treatment_times_m=[None, 3.0],
        grid_size=150,
    )

    assert not surv_out.empty
    assert not rmst_out.empty
    assert surv_out["mean_survival"].between(0.0, 1.0).all()
    assert (rmst_out["mean_rmst"] >= 0.0).all()
    assert (rmst_out["mean_rmst"] <= 60.0).all()