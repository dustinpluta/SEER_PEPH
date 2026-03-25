from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from seer_peph.data.prep import (
    DEFAULT_POST_TTT_BREAKS,
    DEFAULT_SURV_BREAKS,
    DEFAULT_TTT_BREAKS,
    build_survival_long,
    build_treatment_long,
)
from seer_peph.fitting.extract import extract_spatial_fields, extract_survival_effects
from seer_peph.fitting.fit_models import fit_survival_model
from seer_peph.fitting.io import save_survival_fit
from seer_peph.graphs import make_ring_lattice
from seer_peph.inference.run import InferenceConfig
from seer_peph.predict.survival import predict_rmst, predict_survival_at_times
from seer_peph.predict.survival_contrasts import (
    predict_rmst_contrast_summary,
    predict_survival_contrast_summary,
)


@dataclass(frozen=True)
class SurvivalPredictionConfig:
    eval_times_m: tuple[float, ...] = (12.0, 24.0, 36.0, 60.0)
    horizon_m: float = 60.0
    grid_size: int = 400
    treatment_times_m: tuple[float | None, ...] = (None, 1.0, 3.0, 6.0)
    contrast_pairs_m: tuple[tuple[float | None, float | None], ...] = (
        (1.0, 3.0),
        (3.0, 6.0),
        (1.0, 6.0),
    )


@dataclass(frozen=True)
class SurvivalAnalysisConfig:
    input_path: str
    out_dir: str = "artifacts/standalone_survival_analysis"

    # Graph settings
    graph_mode: str = "from_area_id_ring"
    graph_A: int | None = None
    graph_k: int = 4

    # Interval grids
    surv_breaks: tuple[float, ...] = tuple(DEFAULT_SURV_BREAKS)
    ttt_breaks: tuple[float, ...] = tuple(DEFAULT_TTT_BREAKS)
    post_ttt_breaks: tuple[float, ...] = tuple(DEFAULT_POST_TTT_BREAKS)

    # Survival covariates
    surv_x_cols: tuple[str, ...] = (
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "stage_II",
        "stage_III",
    )

    # Inference
    rng_seed: int = 123
    inference: dict[str, Any] = field(
        default_factory=lambda: {
            "num_chains": 2,
            "num_warmup": 500,
            "num_samples": 500,
            "target_accept_prob": 0.95,
            "dense_mass": False,
            "max_tree_depth": 10,
            "progress_bar": True,
        }
    )

    # Prediction
    prediction: SurvivalPredictionConfig = field(default_factory=SurvivalPredictionConfig)

    # Optional profile selection for prediction
    prediction_profile: str = "first_row"
    prediction_area_id: int | None = None


def run_survival_analysis(config: SurvivalAnalysisConfig) -> Path:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wide = _load_wide_data(config.input_path)
    wide_encoded = _encode_like_prep(wide)

    graph = _build_graph(config, wide_encoded)
    surv_long = build_survival_long(
        wide_encoded,
        surv_breaks=config.surv_breaks,
        post_ttt_breaks=config.post_ttt_breaks,
    )
    ttt_long = build_treatment_long(
        wide_encoded,
        ttt_breaks=config.ttt_breaks,
    )

    surv_long.to_csv(out_dir / "surv_long.csv", index=False)
    ttt_long.to_csv(out_dir / "ttt_long.csv", index=False)

    infer_cfg = InferenceConfig(**config.inference)

    fit = fit_survival_model(
        surv_long=surv_long,
        ttt_long=ttt_long,
        graph=graph,
        surv_x_cols=list(config.surv_x_cols),
        surv_breaks=list(config.surv_breaks),
        ttt_breaks=list(config.ttt_breaks),
        post_ttt_breaks=list(config.post_ttt_breaks),
        rng_seed=config.rng_seed,
        inference_config=infer_cfg,
        extra_fields=("diverging",),
        extra_metadata={
            "input_path": config.input_path,
            "graph_mode": config.graph_mode,
        },
    )

    fit_dir = out_dir / "fit_bundle"
    save_survival_fit(fit, fit_dir)

    _write_survival_extractions(fit, out_dir)
    _write_prediction_artifacts(fit, wide_encoded, out_dir, config)

    _write_json(out_dir / "analysis_config.json", _json_ready(asdict(config)))
    _write_json(
        out_dir / "run_manifest.json",
        {
            "model_name": fit.model_name,
            "fit_dir": str(fit_dir),
            "n_subjects": int(len(wide_encoded)),
            "n_surv_rows": int(len(surv_long)),
            "n_ttt_rows": int(len(ttt_long)),
            "surv_x_cols": list(config.surv_x_cols),
            "surv_breaks": list(config.surv_breaks),
            "ttt_breaks": list(config.ttt_breaks),
            "post_ttt_breaks": list(config.post_ttt_breaks),
            "rng_seed": config.rng_seed,
        },
    )

    return out_dir


def _write_survival_extractions(fit, out_dir: Path) -> None:
    surv = extract_survival_effects(fit, include_draws=False)
    spatial = extract_spatial_fields(fit, include_draws=False)

    if "beta" in surv:
        surv["beta"].to_csv(out_dir / "survival_beta_summary.csv", index=False)
    if "alpha" in surv:
        surv["alpha"].to_csv(out_dir / "survival_alpha_summary.csv", index=False)
    if "delta_post" in surv:
        surv["delta_post"].to_csv(out_dir / "survival_delta_post_summary.csv", index=False)

    if "field" in spatial:
        spatial["field"].to_csv(out_dir / "survival_spatial_field_summary.csv", index=False)
    if "hyperparameters" in spatial:
        spatial["hyperparameters"].to_csv(
            out_dir / "survival_spatial_hyperparameter_summary.csv",
            index=False,
        )


def _write_prediction_artifacts(
    fit,
    wide_encoded: pd.DataFrame,
    out_dir: Path,
    config: SurvivalAnalysisConfig,
) -> None:
    x, area_id = _select_prediction_profile(
        wide_encoded=wide_encoded,
        surv_x_cols=config.surv_x_cols,
        prediction_profile=config.prediction_profile,
        prediction_area_id=config.prediction_area_id,
    )

    eval_times = list(config.prediction.eval_times_m)
    treatment_times = list(config.prediction.treatment_times_m)

    scenario_survival = predict_survival_at_times(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=list(config.surv_breaks),
        post_treatment_breaks=list(config.post_ttt_breaks),
        times=eval_times,
        treatment_times_m=treatment_times,
    )
    scenario_survival.to_csv(out_dir / "predicted_survival_scenarios.csv", index=False)

    scenario_rmst = predict_rmst(
        fit,
        x=x,
        area_id=area_id,
        surv_breaks=list(config.surv_breaks),
        post_treatment_breaks=list(config.post_ttt_breaks),
        horizon_m=float(config.prediction.horizon_m),
        treatment_times_m=treatment_times,
        grid_size=int(config.prediction.grid_size),
    )
    scenario_rmst.to_csv(out_dir / "predicted_rmst_scenarios.csv", index=False)

    contrast_survival_parts: list[pd.DataFrame] = []
    contrast_rmst_parts: list[pd.DataFrame] = []

    for t_a, t_b in config.prediction.contrast_pairs_m:
        surv_contrast = predict_survival_contrast_summary(
            fit,
            x=x,
            area_id=area_id,
            surv_breaks=list(config.surv_breaks),
            post_treatment_breaks=list(config.post_ttt_breaks),
            eval_times=eval_times,
            treatment_time_m_a=t_a,
            treatment_time_m_b=t_b,
        )
        contrast_survival_parts.append(surv_contrast)

        rmst_contrast = predict_rmst_contrast_summary(
            fit,
            x=x,
            area_id=area_id,
            surv_breaks=list(config.surv_breaks),
            post_treatment_breaks=list(config.post_ttt_breaks),
            horizon_m=float(config.prediction.horizon_m),
            treatment_time_m_a=t_a,
            treatment_time_m_b=t_b,
            grid_size=int(config.prediction.grid_size),
        )
        contrast_rmst_parts.append(rmst_contrast)

    if contrast_survival_parts:
        pd.concat(contrast_survival_parts, ignore_index=True).to_csv(
            out_dir / "predicted_survival_contrasts.csv",
            index=False,
        )

    if contrast_rmst_parts:
        pd.concat(contrast_rmst_parts, ignore_index=True).to_csv(
            out_dir / "predicted_rmst_contrasts.csv",
            index=False,
        )

    _write_json(
        out_dir / "prediction_profile.json",
        {
            "area_id": int(area_id),
            "surv_x_cols": list(config.surv_x_cols),
            "x": [float(v) for v in x],
            "eval_times_m": eval_times,
            "horizon_m": float(config.prediction.horizon_m),
            "treatment_times_m": [_json_ready(v) for v in treatment_times],
            "contrast_pairs_m": [
                [_json_ready(a), _json_ready(b)]
                for a, b in config.prediction.contrast_pairs_m
            ],
            "surv_breaks": list(config.surv_breaks),
            "post_ttt_breaks": list(config.post_ttt_breaks),
        },
    )


def _select_prediction_profile(
    *,
    wide_encoded: pd.DataFrame,
    surv_x_cols: Sequence[str],
    prediction_profile: str,
    prediction_area_id: int | None,
) -> tuple[list[float], int]:
    if prediction_profile == "first_row":
        row = wide_encoded.iloc[0]
    elif prediction_profile == "mean_profile":
        numeric = wide_encoded[list(surv_x_cols) + ["area_id"]].copy()
        row = numeric.mean(numeric_only=True)
    else:
        raise ValueError(f"Unsupported prediction_profile: {prediction_profile}")

    x = [float(row[col]) for col in surv_x_cols]

    if prediction_area_id is not None:
        area_id = int(prediction_area_id)
    else:
        area_id = int(row["area_id"])

    return x, area_id


def _build_graph(config: SurvivalAnalysisConfig, wide_encoded: pd.DataFrame):
    if config.graph_mode != "from_area_id_ring":
        raise ValueError(f"Unsupported graph_mode: {config.graph_mode}")

    if config.graph_A is not None:
        A = int(config.graph_A)
    else:
        if "area_id" not in wide_encoded.columns:
            raise ValueError("area_id column is required to build the graph.")
        A = int(wide_encoded["area_id"].nunique())

    return make_ring_lattice(A=A, k=int(config.graph_k))


def _load_wide_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)

    raise ValueError("Input path must be a .csv or .parquet file.")


def _encode_like_prep(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()

    if "time_m" not in df.columns and "time" in df.columns:
        df["time_m"] = df["time"] / 30.4375

    if "treatment_time_m" not in df.columns and "treatment_time" in df.columns:
        df["treatment_time_m"] = df["treatment_time"] / 30.4375

    if "treatment_time_obs_m" not in df.columns and "treatment_time_obs" in df.columns:
        df["treatment_time_obs_m"] = df["treatment_time_obs"] / 30.4375

    if "sex_male" not in df.columns and "sex" in df.columns:
        df["sex_male"] = (df["sex"] == "M").astype("int8")

    if "stage_II" not in df.columns and "stage" in df.columns:
        df["stage_II"] = (df["stage"] == "II").astype("int8")

    if "stage_III" not in df.columns and "stage" in df.columns:
        df["stage_III"] = (df["stage"] == "III").astype("int8")

    if "area_id" not in df.columns:
        if "zip" in df.columns:
            area_map = {z: i for i, z in enumerate(sorted(df["zip"].unique()))}
            df["area_id"] = df["zip"].map(area_map).astype("int16")
        else:
            raise ValueError("Input data must contain either area_id or zip.")

    required = [
        "id",
        "time_m",
        "event",
        "treatment_time_m",
        "treatment_time_obs_m",
        "treatment_event",
        "area_id",
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "stage_II",
        "stage_III",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")

    return df


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value