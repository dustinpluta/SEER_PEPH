from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from seer_peph.data.prep import (
    DAYS_PER_MONTH,
    DEFAULT_POST_TTT_BREAKS,
    DEFAULT_SURV_BREAKS,
    DEFAULT_TTT_BREAKS,
    build_survival_long,
    build_treatment_long,
)
from seer_peph.fitting.extract import (
    extract_joint_coupling,
    extract_spatial_fields,
    extract_survival_effects,
    extract_treatment_effects,
)
from seer_peph.fitting.fit_models import fit_joint_model
from seer_peph.fitting.io import save_joint_fit
from seer_peph.graphs import make_ring_lattice
from seer_peph.inference.run import InferenceConfig


@dataclass(frozen=True)
class InputColumnConfig:
    id_col: str = "id"
    time_days_col: str = "time"
    event_col: str = "event"
    treatment_time_days_col: str = "treatment_time"
    treatment_time_obs_days_col: str = "treatment_time_obs"
    treatment_event_col: str = "treatment_event"
    zip_col: str = "zip"
    sex_col: str = "sex"
    stage_col: str = "stage"


@dataclass(frozen=True)
class DerivedColumnConfig:
    time_m_col: str = "time_m"
    treatment_time_m_col: str = "treatment_time_m"
    treatment_time_obs_m_col: str = "treatment_time_obs_m"
    area_id_col: str = "area_id"
    sex_male_col: str = "sex_male"
    stage_ii_col: str = "stage_II"
    stage_iii_col: str = "stage_III"


@dataclass(frozen=True)
class JointAnalysisConfig:
    input_path: str
    out_dir: str = "artifacts/joint_analysis"

    # Graph settings
    graph_mode: str = "from_area_id_ring"
    graph_A: int | None = None
    graph_k: int = 4

    # Column schema
    input_columns: InputColumnConfig = field(default_factory=InputColumnConfig)
    derived_columns: DerivedColumnConfig = field(default_factory=DerivedColumnConfig)

    # Interval grids
    surv_breaks: tuple[float, ...] = tuple(DEFAULT_SURV_BREAKS)
    ttt_breaks: tuple[float, ...] = tuple(DEFAULT_TTT_BREAKS)
    post_ttt_breaks: tuple[float, ...] = tuple(DEFAULT_POST_TTT_BREAKS)

    # Model covariates (post-encoding names, before canonicalization)
    surv_x_cols: tuple[str, ...] = (
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "stage_II",
        "stage_III",
    )
    ttt_x_cols: tuple[str, ...] = (
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "ses",
        "sex_male",
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


def run_joint_analysis(config: JointAnalysisConfig) -> Path:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wide = _load_wide_data(config.input_path)
    wide_encoded = _encode_like_prep(
        wide,
        input_cols=config.input_columns,
        derived_cols=config.derived_columns,
        required_covariates=tuple(sorted(set(config.surv_x_cols) | set(config.ttt_x_cols))),
    )

    graph = _build_graph(config, wide_encoded)

    surv_long = build_survival_long(
        wide_encoded,
        x_cols=list(config.surv_x_cols),
        surv_breaks=config.surv_breaks,
        post_ttt_breaks=config.post_ttt_breaks,
    )
    ttt_long = build_treatment_long(
        wide_encoded,
        x_cols=list(config.ttt_x_cols),
        ttt_breaks=config.ttt_breaks,
    )

    surv_long.to_csv(out_dir / "surv_long.csv", index=False)
    ttt_long.to_csv(out_dir / "ttt_long.csv", index=False)

    infer_cfg = InferenceConfig(**config.inference)

    fit = fit_joint_model(
        surv_long=surv_long,
        ttt_long=ttt_long,
        graph=graph,
        surv_x_cols=list(config.surv_x_cols),
        ttt_x_cols=list(config.ttt_x_cols),
        surv_breaks=list(config.surv_breaks),
        ttt_breaks=list(config.ttt_breaks),
        post_ttt_breaks=list(config.post_ttt_breaks),
        rng_seed=config.rng_seed,
        inference_config=infer_cfg,
        extra_fields=("diverging",),
        extra_metadata={
            "input_path": config.input_path,
            "graph_mode": config.graph_mode,
            "input_columns": asdict(config.input_columns),
            "derived_columns": asdict(config.derived_columns),
        },
    )

    fit_dir = out_dir / "fit_bundle"
    save_joint_fit(fit, fit_dir)

    _write_joint_extractions(fit, out_dir)

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
            "ttt_x_cols": list(config.ttt_x_cols),
            "surv_breaks": list(config.surv_breaks),
            "ttt_breaks": list(config.ttt_breaks),
            "post_ttt_breaks": list(config.post_ttt_breaks),
            "input_columns": asdict(config.input_columns),
            "derived_columns": asdict(config.derived_columns),
            "rng_seed": config.rng_seed,
        },
    )

    return out_dir


def _write_joint_extractions(fit, out_dir: Path) -> None:
    surv = extract_survival_effects(fit, include_draws=False)
    ttt = extract_treatment_effects(fit, include_draws=False)
    spatial = extract_spatial_fields(fit, include_draws=False)
    coupling = extract_joint_coupling(fit, include_draws=False)

    if "beta" in surv:
        surv["beta"].to_csv(out_dir / "joint_survival_beta_summary.csv", index=False)
    if "alpha" in surv:
        surv["alpha"].to_csv(out_dir / "joint_survival_alpha_summary.csv", index=False)
    if "delta_post" in surv:
        surv["delta_post"].to_csv(out_dir / "joint_survival_delta_post_summary.csv", index=False)

    if "theta" in ttt:
        ttt["theta"].to_csv(out_dir / "joint_treatment_theta_summary.csv", index=False)
    if "gamma" in ttt:
        ttt["gamma"].to_csv(out_dir / "joint_treatment_gamma_summary.csv", index=False)

    for name in ["u_surv", "u_ttt", "u_ttt_ind", "s_surv", "s_ttt"]:
        if name in spatial:
            spatial[name].to_csv(out_dir / f"{name}_summary.csv", index=False)

    if "hyperparameters" in spatial:
        spatial["hyperparameters"].to_csv(
            out_dir / "joint_spatial_hyperparameter_summary.csv",
            index=False,
        )

    if "coupling" in coupling:
        coupling["coupling"].to_csv(out_dir / "joint_coupling_summary.csv", index=False)
    if "field_correlations" in coupling:
        coupling["field_correlations"].to_csv(
            out_dir / "joint_field_correlations_summary.csv",
            index=False,
        )


def _build_graph(config: JointAnalysisConfig, wide_encoded: pd.DataFrame):
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


def _encode_like_prep(
    wide: pd.DataFrame,
    *,
    input_cols: InputColumnConfig,
    derived_cols: DerivedColumnConfig,
    required_covariates: Sequence[str],
) -> pd.DataFrame:
    """
    Convert configurable raw-schema wide data into the canonical internal schema
    expected by prep/model code.

    External schema is controlled by:
      - input_cols
      - derived_cols

    Internal canonical columns after this function:
      - id
      - time_m
      - event
      - treatment_time_m
      - treatment_time_obs_m
      - treatment_event
      - area_id

    plus any modeled covariates named in required_covariates.
    """
    df = wide.copy()

    if derived_cols.time_m_col not in df.columns and input_cols.time_days_col in df.columns:
        df[derived_cols.time_m_col] = df[input_cols.time_days_col] / DAYS_PER_MONTH

    if (
        derived_cols.treatment_time_m_col not in df.columns
        and input_cols.treatment_time_days_col in df.columns
    ):
        df[derived_cols.treatment_time_m_col] = (
            df[input_cols.treatment_time_days_col] / DAYS_PER_MONTH
        )

    if (
        derived_cols.treatment_time_obs_m_col not in df.columns
        and input_cols.treatment_time_obs_days_col in df.columns
    ):
        df[derived_cols.treatment_time_obs_m_col] = (
            df[input_cols.treatment_time_obs_days_col] / DAYS_PER_MONTH
        )

    if derived_cols.sex_male_col not in df.columns and input_cols.sex_col in df.columns:
        df[derived_cols.sex_male_col] = (df[input_cols.sex_col] == "M").astype("int8")

    if derived_cols.stage_ii_col not in df.columns and input_cols.stage_col in df.columns:
        df[derived_cols.stage_ii_col] = (df[input_cols.stage_col] == "II").astype("int8")

    if derived_cols.stage_iii_col not in df.columns and input_cols.stage_col in df.columns:
        df[derived_cols.stage_iii_col] = (df[input_cols.stage_col] == "III").astype("int8")

    if derived_cols.area_id_col not in df.columns:
        if input_cols.zip_col in df.columns:
            area_map = {z: i for i, z in enumerate(sorted(df[input_cols.zip_col].unique()))}
            df[derived_cols.area_id_col] = df[input_cols.zip_col].map(area_map).astype("int16")
        else:
            raise ValueError(
                f"Input data must contain either {derived_cols.area_id_col} "
                f"or {input_cols.zip_col}."
            )

    required_before_rename = [
        input_cols.id_col,
        derived_cols.time_m_col,
        input_cols.event_col,
        derived_cols.treatment_time_m_col,
        derived_cols.treatment_time_obs_m_col,
        input_cols.treatment_event_col,
        derived_cols.area_id_col,
        *list(required_covariates),
    ]
    missing = [c for c in required_before_rename if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required columns after encoding: {missing}")

    rename_map = {
        input_cols.id_col: "id",
        derived_cols.time_m_col: "time_m",
        input_cols.event_col: "event",
        derived_cols.treatment_time_m_col: "treatment_time_m",
        derived_cols.treatment_time_obs_m_col: "treatment_time_obs_m",
        input_cols.treatment_event_col: "treatment_event",
        derived_cols.area_id_col: "area_id",
    }
    df = df.rename(columns=rename_map)

    required_after_rename = [
        "id",
        "time_m",
        "event",
        "treatment_time_m",
        "treatment_time_obs_m",
        "treatment_event",
        "area_id",
        *list(required_covariates),
    ]
    missing_after = [c for c in required_after_rename if c not in df.columns]
    if missing_after:
        raise ValueError(f"Canonicalized data missing required columns: {missing_after}")

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