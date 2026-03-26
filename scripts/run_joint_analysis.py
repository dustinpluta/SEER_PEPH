from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seer_peph.analysis.joint_analysis import (
    DerivedColumnConfig,
    InputColumnConfig,
    JointAnalysisConfig,
    run_joint_analysis,
)
from seer_peph.data.prep import (
    DEFAULT_POST_TTT_BREAKS,
    DEFAULT_SURV_BREAKS,
    DEFAULT_TTT_BREAKS,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Top-level config JSON must be an object.")
    return obj


def _parse_breaks(
    obj: Any,
    *,
    default: tuple[float, ...] | list[float],
    name: str,
) -> tuple[float, ...]:
    vals = default if obj is None else obj
    if not isinstance(vals, (list, tuple)):
        raise ValueError(f"{name} must be a JSON array.")
    out = tuple(float(v) for v in vals)
    if len(out) < 2:
        raise ValueError(f"{name} must contain at least two values.")
    return out


def _parse_input_columns(obj: Any) -> InputColumnConfig:
    default_cfg = InputColumnConfig()

    if obj is None:
        return default_cfg
    if not isinstance(obj, dict):
        raise ValueError("input_columns must be a JSON object.")

    return InputColumnConfig(
        id_col=str(obj.get("id_col", default_cfg.id_col)),
        time_days_col=str(obj.get("time_days_col", default_cfg.time_days_col)),
        event_col=str(obj.get("event_col", default_cfg.event_col)),
        treatment_time_days_col=str(
            obj.get("treatment_time_days_col", default_cfg.treatment_time_days_col)
        ),
        treatment_time_obs_days_col=str(
            obj.get("treatment_time_obs_days_col", default_cfg.treatment_time_obs_days_col)
        ),
        treatment_event_col=str(
            obj.get("treatment_event_col", default_cfg.treatment_event_col)
        ),
        zip_col=str(obj.get("zip_col", default_cfg.zip_col)),
        sex_col=str(obj.get("sex_col", default_cfg.sex_col)),
        stage_col=str(obj.get("stage_col", default_cfg.stage_col)),
    )


def _parse_derived_columns(obj: Any) -> DerivedColumnConfig:
    default_cfg = DerivedColumnConfig()

    if obj is None:
        return default_cfg
    if not isinstance(obj, dict):
        raise ValueError("derived_columns must be a JSON object.")

    return DerivedColumnConfig(
        time_m_col=str(obj.get("time_m_col", default_cfg.time_m_col)),
        treatment_time_m_col=str(
            obj.get("treatment_time_m_col", default_cfg.treatment_time_m_col)
        ),
        treatment_time_obs_m_col=str(
            obj.get("treatment_time_obs_m_col", default_cfg.treatment_time_obs_m_col)
        ),
        area_id_col=str(obj.get("area_id_col", default_cfg.area_id_col)),
        sex_male_col=str(obj.get("sex_male_col", default_cfg.sex_male_col)),
        stage_ii_col=str(obj.get("stage_ii_col", default_cfg.stage_ii_col)),
        stage_iii_col=str(obj.get("stage_iii_col", default_cfg.stage_iii_col)),
    )


def _parse_joint_analysis_config(obj: dict[str, Any]) -> JointAnalysisConfig:
    if "input_path" not in obj:
        raise ValueError("Config must include input_path.")

    default_cfg = JointAnalysisConfig(input_path=str(obj["input_path"]))
    input_columns_cfg = _parse_input_columns(obj.get("input_columns"))
    derived_columns_cfg = _parse_derived_columns(obj.get("derived_columns"))

    surv_breaks = _parse_breaks(
        obj.get("surv_breaks"),
        default=tuple(DEFAULT_SURV_BREAKS),
        name="surv_breaks",
    )
    ttt_breaks = _parse_breaks(
        obj.get("ttt_breaks"),
        default=tuple(DEFAULT_TTT_BREAKS),
        name="ttt_breaks",
    )
    post_ttt_breaks = _parse_breaks(
        obj.get("post_ttt_breaks"),
        default=tuple(DEFAULT_POST_TTT_BREAKS),
        name="post_ttt_breaks",
    )

    return JointAnalysisConfig(
        input_path=str(obj["input_path"]),
        out_dir=str(obj.get("out_dir", default_cfg.out_dir)),
        graph_mode=str(obj.get("graph_mode", default_cfg.graph_mode)),
        graph_A=None if obj.get("graph_A") is None else int(obj["graph_A"]),
        graph_k=int(obj.get("graph_k", default_cfg.graph_k)),
        input_columns=input_columns_cfg,
        derived_columns=derived_columns_cfg,
        surv_breaks=surv_breaks,
        ttt_breaks=ttt_breaks,
        post_ttt_breaks=post_ttt_breaks,
        surv_x_cols=tuple(obj.get("surv_x_cols", default_cfg.surv_x_cols)),
        ttt_x_cols=tuple(obj.get("ttt_x_cols", default_cfg.ttt_x_cols)),
        rng_seed=int(obj.get("rng_seed", default_cfg.rng_seed)),
        inference=dict(obj.get("inference", default_cfg.inference)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run joint treatment-survival analysis from a JSON config file."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to joint analysis JSON config file.",
    )
    args = parser.parse_args()

    raw_cfg = _load_json(args.config_path)
    cfg = _parse_joint_analysis_config(raw_cfg)
    out_dir = run_joint_analysis(cfg)
    print(f"Joint analysis completed. Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()