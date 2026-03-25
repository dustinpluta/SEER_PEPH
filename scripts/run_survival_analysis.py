from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seer_peph.analysis.survival_analysis import (
    SurvivalAnalysisConfig,
    SurvivalPredictionConfig,
    run_survival_analysis,
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


def _parse_prediction_config(obj: Any) -> SurvivalPredictionConfig:
    default_cfg = SurvivalPredictionConfig()

    if obj is None:
        return default_cfg
    if not isinstance(obj, dict):
        raise ValueError("prediction must be a JSON object.")

    eval_times_m = tuple(obj.get("eval_times_m", default_cfg.eval_times_m))
    horizon_m = float(obj.get("horizon_m", default_cfg.horizon_m))
    grid_size = int(obj.get("grid_size", default_cfg.grid_size))
    treatment_times_m = tuple(obj.get("treatment_times_m", default_cfg.treatment_times_m))

    raw_pairs = obj.get("contrast_pairs_m", default_cfg.contrast_pairs_m)
    contrast_pairs_m: tuple[tuple[float | None, float | None], ...] = tuple(
        tuple(pair) for pair in raw_pairs
    )

    return SurvivalPredictionConfig(
        eval_times_m=eval_times_m,
        horizon_m=horizon_m,
        grid_size=grid_size,
        treatment_times_m=treatment_times_m,
        contrast_pairs_m=contrast_pairs_m,
    )


def _parse_survival_analysis_config(obj: dict[str, Any]) -> SurvivalAnalysisConfig:
    if "input_path" not in obj:
        raise ValueError("Config must include input_path.")

    default_cfg = SurvivalAnalysisConfig(input_path=str(obj["input_path"]))
    prediction_cfg = _parse_prediction_config(obj.get("prediction"))

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

    return SurvivalAnalysisConfig(
        input_path=str(obj["input_path"]),
        out_dir=str(obj.get("out_dir", default_cfg.out_dir)),
        graph_mode=str(obj.get("graph_mode", default_cfg.graph_mode)),
        graph_A=None if obj.get("graph_A") is None else int(obj["graph_A"]),
        graph_k=int(obj.get("graph_k", default_cfg.graph_k)),
        surv_breaks=surv_breaks,
        ttt_breaks=ttt_breaks,
        post_ttt_breaks=post_ttt_breaks,
        surv_x_cols=tuple(obj.get("surv_x_cols", default_cfg.surv_x_cols)),
        rng_seed=int(obj.get("rng_seed", default_cfg.rng_seed)),
        inference=dict(obj.get("inference", default_cfg.inference)),
        prediction=prediction_cfg,
        prediction_profile=str(obj.get("prediction_profile", default_cfg.prediction_profile)),
        prediction_area_id=(
            None if obj.get("prediction_area_id") is None else int(obj["prediction_area_id"])
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run standalone survival analysis from a JSON config file."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to survival analysis JSON config file.",
    )
    args = parser.parse_args()

    raw_cfg = _load_json(args.config_path)
    cfg = _parse_survival_analysis_config(raw_cfg)
    out_dir = run_survival_analysis(cfg)
    print(f"Survival analysis completed. Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()