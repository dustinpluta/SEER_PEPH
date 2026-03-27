from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_SCENARIOS = ["J00", "J01", "J02", "J03", "J04", "J05", "J06"]


@dataclass(frozen=True)
class JointValidationStudyConfig:
    out_dir: str
    scenarios: tuple[str, ...] = tuple(DEFAULT_SCENARIOS)
    seeds: tuple[int, ...] = (101, 202, 303)
    inference: dict[str, Any] = field(
        default_factory=lambda: {
            "num_chains": 1,
            "num_warmup": 200,
            "num_samples": 200,
            "target_accept_prob": 0.9,
            "dense_mass": False,
            "max_tree_depth": 8,
            "progress_bar": False,
        }
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.out_dir or not str(self.out_dir).strip():
            raise ValueError("out_dir must be a non-empty string.")

        if len(self.scenarios) == 0:
            raise ValueError("scenarios must contain at least one scenario name.")
        if any((not isinstance(s, str)) or (not s.strip()) for s in self.scenarios):
            raise ValueError("All scenarios must be non-empty strings.")

        if len(self.seeds) == 0:
            raise ValueError("seeds must contain at least one integer seed.")
        if any((not isinstance(s, int)) for s in self.seeds):
            raise ValueError("All seeds must be integers.")

        required_inference_keys = {
            "num_chains",
            "num_warmup",
            "num_samples",
            "target_accept_prob",
            "dense_mass",
            "max_tree_depth",
            "progress_bar",
        }
        missing = required_inference_keys - set(self.inference.keys())
        if missing:
            raise ValueError(f"inference is missing required keys: {sorted(missing)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "scenarios": list(self.scenarios),
            "seeds": list(self.seeds),
            "inference": dict(self.inference),
            "metadata": dict(self.metadata),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full model-matched joint validation study from a JSON config file."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to a joint validation study JSON config file.",
    )
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Top-level config JSON must be an object.")
    return obj


def _parse_inference_config(obj: Any) -> dict[str, Any]:
    default = JointValidationStudyConfig(out_dir="DUMMY").inference

    if obj is None:
        return dict(default)
    if not isinstance(obj, dict):
        raise ValueError("inference must be a JSON object.")

    return {
        "num_chains": int(obj.get("num_chains", default["num_chains"])),
        "num_warmup": int(obj.get("num_warmup", default["num_warmup"])),
        "num_samples": int(obj.get("num_samples", default["num_samples"])),
        "target_accept_prob": float(
            obj.get("target_accept_prob", default["target_accept_prob"])
        ),
        "dense_mass": bool(obj.get("dense_mass", default["dense_mass"])),
        "max_tree_depth": int(obj.get("max_tree_depth", default["max_tree_depth"])),
        "progress_bar": bool(obj.get("progress_bar", default["progress_bar"])),
    }


def _parse_study_config(obj: dict[str, Any]) -> JointValidationStudyConfig:
    if "out_dir" not in obj:
        raise ValueError("Config must include out_dir.")

    scenarios_raw = obj.get("scenarios", DEFAULT_SCENARIOS)
    seeds_raw = obj.get("seeds")
    if seeds_raw is None:
        raise ValueError("Config must include seeds.")

    if not isinstance(scenarios_raw, list) or len(scenarios_raw) == 0:
        raise ValueError("scenarios must be a non-empty JSON array of strings.")
    if not isinstance(seeds_raw, list) or len(seeds_raw) == 0:
        raise ValueError("seeds must be a non-empty JSON array of integers.")

    metadata = obj.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object when provided.")

    return JointValidationStudyConfig(
        out_dir=str(obj["out_dir"]),
        scenarios=tuple(str(s) for s in scenarios_raw),
        seeds=tuple(int(s) for s in seeds_raw),
        inference=_parse_inference_config(obj.get("inference")),
        metadata=dict(metadata),
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _seed_runner_path() -> Path:
    path = _repo_root() / "scripts" / "run_joint_validation_seed_diagnostics.py"
    if not path.exists():
        raise FileNotFoundError(f"Could not find seed runner at: {path}")
    return path


def _run_one_replicate(
    *,
    scenario: str,
    seed: int,
    out_dir: Path,
    study_cfg: JointValidationStudyConfig,
) -> None:
    script_path = _seed_runner_path()
    inf = study_cfg.inference

    cmd = [
        sys.executable,
        str(script_path),
        "--scenario",
        str(scenario),
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
        "--num-chains",
        str(int(inf["num_chains"])),
        "--num-warmup",
        str(int(inf["num_warmup"])),
        "--num-samples",
        str(int(inf["num_samples"])),
        "--target-accept-prob",
        str(float(inf["target_accept_prob"])),
        "--max-tree-depth",
        str(int(inf["max_tree_depth"])),
    ]
    if bool(inf["dense_mass"]):
        cmd.append("--dense-mass")
    if bool(inf["progress_bar"]):
        cmd.append("--progress-bar")

    result = subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Joint validation seed run failed.\n"
            f"Scenario: {scenario}\n"
            f"Seed: {seed}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _add_run_keys(df: pd.DataFrame | None, *, scenario: str, seed: int) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    out.insert(0, "seed", int(seed))
    out.insert(0, "scenario", str(scenario))
    return out


def _collect_per_replicate_tables(
    *,
    scenario: str,
    seed: int,
    run_dir: Path,
) -> dict[str, pd.DataFrame]:
    files = {
        "survival_beta": "recovery_survival_beta.csv",
        "treatment_theta": "recovery_treatment_theta.csv",
        "survival_alpha": "recovery_survival_alpha.csv",
        "treatment_gamma": "recovery_treatment_gamma.csv",
        "survival_delta_post": "recovery_survival_delta_post.csv",
        "area_fields": "recovery_area_fields.csv",
        "area_field_summary": "recovery_area_field_summary.csv",
        "fit_diagnostics": "fit_diagnostics.csv",
        "survival_alpha_support": "recovery_survival_alpha_support.csv",
        "treatment_gamma_support_observed": "recovery_treatment_gamma_support_observed.csv",
        "treatment_gamma_support_true": "recovery_treatment_gamma_support_true.csv",
        "survival_delta_post_support": "recovery_survival_delta_post_support.csv",
        "coupling_hyperparameter_diagnostics": "coupling_hyperparameter_diagnostics.csv",
    }

    out: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        df = _read_csv_if_exists(run_dir / filename)
        df = _add_run_keys(df, scenario=scenario, seed=seed)
        if df is not None:
            out[key] = df
    return out


def _concat_tables(runs: list[pd.DataFrame]) -> pd.DataFrame:
    if not runs:
        return pd.DataFrame()
    return pd.concat(runs, ignore_index=True)


def _safe_rmse(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(math.sqrt(np_mean(vals**2)))


def np_mean(x) -> float:
    x = list(x)
    if len(x) == 0:
        return float("nan")
    return float(sum(float(v) for v in x) / len(x))


def _summarize_scalar_or_vector_recovery(df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(group_cols, dropna=False)

    rows: list[dict[str, Any]] = []
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: key for col, key in zip(group_cols, keys)}
        bias = pd.to_numeric(sub["bias_mean"], errors="coerce")
        abs_err = pd.to_numeric(sub["abs_error_mean"], errors="coerce")
        covered = pd.to_numeric(sub["covered_90"], errors="coerce")

        row.update(
            {
                "n_runs": int(sub[["scenario", "seed"]].drop_duplicates().shape[0]),
                "mean_bias": float(bias.mean()),
                "mean_abs_error": float(abs_err.mean()),
                "rmse": _safe_rmse(bias),
                "coverage_90": float(covered.mean()),
            }
        )
        if "truth" in sub.columns:
            row["truth_mean"] = float(pd.to_numeric(sub["truth"], errors="coerce").mean())
        rows.append(row)

    return pd.DataFrame(rows)


def _summarize_area_field_recovery(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(["scenario", "field"], dropna=False)
    rows: list[dict[str, Any]] = []
    for (scenario, field), sub in grouped:
        rows.append(
            {
                "scenario": scenario,
                "field": field,
                "n_runs": int(sub["seed"].nunique()),
                "mean_pearson_corr": float(pd.to_numeric(sub["pearson_corr"], errors="coerce").mean()),
                "mean_spearman_corr": float(pd.to_numeric(sub["spearman_corr"], errors="coerce").mean()),
                "mean_rmse": float(pd.to_numeric(sub["rmse"], errors="coerce").mean()),
                "mean_abs_error": float(pd.to_numeric(sub["mean_abs_error"], errors="coerce").mean()),
                "coverage_90": float(pd.to_numeric(sub["coverage_90"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_fit_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(["scenario", "metric"], dropna=False)
    rows: list[dict[str, Any]] = []
    for (scenario, metric), sub in grouped:
        vals = pd.to_numeric(sub["value"], errors="coerce")
        rows.append(
            {
                "scenario": scenario,
                "metric": metric,
                "n_runs": int(sub["seed"].nunique()),
                "mean_value": float(vals.mean()),
                "median_value": float(vals.median()),
                "min_value": float(vals.min()),
                "max_value": float(vals.max()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_support_linked_recovery(df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(group_cols, dropna=False)
    rows: list[dict[str, Any]] = []
    support_cols = [c for c in ["interval_start", "interval_end", "count", "count_total", "count_fraction", "subject_count", "subject_count_total", "subject_count_fraction"] if c in df.columns]

    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        bias = pd.to_numeric(sub["bias_mean"], errors="coerce")
        abs_err = pd.to_numeric(sub["abs_error_mean"], errors="coerce")
        covered = pd.to_numeric(sub["covered_90"], errors="coerce")
        row.update({
            "n_runs": int(sub[["scenario", "seed"]].drop_duplicates().shape[0]),
            "mean_bias": float(bias.mean()),
            "mean_abs_error": float(abs_err.mean()),
            "rmse": _safe_rmse(bias),
            "coverage_90": float(covered.mean()),
        })
        if "truth" in sub.columns:
            row["truth_mean"] = float(pd.to_numeric(sub["truth"], errors="coerce").mean())
        for col in support_cols:
            row[f"mean_{col}"] = float(pd.to_numeric(sub[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_coupling_hyperparameter_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = ["scenario", "section", "metric_name", "truth_parameter", "estimate_parameter", "comparable_to_truth"]
    grouped = df.groupby(group_cols, dropna=False)
    rows: list[dict[str, Any]] = []
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({
            "n_runs": int(sub[["scenario", "seed"]].drop_duplicates().shape[0]),
            "truth_mean": float(pd.to_numeric(sub["truth"], errors="coerce").mean()),
            "posterior_mean_mean": float(pd.to_numeric(sub["posterior_mean"], errors="coerce").mean()),
            "posterior_median_mean": float(pd.to_numeric(sub["posterior_median"], errors="coerce").mean()),
            "mean_bias": float(pd.to_numeric(sub["bias_mean"], errors="coerce").mean()),
            "mean_abs_error": float(pd.to_numeric(sub["abs_error_mean"], errors="coerce").mean()),
            "rmse": _safe_rmse(pd.to_numeric(sub["bias_mean"], errors="coerce")),
            "coverage_90": float(pd.to_numeric(sub["covered_90"], errors="coerce").mean()),
            "mean_q05": float(pd.to_numeric(sub["q05"], errors="coerce").mean()),
            "mean_q95": float(pd.to_numeric(sub["q95"], errors="coerce").mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_json_ready(obj), indent=2), encoding="utf-8")


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


def main() -> None:
    args = _parse_args()
    raw_cfg = _load_json(args.config_path)
    study_cfg = _parse_study_config(raw_cfg)

    study_dir = Path(study_cfg.out_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    all_survival_beta: list[pd.DataFrame] = []
    all_treatment_theta: list[pd.DataFrame] = []
    all_survival_alpha: list[pd.DataFrame] = []
    all_treatment_gamma: list[pd.DataFrame] = []
    all_survival_delta_post: list[pd.DataFrame] = []
    all_area_fields: list[pd.DataFrame] = []
    all_area_field_summary: list[pd.DataFrame] = []
    all_fit_diagnostics: list[pd.DataFrame] = []
    all_survival_alpha_support: list[pd.DataFrame] = []
    all_treatment_gamma_support_observed: list[pd.DataFrame] = []
    all_treatment_gamma_support_true: list[pd.DataFrame] = []
    all_survival_delta_post_support: list[pd.DataFrame] = []
    all_coupling_hyperparameter_diagnostics: list[pd.DataFrame] = []

    completed_runs: list[dict[str, Any]] = []

    for scenario in study_cfg.scenarios:
        for seed in study_cfg.seeds:
            run_dir = study_dir / scenario / f"seed_{int(seed)}"
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"Running scenario={scenario}, seed={seed}")
            _run_one_replicate(
                scenario=str(scenario),
                seed=int(seed),
                out_dir=run_dir,
                study_cfg=study_cfg,
            )

            tables = _collect_per_replicate_tables(
                scenario=str(scenario),
                seed=int(seed),
                run_dir=run_dir,
            )

            if "survival_beta" in tables:
                all_survival_beta.append(tables["survival_beta"])
            if "treatment_theta" in tables:
                all_treatment_theta.append(tables["treatment_theta"])
            if "survival_alpha" in tables:
                all_survival_alpha.append(tables["survival_alpha"])
            if "treatment_gamma" in tables:
                all_treatment_gamma.append(tables["treatment_gamma"])
            if "survival_delta_post" in tables:
                all_survival_delta_post.append(tables["survival_delta_post"])
            if "area_fields" in tables:
                all_area_fields.append(tables["area_fields"])
            if "area_field_summary" in tables:
                all_area_field_summary.append(tables["area_field_summary"])
            if "fit_diagnostics" in tables:
                all_fit_diagnostics.append(tables["fit_diagnostics"])
            if "survival_alpha_support" in tables:
                all_survival_alpha_support.append(tables["survival_alpha_support"])
            if "treatment_gamma_support_observed" in tables:
                all_treatment_gamma_support_observed.append(tables["treatment_gamma_support_observed"])
            if "treatment_gamma_support_true" in tables:
                all_treatment_gamma_support_true.append(tables["treatment_gamma_support_true"])
            if "survival_delta_post_support" in tables:
                all_survival_delta_post_support.append(tables["survival_delta_post_support"])
            if "coupling_hyperparameter_diagnostics" in tables:
                all_coupling_hyperparameter_diagnostics.append(tables["coupling_hyperparameter_diagnostics"])

            completed_runs.append(
                {
                    "scenario": str(scenario),
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                }
            )

    per_seed_dir = study_dir / "per_seed_tables"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    survival_beta_all = _concat_tables(all_survival_beta)
    treatment_theta_all = _concat_tables(all_treatment_theta)
    survival_alpha_all = _concat_tables(all_survival_alpha)
    treatment_gamma_all = _concat_tables(all_treatment_gamma)
    survival_delta_post_all = _concat_tables(all_survival_delta_post)
    area_fields_all = _concat_tables(all_area_fields)
    area_field_summary_all = _concat_tables(all_area_field_summary)
    fit_diagnostics_all = _concat_tables(all_fit_diagnostics)
    survival_alpha_support_all = _concat_tables(all_survival_alpha_support)
    treatment_gamma_support_observed_all = _concat_tables(all_treatment_gamma_support_observed)
    treatment_gamma_support_true_all = _concat_tables(all_treatment_gamma_support_true)
    survival_delta_post_support_all = _concat_tables(all_survival_delta_post_support)
    coupling_hyperparameter_diagnostics_all = _concat_tables(all_coupling_hyperparameter_diagnostics)

    if not survival_beta_all.empty:
        survival_beta_all.to_csv(per_seed_dir / "recovery_survival_beta_all.csv", index=False)
    if not treatment_theta_all.empty:
        treatment_theta_all.to_csv(per_seed_dir / "recovery_treatment_theta_all.csv", index=False)
    if not survival_alpha_all.empty:
        survival_alpha_all.to_csv(per_seed_dir / "recovery_survival_alpha_all.csv", index=False)
    if not treatment_gamma_all.empty:
        treatment_gamma_all.to_csv(per_seed_dir / "recovery_treatment_gamma_all.csv", index=False)
    if not survival_delta_post_all.empty:
        survival_delta_post_all.to_csv(per_seed_dir / "recovery_survival_delta_post_all.csv", index=False)
    if not area_fields_all.empty:
        area_fields_all.to_csv(per_seed_dir / "recovery_area_fields_all.csv", index=False)
    if not area_field_summary_all.empty:
        area_field_summary_all.to_csv(per_seed_dir / "recovery_area_field_summary_all.csv", index=False)
    if not fit_diagnostics_all.empty:
        fit_diagnostics_all.to_csv(per_seed_dir / "fit_diagnostics_all.csv", index=False)
    if not survival_alpha_support_all.empty:
        survival_alpha_support_all.to_csv(per_seed_dir / "recovery_survival_alpha_support_all.csv", index=False)
    if not treatment_gamma_support_observed_all.empty:
        treatment_gamma_support_observed_all.to_csv(per_seed_dir / "recovery_treatment_gamma_support_observed_all.csv", index=False)
    if not treatment_gamma_support_true_all.empty:
        treatment_gamma_support_true_all.to_csv(per_seed_dir / "recovery_treatment_gamma_support_true_all.csv", index=False)
    if not survival_delta_post_support_all.empty:
        survival_delta_post_support_all.to_csv(per_seed_dir / "recovery_survival_delta_post_support_all.csv", index=False)
    if not coupling_hyperparameter_diagnostics_all.empty:
        coupling_hyperparameter_diagnostics_all.to_csv(per_seed_dir / "coupling_hyperparameter_diagnostics_all.csv", index=False)

    summary_dir = study_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    survival_beta_summary = _summarize_scalar_or_vector_recovery(
        survival_beta_all,
        group_cols=["scenario", "parameter", "label"],
    )
    treatment_theta_summary = _summarize_scalar_or_vector_recovery(
        treatment_theta_all,
        group_cols=["scenario", "parameter", "label"],
    )
    survival_alpha_summary = _summarize_scalar_or_vector_recovery(
        survival_alpha_all,
        group_cols=["scenario", "group", "index"],
    )
    treatment_gamma_summary = _summarize_scalar_or_vector_recovery(
        treatment_gamma_all,
        group_cols=["scenario", "group", "index"],
    )
    survival_delta_post_summary = _summarize_scalar_or_vector_recovery(
        survival_delta_post_all,
        group_cols=["scenario", "group", "index"],
    )
    area_field_summary = _summarize_area_field_recovery(area_field_summary_all)
    fit_diagnostics_summary = _summarize_fit_diagnostics(fit_diagnostics_all)
    survival_alpha_support_summary = _summarize_support_linked_recovery(
        survival_alpha_support_all,
        group_cols=["scenario", "group", "index", "support_source"],
    )
    treatment_gamma_support_observed_summary = _summarize_support_linked_recovery(
        treatment_gamma_support_observed_all,
        group_cols=["scenario", "group", "index", "support_source"],
    )
    treatment_gamma_support_true_summary = _summarize_support_linked_recovery(
        treatment_gamma_support_true_all,
        group_cols=["scenario", "group", "index", "support_source"],
    )
    survival_delta_post_support_summary = _summarize_support_linked_recovery(
        survival_delta_post_support_all,
        group_cols=["scenario", "group", "index", "support_source"],
    )
    coupling_hyperparameter_diagnostics_summary = _summarize_coupling_hyperparameter_diagnostics(
        coupling_hyperparameter_diagnostics_all
    )

    if not survival_beta_summary.empty:
        survival_beta_summary.to_csv(summary_dir / "summary_survival_beta.csv", index=False)
    if not treatment_theta_summary.empty:
        treatment_theta_summary.to_csv(summary_dir / "summary_treatment_theta.csv", index=False)
    if not survival_alpha_summary.empty:
        survival_alpha_summary.to_csv(summary_dir / "summary_survival_alpha.csv", index=False)
    if not treatment_gamma_summary.empty:
        treatment_gamma_summary.to_csv(summary_dir / "summary_treatment_gamma.csv", index=False)
    if not survival_delta_post_summary.empty:
        survival_delta_post_summary.to_csv(summary_dir / "summary_survival_delta_post.csv", index=False)
    if not area_field_summary.empty:
        area_field_summary.to_csv(summary_dir / "summary_area_fields.csv", index=False)
    if not fit_diagnostics_summary.empty:
        fit_diagnostics_summary.to_csv(summary_dir / "summary_fit_diagnostics.csv", index=False)
    if not survival_alpha_support_summary.empty:
        survival_alpha_support_summary.to_csv(summary_dir / "summary_survival_alpha_support.csv", index=False)
    if not treatment_gamma_support_observed_summary.empty:
        treatment_gamma_support_observed_summary.to_csv(summary_dir / "summary_treatment_gamma_support_observed.csv", index=False)
    if not treatment_gamma_support_true_summary.empty:
        treatment_gamma_support_true_summary.to_csv(summary_dir / "summary_treatment_gamma_support_true.csv", index=False)
    if not survival_delta_post_support_summary.empty:
        survival_delta_post_support_summary.to_csv(summary_dir / "summary_survival_delta_post_support.csv", index=False)
    if not coupling_hyperparameter_diagnostics_summary.empty:
        coupling_hyperparameter_diagnostics_summary.to_csv(summary_dir / "summary_coupling_hyperparameter_diagnostics.csv", index=False)

    study_manifest = {
        "study_dir": str(study_dir),
        "scenarios": list(study_cfg.scenarios),
        "seeds": [int(s) for s in study_cfg.seeds],
        "num_runs": int(len(completed_runs)),
        "inference": dict(study_cfg.inference),
        "metadata": dict(study_cfg.metadata),
    }
    _write_json(study_dir / "study_manifest.json", study_manifest)
    _write_json(study_dir / "completed_runs.json", completed_runs)
    _write_json(study_dir / "study_config_resolved.json", study_cfg.to_dict())

    print("Joint validation study completed.")
    print(f"Study artifacts written to: {study_dir}")


if __name__ == "__main__":
    main()