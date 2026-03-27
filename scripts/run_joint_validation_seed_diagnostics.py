from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from seer_peph.validation.joint_scenarios import (
    JointSimulationScenario,
    baseline_joint_scenario,
    default_joint_validation_scenarios,
    high_coupling_scenario,
    high_spatial_signal_scenario,
    low_coupling_scenario,
    low_spatial_signal_scenario,
    stronger_censoring_scenario,
    weak_post_treatment_effect_scenario,
)
from seer_peph.validation.simulate_joint import simulate_joint_scenario
from seer_peph.data.prep import build_survival_long, build_treatment_long
from seer_peph.data.prep import DAYS_PER_MONTH, build_survival_long, build_treatment_long


SCENARIO_REGISTRY: dict[str, callable] = {
    "J00": baseline_joint_scenario,
    "J01": low_spatial_signal_scenario,
    "J02": high_spatial_signal_scenario,
    "J03": low_coupling_scenario,
    "J04": high_coupling_scenario,
    "J05": weak_post_treatment_effect_scenario,
    "J06": stronger_censoring_scenario,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one model-matched joint validation replicate for a named scenario and seed."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario name, for example J00, J01, ..., J06.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Simulation RNG seed for this replicate.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for this replicate.",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=1,
        help="Number of MCMC chains.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=500,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of post-warmup samples.",
    )
    parser.add_argument(
        "--target-accept-prob",
        type=float,
        default=0.95,
        help="NUTS target acceptance probability.",
    )
    parser.add_argument(
        "--dense-mass",
        action="store_true",
        help="Use dense mass matrix adaptation.",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=10,
        help="NUTS max tree depth.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Enable progress bar during sampling.",
    )
    return parser.parse_args()


def _get_scenario(name: str) -> JointSimulationScenario:
    if name in SCENARIO_REGISTRY:
        return SCENARIO_REGISTRY[name]()
    available = ", ".join(sorted(SCENARIO_REGISTRY))
    raise ValueError(f"Unknown scenario {name!r}. Available scenarios: {available}")


def _build_graph_for_scenario(scenario: JointSimulationScenario):
    if scenario.graph_name != "ring_lattice":
        raise ValueError(f"Unsupported graph_name: {scenario.graph_name}")
    k = int(scenario.graph_kwargs.get("k", 4))
    return make_ring_lattice(A=int(scenario.n_areas), k=k)


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


def _build_inference_config(args: argparse.Namespace) -> InferenceConfig:
    return InferenceConfig(
        num_chains=int(args.num_chains),
        num_warmup=int(args.num_warmup),
        num_samples=int(args.num_samples),
        target_accept_prob=float(args.target_accept_prob),
        dense_mass=bool(args.dense_mass),
        max_tree_depth=int(args.max_tree_depth),
        progress_bar=bool(args.progress_bar),
    )


def _truth_table_to_map(
    parameter_truth: pd.DataFrame,
    *,
    group: str,
    parameter_col: str = "parameter",
    truth_col: str = "truth",
) -> dict[str, float]:
    sub = parameter_truth.loc[parameter_truth["group"] == group].copy()
    return {
        str(row[parameter_col]): float(row[truth_col])
        for _, row in sub.iterrows()
    }


def _vector_truth_map(
    parameter_truth: pd.DataFrame,
    *,
    parameter_name: str,
) -> dict[int, float]:
    sub = parameter_truth.loc[parameter_truth["parameter"] == parameter_name].copy()
    if sub.empty:
        return {}
    if "index" not in sub.columns:
        raise ValueError("parameter_truth must contain an 'index' column for vector truth maps.")
    return {
        int(row["index"]): float(row["truth"])
        for _, row in sub.iterrows()
    }


def _make_scalar_recovery_table(
    *,
    scalar_truth: dict[str, float],
    posterior_summary: dict[str, pd.DataFrame],
    df_name: str,
    parameter_prefix: str | None = None,
) -> pd.DataFrame:
    df = posterior_summary[df_name].copy()
    truth_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        label = str(row.get("label", row.get("parameter", "")))
        parameter = str(row.get("parameter", label))
        key = parameter
        if parameter_prefix is not None and not key.startswith(parameter_prefix):
            key = f"{parameter_prefix}{label}"

        if key not in scalar_truth and parameter in scalar_truth:
            key = parameter
        if key not in scalar_truth and label in scalar_truth:
            key = label

        truth = scalar_truth.get(key)
        if truth is None:
            continue

        mean_est = float(row["mean"])
        median_est = float(row.get("median", mean_est))
        q05 = float(row.get("q05", row.get("p05", row.get("ci_lower", mean_est))))
        q95 = float(row.get("q95", row.get("p95", row.get("ci_upper", mean_est))))

        truth_rows.append(
            {
                "parameter": parameter,
                "label": label,
                "truth": truth,
                "posterior_mean": mean_est,
                "posterior_median": median_est,
                "bias_mean": mean_est - truth,
                "abs_error_mean": abs(mean_est - truth),
                "covered_90": int(q05 <= truth <= q95),
                "q05": q05,
                "q95": q95,
            }
        )

    return pd.DataFrame(truth_rows)


def _make_vector_recovery_table(
    *,
    truth_map: dict[int, float],
    summary_df: pd.DataFrame,
    index_col_candidates: tuple[str, ...],
    group_name: str,
) -> pd.DataFrame:
    df = summary_df.copy()

    index_col = None
    for candidate in index_col_candidates:
        if candidate in df.columns:
            index_col = candidate
            break
    if index_col is None:
        raise ValueError(f"Could not find an index column for {group_name} in columns {list(df.columns)}")

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        idx = int(row[index_col])
        if idx not in truth_map:
            continue

        truth = float(truth_map[idx])
        mean_est = float(row["mean"])
        median_est = float(row.get("median", mean_est))
        q05 = float(row.get("q05", row.get("p05", row.get("ci_lower", mean_est))))
        q95 = float(row.get("q95", row.get("p95", row.get("ci_upper", mean_est))))

        rows.append(
            {
                "group": group_name,
                "index": idx,
                "truth": truth,
                "posterior_mean": mean_est,
                "posterior_median": median_est,
                "bias_mean": mean_est - truth,
                "abs_error_mean": abs(mean_est - truth),
                "covered_90": int(q05 <= truth <= q95),
                "q05": q05,
                "q95": q95,
            }
        )

    return pd.DataFrame(rows)


def _make_area_recovery_table(
    *,
    area_truth: pd.DataFrame,
    spatial_summary: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    truth = area_truth.copy()
    truth = truth.sort_values("area_id").reset_index(drop=True)

    rows: list[pd.DataFrame] = []

    field_map = {
        "u_surv": "u_surv_true",
        "u_ttt": "u_ttt_true",
        "u_ttt_ind": "u_ttt_ind_true",
    }

    for field_name, truth_col in field_map.items():
        if field_name not in spatial_summary:
            continue

        est = spatial_summary[field_name].copy()
        if "area_id" not in est.columns:
            continue

        merged = est.merge(
            truth[["area_id", truth_col]],
            on="area_id",
            how="inner",
        )
        merged["field"] = field_name
        merged["truth"] = merged[truth_col].astype(float)
        merged["posterior_mean"] = merged["mean"].astype(float)
        merged["posterior_median"] = merged.get("median", merged["mean"]).astype(float)
        merged["bias_mean"] = merged["posterior_mean"] - merged["truth"]
        merged["abs_error_mean"] = (merged["posterior_mean"] - merged["truth"]).abs()

        q05 = merged["q05"] if "q05" in merged.columns else merged.get("p05", merged["posterior_mean"])
        q95 = merged["q95"] if "q95" in merged.columns else merged.get("p95", merged["posterior_mean"])
        merged["covered_90"] = ((q05 <= merged["truth"]) & (merged["truth"] <= q95)).astype(int)

        rows.append(
            merged[
                [
                    "field",
                    "area_id",
                    "truth",
                    "posterior_mean",
                    "posterior_median",
                    "bias_mean",
                    "abs_error_mean",
                    "covered_90",
                ]
            ].copy()
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "field",
                "area_id",
                "truth",
                "posterior_mean",
                "posterior_median",
                "bias_mean",
                "abs_error_mean",
                "covered_90",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def _make_area_field_summary(area_recovery: pd.DataFrame) -> pd.DataFrame:
    if area_recovery.empty:
        return pd.DataFrame(
            columns=[
                "field",
                "pearson_corr",
                "spearman_corr",
                "rmse",
                "mean_abs_error",
                "coverage_90",
            ]
        )

    rows: list[dict[str, Any]] = []
    for field, sub in area_recovery.groupby("field"):
        truth = sub["truth"].astype(float)
        est = sub["posterior_mean"].astype(float)

        pearson_corr = float(truth.corr(est, method="pearson"))
        spearman_corr = float(truth.corr(est, method="spearman"))
        rmse = float(((est - truth) ** 2).mean() ** 0.5)
        mae = float((est - truth).abs().mean())
        coverage = float(sub["covered_90"].mean())

        rows.append(
            {
                "field": field,
                "pearson_corr": pearson_corr,
                "spearman_corr": spearman_corr,
                "rmse": rmse,
                "mean_abs_error": mae,
                "coverage_90": coverage,
            }
        )
    return pd.DataFrame(rows)


def _make_fit_diagnostics_table(fit) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    metadata = fit.metadata
    rows.append(
        {
            "metric": "n_surv_rows",
            "value": metadata.n_surv,
        }
    )
    rows.append(
        {
            "metric": "n_ttt_rows",
            "value": metadata.n_ttt,
        }
    )
    rows.append(
        {
            "metric": "graph_A",
            "value": metadata.graph_A,
        }
    )
    rows.append(
        {
            "metric": "rng_seed",
            "value": metadata.rng_seed,
        }
    )

    # Divergences from extra sampler fields saved in samples.
    div = None
    if hasattr(fit, "samples") and isinstance(fit.samples, dict):
        div = fit.samples.get("diverging")

    if div is not None:
        div_arr = pd.Series(div.ravel() if hasattr(div, "ravel") else div)
        rows.append(
            {
                "metric": "num_divergences",
                "value": int(div_arr.sum()),
            }
        )
        rows.append(
            {
                "metric": "any_divergence",
                "value": int(div_arr.sum() > 0),
            }
        )

    return pd.DataFrame(rows)
    


def _make_support_linked_recovery_table(
    *,
    recovery_df: pd.DataFrame,
    support_df: pd.DataFrame,
    recovery_index_col: str,
    support_index_col: str,
    support_source: str,
    support_value_cols: tuple[str, ...],
) -> pd.DataFrame:
    if recovery_df.empty:
        return recovery_df.copy()

    out = recovery_df.copy()
    support = support_df.copy()
    if support.empty:
        out["support_source"] = support_source
        for col in support_value_cols:
            out[col] = np.nan
            out[f"{col}_total"] = np.nan
            out[f"{col}_fraction"] = np.nan
        return out

    support = support.rename(columns={support_index_col: recovery_index_col})
    keep_cols = [recovery_index_col]
    for col in ["interval_start", "interval_end", *support_value_cols]:
        if col in support.columns:
            keep_cols.append(col)

    out = out.merge(support[keep_cols], on=recovery_index_col, how="left")
    out["support_source"] = support_source

    for col in support_value_cols:
        total = float(pd.to_numeric(support[col], errors="coerce").sum()) if col in support.columns else np.nan
        out[f"{col}_total"] = total
        numer = pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(np.nan, index=out.index)
        out[f"{col}_fraction"] = numer / total if np.isfinite(total) and total > 0.0 else np.nan

    return out


def _safe_series_corr(x: pd.Series, y: pd.Series) -> float:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    keep = x_num.notna() & y_num.notna()
    if int(keep.sum()) < 2:
        return float("nan")
    if float(x_num[keep].std(ddof=0)) == 0.0 or float(y_num[keep].std(ddof=0)) == 0.0:
        return float("nan")
    return float(x_num[keep].corr(y_num[keep], method="pearson"))


def _extract_metric_value(df: pd.DataFrame, metric: str) -> float | None:
    if df.empty or "metric" not in df.columns or "value" not in df.columns:
        return None
    sub = df.loc[df["metric"].astype(str) == str(metric), "value"]
    if sub.empty:
        return None
    return float(sub.iloc[0])


def _make_coupling_hyperparameter_diagnostics(
    *,
    parameter_truth: pd.DataFrame,
    area_truth: pd.DataFrame,
    spatial_summary: dict[str, pd.DataFrame],
    coupling_summary: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    truth_map = _truth_table_to_map(parameter_truth, group="spatial_hyperparameter")

    for parameter in ["phi_surv", "phi_ttt", "sigma_surv", "sigma_ttt", "rho_u"]:
        rows.append(
            {
                "section": "scenario_truth_hyperparameter",
                "metric_name": parameter,
                "truth_parameter": parameter,
                "estimate_parameter": None,
                "comparable_to_truth": 0,
                "truth": truth_map.get(parameter),
                "posterior_mean": np.nan,
                "posterior_median": np.nan,
                "bias_mean": np.nan,
                "abs_error_mean": np.nan,
                "covered_90": np.nan,
                "q05": np.nan,
                "q95": np.nan,
            }
        )

    spatial_hyper = spatial_summary.get("hyperparameters", pd.DataFrame())
    if not spatial_hyper.empty:
        for _, row in spatial_hyper.iterrows():
            rows.append(
                {
                    "section": "fitted_model_hyperparameter",
                    "metric_name": str(row["parameter"]),
                    "truth_parameter": None,
                    "estimate_parameter": str(row["parameter"]),
                    "comparable_to_truth": 0,
                    "truth": np.nan,
                    "posterior_mean": float(row["mean"]),
                    "posterior_median": float(row.get("median", row["mean"])),
                    "bias_mean": np.nan,
                    "abs_error_mean": np.nan,
                    "covered_90": np.nan,
                    "q05": float(row.get("q05", np.nan)),
                    "q95": float(row.get("q95", np.nan)),
                }
            )

    coupling_df = coupling_summary.get("coupling", pd.DataFrame())
    if not coupling_df.empty:
        for _, row in coupling_df.iterrows():
            parameter = str(row["parameter"])
            if parameter == "rho_u_cross" and "rho_u" in truth_map:
                truth = float(truth_map["rho_u"])
                mean_est = float(row["mean"])
                q05 = float(row.get("q05", np.nan))
                q95 = float(row.get("q95", np.nan))
                rows.append(
                    {
                        "section": "direct_parameter_recovery",
                        "metric_name": "rho_u_cross_vs_rho_u_true",
                        "truth_parameter": "rho_u",
                        "estimate_parameter": parameter,
                        "comparable_to_truth": 1,
                        "truth": truth,
                        "posterior_mean": mean_est,
                        "posterior_median": float(row.get("median", mean_est)),
                        "bias_mean": mean_est - truth,
                        "abs_error_mean": abs(mean_est - truth),
                        "covered_90": float(q05 <= truth <= q95) if np.isfinite(q05) and np.isfinite(q95) else np.nan,
                        "q05": q05,
                        "q95": q95,
                    }
                )

            rows.append(
                {
                    "section": "fitted_model_hyperparameter",
                    "metric_name": parameter,
                    "truth_parameter": None,
                    "estimate_parameter": parameter,
                    "comparable_to_truth": 0,
                    "truth": np.nan,
                    "posterior_mean": float(row["mean"]),
                    "posterior_median": float(row.get("median", row["mean"])),
                    "bias_mean": np.nan,
                    "abs_error_mean": np.nan,
                    "covered_90": np.nan,
                    "q05": float(row.get("q05", np.nan)),
                    "q95": float(row.get("q95", np.nan)),
                }
            )

    area_truth_sorted = area_truth.sort_values("area_id").reset_index(drop=True)
    for field_name, truth_col in [("u_surv", "u_surv_true"), ("u_ttt", "u_ttt_true"), ("u_ttt_ind", "u_ttt_ind_true")]:
        est_df = spatial_summary.get(field_name)
        if est_df is None or est_df.empty or truth_col not in area_truth_sorted.columns or "area_id" not in est_df.columns:
            continue
        est_sorted = est_df.sort_values("area_id").reset_index(drop=True)
        merged = est_sorted.merge(area_truth_sorted[["area_id", truth_col]], on="area_id", how="inner")
        if merged.empty:
            continue
        truth_vals = pd.to_numeric(merged[truth_col], errors="coerce")
        est_vals = pd.to_numeric(merged["mean"], errors="coerce")
        truth_sd = float(truth_vals.std(ddof=0))
        est_sd = float(est_vals.std(ddof=0))
        rows.append(
            {
                "section": "field_scale_diagnostic",
                "metric_name": f"empirical_sd_{field_name}",
                "truth_parameter": f"empirical_sd_{truth_col}",
                "estimate_parameter": f"empirical_sd_{field_name}_posterior_mean",
                "comparable_to_truth": 1,
                "truth": truth_sd,
                "posterior_mean": est_sd,
                "posterior_median": np.nan,
                "bias_mean": est_sd - truth_sd,
                "abs_error_mean": abs(est_sd - truth_sd),
                "covered_90": np.nan,
                "q05": np.nan,
                "q95": np.nan,
            }
        )

    truth_corr = _safe_series_corr(area_truth_sorted["u_surv_true"], area_truth_sorted["u_ttt_true"])
    if "u_surv" in spatial_summary and "u_ttt" in spatial_summary:
        surv_est = spatial_summary["u_surv"].sort_values("area_id").reset_index(drop=True)
        ttt_est = spatial_summary["u_ttt"].sort_values("area_id").reset_index(drop=True)
        merged = surv_est[["area_id", "mean"]].merge(ttt_est[["area_id", "mean"]], on="area_id", how="inner", suffixes=("_surv", "_ttt"))
        if not merged.empty:
            est_corr = _safe_series_corr(merged["mean_surv"], merged["mean_ttt"])
            rows.append(
                {
                    "section": "field_correlation_diagnostic",
                    "metric_name": "corr_u_surv_u_ttt_posterior_mean",
                    "truth_parameter": "corr_u_surv_true_u_ttt_true",
                    "estimate_parameter": "corr_u_surv_u_ttt_posterior_mean",
                    "comparable_to_truth": 1,
                    "truth": truth_corr,
                    "posterior_mean": est_corr,
                    "posterior_median": np.nan,
                    "bias_mean": est_corr - truth_corr if np.isfinite(est_corr) and np.isfinite(truth_corr) else np.nan,
                    "abs_error_mean": abs(est_corr - truth_corr) if np.isfinite(est_corr) and np.isfinite(truth_corr) else np.nan,
                    "covered_90": np.nan,
                    "q05": np.nan,
                    "q95": np.nan,
                }
            )

    field_corr = coupling_summary.get("field_correlations", pd.DataFrame())
    draw_mean = _extract_metric_value(field_corr, "corr_u_surv_u_ttt_draw_mean")
    draw_median = _extract_metric_value(field_corr, "corr_u_surv_u_ttt_draw_median")
    draw_q05 = _extract_metric_value(field_corr, "corr_u_surv_u_ttt_draw_q05")
    draw_q95 = _extract_metric_value(field_corr, "corr_u_surv_u_ttt_draw_q95")
    if draw_mean is not None:
        rows.append(
            {
                "section": "field_correlation_diagnostic",
                "metric_name": "corr_u_surv_u_ttt_draw_mean",
                "truth_parameter": "corr_u_surv_true_u_ttt_true",
                "estimate_parameter": "corr_u_surv_u_ttt_draw_mean",
                "comparable_to_truth": 1,
                "truth": truth_corr,
                "posterior_mean": float(draw_mean),
                "posterior_median": float(draw_median) if draw_median is not None else np.nan,
                "bias_mean": float(draw_mean) - truth_corr if np.isfinite(truth_corr) else np.nan,
                "abs_error_mean": abs(float(draw_mean) - truth_corr) if np.isfinite(truth_corr) else np.nan,
                "covered_90": float(draw_q05 <= truth_corr <= draw_q95) if draw_q05 is not None and draw_q95 is not None and np.isfinite(truth_corr) else np.nan,
                "q05": float(draw_q05) if draw_q05 is not None else np.nan,
                "q95": float(draw_q95) if draw_q95 is not None else np.nan,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    scenario = _get_scenario(args.scenario)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inference_config = _build_inference_config(args)
    sim = simulate_joint_scenario(scenario, seed=int(args.seed))
    wide = sim.wide.copy()
    wide["time_m"] = wide["time"] / DAYS_PER_MONTH
    wide["treatment_time_m"] = wide["treatment_time"] / DAYS_PER_MONTH
    wide["treatment_time_obs_m"] = wide["treatment_time_obs"] / DAYS_PER_MONTH

    graph = _build_graph_for_scenario(scenario)

    surv_x_cols = list(sim.scenario.beta_surv.keys())
    ttt_x_cols = list(sim.scenario.theta_ttt.keys())

    surv_long = build_survival_long(
        wide,
        x_cols=surv_x_cols,
        surv_breaks=sim.scenario.surv_breaks,
        post_ttt_breaks=sim.scenario.post_ttt_breaks,
    )

    ttt_long = build_treatment_long(
        wide,
        x_cols=ttt_x_cols,
        ttt_breaks=sim.scenario.ttt_breaks,
    )

    surv_long.to_csv(out_dir / "surv_long.csv", index=False)
    ttt_long.to_csv(out_dir / "ttt_long.csv", index=False)

    fit = fit_joint_model(
        surv_long=surv_long,
        ttt_long=ttt_long,
        graph=graph,
        surv_x_cols=surv_x_cols,
        ttt_x_cols=ttt_x_cols,
        surv_breaks=list(sim.scenario.surv_breaks),
        ttt_breaks=list(sim.scenario.ttt_breaks),
        post_ttt_breaks=list(sim.scenario.post_ttt_breaks),
        rng_seed=int(args.seed),
        inference_config=inference_config,
        extra_fields=("diverging",),
        extra_metadata={
            "scenario_name": sim.scenario.name,
            "scenario_description": sim.scenario.description,
            "simulation_seed": int(args.seed),
        },
    )

    wide.to_csv(out_dir / "simulated_wide.csv", index=False)
    sim.parameter_truth.to_csv(out_dir / "parameter_truth.csv", index=False)
    sim.area_truth.to_csv(out_dir / "area_truth.csv", index=False)

    for name, df in sim.support_diagnostics.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    fit_dir = out_dir / "fit_bundle"
    save_joint_fit(fit, fit_dir)

    surv_summary = extract_survival_effects(fit, include_draws=False)
    ttt_summary = extract_treatment_effects(fit, include_draws=False)
    spatial_summary = extract_spatial_fields(fit, include_draws=False)
    coupling_summary = extract_joint_coupling(fit, include_draws=False)

    for name, df in surv_summary.items():
        df.to_csv(out_dir / f"survival_{name}_summary.csv", index=False)
    for name, df in ttt_summary.items():
        df.to_csv(out_dir / f"treatment_{name}_summary.csv", index=False)
    for name, df in spatial_summary.items():
        df.to_csv(out_dir / f"spatial_{name}_summary.csv", index=False)
    for name, df in coupling_summary.items():
        df.to_csv(out_dir / f"coupling_{name}_summary.csv", index=False)

    beta_truth = {
        str(row["label"]): float(row["truth"])
        for _, row in sim.parameter_truth.loc[sim.parameter_truth["group"] == "survival_beta"].iterrows()
    }
    theta_truth = {
        str(row["label"]): float(row["truth"])
        for _, row in sim.parameter_truth.loc[sim.parameter_truth["group"] == "treatment_theta"].iterrows()
    }
    alpha_truth = _vector_truth_map(sim.parameter_truth, parameter_name="alpha_surv")
    gamma_truth = _vector_truth_map(sim.parameter_truth, parameter_name="gamma_ttt")
    delta_truth = _vector_truth_map(sim.parameter_truth, parameter_name="delta_post")

    survival_beta_recovery = _make_scalar_recovery_table(
        scalar_truth=beta_truth,
        posterior_summary=surv_summary,
        df_name="beta",
    )
    treatment_theta_recovery = _make_scalar_recovery_table(
        scalar_truth=theta_truth,
        posterior_summary=ttt_summary,
        df_name="theta",
    )
    survival_alpha_recovery = _make_vector_recovery_table(
        truth_map=alpha_truth,
        summary_df=surv_summary["alpha"],
        index_col_candidates=("index", "k", "interval_index"),
        group_name="alpha_surv",
    )
    treatment_gamma_recovery = _make_vector_recovery_table(
        truth_map=gamma_truth,
        summary_df=ttt_summary["gamma"],
        index_col_candidates=("index", "k", "interval_index"),
        group_name="gamma_ttt",
    )
    survival_delta_recovery = _make_vector_recovery_table(
        truth_map=delta_truth,
        summary_df=surv_summary["delta_post"],
        index_col_candidates=("index", "k_post", "post_interval_index"),
        group_name="delta_post",
    )
    area_recovery = _make_area_recovery_table(
        area_truth=sim.area_truth,
        spatial_summary=spatial_summary,
    )
    area_field_summary = _make_area_field_summary(area_recovery)
    fit_diagnostics = _make_fit_diagnostics_table(fit)

    survival_alpha_support = _make_support_linked_recovery_table(
        recovery_df=survival_alpha_recovery,
        support_df=sim.support_diagnostics.get("survival_interval_support", pd.DataFrame()),
        recovery_index_col="index",
        support_index_col="k",
        support_source="survival_interval_support",
        support_value_cols=("count",),
    )
    treatment_gamma_support_observed = _make_support_linked_recovery_table(
        recovery_df=treatment_gamma_recovery,
        support_df=sim.support_diagnostics.get("treatment_interval_support_observed", pd.DataFrame()),
        recovery_index_col="index",
        support_index_col="k",
        support_source="treatment_interval_support_observed",
        support_value_cols=("count",),
    )
    treatment_gamma_support_true = _make_support_linked_recovery_table(
        recovery_df=treatment_gamma_recovery,
        support_df=sim.support_diagnostics.get("treatment_interval_support_true", pd.DataFrame()),
        recovery_index_col="index",
        support_index_col="k",
        support_source="treatment_interval_support_true",
        support_value_cols=("count",),
    )
    survival_delta_support = _make_support_linked_recovery_table(
        recovery_df=survival_delta_recovery,
        support_df=sim.support_diagnostics.get("post_treatment_interval_support", pd.DataFrame()),
        recovery_index_col="index",
        support_index_col="k_post",
        support_source="post_treatment_interval_support",
        support_value_cols=("subject_count",),
    )
    coupling_hyperparameter_diagnostics = _make_coupling_hyperparameter_diagnostics(
        parameter_truth=sim.parameter_truth,
        area_truth=sim.area_truth,
        spatial_summary=spatial_summary,
        coupling_summary=coupling_summary,
    )

    survival_beta_recovery.to_csv(out_dir / "recovery_survival_beta.csv", index=False)
    treatment_theta_recovery.to_csv(out_dir / "recovery_treatment_theta.csv", index=False)
    survival_alpha_recovery.to_csv(out_dir / "recovery_survival_alpha.csv", index=False)
    treatment_gamma_recovery.to_csv(out_dir / "recovery_treatment_gamma.csv", index=False)
    survival_delta_recovery.to_csv(out_dir / "recovery_survival_delta_post.csv", index=False)
    area_recovery.to_csv(out_dir / "recovery_area_fields.csv", index=False)
    area_field_summary.to_csv(out_dir / "recovery_area_field_summary.csv", index=False)
    fit_diagnostics.to_csv(out_dir / "fit_diagnostics.csv", index=False)
    survival_alpha_support.to_csv(out_dir / "recovery_survival_alpha_support.csv", index=False)
    treatment_gamma_support_observed.to_csv(out_dir / "recovery_treatment_gamma_support_observed.csv", index=False)
    treatment_gamma_support_true.to_csv(out_dir / "recovery_treatment_gamma_support_true.csv", index=False)
    survival_delta_support.to_csv(out_dir / "recovery_survival_delta_post_support.csv", index=False)
    coupling_hyperparameter_diagnostics.to_csv(out_dir / "coupling_hyperparameter_diagnostics.csv", index=False)

    manifest = {
        "scenario": sim.scenario.name,
        "seed": int(args.seed),
        "out_dir": str(out_dir),
        "fit_dir": str(fit_dir),
        "n_subjects": int(len(wide)),
        "n_areas": int(sim.scenario.n_areas),
        "surv_x_cols": list(sim.scenario.beta_surv.keys()),
        "ttt_x_cols": list(sim.scenario.theta_ttt.keys()),
        "surv_breaks": list(sim.scenario.surv_breaks),
        "ttt_breaks": list(sim.scenario.ttt_breaks),
        "post_ttt_breaks": list(sim.scenario.post_ttt_breaks),
        "inference": _json_ready(asdict(inference_config)),
    }
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "scenario.json", sim.scenario.to_dict())
    _write_json(out_dir / "simulation_metadata.json", sim.metadata)

    print(f"Joint validation seed run completed for scenario={sim.scenario.name}, seed={args.seed}")
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()