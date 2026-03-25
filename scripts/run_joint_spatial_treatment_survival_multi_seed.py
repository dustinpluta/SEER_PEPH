from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.random as random
import numpy as np
import pandas as pd

from seer_peph.data.model_data import make_model_data
from seer_peph.data.prep import build_survival_long, build_treatment_long
from seer_peph.graphs import make_ring_lattice
from seer_peph.inference.run import InferenceConfig, run_mcmc, summarise_samples
from seer_peph.models.joint_spatial_treatment_survival import (
    model as joint_spatial_treatment_survival_model,
)
from seer_peph.validation.simulate import DEFAULT_GAMMA_TTT, simulate_joint


@dataclass(frozen=True)
class MultiSeedJointSpatialRecoveryConfig:
    seeds: tuple[int, ...] = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1001)

    A: int = 20
    k: int = 4
    n_per_area: int = 200

    rho_u: float = 0.50
    phi_surv: float = 0.80
    phi_ttt: float = 0.80
    sigma_surv: float = 0.50
    sigma_ttt: float = 0.30

    gamma_ttt_true: tuple[float, ...] = tuple(DEFAULT_GAMMA_TTT.tolist())

    num_chains: int = 2
    num_warmup: int = 500
    num_samples: int = 500
    target_accept_prob: float = 0.95
    dense_mass: bool = False
    max_tree_depth: int = 10

    out_dir: str = "artifacts/joint_spatial_multi_seed_recovery"


def _encode_like_prep(wide: pd.DataFrame) -> pd.DataFrame:
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


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(x - y))))


def _extract_truth(
    wide: pd.DataFrame,
    gamma_ttt_true: tuple[float, ...],
    rho_u_true: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    first = wide.iloc[0]
    beta_td_true = float(first["beta_td_true"])

    param_truth: dict[str, float] = {
        "beta_age_per10_centered": float(first["beta_surv_age_per10_centered_true"]),
        "beta_cci": float(first["beta_surv_cci_true"]),
        "beta_tumor_size_log": float(first["beta_surv_tumor_size_log_true"]),
        "beta_stage_II": float(first["beta_surv_stage_II_true"]),
        "beta_stage_III": float(first["beta_surv_stage_III_true"]),
        "theta_age_per10_centered": float(first["theta_ttt_age_per10_centered_true"]),
        "theta_cci": float(first["theta_ttt_cci_true"]),
        "theta_ses": float(first["theta_ttt_ses_true"]),
        "theta_sex_male": float(first["theta_ttt_sex_male_true"]),
        "theta_stage_II": float(first["theta_ttt_stage_II_true"]),
        "theta_stage_III": float(first["theta_ttt_stage_III_true"]),
        "phi_surv": float(first["phi_surv_true"]),
        "sigma_surv": float(first["sigma_surv_true"]),
        "phi_ttt": float(first["phi_ttt_true"]),
        "sigma_ttt": float(first["sigma_ttt_true"]),
        "rho_u_cross": float(rho_u_true),
    }

    delta_cols = sorted(
        [c for c in first.index if c.startswith("delta_post_") and c.endswith("_true")],
        key=lambda s: int(s.split("_")[2]),
    )
    for col in delta_cols:
        j = int(col.split("_")[2])
        param_truth[f"delta_post[{j}]"] = beta_td_true + float(first[col])

    gamma_cols = sorted(
        [c for c in first.index if c.startswith("gamma_ttt_") and c.endswith("_true")],
        key=lambda s: int(s.split("_")[2]),
    )
    if gamma_cols:
        for col in gamma_cols:
            j = int(col.split("_")[2])
            param_truth[f"gamma[{j}]"] = float(first[col])
    else:
        for j, val in enumerate(gamma_ttt_true):
            param_truth[f"gamma[{j}]"] = float(val)

    area_truth = (
        wide.groupby("area_id_true", as_index=False)
        .agg(
            u_surv_true=("u_surv_true", "first"),
            u_ttt_true=("u_ttt_true", "first"),
        )
        .sort_values("area_id_true")
        .reset_index(drop=True)
    )
    return param_truth, area_truth


def _param_to_summary_name(param: str) -> str:
    mapping = {
        "beta_age_per10_centered": "beta[0]",
        "beta_cci": "beta[1]",
        "beta_tumor_size_log": "beta[2]",
        "beta_stage_II": "beta[3]",
        "beta_stage_III": "beta[4]",
        "theta_age_per10_centered": "theta[0]",
        "theta_cci": "theta[1]",
        "theta_ses": "theta[2]",
        "theta_sex_male": "theta[3]",
        "theta_stage_II": "theta[4]",
        "theta_stage_III": "theta[5]",
        "phi_surv": "rho_surv",
        "sigma_surv": "tau_surv",
        "phi_ttt": "rho_ttt",
        "sigma_ttt": "tau_ttt",
        "rho_u_cross": "rho_u_cross",
    }
    return mapping.get(param, param)


def _extract_param_estimates(summary_dict: dict[str, dict[str, float]]) -> dict[str, float]:
    out = {
        "beta_age_per10_centered": float(summary_dict["beta[0]"]["mean"]),
        "beta_cci": float(summary_dict["beta[1]"]["mean"]),
        "beta_tumor_size_log": float(summary_dict["beta[2]"]["mean"]),
        "beta_stage_II": float(summary_dict["beta[3]"]["mean"]),
        "beta_stage_III": float(summary_dict["beta[4]"]["mean"]),
        "theta_age_per10_centered": float(summary_dict["theta[0]"]["mean"]),
        "theta_cci": float(summary_dict["theta[1]"]["mean"]),
        "theta_ses": float(summary_dict["theta[2]"]["mean"]),
        "theta_sex_male": float(summary_dict["theta[3]"]["mean"]),
        "theta_stage_II": float(summary_dict["theta[4]"]["mean"]),
        "theta_stage_III": float(summary_dict["theta[5]"]["mean"]),
        "phi_surv": float(summary_dict["rho_surv"]["mean"]),
        "sigma_surv": float(summary_dict["tau_surv"]["mean"]),
        "phi_ttt": float(summary_dict["rho_ttt"]["mean"]),
        "sigma_ttt": float(summary_dict["tau_ttt"]["mean"]),
        "rho_u_cross": float(summary_dict["rho_u_cross"]["mean"]),
    }

    delta_keys = sorted(
        (k for k in summary_dict if k.startswith("delta_post[")),
        key=lambda s: int(s.split("[", 1)[1].split("]", 1)[0]),
    )
    for k in delta_keys:
        out[k] = float(summary_dict[k]["mean"])

    gamma_keys = sorted(
        (k for k in summary_dict if k.startswith("gamma[")),
        key=lambda s: int(s.split("[", 1)[1].split("]", 1)[0]),
    )
    for k in gamma_keys:
        out[k] = float(summary_dict[k]["mean"])

    return out


def _make_interval_support(seed: int, long_df: pd.DataFrame, treated_col: str | None = None) -> pd.DataFrame:
    agg_dict = {
        "rows": ("id", "size"),
        "unique_subjects": ("id", "nunique"),
        "events": ("event", "sum"),
        "exposure_pm": ("exposure", "sum"),
    }
    if treated_col is not None:
        agg_dict["treated_rows"] = (treated_col, "sum")

    out = (
        long_df.groupby("k", dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .sort_values("k")
        .reset_index(drop=True)
    )
    out["event_rate_per_100pm"] = np.where(
        out["exposure_pm"] > 0,
        100.0 * out["events"] / out["exposure_pm"],
        np.nan,
    )
    out["seed"] = seed
    return out


def _summarise_interval_support(interval_support_all: pd.DataFrame) -> pd.DataFrame:
    out = (
        interval_support_all.groupby("k", sort=True)
        .agg(
            mean_rows=("rows", "mean"),
            median_rows=("rows", "median"),
            mean_unique_subjects=("unique_subjects", "mean"),
            median_unique_subjects=("unique_subjects", "median"),
            mean_events=("events", "mean"),
            median_events=("events", "median"),
            min_events=("events", "min"),
            max_events=("events", "max"),
            mean_exposure_pm=("exposure_pm", "mean"),
            median_exposure_pm=("exposure_pm", "median"),
            min_exposure_pm=("exposure_pm", "min"),
            max_exposure_pm=("exposure_pm", "max"),
            mean_event_rate_per_100pm=("event_rate_per_100pm", "mean"),
            median_event_rate_per_100pm=("event_rate_per_100pm", "median"),
        )
        .reset_index()
        .sort_values("k")
        .reset_index(drop=True)
    )
    return out


def _one_seed_recovery(
    seed: int,
    cfg: MultiSeedJointSpatialRecoveryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    graph = make_ring_lattice(A=cfg.A, k=cfg.k)

    wide = simulate_joint(
        graph,
        n_per_area=cfg.n_per_area,
        rho_u=cfg.rho_u,
        phi_surv=cfg.phi_surv,
        phi_ttt=cfg.phi_ttt,
        sigma_surv=cfg.sigma_surv,
        sigma_ttt=cfg.sigma_ttt,
        gamma_ttt=cfg.gamma_ttt_true,
        seed=seed,
    )
    df = _encode_like_prep(wide)
    surv_long = build_survival_long(df)
    ttt_long = build_treatment_long(df)

    surv_interval_support = _make_interval_support(seed, surv_long, treated_col="treated_td")
    ttt_interval_support = _make_interval_support(seed, ttt_long)

    data = make_model_data(
        surv_long,
        ttt_long,
        graph,
        surv_x_cols=[
            "age_per10_centered",
            "cci",
            "tumor_size_log",
            "stage_II",
            "stage_III",
        ],
        ttt_x_cols=[
            "age_per10_centered",
            "cci",
            "ses",
            "sex_male",
            "stage_II",
            "stage_III",
        ],
        as_jax=True,
    )

    infer_cfg = InferenceConfig(
        num_chains=cfg.num_chains,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        target_accept_prob=cfg.target_accept_prob,
        dense_mass=cfg.dense_mass,
        max_tree_depth=cfg.max_tree_depth,
        progress_bar=True,
    )

    result = run_mcmc(
        joint_spatial_treatment_survival_model,
        data,
        rng_key=random.PRNGKey(seed + 1000),
        config=infer_cfg,
        extra_fields=("diverging",),
    )

    posterior_summary = summarise_samples(result.samples)
    param_truth, area_truth = _extract_truth(wide, cfg.gamma_ttt_true, cfg.rho_u)
    param_est = _extract_param_estimates(posterior_summary)

    param_rows: list[dict[str, Any]] = []
    missing_params: list[str] = []

    for param, true_val in param_truth.items():
        if param not in param_est:
            missing_params.append(param)
            continue

        est_val = float(param_est[param])
        summary_name = _param_to_summary_name(param)

        if summary_name in posterior_summary:
            q05 = float(posterior_summary[summary_name]["q05"])
            q95 = float(posterior_summary[summary_name]["q95"])
            covered = int(q05 <= true_val <= q95)
        else:
            q05 = np.nan
            q95 = np.nan
            covered = np.nan

        param_rows.append(
            {
                "seed": seed,
                "parameter": param,
                "summary_name": summary_name,
                "truth": true_val,
                "posterior_mean": est_val,
                "q05": q05,
                "q95": q95,
                "error": est_val - true_val,
                "abs_error": abs(est_val - true_val),
                "covered_by_q05_q95": covered,
            }
        )

    param_df = pd.DataFrame(param_rows)
    if missing_params:
        pd.DataFrame({"seed": seed, "missing_parameter": missing_params}).to_csv(
            Path(cfg.out_dir) / f"missing_parameter_truths_seed_{seed}.csv",
            index=False,
        )

    u_surv_post_mean = np.asarray(result.samples["u_surv"]).mean(axis=0)
    u_ttt_post_mean = np.asarray(result.samples["u_ttt"]).mean(axis=0)
    u_ttt_ind_post_mean = np.asarray(result.samples["u_ttt_ind"]).mean(axis=0)
    s_surv_post_mean = np.asarray(result.samples["s_surv"]).mean(axis=0)
    s_ttt_post_mean = np.asarray(result.samples["s_ttt"]).mean(axis=0)

    spatial_df = pd.DataFrame(
        {
            "seed": seed,
            "area_id_true": np.arange(graph.A, dtype=int),
            "u_surv_true": area_truth["u_surv_true"].to_numpy(dtype=float),
            "u_ttt_true": area_truth["u_ttt_true"].to_numpy(dtype=float),
            "u_surv_post_mean": u_surv_post_mean,
            "u_ttt_post_mean": u_ttt_post_mean,
            "u_ttt_ind_post_mean": u_ttt_ind_post_mean,
            "s_surv_post_mean": s_surv_post_mean,
            "s_ttt_post_mean": s_ttt_post_mean,
        }
    )
    spatial_df["u_surv_error"] = spatial_df["u_surv_post_mean"] - spatial_df["u_surv_true"]
    spatial_df["u_ttt_error"] = spatial_df["u_ttt_post_mean"] - spatial_df["u_ttt_true"]
    spatial_df["u_surv_abs_error"] = np.abs(spatial_df["u_surv_error"])
    spatial_df["u_ttt_abs_error"] = np.abs(spatial_df["u_ttt_error"])

    divergences = int(np.asarray(result.mcmc.get_extra_fields()["diverging"]).sum())

    beta_rows = param_df[param_df["parameter"].str.startswith("beta_")].copy()
    theta_rows = param_df[param_df["parameter"].str.startswith("theta_")].copy()
    delta_rows = param_df[param_df["parameter"].str.startswith("delta_post[")].copy()
    gamma_rows = param_df[param_df["parameter"].str.startswith("gamma[")].copy()
    hyper_rows = param_df[param_df["parameter"].isin(
        ["phi_surv", "sigma_surv", "phi_ttt", "sigma_ttt", "rho_u_cross"]
    )].copy()

    def _block_stats(df: pd.DataFrame, label: str) -> dict[str, float]:
        if len(df) == 0:
            return {
                f"{label}_mae": np.nan,
                f"{label}_rmse": np.nan,
                f"{label}_90pct_coverage": np.nan,
            }
        return {
            f"{label}_mae": float(df["abs_error"].mean()),
            f"{label}_rmse": float(np.sqrt(np.mean(np.square(df["error"])))),
            f"{label}_90pct_coverage": float(100.0 * df["covered_by_q05_q95"].mean()),
        }

    seed_meta: dict[str, Any] = {
        "seed": seed,
        "n_surv_rows": int(len(surv_long)),
        "n_ttt_rows": int(len(ttt_long)),
        "n_surv_events": int(surv_long["event"].sum()),
        "n_ttt_events": int(ttt_long["event"].sum()),
        "divergences": divergences,
        "overall_param_mae": float(param_df["abs_error"].mean()) if len(param_df) > 0 else np.nan,
        "overall_param_rmse": float(np.sqrt(np.mean(np.square(param_df["error"])))) if len(param_df) > 0 else np.nan,
        "corr_u_surv_true_vs_post_mean": _corr(spatial_df["u_surv_true"], spatial_df["u_surv_post_mean"]),
        "rmse_u_surv_true_vs_post_mean": _rmse(spatial_df["u_surv_true"], spatial_df["u_surv_post_mean"]),
        "corr_u_surv_true_vs_s_surv_post_mean": _corr(spatial_df["u_surv_true"], spatial_df["s_surv_post_mean"]),
        "corr_u_ttt_true_vs_post_mean": _corr(spatial_df["u_ttt_true"], spatial_df["u_ttt_post_mean"]),
        "rmse_u_ttt_true_vs_post_mean": _rmse(spatial_df["u_ttt_true"], spatial_df["u_ttt_post_mean"]),
        "corr_u_ttt_true_vs_s_ttt_post_mean": _corr(spatial_df["u_ttt_true"], spatial_df["s_ttt_post_mean"]),
        "corr_u_surv_post_mean_vs_u_ttt_post_mean": _corr(
            spatial_df["u_surv_post_mean"], spatial_df["u_ttt_post_mean"]
        ),
        "corr_u_surv_true_vs_u_ttt_post_mean": _corr(
            spatial_df["u_surv_true"], spatial_df["u_ttt_post_mean"]
        ),
        "corr_u_ttt_true_vs_u_surv_post_mean": _corr(
            spatial_df["u_ttt_true"], spatial_df["u_surv_post_mean"]
        ),
    }
    seed_meta.update(_block_stats(beta_rows, "beta"))
    seed_meta.update(_block_stats(theta_rows, "theta"))
    seed_meta.update(_block_stats(delta_rows, "delta_post"))
    seed_meta.update(_block_stats(gamma_rows, "gamma"))
    seed_meta.update(_block_stats(hyper_rows, "hyper"))

    return param_df, spatial_df, seed_meta, surv_interval_support, ttt_interval_support


def _aggregate_param_results(param_results: pd.DataFrame) -> pd.DataFrame:
    grouped = param_results.groupby(["parameter", "summary_name"], sort=False)

    summary = grouped.agg(
        truth=("truth", "first"),
        mean_posterior_mean=("posterior_mean", "mean"),
        sd_posterior_mean=("posterior_mean", "std"),
        mean_error=("error", "mean"),
        median_error=("error", "median"),
        rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        mae=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
        coverage_q05_q95=("covered_by_q05_q95", "mean"),
        mean_q05=("q05", "mean"),
        mean_q95=("q95", "mean"),
    ).reset_index()

    summary["coverage_q05_q95"] = 100.0 * summary["coverage_q05_q95"]
    return summary


def _aggregate_param_blocks(agg: pd.DataFrame) -> pd.DataFrame:
    def block_of(param: str) -> str:
        if param.startswith("beta_"):
            return "beta"
        if param.startswith("theta_"):
            return "theta"
        if param.startswith("delta_post["):
            return "delta_post"
        if param.startswith("gamma["):
            return "gamma"
        if param in {"phi_surv", "sigma_surv", "phi_ttt", "sigma_ttt", "rho_u_cross"}:
            return "hyper"
        return "other"

    tmp = agg.copy()
    tmp["block"] = tmp["parameter"].map(block_of)

    out = tmp.groupby("block", sort=False).agg(
        n_parameters=("parameter", "size"),
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_abs_bias=("mean_error", lambda x: float(np.mean(np.abs(x)))),
        mean_coverage_q05_q95=("coverage_q05_q95", "mean"),
    ).reset_index()

    return out


def _aggregate_seed_meta(seed_meta_df: pd.DataFrame) -> dict[str, float]:
    numeric_cols = [
        c for c in seed_meta_df.columns
        if c != "seed" and pd.api.types.is_numeric_dtype(seed_meta_df[c])
    ]
    out: dict[str, float] = {}
    for col in numeric_cols:
        if col == "divergences":
            out["mean_divergences"] = float(seed_meta_df[col].mean())
            out["total_divergences"] = int(seed_meta_df[col].sum())
        else:
            out[f"mean_{col}"] = float(seed_meta_df[col].mean())
            out[f"median_{col}"] = float(seed_meta_df[col].median())
    return out


def main(config: MultiSeedJointSpatialRecoveryConfig | None = None) -> None:
    cfg = MultiSeedJointSpatialRecoveryConfig() if config is None else config
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_param_rows: list[pd.DataFrame] = []
    all_spatial_rows: list[pd.DataFrame] = []
    all_surv_interval_rows: list[pd.DataFrame] = []
    all_ttt_interval_rows: list[pd.DataFrame] = []
    seed_meta_rows: list[dict[str, Any]] = []

    print("\nRunning multi-seed joint spatial recovery study")
    print("===============================================")
    print(f"Seeds: {list(cfg.seeds)}")
    print(f"Output directory: {out_dir}")

    for i, seed in enumerate(cfg.seeds, start=1):
        print(f"\n[{i}/{len(cfg.seeds)}] Seed {seed}")

        param_df, spatial_df, seed_meta, surv_interval_support, ttt_interval_support = _one_seed_recovery(seed, cfg)

        all_param_rows.append(param_df)
        all_spatial_rows.append(spatial_df)
        all_surv_interval_rows.append(surv_interval_support)
        all_ttt_interval_rows.append(ttt_interval_support)
        seed_meta_rows.append(seed_meta)

        print(
            f"  divergences={seed_meta['divergences']}, "
            f"surv_events={seed_meta['n_surv_events']}, "
            f"ttt_events={seed_meta['n_ttt_events']}, "
            f"beta_mae={seed_meta['beta_mae']:.4f}, "
            f"theta_mae={seed_meta['theta_mae']:.4f}, "
            f"delta_mae={seed_meta['delta_post_mae']:.4f}, "
            f"gamma_mae={seed_meta['gamma_mae']:.4f}, "
            f"hyper_mae={seed_meta['hyper_mae']:.4f}, "
            f"corr_u_surv={seed_meta['corr_u_surv_true_vs_post_mean']:.4f}, "
            f"corr_u_ttt={seed_meta['corr_u_ttt_true_vs_post_mean']:.4f}"
        )

    param_results = pd.concat(all_param_rows, ignore_index=True)
    spatial_results = pd.concat(all_spatial_rows, ignore_index=True)
    surv_interval_support_all = pd.concat(all_surv_interval_rows, ignore_index=True)
    ttt_interval_support_all = pd.concat(all_ttt_interval_rows, ignore_index=True)
    seed_meta_df = pd.DataFrame(seed_meta_rows)

    param_summary = _aggregate_param_results(param_results)
    block_summary = _aggregate_param_blocks(param_summary)
    seed_summary = _aggregate_seed_meta(seed_meta_df)
    surv_interval_summary = _summarise_interval_support(surv_interval_support_all)
    ttt_interval_summary = _summarise_interval_support(ttt_interval_support_all)

    param_results.to_csv(out_dir / "parameter_recovery_by_seed.csv", index=False)
    param_summary.to_csv(out_dir / "parameter_recovery_summary.csv", index=False)
    block_summary.to_csv(out_dir / "parameter_recovery_block_summary.csv", index=False)
    spatial_results.to_csv(out_dir / "spatial_recovery_by_seed_area.csv", index=False)
    seed_meta_df.to_csv(out_dir / "seed_level_summary.csv", index=False)
    surv_interval_support_all.to_csv(out_dir / "survival_interval_support_by_seed.csv", index=False)
    surv_interval_summary.to_csv(out_dir / "survival_interval_support_summary.csv", index=False)
    ttt_interval_support_all.to_csv(out_dir / "treatment_interval_support_by_seed.csv", index=False)
    ttt_interval_summary.to_csv(out_dir / "treatment_interval_support_summary.csv", index=False)

    with open(out_dir / "seed_level_summary_aggregates.json", "w", encoding="utf-8") as f:
        json.dump(seed_summary, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("\nParameter-level average recovery")
    print("-------------------------------")
    print(param_summary.to_string(index=False))

    print("\nParameter block summary")
    print("-----------------------")
    print(block_summary.to_string(index=False))

    print("\nSeed-level summary")
    print("------------------")
    print(seed_meta_df.to_string(index=False))

    print("\nSurvival interval support summary")
    print("---------------------------------")
    print(surv_interval_summary.to_string(index=False))

    print("\nTreatment interval support summary")
    print("----------------------------------")
    print(ttt_interval_summary.to_string(index=False))

    print("\nAggregate seed-level summary")
    print("----------------------------")
    for k, v in seed_summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\nOverall")
    print("-------")
    print(f"Overall parameter MAE: {param_results['abs_error'].mean():.4f}")
    print(f"Overall parameter RMSE: {np.sqrt(np.mean(np.square(param_results['error']))):.4f}")
    print(f"Mean corr(u_surv_true, u_surv_post_mean): {seed_meta_df['corr_u_surv_true_vs_post_mean'].mean():.4f}")
    print(f"Mean corr(u_ttt_true, u_ttt_post_mean): {seed_meta_df['corr_u_ttt_true_vs_post_mean'].mean():.4f}")
    print(f"Mean RMSE(u_surv_true, u_surv_post_mean): {seed_meta_df['rmse_u_surv_true_vs_post_mean'].mean():.4f}")
    print(f"Mean RMSE(u_ttt_true, u_ttt_post_mean): {seed_meta_df['rmse_u_ttt_true_vs_post_mean'].mean():.4f}")
    print(f"Mean rho_u_cross MAE: {param_results.loc[param_results['parameter'] == 'rho_u_cross', 'abs_error'].mean():.4f}")


if __name__ == "__main__":
    main()