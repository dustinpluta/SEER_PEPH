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
from seer_peph.models.treatment_spatial_pe import model as treatment_spatial_pe_model
from seer_peph.validation.simulate import DEFAULT_GAMMA_TTT, simulate_joint


@dataclass(frozen=True)
class MultiSeedTreatmentSpatialPERecoveryConfig:
    seeds: tuple[int, ...] = (101, 202, 303, 404, 505)

    A: int = 20
    k: int = 4
    n_per_area: int = 200

    rho_u: float = 0.5
    phi_surv: float = 0.8
    phi_ttt: float = 0.8
    sigma_surv: float = 0.5
    sigma_ttt: float = 0.30

    gamma_ttt_true: tuple[float, ...] = tuple(DEFAULT_GAMMA_TTT.tolist())

    num_chains: int = 2
    num_warmup: int = 500
    num_samples: int = 500
    target_accept_prob: float = 0.95
    dense_mass: bool = False
    max_tree_depth: int = 10

    out_dir: str = "artifacts/treatment_spatial_pe_multi_seed_recovery"


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


def _extract_truth(
    wide: pd.DataFrame,
    gamma_ttt_true: tuple[float, ...],
) -> tuple[dict[str, float], pd.DataFrame]:
    first = wide.iloc[0]

    param_truth = {
        "theta_age_per10_centered": float(first["theta_ttt_age_per10_centered_true"]),
        "theta_cci": float(first["theta_ttt_cci_true"]),
        "theta_ses": float(first["theta_ttt_ses_true"]),
        "theta_sex_male": float(first["theta_ttt_sex_male_true"]),
        "theta_stage_II": float(first["theta_ttt_stage_II_true"]),
        "theta_stage_III": float(first["theta_ttt_stage_III_true"]),
        "phi_ttt": float(first["phi_ttt_true"]),
        "sigma_ttt": float(first["sigma_ttt_true"]),
    }

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
        "theta_age_per10_centered": "theta[0]",
        "theta_cci": "theta[1]",
        "theta_ses": "theta[2]",
        "theta_sex_male": "theta[3]",
        "theta_stage_II": "theta[4]",
        "theta_stage_III": "theta[5]",
        "phi_ttt": "rho",
        "sigma_ttt": "tau",
    }
    return mapping.get(param, param)


def _extract_param_estimates(summary_dict: dict[str, dict[str, float]]) -> dict[str, float]:
    out = {
        "theta_age_per10_centered": summary_dict["theta[0]"]["mean"],
        "theta_cci": summary_dict["theta[1]"]["mean"],
        "theta_ses": summary_dict["theta[2]"]["mean"],
        "theta_sex_male": summary_dict["theta[3]"]["mean"],
        "theta_stage_II": summary_dict["theta[4]"]["mean"],
        "theta_stage_III": summary_dict["theta[5]"]["mean"],
        "phi_ttt": summary_dict["rho"]["mean"],
        "sigma_ttt": summary_dict["tau"]["mean"],
    }

    gamma_keys = sorted(
        (k for k in summary_dict if k.startswith("gamma[")),
        key=lambda s: int(s.split("[", 1)[1].split("]", 1)[0]),
    )
    for k in gamma_keys:
        out[k] = summary_dict[k]["mean"]

    return out


def _make_interval_support(seed: int, ttt_long: pd.DataFrame) -> pd.DataFrame:
    interval_support = (
        ttt_long.groupby("k", dropna=False)
        .agg(
            rows=("id", "size"),
            unique_subjects=("id", "nunique"),
            events=("event", "sum"),
            exposure_pm=("exposure", "sum"),
        )
        .reset_index()
        .sort_values("k")
        .reset_index(drop=True)
    )
    interval_support["event_rate_per_100pm"] = np.where(
        interval_support["exposure_pm"] > 0,
        100.0 * interval_support["events"] / interval_support["exposure_pm"],
        np.nan,
    )
    interval_support["seed"] = seed
    return interval_support


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
    cfg: MultiSeedTreatmentSpatialPERecoveryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
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

    interval_support = _make_interval_support(seed, ttt_long)

    data = make_model_data(
        surv_long,
        ttt_long,
        graph,
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
        treatment_spatial_pe_model,
        data,
        rng_key=random.PRNGKey(seed + 10000),
        config=infer_cfg,
        extra_fields=("diverging",),
    )

    posterior_summary = summarise_samples(result.samples)
    param_truth, area_truth = _extract_truth(wide, cfg.gamma_ttt_true)
    param_est = _extract_param_estimates(posterior_summary)

    param_rows: list[dict[str, Any]] = []
    for param, true_val in param_truth.items():
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

    u_post_mean = np.asarray(result.samples["u"]).mean(axis=0)
    s_post_mean = np.asarray(result.samples["s"]).mean(axis=0)

    spatial_df = pd.DataFrame(
        {
            "seed": seed,
            "area_id_true": np.arange(graph.A, dtype=int),
            "u_surv_true": area_truth["u_surv_true"].to_numpy(dtype=float),
            "u_ttt_true": area_truth["u_ttt_true"].to_numpy(dtype=float),
            "u_ttt_post_mean": u_post_mean,
            "s_post_mean": s_post_mean,
        }
    )
    spatial_df["u_error"] = spatial_df["u_ttt_post_mean"] - spatial_df["u_ttt_true"]
    spatial_df["u_abs_error"] = np.abs(spatial_df["u_error"])

    divergences = int(np.asarray(result.mcmc.get_extra_fields()["diverging"]).sum())

    theta_rows = param_df[param_df["parameter"].str.startswith("theta_")].copy()
    gamma_rows = param_df[param_df["parameter"].str.startswith("gamma[")].copy()

    seed_meta = {
        "seed": seed,
        "n_ttt_rows": int(len(ttt_long)),
        "n_ttt_events": int(ttt_long["event"].sum()),
        "divergences": divergences,
        "theta_mae": float(theta_rows["abs_error"].mean()),
        "theta_rmse": float(np.sqrt(np.mean(np.square(theta_rows["error"])))),
        "theta_90pct_coverage": float(100.0 * theta_rows["covered_by_q05_q95"].mean()),
        "gamma_mae": float(gamma_rows["abs_error"].mean()),
        "gamma_rmse": float(np.sqrt(np.mean(np.square(gamma_rows["error"])))),
        "gamma_90pct_coverage": float(100.0 * gamma_rows["covered_by_q05_q95"].mean()),
        "overall_param_mae": float(param_df["abs_error"].mean()),
        "overall_param_rmse": float(np.sqrt(np.mean(np.square(param_df["error"])))),
        "corr_u_true_vs_post_mean": _corr(spatial_df["u_ttt_true"], spatial_df["u_ttt_post_mean"]),
        "rmse_u_true_vs_post_mean": float(np.sqrt(np.mean(np.square(spatial_df["u_error"])))),
        "corr_u_true_vs_s_post_mean": _corr(spatial_df["u_ttt_true"], spatial_df["s_post_mean"]),
        "corr_u_surv_true_vs_u_ttt_post_mean": _corr(spatial_df["u_surv_true"], spatial_df["u_ttt_post_mean"]),
    }

    return param_df, spatial_df, seed_meta, interval_support


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
        if param.startswith("theta_"):
            return "theta"
        if param.startswith("gamma["):
            return "gamma"
        if param in {"phi_ttt", "sigma_ttt"}:
            return "spatial_hyper"
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


def _aggregate_spatial_seed_meta(seed_meta_df: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_divergences": float(seed_meta_df["divergences"].mean()),
        "total_divergences": int(seed_meta_df["divergences"].sum()),
        "mean_n_ttt_events": float(seed_meta_df["n_ttt_events"].mean()),
        "mean_theta_mae": float(seed_meta_df["theta_mae"].mean()),
        "mean_gamma_mae": float(seed_meta_df["gamma_mae"].mean()),
        "mean_overall_param_mae": float(seed_meta_df["overall_param_mae"].mean()),
        "mean_theta_90pct_coverage": float(seed_meta_df["theta_90pct_coverage"].mean()),
        "mean_gamma_90pct_coverage": float(seed_meta_df["gamma_90pct_coverage"].mean()),
        "mean_corr_u_true_vs_post_mean": float(seed_meta_df["corr_u_true_vs_post_mean"].mean()),
        "median_corr_u_true_vs_post_mean": float(seed_meta_df["corr_u_true_vs_post_mean"].median()),
        "mean_rmse_u_true_vs_post_mean": float(seed_meta_df["rmse_u_true_vs_post_mean"].mean()),
        "median_rmse_u_true_vs_post_mean": float(seed_meta_df["rmse_u_true_vs_post_mean"].median()),
        "mean_corr_u_true_vs_s_post_mean": float(seed_meta_df["corr_u_true_vs_s_post_mean"].mean()),
        "mean_corr_u_surv_true_vs_u_ttt_post_mean": float(seed_meta_df["corr_u_surv_true_vs_u_ttt_post_mean"].mean()),
    }


def main(config: MultiSeedTreatmentSpatialPERecoveryConfig | None = None) -> None:
    cfg = MultiSeedTreatmentSpatialPERecoveryConfig() if config is None else config
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_param_rows: list[pd.DataFrame] = []
    all_spatial_rows: list[pd.DataFrame] = []
    all_interval_support_rows: list[pd.DataFrame] = []
    seed_meta_rows: list[dict[str, Any]] = []

    print("\nRunning multi-seed treatment_spatial_pe recovery study")
    print("======================================================")
    print(f"Seeds: {list(cfg.seeds)}")
    print(f"Output directory: {out_dir}")

    for i, seed in enumerate(cfg.seeds, start=1):
        print(f"\n[{i}/{len(cfg.seeds)}] Seed {seed}")
        param_df, spatial_df, seed_meta, interval_support = _one_seed_recovery(seed, cfg)

        all_param_rows.append(param_df)
        all_spatial_rows.append(spatial_df)
        all_interval_support_rows.append(interval_support)
        seed_meta_rows.append(seed_meta)

        tail_support = interval_support[interval_support["k"] >= 7].copy()
        tail_events = float(tail_support["events"].sum()) if len(tail_support) > 0 else 0.0
        tail_exposure = float(tail_support["exposure_pm"].sum()) if len(tail_support) > 0 else 0.0

        print(
            f"  divergences={seed_meta['divergences']}, "
            f"ttt_events={seed_meta['n_ttt_events']}, "
            f"theta_mae={seed_meta['theta_mae']:.4f}, "
            f"gamma_mae={seed_meta['gamma_mae']:.4f}, "
            f"theta_cov90={seed_meta['theta_90pct_coverage']:.1f}, "
            f"gamma_cov90={seed_meta['gamma_90pct_coverage']:.1f}, "
            f"corr_u={seed_meta['corr_u_true_vs_post_mean']:.4f}, "
            f"tail_events={tail_events:.0f}, "
            f"tail_exposure={tail_exposure:.2f}"
        )

    param_results = pd.concat(all_param_rows, ignore_index=True)
    spatial_results = pd.concat(all_spatial_rows, ignore_index=True)
    interval_support_all = pd.concat(all_interval_support_rows, ignore_index=True)
    interval_support_summary = _summarise_interval_support(interval_support_all)
    seed_meta_df = pd.DataFrame(seed_meta_rows)

    param_summary = _aggregate_param_results(param_results)
    block_summary = _aggregate_param_blocks(param_summary)
    spatial_summary = _aggregate_spatial_seed_meta(seed_meta_df)

    param_results.to_csv(out_dir / "parameter_recovery_by_seed.csv", index=False)
    param_summary.to_csv(out_dir / "parameter_recovery_summary.csv", index=False)
    block_summary.to_csv(out_dir / "parameter_recovery_block_summary.csv", index=False)
    spatial_results.to_csv(out_dir / "spatial_recovery_by_seed_area.csv", index=False)
    seed_meta_df.to_csv(out_dir / "seed_level_spatial_summary.csv", index=False)
    interval_support_all.to_csv(out_dir / "treatment_interval_support_by_seed.csv", index=False)
    interval_support_summary.to_csv(out_dir / "treatment_interval_support_summary.csv", index=False)

    with open(out_dir / "spatial_recovery_summary.json", "w", encoding="utf-8") as f:
        json.dump(spatial_summary, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("\nParameter-level average recovery")
    print("-------------------------------")
    print(param_summary.to_string(index=False))

    print("\nParameter block summary")
    print("-----------------------")
    print(block_summary.to_string(index=False))

    print("\nSeed-level spatial summary")
    print("--------------------------")
    print(seed_meta_df.to_string(index=False))

    print("\nTreatment interval support summary")
    print("---------------------------------")
    print(interval_support_summary.to_string(index=False))

    print("\nSpatial recovery summary")
    print("------------------------")
    for k, v in spatial_summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\nOverall")
    print("-------")
    print(f"Overall parameter MAE: {param_results['abs_error'].mean():.4f}")
    print(f"Overall parameter RMSE: {np.sqrt(np.mean(np.square(param_results['error']))):.4f}")
    print(f"Mean theta MAE: {seed_meta_df['theta_mae'].mean():.4f}")
    print(f"Mean gamma MAE: {seed_meta_df['gamma_mae'].mean():.4f}")
    print(f"Mean theta 90% coverage: {seed_meta_df['theta_90pct_coverage'].mean():.2f}")
    print(f"Mean gamma 90% coverage: {seed_meta_df['gamma_90pct_coverage'].mean():.2f}")
    print(f"Mean spatial corr(u_true, u_post_mean): {seed_meta_df['corr_u_true_vs_post_mean'].mean():.4f}")
    print(f"Mean spatial RMSE(u_true, u_post_mean): {seed_meta_df['rmse_u_true_vs_post_mean'].mean():.4f}")


if __name__ == "__main__":
    main()