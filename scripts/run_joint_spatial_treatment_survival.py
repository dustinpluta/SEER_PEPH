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
class JointSpatialRecoveryConfig:
    seed: int = 123

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

    out_dir: str = "artifacts/joint_spatial_recovery"


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
        "rho_u_cross": float(first["rho_true"]) if "rho_true" in first.index else float(first["treatment_rho_true"]),
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


def main(config: JointSpatialRecoveryConfig | None = None) -> None:
    cfg = JointSpatialRecoveryConfig() if config is None else config
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        seed=cfg.seed,
    )
    wide.to_csv(out_dir / "simulated_wide.csv", index=False)

    df = _encode_like_prep(wide)
    surv_long = build_survival_long(df)
    ttt_long = build_treatment_long(df)

    surv_long.to_csv(out_dir / "surv_long.csv", index=False)
    ttt_long.to_csv(out_dir / "ttt_long.csv", index=False)

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
        rng_key=random.PRNGKey(cfg.seed + 1000),
        config=infer_cfg,
        extra_fields=("diverging",),
    )

    posterior_summary = summarise_samples(result.samples)
    param_truth, area_truth = _extract_truth(wide, cfg.gamma_ttt_true)
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

    param_recovery = pd.DataFrame(param_rows).sort_values("parameter").reset_index(drop=True)
    param_recovery.to_csv(out_dir / "parameter_recovery_table.csv", index=False)

    if missing_params:
        pd.DataFrame({"missing_parameter": missing_params}).to_csv(
            out_dir / "missing_parameter_truths.csv", index=False
        )

    u_surv_post_mean = np.asarray(result.samples["u_surv"]).mean(axis=0)
    u_ttt_post_mean = np.asarray(result.samples["u_ttt"]).mean(axis=0)
    u_ttt_ind_post_mean = np.asarray(result.samples["u_ttt_ind"]).mean(axis=0)
    s_surv_post_mean = np.asarray(result.samples["s_surv"]).mean(axis=0)
    s_ttt_post_mean = np.asarray(result.samples["s_ttt"]).mean(axis=0)

    spatial_df = pd.DataFrame(
        {
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
    spatial_df.to_csv(out_dir / "spatial_recovery_by_area.csv", index=False)

    surv_interval_diag = (
        surv_long.groupby("k", dropna=False)
        .agg(
            rows=("id", "size"),
            unique_subjects=("id", "nunique"),
            events=("event", "sum"),
            exposure_pm=("exposure", "sum"),
            treated_rows=("treated_td", "sum"),
        )
        .reset_index()
        .sort_values("k")
        .reset_index(drop=True)
    )
    surv_interval_diag["event_rate_per_100pm"] = np.where(
        surv_interval_diag["exposure_pm"] > 0,
        100.0 * surv_interval_diag["events"] / surv_interval_diag["exposure_pm"],
        np.nan,
    )
    surv_interval_diag.to_csv(out_dir / "survival_interval_diagnostics.csv", index=False)

    ttt_interval_diag = (
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
    ttt_interval_diag["event_rate_per_100pm"] = np.where(
        ttt_interval_diag["exposure_pm"] > 0,
        100.0 * ttt_interval_diag["events"] / ttt_interval_diag["exposure_pm"],
        np.nan,
    )
    ttt_interval_diag.to_csv(out_dir / "treatment_interval_diagnostics.csv", index=False)

    spatial_metrics = {
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

    def _block_mask(df: pd.DataFrame, prefix: str) -> pd.Series:
        return df["parameter"].str.startswith(prefix)

    beta_rows = param_recovery[_block_mask(param_recovery, "beta_")].copy()
    theta_rows = param_recovery[_block_mask(param_recovery, "theta_")].copy()
    delta_rows = param_recovery[_block_mask(param_recovery, "delta_post[")].copy()
    gamma_rows = param_recovery[_block_mask(param_recovery, "gamma[")].copy()
    hyper_rows = param_recovery[param_recovery["parameter"].isin(
        ["phi_surv", "sigma_surv", "phi_ttt", "sigma_ttt", "rho_u_cross"]
    )].copy()

    def _block_metrics(df: pd.DataFrame, label: str) -> dict[str, float]:
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

    aggregate_metrics: dict[str, float] = {
        "overall_mae": float(param_recovery["abs_error"].mean()) if len(param_recovery) > 0 else np.nan,
        "overall_rmse": float(np.sqrt(np.mean(np.square(param_recovery["error"])))) if len(param_recovery) > 0 else np.nan,
    }
    aggregate_metrics.update(_block_metrics(beta_rows, "beta"))
    aggregate_metrics.update(_block_metrics(theta_rows, "theta"))
    aggregate_metrics.update(_block_metrics(delta_rows, "delta_post"))
    aggregate_metrics.update(_block_metrics(gamma_rows, "gamma"))
    aggregate_metrics.update(_block_metrics(hyper_rows, "hyper"))

    divergences = int(np.asarray(result.mcmc.get_extra_fields()["diverging"]).sum())

    with open(out_dir / "posterior_summary.json", "w", encoding="utf-8") as f:
        json.dump(posterior_summary, f, indent=2)
    with open(out_dir / "spatial_recovery_metrics.json", "w", encoding="utf-8") as f:
        json.dump(spatial_metrics, f, indent=2)
    with open(out_dir / "aggregate_recovery_metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("\nJoint spatial treatment-survival recovery check")
    print("------------------------------------------------")
    print(f"Rows in survival long data:  {len(surv_long):,}")
    print(f"Rows in treatment long data: {len(ttt_long):,}")
    print(f"Observed survival events:    {int(surv_long['event'].sum()):,}")
    print(f"Observed treatment events:   {int(ttt_long['event'].sum()):,}")
    print(f"Divergences:                 {divergences}")

    print("\nParameter recovery")
    print("------------------")
    print(param_recovery.to_string(index=False))

    print("\nSurvival interval diagnostics")
    print("-----------------------------")
    print(surv_interval_diag.to_string(index=False))

    print("\nTreatment interval diagnostics")
    print("------------------------------")
    print(ttt_interval_diag.to_string(index=False))

    print("\nSpatial recovery metrics")
    print("------------------------")
    for k, v in spatial_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nAggregate error metrics")
    print("-----------------------")
    for k, v in aggregate_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()