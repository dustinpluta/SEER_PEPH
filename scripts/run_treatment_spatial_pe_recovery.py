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
class TreatmentSpatialPERecoveryConfig:
    seed: int = 123

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

    out_dir: str = "artifacts/treatment_spatial_pe_recovery"


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


def _extract_param_estimates(summary_dict: dict[str, dict[str, float]]) -> dict[str, float]:
    out = {
        "theta_age_per10_centered": summary_dict["theta[0]"]["mean"],
        "theta_cci": summary_dict["theta[1]"]["mean"],
        "theta_ses": summary_dict["theta[2]"]["mean"],
        "theta_sex_male": summary_dict["theta[3]"]["mean"],
        "theta_stage_II": summary_dict["theta[4]"]["mean"],
        "theta_stage_III": summary_dict["theta[5]"]["mean"],
        "phi_ttt_proxy_rho": summary_dict["rho"]["mean"],
        "sigma_ttt_proxy_tau": summary_dict["tau"]["mean"],
    }

    gamma_keys = sorted(
        (k for k in summary_dict if k.startswith("gamma[")),
        key=lambda s: int(s.split("[", 1)[1].split("]", 1)[0]),
    )
    for k in gamma_keys:
        out[k] = summary_dict[k]["mean"]

    return out


def main(config: TreatmentSpatialPERecoveryConfig | None = None) -> None:
    cfg = TreatmentSpatialPERecoveryConfig() if config is None else config
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
        rng_key=random.PRNGKey(cfg.seed + 1000),
        config=infer_cfg,
        extra_fields=("diverging",),
    )

    posterior_summary = summarise_samples(result.samples)
    param_truth, area_truth = _extract_truth(wide, cfg.gamma_ttt_true)
    param_est = _extract_param_estimates(posterior_summary)

    name_map = {
        "phi_ttt": "phi_ttt_proxy_rho",
        "sigma_ttt": "sigma_ttt_proxy_tau",
    }

    param_rows: list[dict[str, Any]] = []
    for param, true_val in param_truth.items():
        est_name = name_map.get(param, param)
        est_val = float(param_est[est_name])

        if est_name in posterior_summary:
            q05 = float(posterior_summary[est_name]["q05"])
            q95 = float(posterior_summary[est_name]["q95"])
            covered = int(q05 <= true_val <= q95)
        else:
            q05 = np.nan
            q95 = np.nan
            covered = np.nan

        param_rows.append(
            {
                "parameter": param,
                "estimate_name": est_name,
                "truth": true_val,
                "posterior_mean": est_val,
                "q05": q05,
                "q95": q95,
                "error": est_val - true_val,
                "abs_error": abs(est_val - true_val),
                "covered_by_q05_q95": covered,
            }
        )
    param_recovery = pd.DataFrame(param_rows)
    param_recovery.to_csv(out_dir / "parameter_recovery_table.csv", index=False)

    u_post_mean = np.asarray(result.samples["u"]).mean(axis=0)
    s_post_mean = np.asarray(result.samples["s"]).mean(axis=0)

    spatial_df = pd.DataFrame(
        {
            "area_id_true": np.arange(graph.A, dtype=int),
            "u_surv_true": area_truth["u_surv_true"].to_numpy(dtype=float),
            "u_ttt_true": area_truth["u_ttt_true"].to_numpy(dtype=float),
            "u_ttt_post_mean": u_post_mean,
            "s_post_mean": s_post_mean,
        }
    )
    spatial_df["u_error"] = spatial_df["u_ttt_post_mean"] - spatial_df["u_ttt_true"]
    spatial_df["u_abs_error"] = np.abs(spatial_df["u_error"])
    spatial_df.to_csv(out_dir / "spatial_recovery_by_area.csv", index=False)

    interval_diag = (
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
    if len(interval_diag) > 0:
        interval_diag["event_rate_per_100pm"] = 100.0 * interval_diag["events"] / interval_diag["exposure_pm"]
    interval_diag.to_csv(out_dir / "treatment_interval_diagnostics.csv", index=False)

    spatial_metrics = {
        "corr_u_true_vs_post_mean": _corr(spatial_df["u_ttt_true"], spatial_df["u_ttt_post_mean"]),
        "rmse_u_true_vs_post_mean": float(np.sqrt(np.mean(np.square(spatial_df["u_error"])))),
        "corr_u_true_vs_s_post_mean": _corr(spatial_df["u_ttt_true"], spatial_df["s_post_mean"]),
        "corr_u_surv_true_vs_u_ttt_post_mean": _corr(spatial_df["u_surv_true"], spatial_df["u_ttt_post_mean"]),
    }

    theta_rows = param_recovery[param_recovery["parameter"].str.startswith("theta_")].copy()
    gamma_rows = param_recovery[param_recovery["parameter"].str.startswith("gamma[")].copy()

    aggregate_metrics = {
        "theta_mae": float(theta_rows["abs_error"].mean()),
        "theta_rmse": float(np.sqrt(np.mean(np.square(theta_rows["error"])))),
        "theta_90pct_coverage": float(100.0 * theta_rows["covered_by_q05_q95"].mean()),
        "gamma_mae": float(gamma_rows["abs_error"].mean()),
        "gamma_rmse": float(np.sqrt(np.mean(np.square(gamma_rows["error"])))),
        "gamma_90pct_coverage": float(100.0 * gamma_rows["covered_by_q05_q95"].mean()),
        "overall_mae": float(param_recovery["abs_error"].mean()),
        "overall_rmse": float(np.sqrt(np.mean(np.square(param_recovery["error"])))),
    }

    with open(out_dir / "posterior_summary.json", "w", encoding="utf-8") as f:
        json.dump(posterior_summary, f, indent=2)
    with open(out_dir / "spatial_recovery_metrics.json", "w", encoding="utf-8") as f:
        json.dump(spatial_metrics, f, indent=2)
    with open(out_dir / "aggregate_recovery_metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    divergences = int(np.asarray(result.mcmc.get_extra_fields()["diverging"]).sum())

    print("\nSpatial treatment-time PE recovery check")
    print("---------------------------------------")
    print(f"Rows in treatment long data: {len(ttt_long):,}")
    print(f"Observed treatment events:   {int(ttt_long['event'].sum()):,}")
    print(f"Divergences:                {divergences}")

    print("\nParameter recovery")
    print("------------------")
    print(param_recovery.to_string(index=False))

    print("\nTreatment interval diagnostics")
    print("------------------------------")
    print(interval_diag.to_string(index=False))

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