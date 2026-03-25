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
from seer_peph.models.survival_spatial_delta_only import model as survival_spatial_delta_only_model
from seer_peph.validation.simulate import simulate_joint


@dataclass(frozen=True)
class SpatialDeltaOnlyNullRecoveryConfig:
    seed: int = 123
    A: int = 20
    k: int = 4
    n_per_area: int = 200

    rho_u: float = 0.5
    phi_surv: float = 0.8
    phi_ttt: float = 0.8
    sigma_surv: float = 0.5
    sigma_ttt: float = 0.05

    # Null treatment-history effect
    beta_td_true: float = 0.0
    delta_post_true: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    num_chains: int = 2
    num_warmup: int = 500
    num_samples: int = 500
    target_accept_prob: float = 0.95
    dense_mass: bool = False
    max_tree_depth: int = 10

    out_dir: str = "artifacts/survival_spatial_delta_only_null_recovery"


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


def _extract_truth(wide: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    first = wide.iloc[0]

    param_truth = {
        "beta_age_per10_centered": float(first["beta_surv_age_per10_centered_true"]),
        "beta_cci": float(first["beta_surv_cci_true"]),
        "beta_tumor_size_log": float(first["beta_surv_tumor_size_log_true"]),
        "beta_stage_II": float(first["beta_surv_stage_II_true"]),
        "beta_stage_III": float(first["beta_surv_stage_III_true"]),
        "delta_post[0]": 0.0,
        "delta_post[1]": 0.0,
        "delta_post[2]": 0.0,
        "delta_post[3]": 0.0,
        "delta_post[4]": 0.0,
        "phi_surv": float(first["phi_surv_true"]),
        "sigma_surv": float(first["sigma_surv_true"]),
    }

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
        "beta_age_per10_centered": summary_dict["beta[0]"]["mean"],
        "beta_cci": summary_dict["beta[1]"]["mean"],
        "beta_tumor_size_log": summary_dict["beta[2]"]["mean"],
        "beta_stage_II": summary_dict["beta[3]"]["mean"],
        "beta_stage_III": summary_dict["beta[4]"]["mean"],
        "phi_surv_proxy_rho": summary_dict["rho"]["mean"],
        "sigma_surv_proxy_tau": summary_dict["tau"]["mean"],
    }

    delta_keys = sorted(k for k in summary_dict if k.startswith("delta_post["))
    for k in delta_keys:
        out[k] = summary_dict[k]["mean"]

    return out


def main(config: SpatialDeltaOnlyNullRecoveryConfig | None = None) -> None:
    cfg = SpatialDeltaOnlyNullRecoveryConfig() if config is None else config
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
        beta_td=cfg.beta_td_true,
        delta_post=cfg.delta_post_true,
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
        survival_spatial_delta_only_model,
        data,
        rng_key=random.PRNGKey(cfg.seed + 1000),
        config=infer_cfg,
        extra_fields=("diverging",),
    )

    posterior_summary = summarise_samples(result.samples)
    param_truth, area_truth = _extract_truth(wide)
    param_est = _extract_param_estimates(posterior_summary)

    name_map = {
        "phi_surv": "phi_surv_proxy_rho",
        "sigma_surv": "sigma_surv_proxy_tau",
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
            "u_surv_post_mean": u_post_mean,
            "s_post_mean": s_post_mean,
        }
    )
    spatial_df["u_error"] = spatial_df["u_surv_post_mean"] - spatial_df["u_surv_true"]
    spatial_df["u_abs_error"] = np.abs(spatial_df["u_error"])
    spatial_df.to_csv(out_dir / "spatial_recovery_by_area.csv", index=False)

    treated = surv_long.loc[surv_long["treated_td"] == 1].copy()
    interval_diag = (
        treated.groupby("k_post", dropna=False)
        .agg(
            rows=("id", "size"),
            unique_subjects=("id", "nunique"),
            events=("event", "sum"),
            exposure_pm=("exposure", "sum"),
        )
        .reset_index()
    )
    if len(interval_diag) > 0:
        interval_diag["event_rate_per_100pm"] = 100.0 * interval_diag["events"] / interval_diag["exposure_pm"]
    interval_diag.to_csv(out_dir / "post_treatment_interval_diagnostics.csv", index=False)

    spatial_metrics = {
        "corr_u_true_vs_post_mean": _corr(spatial_df["u_surv_true"], spatial_df["u_surv_post_mean"]),
        "rmse_u_true_vs_post_mean": float(np.sqrt(np.mean(np.square(spatial_df["u_error"])))),
        "corr_u_true_vs_s_post_mean": _corr(spatial_df["u_surv_true"], spatial_df["s_post_mean"]),
    }

    delta_rows = param_recovery[param_recovery["parameter"].str.startswith("delta_post")].copy()
    null_metrics = {
        "mean_abs_delta_bias": float(delta_rows["abs_error"].mean()),
        "max_abs_delta_bias": float(delta_rows["abs_error"].max()),
        "delta_90pct_coverage": float(100.0 * delta_rows["covered_by_q05_q95"].mean()),
    }

    with open(out_dir / "posterior_summary.json", "w", encoding="utf-8") as f:
        json.dump(posterior_summary, f, indent=2)
    with open(out_dir / "spatial_recovery_metrics.json", "w", encoding="utf-8") as f:
        json.dump(spatial_metrics, f, indent=2)
    with open(out_dir / "null_delta_metrics.json", "w", encoding="utf-8") as f:
        json.dump(null_metrics, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    divergences = int(np.asarray(result.mcmc.get_extra_fields()["diverging"]).sum())

    print("\nSpatial survival delta-only NULL recovery check")
    print("-----------------------------------------------")
    print(f"Rows in survival long data: {len(surv_long):,}")
    print(f"Observed survival events:   {int(surv_long['event'].sum()):,}")
    print(f"Observed treated rows:      {int(surv_long['treated_td'].sum()):,}")
    print(f"Divergences:                {divergences}")

    print("\nParameter recovery")
    print("------------------")
    print(param_recovery.to_string(index=False))

    print("\nSpatial recovery metrics")
    print("------------------------")
    for k, v in spatial_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nNull delta metrics")
    print("------------------")
    for k, v in null_metrics.items():
        print(f"{k}: {v:.4f}")

    beta_rows = param_recovery[param_recovery["parameter"].str.startswith("beta_")].copy()
    print("\nAggregate error metrics")
    print("-----------------------")
    print(f"Beta MAE:   {beta_rows['abs_error'].mean():.4f}")
    print(f"Delta MAE:  {delta_rows['abs_error'].mean():.4f}")
    print(f"Overall MAE:{param_recovery['abs_error'].mean():.4f}")


if __name__ == "__main__":
    main()