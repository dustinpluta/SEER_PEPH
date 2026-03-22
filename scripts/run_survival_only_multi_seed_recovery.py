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
from seer_peph.models.survival_only import model as survival_only_model
from seer_peph.validation.simulate import simulate_joint


@dataclass(frozen=True)
class MultiSeedRecoveryConfig:
    seeds: tuple[int, ...] = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1001)

    A: int = 20
    k: int = 4
    n_per_area: int = 200

    # Keep omitted frailty weak for the survival_only sanity study
    rho_u: float = 0.5
    phi_surv: float = 0.8
    phi_ttt: float = 0.8
    sigma_surv: float = 0.05
    sigma_ttt: float = 0.05

    num_chains: int = 2
    num_warmup: int = 500
    num_samples: int = 500
    target_accept_prob: float = 0.85
    dense_mass: bool = False
    max_tree_depth: int = 10

    out_dir: str = "artifacts/survival_only_multi_seed_recovery"


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


def _extract_truth(wide: pd.DataFrame) -> dict[str, float]:
    first = wide.iloc[0]
    return {
        "beta_td": float(first["beta_td_true"]),
        "beta_age_per10_centered": float(first["beta_surv_age_per10_centered_true"]),
        "beta_cci": float(first["beta_surv_cci_true"]),
        "beta_tumor_size_log": float(first["beta_surv_tumor_size_log_true"]),
        "beta_stage_II": float(first["beta_surv_stage_II_true"]),
        "beta_stage_III": float(first["beta_surv_stage_III_true"]),
        "delta_post[0]": float(first["delta_post_0_true"]),
        "delta_post[1]": float(first["delta_post_1_true"]),
        "delta_post[2]": float(first["delta_post_2_true"]),
        "delta_post[3]": float(first["delta_post_3_true"]),
        "delta_post[4]": float(first["delta_post_4_true"]),
    }


def _extract_posterior_summary(summary_dict: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        "beta_td": summary_dict["beta_td"],
        "beta_age_per10_centered": summary_dict["beta[0]"],
        "beta_cci": summary_dict["beta[1]"],
        "beta_tumor_size_log": summary_dict["beta[2]"],
        "beta_stage_II": summary_dict["beta[3]"],
        "beta_stage_III": summary_dict["beta[4]"],
        "delta_post[0]": summary_dict["delta_post[0]"],
        "delta_post[1]": summary_dict["delta_post[1]"],
        "delta_post[2]": summary_dict["delta_post[2]"],
        "delta_post[3]": summary_dict["delta_post[3]"],
        "delta_post[4]": summary_dict["delta_post[4]"],
    }


def _one_seed_recovery(seed: int, cfg: MultiSeedRecoveryConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    graph = make_ring_lattice(A=cfg.A, k=cfg.k)

    wide = simulate_joint(
        graph,
        n_per_area=cfg.n_per_area,
        rho_u=cfg.rho_u,
        phi_surv=cfg.phi_surv,
        phi_ttt=cfg.phi_ttt,
        sigma_surv=cfg.sigma_surv,
        sigma_ttt=cfg.sigma_ttt,
        seed=seed,
    )

    df = _encode_like_prep(wide)
    surv_long = build_survival_long(df)
    ttt_long = build_treatment_long(df)

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
        survival_only_model,
        data,
        rng_key=random.PRNGKey(seed + 10000),
        config=infer_cfg,
        extra_fields=("diverging",),
    )

    posterior_summary = summarise_samples(result.samples)
    truth = _extract_truth(wide)
    est = _extract_posterior_summary(posterior_summary)

    divergences = int(np.asarray(result.mcmc.get_extra_fields()["diverging"]).sum())

    rows: list[dict[str, Any]] = []
    for param, true_val in truth.items():
        est_mean = float(est[param]["mean"])
        est_median = float(est[param]["median"])
        q05 = float(est[param]["q05"])
        q95 = float(est[param]["q95"])
        error = est_mean - true_val
        rows.append(
            {
                "seed": seed,
                "parameter": param,
                "truth": true_val,
                "posterior_mean": est_mean,
                "posterior_median": est_median,
                "q05": q05,
                "q95": q95,
                "error": error,
                "abs_error": abs(error),
                "covered_by_q05_q95": int(q05 <= true_val <= q95),
                "divergences": divergences,
                "n_surv_rows": int(len(surv_long)),
                "n_surv_events": int(surv_long["event"].sum()),
                "n_treated_rows": int(surv_long["treated_td"].sum()),
            }
        )

    seed_meta = {
        "seed": seed,
        "divergences": divergences,
        "n_surv_rows": int(len(surv_long)),
        "n_surv_events": int(surv_long["event"].sum()),
        "n_treated_rows": int(surv_long["treated_td"].sum()),
    }

    return pd.DataFrame(rows), seed_meta


def _aggregate_results(all_results: pd.DataFrame) -> pd.DataFrame:
    grouped = all_results.groupby("parameter", sort=False)

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
        mean_divergences=("divergences", "mean"),
    ).reset_index()

    summary["coverage_q05_q95"] = 100.0 * summary["coverage_q05_q95"]
    return summary


def _aggregate_by_block(agg: pd.DataFrame) -> pd.DataFrame:
    def block_of(param: str) -> str:
        if param.startswith("beta_"):
            return "beta"
        if param.startswith("delta_post"):
            return "delta_post"
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


def main(config: MultiSeedRecoveryConfig | None = None) -> None:
    cfg = MultiSeedRecoveryConfig() if config is None else config
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    seed_meta_rows: list[dict[str, Any]] = []

    print("\nRunning multi-seed survival_only recovery study")
    print("===============================================")
    print(f"Seeds: {list(cfg.seeds)}")
    print(f"Output directory: {out_dir}")

    for i, seed in enumerate(cfg.seeds, start=1):
        print(f"\n[{i}/{len(cfg.seeds)}] Seed {seed}")
        seed_df, seed_meta = _one_seed_recovery(seed, cfg)
        all_rows.append(seed_df)
        seed_meta_rows.append(seed_meta)

        seed_param_mae = float(seed_df["abs_error"].mean())
        print(
            f"  divergences={seed_meta['divergences']}, "
            f"surv_events={seed_meta['n_surv_events']}, "
            f"treated_rows={seed_meta['n_treated_rows']}, "
            f"overall_mae={seed_param_mae:.4f}"
        )

    all_results = pd.concat(all_rows, ignore_index=True)
    seed_meta_df = pd.DataFrame(seed_meta_rows)
    agg = _aggregate_results(all_results)
    block_agg = _aggregate_by_block(agg)

    all_results.to_csv(out_dir / "recovery_by_seed.csv", index=False)
    agg.to_csv(out_dir / "recovery_summary.csv", index=False)
    block_agg.to_csv(out_dir / "recovery_block_summary.csv", index=False)
    seed_meta_df.to_csv(out_dir / "seed_level_summary.csv", index=False)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("\nParameter-level average recovery")
    print("-------------------------------")
    print(agg.to_string(index=False))

    print("\nBlock-level summary")
    print("-------------------")
    print(block_agg.to_string(index=False))

    print("\nSeed-level summary")
    print("------------------")
    print(seed_meta_df.to_string(index=False))

    print("\nOverall")
    print("-------")
    print(f"Mean seed divergences: {seed_meta_df['divergences'].mean():.2f}")
    print(f"Total divergences:     {int(seed_meta_df['divergences'].sum())}")
    print(f"Overall parameter MAE: {all_results['abs_error'].mean():.4f}")
    print(f"Overall parameter RMSE:{np.sqrt(np.mean(np.square(all_results['error']))):.4f}")


if __name__ == "__main__":
    main()