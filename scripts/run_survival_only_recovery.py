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
class RecoveryConfig:
    seed: int = 123
    A: int = 20
    k: int = 4
    n_per_area: int = 200

    # Keep frailty small so the survival_only model is not badly misspecified
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

    out_dir: str = "artifacts/survival_only_recovery"


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
    """
    Pull the survival-only truth from the simulator output.

    These are constant across rows by construction, so we read the first row.
    """
    first = wide.iloc[0]

    truth = {
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
    return truth


def _extract_posterior_means(result_summary: dict[str, dict[str, float]]) -> dict[str, float]:
    """
    Map posterior summaries into the same naming convention as the truth dict.
    """
    est = {
        "beta_td": result_summary["beta_td"]["mean"],
        "beta_age_per10_centered": result_summary["beta[0]"]["mean"],
        "beta_cci": result_summary["beta[1]"]["mean"],
        "beta_tumor_size_log": result_summary["beta[2]"]["mean"],
        "beta_stage_II": result_summary["beta[3]"]["mean"],
        "beta_stage_III": result_summary["beta[4]"]["mean"],
        "delta_post[0]": result_summary["delta_post[0]"]["mean"],
        "delta_post[1]": result_summary["delta_post[1]"]["mean"],
        "delta_post[2]": result_summary["delta_post[2]"]["mean"],
        "delta_post[3]": result_summary["delta_post[3]"]["mean"],
        "delta_post[4]": result_summary["delta_post[4]"]["mean"],
    }
    return est


def _build_recovery_table(truth: dict[str, float], est: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, true_val in truth.items():
        est_val = est[name]
        rows.append(
            {
                "parameter": name,
                "truth": true_val,
                "posterior_mean": est_val,
                "error": est_val - true_val,
                "abs_error": abs(est_val - true_val),
            }
        )
    return pd.DataFrame(rows)


def main(config: RecoveryConfig | None = None) -> None:
    cfg = RecoveryConfig() if config is None else config
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
        survival_only_model,
        data,
        rng_key=random.PRNGKey(cfg.seed + 1000),
        config=infer_cfg,
        extra_fields=("diverging",),
    )

    posterior_summary = summarise_samples(result.samples)
    truth = _extract_truth(wide)
    est = _extract_posterior_means(posterior_summary)
    recovery = _build_recovery_table(truth, est)

    recovery.to_csv(out_dir / "recovery_table.csv", index=False)

    with open(out_dir / "posterior_summary.json", "w", encoding="utf-8") as f:
        json.dump(posterior_summary, f, indent=2)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    divergences = np.asarray(result.mcmc.get_extra_fields()["diverging"]).sum()

    print("\nSurvival-only recovery study")
    print("----------------------------")
    print(f"Rows in survival long data: {len(surv_long):,}")
    print(f"Observed survival events:   {int(surv_long['event'].sum()):,}")
    print(f"Observed treated rows:      {int(surv_long['treated_td'].sum()):,}")
    print(f"Divergences:                {int(divergences)}")
    print("\nRecovery table:")
    print(recovery.to_string(index=False))

    beta_rows = recovery[recovery["parameter"].str.startswith("beta_")].copy()
    delta_rows = recovery[recovery["parameter"].str.startswith("delta_post")].copy()

    print("\nAggregate error metrics")
    print("-----------------------")
    print(f"Beta MAE:   {beta_rows['abs_error'].mean():.4f}")
    print(f"Delta MAE:  {delta_rows['abs_error'].mean():.4f}")
    print(f"Overall MAE:{recovery['abs_error'].mean():.4f}")


if __name__ == "__main__":
    main()