from __future__ import annotations

import json
from pathlib import Path

from seer_peph.analysis.joint_analysis import JointAnalysisConfig, run_joint_analysis


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = _repo_root()

    config_path = root / "configs" / "ga_synth_grant_demo" / "ga_joint_analysis_county_demo_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find demo config at {config_path}. "
            "Update the path in this smoke-test script if needed."
        )

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    if not isinstance(raw_cfg, dict):
        raise ValueError("Smoke-test config JSON must decode to an object.")

    # Make paths absolute relative to repo root so the test is robust
    for key in ["input_path", "graph_edges_path", "graph_lookup_path"]:
        if key in raw_cfg and raw_cfg[key] is not None:
            raw_cfg[key] = str((root / raw_cfg[key]).resolve())

    # Redirect output to a smoke-test-specific artifact directory
    raw_cfg["out_dir"] = str((root / "artifacts" / "joint_analysis_smoke_test").resolve())

    # Fast smoke-test inference budget
    raw_cfg["inference"] = {
        "num_chains": 1,
        "num_warmup": 50,
        "num_samples": 50,
        "target_accept_prob": 0.9,
        "dense_mass": False,
        "max_tree_depth": 8,
        "progress_bar": True,
    }

    # Keep PPC on, but lightweight
    raw_cfg["ppc"] = {
        "enabled": True,
        "draw_indices": None,
        "sample_posterior_predictive": True,
        "random_seed": 123,
    }

    cfg = JointAnalysisConfig.from_dict(raw_cfg)

    print("Running county-based joint-analysis smoke test")
    print(f"Input dataset: {cfg.input_path}")
    print(f"Graph edges:   {cfg.graph_edges_path}")
    print(f"Graph lookup:  {cfg.graph_lookup_path}")
    print(f"Output dir:    {cfg.out_dir}")

    out_dir = run_joint_analysis(cfg)

    expected = [
        "surv_long.csv",
        "ttt_long.csv",
        "joint_survival_beta_summary.csv",
        "joint_survival_alpha_summary.csv",
        "joint_survival_delta_post_summary.csv",
        "joint_survival_delta_post_linear_summary.csv",
        "joint_treatment_theta_summary.csv",
        "joint_treatment_gamma_summary.csv",
        "u_surv_summary.csv",
        "u_ttt_summary.csv",
        "u_ttt_ind_summary.csv",
        "joint_spatial_hyperparameter_summary.csv",
        "joint_coupling_summary.csv",
        "joint_field_correlations_summary.csv",
        "joint_survival_ppc_interval_counts.csv",
        "joint_survival_ppc_area_counts.csv",
        "joint_survival_ppc_interval_by_treatment_counts.csv",
        "joint_treatment_ppc_interval_counts.csv",
        "joint_treatment_ppc_area_counts.csv",
        "analysis_config.json",
        "run_manifest.json",
    ]

    print("\nSmoke test completed.")
    print(f"Artifacts written to: {out_dir}")

    missing = []
    for name in expected:
        path = Path(out_dir) / name
        if path.exists():
            print(f"  OK   {name}")
        else:
            print(f"  MISS {name}")
            missing.append(name)

    fit_dir = Path(out_dir) / "fit_bundle"
    if fit_dir.exists():
        print("  OK   fit_bundle/")
    else:
        print("  MISS fit_bundle/")
        missing.append("fit_bundle/")

    if missing:
        print("\nSmoke test finished with missing expected artifacts:")
        for name in missing:
            print(f"  - {name}")
        raise SystemExit(1)

    print("\nAll expected smoke-test artifacts were found.")


if __name__ == "__main__":
    main()