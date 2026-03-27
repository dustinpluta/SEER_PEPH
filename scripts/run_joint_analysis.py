from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seer_peph.analysis.joint_analysis import JointAnalysisConfig, run_joint_analysis


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Top-level config JSON must be an object.")
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run joint treatment-survival analysis from a JSON config file."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to joint analysis JSON config file.",
    )
    args = parser.parse_args()

    raw_cfg = _load_json(args.config_path)
    cfg = JointAnalysisConfig.from_dict(raw_cfg)
    out_dir = run_joint_analysis(cfg)
    print(f"Joint analysis completed. Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()