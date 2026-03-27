from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from seer_peph.validation.joint_scenarios import JointSimulationScenario


@dataclass(frozen=True)
class JointSimulationResult:
    """
    Container for the output of one model-matched joint simulation run.

    A simulator should return one of these for each (scenario, seed) pair.

    Contents
    --------
    scenario
        The scenario specification used to generate the data.
    seed
        RNG seed for this replicate.
    wide
        Simulated subject-level wide dataframe. This is the dataset passed
        downstream to prep / fitting.
    parameter_truth
        Flat truth table for scalar and vector parameters. Recommended columns:
            - parameter
            - index        (nullable; for vector components)
            - label        (human-readable component label)
            - truth
            - group        (e.g. survival_beta, treatment_theta, alpha, gamma)
    area_truth
        Area-level truth table. Recommended columns:
            - area_id
            - u_surv_true
            - u_ttt_true
            - u_ttt_ind_true
            - s_surv_true    (optional, if stored separately)
            - s_ttt_true     (optional, if stored separately)
    support_diagnostics
        Optional subject/interval support summaries produced at simulation time.
        These are intended to help interpret failures in sparse scenarios.
    metadata
        Optional free-form metadata for provenance and diagnostics.
    """

    scenario: JointSimulationScenario
    seed: int
    wide: pd.DataFrame
    parameter_truth: pd.DataFrame
    area_truth: pd.DataFrame
    support_diagnostics: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_seed()
        self._validate_wide()
        self._validate_parameter_truth()
        self._validate_area_truth()
        self._validate_support_diagnostics()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def n_subjects(self) -> int:
        return int(len(self.wide))

    @property
    def n_areas(self) -> int:
        if "area_id" in self.wide.columns:
            return int(self.wide["area_id"].nunique())
        if "area_id" in self.area_truth.columns:
            return int(self.area_truth["area_id"].nunique())
        return int(self.scenario.n_areas)

    @property
    def scenario_name(self) -> str:
        return self.scenario.name

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def summary_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario.name,
            "seed": int(self.seed),
            "n_subjects": self.n_subjects,
            "n_areas": self.n_areas,
            "n_parameter_truth_rows": int(len(self.parameter_truth)),
            "n_area_truth_rows": int(len(self.area_truth)),
            "support_tables": list(self.support_diagnostics.keys()),
            "metadata_keys": list(self.metadata.keys()),
        }

    def get_parameter_truth_vector(self, group: str) -> pd.DataFrame:
        """
        Return the subset of parameter_truth for a named parameter group.
        """
        if "group" not in self.parameter_truth.columns:
            raise ValueError("parameter_truth does not contain a 'group' column.")
        return (
            self.parameter_truth.loc[self.parameter_truth["group"] == group]
            .copy()
            .reset_index(drop=True)
        )

    def get_scalar_truth(self, parameter: str) -> float:
        """
        Extract a scalar truth value from parameter_truth.

        Requires exactly one matching row.
        """
        if "parameter" not in self.parameter_truth.columns:
            raise ValueError("parameter_truth does not contain a 'parameter' column.")
        if "truth" not in self.parameter_truth.columns:
            raise ValueError("parameter_truth does not contain a 'truth' column.")

        sub = self.parameter_truth.loc[self.parameter_truth["parameter"] == parameter]
        if len(sub) != 1:
            raise ValueError(
                f"Expected exactly one row for parameter={parameter!r}, found {len(sub)}."
            )
        return float(sub["truth"].iloc[0])

    def to_metadata_record(self) -> dict[str, Any]:
        """
        Flatten high-level replicate metadata for study-level summary tables.
        """
        out = {
            "scenario_name": self.scenario.name,
            "seed": int(self.seed),
            "n_subjects": self.n_subjects,
            "n_areas": self.n_areas,
            "n_parameter_truth_rows": int(len(self.parameter_truth)),
            "n_area_truth_rows": int(len(self.area_truth)),
        }
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[str(k)] = v
        return out

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_seed(self) -> None:
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")

    def _validate_wide(self) -> None:
        required = {
            "id",
            "zip",
            "time",
            "event",
            "treatment_time",
            "treatment_time_obs",
            "treatment_event",
        }
        missing = [c for c in required if c not in self.wide.columns]
        if missing:
            raise ValueError(f"wide is missing required columns: {missing}")

        if len(self.wide) == 0:
            raise ValueError("wide must contain at least one row.")

        if "id" in self.wide.columns and self.wide["id"].duplicated().any():
            raise ValueError("wide contains duplicate subject ids.")

        if "event" in self.wide.columns:
            bad = ~self.wide["event"].isin([0, 1])
            if bad.any():
                raise ValueError("wide['event'] must contain only 0/1 values.")

        if "treatment_event" in self.wide.columns:
            bad = ~self.wide["treatment_event"].isin([0, 1])
            if bad.any():
                raise ValueError("wide['treatment_event'] must contain only 0/1 values.")

    def _validate_parameter_truth(self) -> None:
        required = {"parameter", "truth"}
        missing = [c for c in required if c not in self.parameter_truth.columns]
        if missing:
            raise ValueError(f"parameter_truth is missing required columns: {missing}")

        if len(self.parameter_truth) == 0:
            raise ValueError("parameter_truth must contain at least one row.")

        if self.parameter_truth["parameter"].isna().any():
            raise ValueError("parameter_truth['parameter'] contains missing values.")

        if self.parameter_truth["truth"].isna().any():
            raise ValueError("parameter_truth['truth'] contains missing values.")

    def _validate_area_truth(self) -> None:
        required = {"area_id", "u_surv_true", "u_ttt_true"}
        missing = [c for c in required if c not in self.area_truth.columns]
        if missing:
            raise ValueError(f"area_truth is missing required columns: {missing}")

        if len(self.area_truth) == 0:
            raise ValueError("area_truth must contain at least one row.")

        if self.area_truth["area_id"].duplicated().any():
            raise ValueError("area_truth contains duplicate area_id values.")

        if len(self.area_truth) != self.scenario.n_areas:
            raise ValueError(
                "area_truth row count does not match scenario.n_areas: "
                f"{len(self.area_truth)} != {self.scenario.n_areas}"
            )

    def _validate_support_diagnostics(self) -> None:
        for name, df in self.support_diagnostics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("support_diagnostics keys must be non-empty strings.")
            if not isinstance(df, pd.DataFrame):
                raise ValueError(
                    f"support_diagnostics[{name!r}] must be a pandas DataFrame."
                )