# tests/data/test_prep.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seer_peph.data.prep import (
    DAYS_PER_MONTH,
    SURV_BREAKS,
    TTT_BREAKS,
    build_survival_long,
    build_treatment_long,
    load_and_encode,
)


def _days(months: float) -> float:
    return months * DAYS_PER_MONTH


@pytest.fixture
def wide_csv(tmp_path):
    """
    Three-subject toy dataset designed to hit key prep.py branches.

    id=1: treated at 2.5 months, dies at 10 months
    id=2: never treated, censored at 60 months
    id=3: treatment is censored by death at 4 months
    """
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "zip": 30001,
                "age_per10_centered": -0.5,
                "cci": 1,
                "tumor_size_log": 3.0,
                "ses": 0.2,
                "sex": "M",
                "stage": "II",
                "treatment_time": _days(2.5),
                "treatment_time_obs": _days(2.5),
                "treatment_event": 1,
                "time": _days(10.0),
                "event": 1,
            },
            {
                "id": 2,
                "zip": 30003,
                "age_per10_centered": 0.2,
                "cci": 0,
                "tumor_size_log": 2.7,
                "ses": -0.4,
                "sex": "F",
                "stage": "I",
                "treatment_time": np.nan,
                "treatment_time_obs": _days(60.0),
                "treatment_event": 0,
                "time": _days(60.0),
                "event": 0,
            },
            {
                "id": 3,
                "zip": 30001,
                "age_per10_centered": 1.0,
                "cci": 3,
                "tumor_size_log": 3.4,
                "ses": 0.2,
                "sex": "F",
                "stage": "III",
                "treatment_time": _days(8.0),      # latent treatment time
                "treatment_time_obs": _days(4.0),  # death censors treatment
                "treatment_event": 0,
                "time": _days(4.0),
                "event": 1,
            },
        ]
    )
    path = tmp_path / "toy_joint.csv"
    df.to_csv(path, index=False)
    return path


def test_load_and_encode_builds_expected_encoded_columns(wide_csv):
    df = load_and_encode(wide_csv)

    expected_new_cols = {
        "time_m",
        "treatment_time_m",
        "treatment_time_obs_m",
        "sex_male",
        "stage_II",
        "stage_III",
        "area_id",
    }
    assert expected_new_cols.issubset(df.columns)

    # days -> months conversion
    assert df.loc[df["id"] == 1, "time_m"].item() == pytest.approx(10.0)
    assert df.loc[df["id"] == 1, "treatment_time_m"].item() == pytest.approx(2.5)
    assert df.loc[df["id"] == 3, "treatment_time_obs_m"].item() == pytest.approx(4.0)

    # dummy coding
    row1 = df.loc[df["id"] == 1].iloc[0]
    assert row1["sex_male"] == 1
    assert row1["stage_II"] == 1
    assert row1["stage_III"] == 0

    row2 = df.loc[df["id"] == 2].iloc[0]
    assert row2["sex_male"] == 0
    assert row2["stage_II"] == 0
    assert row2["stage_III"] == 0

    row3 = df.loc[df["id"] == 3].iloc[0]
    assert row3["stage_II"] == 0
    assert row3["stage_III"] == 1

    # sorted contiguous area_id from zip
    z_to_area = (
        df[["zip", "area_id"]]
        .drop_duplicates()
        .sort_values("zip")
        .reset_index(drop=True)
    )
    assert z_to_area["zip"].tolist() == [30001, 30003]
    assert z_to_area["area_id"].tolist() == [0, 1]


def test_build_survival_long_has_expected_time_dependent_treatment_behavior(wide_csv):
    df = load_and_encode(wide_csv)
    surv_long = build_survival_long(df)

    expected_cols = {
        "id",
        "k",
        "t0",
        "t1",
        "exposure",
        "event",
        "treated_td",
        "k_post",
        "area_id",
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "ses",
        "sex_male",
        "stage_II",
        "stage_III",
    }
    assert expected_cols.issubset(surv_long.columns)

    # Basic typing / coding constraints
    assert surv_long["event"].isin([0, 1]).all()
    assert surv_long["treated_td"].isin([0, 1]).all()
    assert pd.api.types.is_integer_dtype(surv_long["k"])
    assert pd.api.types.is_integer_dtype(surv_long["k_post"])

    # Exposure conservation: subject-level summed exposure should match observed survival time,
    # truncated at the survival grid maximum.
    exposure_by_id = surv_long.groupby("id")["exposure"].sum().sort_index()
    expected = (
        df.set_index("id")["time_m"]
        .clip(upper=float(SURV_BREAKS[-1]))
        .sort_index()
    )
    assert np.allclose(exposure_by_id.values, expected.values)

    # One survival event row for subjects 1 and 3, none for subject 2
    event_counts = surv_long.groupby("id")["event"].sum().to_dict()
    assert event_counts == {1: 1, 2: 0, 3: 1}

    # Subject 1 is untreated before 2.5 months, then treated afterward
    s1 = surv_long.loc[surv_long["id"] == 1].sort_values(["t0", "t1"]).reset_index(drop=True)
    assert (s1.loc[s1["t0"] < 2.5, "treated_td"] == 0).all()
    assert (s1.loc[s1["t0"] >= 2.5, "treated_td"] == 1).all()

    # Pre-treatment rows should have k_post = -1; post-treatment rows should be >= 0
    assert (s1.loc[s1["treated_td"] == 0, "k_post"] == -1).all()
    assert (s1.loc[s1["treated_td"] == 1, "k_post"] >= 0).all()

    # Treatment at 2.5 months means a row starts exactly at 2.5
    assert np.isclose(s1["t0"].to_numpy(), 2.5).any()

    # Subject 3 never receives observed treatment, so all rows remain untreated
    s3 = surv_long.loc[surv_long["id"] == 3]
    assert (s3["treated_td"] == 0).all()
    assert (s3["k_post"] == -1).all()


def test_build_treatment_long_respects_observed_treatment_followup(wide_csv):
    df = load_and_encode(wide_csv)
    ttt_long = build_treatment_long(df)

    expected_cols = {
        "id",
        "k",
        "t0",
        "t1",
        "exposure",
        "event",
        "area_id",
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "ses",
        "sex_male",
        "stage_II",
        "stage_III",
    }
    assert expected_cols.issubset(ttt_long.columns)

    assert ttt_long["event"].isin([0, 1]).all()
    assert pd.api.types.is_integer_dtype(ttt_long["k"])

    # Exposure conservation: subject-level summed exposure should match observed treatment follow-up,
    # truncated at the treatment grid maximum.
    exposure_by_id = ttt_long.groupby("id")["exposure"].sum().sort_index()
    expected = (
        df.set_index("id")["treatment_time_obs_m"]
        .clip(upper=float(TTT_BREAKS[-1]))
        .sort_index()
    )
    assert np.allclose(exposure_by_id.values, expected.values)

    # One treatment event for subject 1, none for subjects 2 and 3
    event_counts = ttt_long.groupby("id")["event"].sum().to_dict()
    assert event_counts == {1: 1, 2: 0, 3: 0}

    # Subject 1 should have final event row ending at 2.5 months
    s1 = ttt_long.loc[ttt_long["id"] == 1].sort_values(["t0", "t1"]).reset_index(drop=True)
    s1_event_row = s1.loc[s1["event"] == 1]
    assert len(s1_event_row) == 1
    assert np.isclose(s1_event_row["t1"].item(), 2.5)

    # Subject 3 has latent treatment_time = 8 months but observed treatment follow-up is 4 months,
    # so treatment is censored by death and no row extends past 4 months.
    s3 = ttt_long.loc[ttt_long["id"] == 3].sort_values(["t0", "t1"]).reset_index(drop=True)
    assert s3["event"].sum() == 0
    assert s3["exposure"].sum() == pytest.approx(4.0)
    assert np.isclose(s3["t1"].max(), 4.0)