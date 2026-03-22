# tests/test_model_data.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seer_peph.data.model_data import (
    get_default_surv_x_cols,
    get_default_ttt_x_cols,
    make_model_data,
)
from seer_peph.data.prep import (
    DAYS_PER_MONTH,
    build_survival_long,
    build_treatment_long,
    load_and_encode,
)
from seer_peph.graphs import from_adjacency


def _days(months: float) -> float:
    return months * DAYS_PER_MONTH


@pytest.fixture
def wide_csv(tmp_path):
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
                "treatment_time": _days(8.0),
                "treatment_time_obs": _days(4.0),
                "treatment_event": 0,
                "time": _days(4.0),
                "event": 1,
            },
        ]
    )
    path = tmp_path / "toy_joint.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def encoded_df(wide_csv):
    return load_and_encode(wide_csv)


@pytest.fixture
def surv_long(encoded_df):
    return build_survival_long(encoded_df)


@pytest.fixture
def ttt_long(encoded_df):
    return build_treatment_long(encoded_df)


@pytest.fixture
def graph():
    # The toy dataset uses exactly two areas, so use the minimal connected
    # 2-node graph directly instead of a ring lattice.
    W = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=float,
    )
    return from_adjacency(W, name="two_area_graph")


def test_default_x_col_helpers_return_expected_columns(surv_long, ttt_long):
    surv_x = get_default_surv_x_cols(surv_long)
    ttt_x = get_default_ttt_x_cols(ttt_long)

    assert surv_x == [
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "stage_II",
        "stage_III",
    ]
    assert ttt_x == [
        "age_per10_centered",
        "cci",
        "ses",
        "sex_male",
        "stage_II",
        "stage_III",
    ]


def test_make_model_data_returns_expected_keys(surv_long, ttt_long, graph):
    data = make_model_data(surv_long, ttt_long, graph, as_jax=False)

    expected_keys = {
        "y_surv",
        "log_exposure_surv",
        "k_surv",
        "k_post",
        "treated_td",
        "area_id_surv",
        "X_surv",
        "y_ttt",
        "log_exposure_ttt",
        "k_ttt",
        "area_id_ttt",
        "X_ttt",
        "node1",
        "node2",
        "scaling_factor",
        "A",
        "surv_x_cols",
        "ttt_x_cols",
        "N_surv",
        "N_ttt",
        "P_surv",
        "P_ttt",
    }
    assert expected_keys.issubset(data.keys())


def test_make_model_data_default_shapes_and_metadata(surv_long, ttt_long, graph):
    data = make_model_data(surv_long, ttt_long, graph, as_jax=False)

    n_surv = len(surv_long)
    n_ttt = len(ttt_long)

    assert data["N_surv"] == n_surv
    assert data["N_ttt"] == n_ttt
    assert data["P_surv"] == 5
    assert data["P_ttt"] == 6

    assert np.asarray(data["y_surv"]).shape == (n_surv,)
    assert np.asarray(data["log_exposure_surv"]).shape == (n_surv,)
    assert np.asarray(data["k_surv"]).shape == (n_surv,)
    assert np.asarray(data["k_post"]).shape == (n_surv,)
    assert np.asarray(data["treated_td"]).shape == (n_surv,)
    assert np.asarray(data["area_id_surv"]).shape == (n_surv,)
    assert np.asarray(data["X_surv"]).shape == (n_surv, 5)

    assert np.asarray(data["y_ttt"]).shape == (n_ttt,)
    assert np.asarray(data["log_exposure_ttt"]).shape == (n_ttt,)
    assert np.asarray(data["k_ttt"]).shape == (n_ttt,)
    assert np.asarray(data["area_id_ttt"]).shape == (n_ttt,)
    assert np.asarray(data["X_ttt"]).shape == (n_ttt, 6)

    assert tuple(data["surv_x_cols"]) == (
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "stage_II",
        "stage_III",
    )
    assert tuple(data["ttt_x_cols"]) == (
        "age_per10_centered",
        "cci",
        "ses",
        "sex_male",
        "stage_II",
        "stage_III",
    )

    assert data["A"] == graph.A
    assert np.asarray(data["node1"]).shape == (graph.n_edges,)
    assert np.asarray(data["node2"]).shape == (graph.n_edges,)


def test_make_model_data_default_arrays_match_long_data(surv_long, ttt_long, graph):
    data = make_model_data(surv_long, ttt_long, graph, as_jax=False)

    assert np.array_equal(np.asarray(data["y_surv"]), surv_long["event"].to_numpy(dtype=np.int32))
    assert np.allclose(
        np.asarray(data["log_exposure_surv"]),
        np.log(surv_long["exposure"].to_numpy(dtype=float)),
    )
    assert np.array_equal(np.asarray(data["k_surv"]), surv_long["k"].to_numpy(dtype=np.int32))
    assert np.array_equal(np.asarray(data["k_post"]), surv_long["k_post"].to_numpy(dtype=np.int32))
    assert np.array_equal(
        np.asarray(data["treated_td"]),
        surv_long["treated_td"].to_numpy(dtype=np.int32),
    )
    assert np.array_equal(
        np.asarray(data["area_id_surv"]),
        surv_long["area_id"].to_numpy(dtype=np.int32),
    )

    X_surv_expected = surv_long.loc[
        :,
        ["age_per10_centered", "cci", "tumor_size_log", "stage_II", "stage_III"],
    ].to_numpy(dtype=float)
    assert np.allclose(np.asarray(data["X_surv"]), X_surv_expected)

    assert np.array_equal(np.asarray(data["y_ttt"]), ttt_long["event"].to_numpy(dtype=np.int32))
    assert np.allclose(
        np.asarray(data["log_exposure_ttt"]),
        np.log(ttt_long["exposure"].to_numpy(dtype=float)),
    )
    assert np.array_equal(np.asarray(data["k_ttt"]), ttt_long["k"].to_numpy(dtype=np.int32))
    assert np.array_equal(
        np.asarray(data["area_id_ttt"]),
        ttt_long["area_id"].to_numpy(dtype=np.int32),
    )

    X_ttt_expected = ttt_long.loc[
        :,
        ["age_per10_centered", "cci", "ses", "sex_male", "stage_II", "stage_III"],
    ].to_numpy(dtype=float)
    assert np.allclose(np.asarray(data["X_ttt"]), X_ttt_expected)


def test_make_model_data_custom_x_cols_are_respected(encoded_df, graph):
    surv_long = build_survival_long(
        encoded_df,
        x_cols=["area_id", "age_per10_centered", "cci", "stage_III"],
    )
    ttt_long = build_treatment_long(
        encoded_df,
        x_cols=["area_id", "ses", "sex_male", "stage_II"],
    )

    data = make_model_data(
        surv_long,
        ttt_long,
        graph,
        surv_x_cols=["age_per10_centered", "cci", "stage_III"],
        ttt_x_cols=["ses", "sex_male", "stage_II"],
        as_jax=False,
    )

    assert data["P_surv"] == 3
    assert data["P_ttt"] == 3

    assert tuple(data["surv_x_cols"]) == ("age_per10_centered", "cci", "stage_III")
    assert tuple(data["ttt_x_cols"]) == ("ses", "sex_male", "stage_II")

    X_surv_expected = surv_long.loc[:, ["age_per10_centered", "cci", "stage_III"]].to_numpy(dtype=float)
    X_ttt_expected = ttt_long.loc[:, ["ses", "sex_male", "stage_II"]].to_numpy(dtype=float)

    assert np.allclose(np.asarray(data["X_surv"]), X_surv_expected)
    assert np.allclose(np.asarray(data["X_ttt"]), X_ttt_expected)


def test_make_model_data_area_indices_are_within_graph_range(surv_long, ttt_long, graph):
    data = make_model_data(surv_long, ttt_long, graph, as_jax=False)

    area_surv = np.asarray(data["area_id_surv"])
    area_ttt = np.asarray(data["area_id_ttt"])

    assert np.all(area_surv >= 0)
    assert np.all(area_surv < graph.A)
    assert np.all(area_ttt >= 0)
    assert np.all(area_ttt < graph.A)


def test_make_model_data_preserves_event_totals(surv_long, ttt_long, graph):
    data = make_model_data(surv_long, ttt_long, graph, as_jax=False)

    assert int(np.asarray(data["y_surv"]).sum()) == int(surv_long["event"].sum())
    assert int(np.asarray(data["y_ttt"]).sum()) == int(ttt_long["event"].sum())


def test_make_model_data_missing_required_survival_column_raises(surv_long, ttt_long, graph):
    bad_surv = surv_long.drop(columns=["treated_td"])

    with pytest.raises(ValueError, match="missing required columns"):
        make_model_data(bad_surv, ttt_long, graph, as_jax=False)


def test_make_model_data_missing_required_treatment_column_raises(surv_long, ttt_long, graph):
    bad_ttt = ttt_long.drop(columns=["area_id"])

    with pytest.raises(ValueError, match="missing required columns"):
        make_model_data(surv_long, bad_ttt, graph, as_jax=False)


def test_make_model_data_nonpositive_exposure_raises(surv_long, ttt_long, graph):
    bad_surv = surv_long.copy()
    bad_surv.loc[bad_surv.index[0], "exposure"] = 0.0

    with pytest.raises(ValueError, match="exposure must be strictly positive"):
        make_model_data(bad_surv, ttt_long, graph, as_jax=False)


def test_make_model_data_invalid_binary_survival_field_raises(surv_long, ttt_long, graph):
    bad_surv = surv_long.copy()
    bad_surv.loc[bad_surv.index[0], "treated_td"] = 2

    with pytest.raises(ValueError, match="treated_td must contain only 0/1"):
        make_model_data(bad_surv, ttt_long, graph, as_jax=False)


def test_make_model_data_invalid_k_post_logic_raises(surv_long, ttt_long, graph):
    bad_surv = surv_long.copy()
    untreated_rows = bad_surv["treated_td"] == 0
    bad_surv.loc[untreated_rows.idxmax(), "k_post"] = 0

    with pytest.raises(ValueError, match="treated_td == 0 must have k_post == -1"):
        make_model_data(bad_surv, ttt_long, graph, as_jax=False)


def test_make_model_data_area_id_out_of_range_raises(surv_long, ttt_long, graph):
    bad_surv = surv_long.copy()
    bad_surv.loc[bad_surv.index[0], "area_id"] = graph.A

    with pytest.raises(ValueError, match="area_id contains values >= graph.A"):
        make_model_data(bad_surv, ttt_long, graph, as_jax=False)


def test_make_model_data_missing_requested_x_col_raises(surv_long, ttt_long, graph):
    with pytest.raises(ValueError, match="survival x_cols contain missing columns"):
        make_model_data(
            surv_long,
            ttt_long,
            graph,
            surv_x_cols=["age_per10_centered", "not_a_column"],
            as_jax=False,
        )

    with pytest.raises(ValueError, match="treatment x_cols contain missing columns"):
        make_model_data(
            surv_long,
            ttt_long,
            graph,
            ttt_x_cols=["ses", "not_a_column"],
            as_jax=False,
        )