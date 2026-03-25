from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from seer_peph.fitting.fit_models import (
    FitMetadata,
    JointFit,
    SurvivalFit,
    TreatmentFit,
)
from seer_peph.fitting.io import (
    load_fit,
    load_joint_fit,
    load_survival_fit,
    load_treatment_fit,
    save_fit,
    save_joint_fit,
    save_survival_fit,
    save_treatment_fit,
)
from seer_peph.inference.run import InferenceConfig, InferenceResult


def _dummy_inference_result(samples: dict, summary: dict | None = None) -> InferenceResult:
    return InferenceResult(
        mcmc=None,
        samples=samples,
        summary=summary if summary is not None else {"ok": {"mean": 1.0}},
        config=InferenceConfig(
            num_chains=1,
            num_warmup=10,
            num_samples=20,
            target_accept_prob=0.9,
            dense_mass=False,
            max_tree_depth=5,
            progress_bar=False,
        ),
    )


def _make_survival_fit() -> SurvivalFit:
    samples = {
        "alpha": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        "beta": np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], dtype=float),
        "delta_post": np.array([[-0.2, -0.1], [-0.3, -0.15]], dtype=float),
        "u": np.array([[0.2, -0.2], [0.25, -0.25]], dtype=float),
        "rho": np.array([0.7, 0.75], dtype=float),
        "tau": np.array([0.4, 0.45], dtype=float),
    }
    summary = {"alpha[0]": {"mean": 0.2}, "beta[0]": {"mean": 0.65}}
    scalar_summary = {
        "alpha[0]": {"mean": 0.2, "sd": 0.1, "median": 0.2, "q05": 0.11, "q95": 0.29},
        "beta[0]": {"mean": 0.65, "sd": 0.15, "median": 0.65, "q05": 0.52, "q95": 0.78},
        "rho": {"mean": 0.725, "sd": 0.025, "median": 0.725, "q05": 0.70, "q95": 0.75},
        "tau": {"mean": 0.425, "sd": 0.025, "median": 0.425, "q05": 0.40, "q95": 0.45},
    }
    data = {
        "A": 2,
        "N_surv": 4,
        "N_ttt": 3,
        "P_surv": 3,
        "P_ttt": 2,
        "node1": np.array([0, 1], dtype=int),
        "node2": np.array([1, 0], dtype=int),
        "surv_x_cols": ("age", "cci", "tumor_size_log"),
        "ttt_x_cols": ("age", "cci"),
        "y_surv": np.array([0, 1, 0, 1], dtype=int),
        "log_exposure_surv": np.array([0.0, 0.1, 0.2, 0.3], dtype=float),
    }
    metadata = FitMetadata(
        surv_x_cols=("age", "cci", "tumor_size_log"),
        ttt_x_cols=("age", "cci"),
        surv_breaks=(0.0, 3.0, 6.0),
        ttt_breaks=(0.0, 1.0, 2.0),
        post_ttt_breaks=(0.0, 3.0, 6.0),
        graph_A=2,
        graph_n_edges=2,
        n_surv=4,
        n_ttt=3,
        p_surv=3,
        p_ttt=2,
        rng_seed=123,
    )
    return SurvivalFit(
        model_name="survival_spatial_delta_only",
        inference_result=_dummy_inference_result(samples, summary),
        samples=samples,
        summary=summary,
        scalar_summary=scalar_summary,
        data=data,
        metadata=metadata,
        extra={"note": "survival test"},
    )


def _make_treatment_fit() -> TreatmentFit:
    samples = {
        "gamma": np.array([[-2.0, -2.1], [-2.2, -2.3]], dtype=float),
        "theta": np.array([[0.1, 0.2], [0.15, 0.25]], dtype=float),
        "u": np.array([[0.05, -0.05], [0.1, -0.1]], dtype=float),
        "rho": np.array([0.6, 0.65], dtype=float),
        "tau": np.array([0.3, 0.35], dtype=float),
    }
    summary = {"gamma[0]": {"mean": -2.1}, "theta[0]": {"mean": 0.125}}
    scalar_summary = {
        "gamma[0]": {"mean": -2.1, "sd": 0.1, "median": -2.1, "q05": -2.19, "q95": -2.01},
        "theta[0]": {"mean": 0.125, "sd": 0.025, "median": 0.125, "q05": 0.102, "q95": 0.148},
        "rho": {"mean": 0.625, "sd": 0.025, "median": 0.625, "q05": 0.60, "q95": 0.65},
        "tau": {"mean": 0.325, "sd": 0.025, "median": 0.325, "q05": 0.30, "q95": 0.35},
    }
    data = {
        "A": 2,
        "N_surv": 4,
        "N_ttt": 3,
        "P_surv": 3,
        "P_ttt": 2,
        "node1": np.array([0, 1], dtype=int),
        "node2": np.array([1, 0], dtype=int),
        "surv_x_cols": ("age", "cci", "tumor_size_log"),
        "ttt_x_cols": ("age", "cci"),
        "y_ttt": np.array([1, 0, 1], dtype=int),
        "log_exposure_ttt": np.array([0.0, 0.2, 0.4], dtype=float),
    }
    metadata = FitMetadata(
        surv_x_cols=("age", "cci", "tumor_size_log"),
        ttt_x_cols=("age", "cci"),
        graph_A=2,
        graph_n_edges=2,
        n_surv=4,
        n_ttt=3,
        p_surv=3,
        p_ttt=2,
        rng_seed=456,
    )
    return TreatmentFit(
        model_name="treatment_spatial_pe",
        inference_result=_dummy_inference_result(samples, summary),
        samples=samples,
        summary=summary,
        scalar_summary=scalar_summary,
        data=data,
        metadata=metadata,
        extra={"note": "treatment test"},
    )


def _make_joint_fit() -> JointFit:
    samples = {
        "alpha": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        "beta": np.array([[0.5, 0.6], [0.7, 0.8]], dtype=float),
        "delta_post": np.array([[-0.1, -0.2], [-0.15, -0.25]], dtype=float),
        "gamma": np.array([[-2.0, -2.2], [-2.1, -2.3]], dtype=float),
        "theta": np.array([[0.05, 0.15, 0.25], [0.10, 0.20, 0.30]], dtype=float),
        "u_surv": np.array([[0.2, -0.2], [0.25, -0.25]], dtype=float),
        "u_ttt": np.array([[0.1, -0.1], [0.15, -0.15]], dtype=float),
        "u_ttt_ind": np.array([[0.05, -0.05], [0.10, -0.10]], dtype=float),
        "s_surv": np.array([[1.0, -1.0], [0.9, -0.9]], dtype=float),
        "s_ttt": np.array([[0.6, -0.6], [0.5, -0.5]], dtype=float),
        "rho_surv": np.array([0.8, 0.82], dtype=float),
        "tau_surv": np.array([0.45, 0.50], dtype=float),
        "rho_ttt": np.array([0.7, 0.72], dtype=float),
        "tau_ttt": np.array([0.30, 0.35], dtype=float),
        "rho_u_cross": np.array([0.4, 0.5], dtype=float),
    }
    summary = {"beta[0]": {"mean": 0.6}, "rho_u_cross": {"mean": 0.45}}
    scalar_summary = {
        "beta[0]": {"mean": 0.6, "sd": 0.1, "median": 0.6, "q05": 0.51, "q95": 0.69},
        "rho_u_cross": {"mean": 0.45, "sd": 0.05, "median": 0.45, "q05": 0.40, "q95": 0.50},
        "rho_surv": {"mean": 0.81, "sd": 0.01, "median": 0.81, "q05": 0.80, "q95": 0.82},
        "tau_surv": {"mean": 0.475, "sd": 0.025, "median": 0.475, "q05": 0.45, "q95": 0.50},
        "rho_ttt": {"mean": 0.71, "sd": 0.01, "median": 0.71, "q05": 0.70, "q95": 0.72},
        "tau_ttt": {"mean": 0.325, "sd": 0.025, "median": 0.325, "q05": 0.30, "q95": 0.35},
    }
    data = {
        "A": 2,
        "N_surv": 4,
        "N_ttt": 3,
        "P_surv": 2,
        "P_ttt": 3,
        "node1": np.array([0, 1], dtype=int),
        "node2": np.array([1, 0], dtype=int),
        "surv_x_cols": ("age", "tumor_size_log"),
        "ttt_x_cols": ("age", "cci", "ses"),
        "y_surv": np.array([0, 1, 0, 1], dtype=int),
        "y_ttt": np.array([1, 0, 1], dtype=int),
    }
    metadata = FitMetadata(
        surv_x_cols=("age", "tumor_size_log"),
        ttt_x_cols=("age", "cci", "ses"),
        graph_A=2,
        graph_n_edges=2,
        n_surv=4,
        n_ttt=3,
        p_surv=2,
        p_ttt=3,
        rng_seed=789,
    )
    return JointFit(
        model_name="joint_spatial_treatment_survival",
        inference_result=_dummy_inference_result(samples, summary),
        samples=samples,
        summary=summary,
        scalar_summary=scalar_summary,
        data=data,
        metadata=metadata,
        extra={"note": "joint test", "version": 1},
    )


def _assert_same_arrays_dict(expected: dict, actual: dict) -> None:
    assert set(expected.keys()) == set(actual.keys())
    for key in expected:
        exp = expected[key]
        got = actual[key]
        if isinstance(exp, np.ndarray):
            assert isinstance(got, np.ndarray)
            np.testing.assert_array_equal(exp, got)
        else:
            assert exp == got


def _assert_inference_config_equal(expected: InferenceConfig, actual: InferenceConfig) -> None:
    assert expected == actual


def _assert_fit_round_trip(expected, loaded) -> None:
    assert type(loaded) is type(expected)
    assert loaded.model_name == expected.model_name
    assert loaded.summary == expected.summary
    assert loaded.scalar_summary == expected.scalar_summary
    assert loaded.metadata == expected.metadata
    assert loaded.extra == expected.extra

    _assert_same_arrays_dict(expected.samples, loaded.samples)
    _assert_same_arrays_dict(expected.data, loaded.data)

    assert loaded.inference_result.mcmc is None
    _assert_same_arrays_dict(expected.inference_result.samples, loaded.inference_result.samples)
    assert loaded.inference_result.summary == expected.inference_result.summary
    _assert_inference_config_equal(expected.inference_result.config, loaded.inference_result.config)


def test_save_load_survival_fit_round_trip(tmp_path: Path) -> None:
    fit = _make_survival_fit()
    out_dir = tmp_path / "survival_fit"

    returned = save_fit(fit, out_dir)
    assert returned == out_dir
    assert out_dir.exists()

    loaded = load_fit(out_dir)
    _assert_fit_round_trip(fit, loaded)

    typed = load_survival_fit(out_dir)
    _assert_fit_round_trip(fit, typed)


def test_save_load_treatment_fit_round_trip(tmp_path: Path) -> None:
    fit = _make_treatment_fit()
    out_dir = tmp_path / "treatment_fit"

    returned = save_treatment_fit(fit, out_dir)
    assert returned == out_dir
    assert out_dir.exists()

    loaded = load_fit(out_dir)
    _assert_fit_round_trip(fit, loaded)

    typed = load_treatment_fit(out_dir)
    _assert_fit_round_trip(fit, typed)


def test_save_load_joint_fit_round_trip(tmp_path: Path) -> None:
    fit = _make_joint_fit()
    out_dir = tmp_path / "joint_fit"

    returned = save_joint_fit(fit, out_dir)
    assert returned == out_dir
    assert out_dir.exists()

    loaded = load_fit(out_dir)
    _assert_fit_round_trip(fit, loaded)

    typed = load_joint_fit(out_dir)
    _assert_fit_round_trip(fit, typed)


def test_manifest_and_expected_files_exist(tmp_path: Path) -> None:
    fit = _make_joint_fit()
    out_dir = tmp_path / "joint_fit"

    save_fit(fit, out_dir)

    expected_files = {
        "manifest.json",
        "samples.npz",
        "data_arrays.npz",
        "summary.json",
        "scalar_summary.json",
    }
    found_files = {p.name for p in out_dir.iterdir()}
    assert expected_files.issubset(found_files)

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fit_type"] == "JointFit"
    assert manifest["model_name"] == "joint_spatial_treatment_survival"
    assert manifest["metadata"]["graph_A"] == 2
    assert manifest["extra"]["note"] == "joint test"


def test_typed_loaders_raise_on_wrong_fit_type(tmp_path: Path) -> None:
    fit = _make_survival_fit()
    out_dir = tmp_path / "survival_fit"
    save_fit(fit, out_dir)

    with pytest.raises(TypeError, match="Expected TreatmentFit"):
        load_treatment_fit(out_dir)

    with pytest.raises(TypeError, match="Expected JointFit"):
        load_joint_fit(out_dir)


def test_save_fit_overwrites_existing_directory_contents_safely(tmp_path: Path) -> None:
    fit1 = _make_survival_fit()
    fit2 = _make_treatment_fit()
    out_dir = tmp_path / "fit_bundle"

    save_fit(fit1, out_dir)
    first_loaded = load_fit(out_dir)
    _assert_fit_round_trip(fit1, first_loaded)

    save_fit(fit2, out_dir)
    second_loaded = load_fit(out_dir)
    _assert_fit_round_trip(fit2, second_loaded)