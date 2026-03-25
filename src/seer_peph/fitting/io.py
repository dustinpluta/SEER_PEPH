from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from seer_peph.fitting.fit_models import FitMetadata, JointFit, SurvivalFit, TreatmentFit
from seer_peph.inference.run import InferenceConfig, InferenceResult


FIT_CLASS_TO_NAME = {
    SurvivalFit: "SurvivalFit",
    TreatmentFit: "TreatmentFit",
    JointFit: "JointFit",
}

FIT_NAME_TO_CLASS = {
    "SurvivalFit": SurvivalFit,
    "TreatmentFit": TreatmentFit,
    "JointFit": JointFit,
}


def save_fit(fit: SurvivalFit | TreatmentFit | JointFit, path: str | Path) -> Path:
    """
    Save a fitted model object to a directory bundle.

    The saved representation is designed for downstream analysis, prediction,
    and reporting. It preserves:

    - posterior samples
    - raw summary
    - scalar summary
    - fit metadata
    - extra metadata
    - model-ready data dictionary
    - inference config

    It does not attempt to serialize the live `mcmc` sampler object.
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_type = _fit_type_name(fit)

    samples_np = _coerce_mapping_to_numpy(fit.samples)
    data_np = _coerce_mapping_to_numpy(fit.data)

    sample_arrays, sample_json = _split_array_and_json_values(samples_np, prefix="sample")
    data_arrays, data_json = _split_array_and_json_values(data_np, prefix="data")

    np.savez_compressed(out_dir / "samples.npz", **sample_arrays)
    np.savez_compressed(out_dir / "data_arrays.npz", **data_arrays)

    _write_json(out_dir / "summary.json", _json_ready(fit.summary))
    _write_json(out_dir / "scalar_summary.json", _json_ready(fit.scalar_summary))

    manifest = {
        "fit_type": fit_type,
        "model_name": fit.model_name,
        "metadata": _json_ready(asdict(fit.metadata)),
        "extra": _json_ready(fit.extra),
        "inference_config": _json_ready(asdict(fit.inference_result.config)),
        "sample_json": sample_json,
        "data_json": data_json,
    }
    _write_json(out_dir / "manifest.json", manifest)

    return out_dir


def load_fit(path: str | Path) -> SurvivalFit | TreatmentFit | JointFit:
    """
    Load a fit bundle written by `save_fit(...)`.

    The reconstructed fit object has:
    - `samples`
    - `summary`
    - `scalar_summary`
    - `data`
    - `metadata`
    - `inference_result`

    The reconstructed `inference_result.mcmc` is set to `None`, because the
    live sampler object is not serialized.
    """
    in_dir = Path(path)
    manifest = _read_json(in_dir / "manifest.json")
    summary = _read_json(in_dir / "summary.json")
    scalar_summary = _read_json(in_dir / "scalar_summary.json")

    sample_arrays = _load_npz_dict(in_dir / "samples.npz")
    data_arrays = _load_npz_dict(in_dir / "data_arrays.npz")

    samples = _merge_array_and_json_values(sample_arrays, manifest["sample_json"])
    data = _merge_array_and_json_values(data_arrays, manifest["data_json"])

    for key in ["surv_x_cols", "ttt_x_cols"]:
        if key in data and data[key] is not None:
            data[key] = tuple(data[key])
    
    metadata_dict = dict(manifest["metadata"])

    for key in ["surv_x_cols", "ttt_x_cols", "surv_breaks", "ttt_breaks", "post_ttt_breaks"]:
        if metadata_dict.get(key) is not None:
            metadata_dict[key] = tuple(metadata_dict[key])

    metadata = FitMetadata(**metadata_dict)
    inference_config = InferenceConfig(**manifest["inference_config"])

    inference_result = InferenceResult(
        mcmc=None,
        samples=samples,
        summary=summary,
        config=inference_config,
    )

    fit_cls = FIT_NAME_TO_CLASS[manifest["fit_type"]]

    return fit_cls(
        model_name=manifest["model_name"],
        inference_result=inference_result,
        samples=samples,
        summary=summary,
        scalar_summary=scalar_summary,
        data=data,
        metadata=metadata,
        extra=manifest["extra"],
    )


def save_survival_fit(fit: SurvivalFit, path: str | Path) -> Path:
    return save_fit(fit, path)


def save_treatment_fit(fit: TreatmentFit, path: str | Path) -> Path:
    return save_fit(fit, path)


def save_joint_fit(fit: JointFit, path: str | Path) -> Path:
    return save_fit(fit, path)


def load_survival_fit(path: str | Path) -> SurvivalFit:
    fit = load_fit(path)
    if not isinstance(fit, SurvivalFit):
        raise TypeError(f"Expected SurvivalFit bundle, got {type(fit).__name__}.")
    return fit


def load_treatment_fit(path: str | Path) -> TreatmentFit:
    fit = load_fit(path)
    if not isinstance(fit, TreatmentFit):
        raise TypeError(f"Expected TreatmentFit bundle, got {type(fit).__name__}.")
    return fit


def load_joint_fit(path: str | Path) -> JointFit:
    fit = load_fit(path)
    if not isinstance(fit, JointFit):
        raise TypeError(f"Expected JointFit bundle, got {type(fit).__name__}.")
    return fit


def _fit_type_name(fit: SurvivalFit | TreatmentFit | JointFit) -> str:
    for cls, name in FIT_CLASS_TO_NAME.items():
        if isinstance(fit, cls):
            return name
    raise TypeError(f"Unsupported fit type: {type(fit).__name__}")


def _coerce_mapping_to_numpy(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k): _coerce_value_to_numpy(v) for k, v in mapping.items()}


def _coerce_value_to_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value

    if is_dataclass(value):
        return asdict(value)

    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()

    if isinstance(value, (list, tuple)):
        return [_coerce_value_to_numpy(v) for v in value]

    if isinstance(value, dict):
        return {str(k): _coerce_value_to_numpy(v) for k, v in value.items()}

    # Handle JAX arrays and other array-like objects.
    try:
        arr = np.asarray(value)
        if arr.dtype != object:
            if arr.ndim == 0:
                return arr.item()
            return arr
    except Exception:
        pass

    return value


def _split_array_and_json_values(
    mapping: Mapping[str, Any],
    *,
    prefix: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    json_values: dict[str, Any] = {}

    for key, value in mapping.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            json_values[key] = _json_ready(value)

    return arrays, json_values


def _merge_array_and_json_values(
    arrays: Mapping[str, np.ndarray],
    json_values: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(json_values)
    out.update(arrays)
    return out


def _load_npz_dict(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        return {}
    with np.load(p, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_ready(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()

    if is_dataclass(value):
        return _json_ready(asdict(value))

    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    # Handle JAX arrays and other array-like values that can be converted.
    try:
        arr = np.asarray(value)
        if arr.dtype != object:
            if arr.ndim == 0:
                return arr.item()
            return arr.tolist()
    except Exception:
        pass

    return value