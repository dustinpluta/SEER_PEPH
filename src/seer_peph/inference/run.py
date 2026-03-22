"""
seer_peph/inference/run.py
==========================

Thin inference runner for NumPyro models.

This module provides a minimal, stable interface for fitting models against the
`make_model_data(...)` contract without embedding notebook-specific boilerplate
everywhere in the codebase.

Current scope
-------------
- Run NUTS / MCMC for a supplied NumPyro model.
- Return posterior samples, a lightweight summary table, and fitted objects.
- Keep defaults aligned with current project preferences:
    2 chains, 500 warmup, 500 posterior draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS


ModelFn = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class InferenceConfig:
    """
    Configuration for MCMC inference.

    Parameters
    ----------
    num_chains
        Number of MCMC chains.
    num_warmup
        Number of warmup iterations per chain.
    num_samples
        Number of posterior draws per chain.
    target_accept_prob
        NUTS target acceptance probability.
    dense_mass
        Whether to use a dense mass matrix.
    max_tree_depth
        Maximum NUTS tree depth.
    progress_bar
        Whether to display the NumPyro progress bar.
    """

    num_chains: int = 2
    num_warmup: int = 500
    num_samples: int = 500
    target_accept_prob: float = 0.85
    dense_mass: bool = False
    max_tree_depth: int = 10
    progress_bar: bool = True


@dataclass
class InferenceResult:
    """
    Container for fitted inference output.

    Attributes
    ----------
    mcmc
        The fitted NumPyro MCMC object.
    samples
        Posterior samples as returned by `mcmc.get_samples()`.
    summary
        Lightweight summary dictionary from `numpyro.diagnostics.summary`.
    config
        Configuration actually used for fitting.
    """

    mcmc: MCMC
    samples: dict[str, Any]
    summary: dict[str, Any]
    config: InferenceConfig


def run_mcmc(
    model: ModelFn,
    data: Mapping[str, Any],
    *,
    rng_key: Any,
    config: InferenceConfig | None = None,
    init_strategy: Any | None = None,
    extra_fields: tuple[str, ...] = (),
) -> InferenceResult:
    """
    Run NUTS / MCMC for a NumPyro model.

    Parameters
    ----------
    model
        NumPyro model function with signature `model(data)`.
    data
        Model input dictionary, typically from `seer_peph.data.model_data.make_model_data`.
    rng_key
        JAX PRNG key.
    config
        Optional inference configuration. If None, uses project defaults.
    init_strategy
        Optional NumPyro init strategy to pass into NUTS.
    extra_fields
        Optional extra fields to collect during sampling.

    Returns
    -------
    InferenceResult
        Fitted MCMC object, posterior samples, summary, and config.
    """
    cfg = InferenceConfig() if config is None else config
    _validate_config(cfg)
    _validate_run_inputs(model=model, data=data)

    kernel_kwargs: dict[str, Any] = {
        "target_accept_prob": cfg.target_accept_prob,
        "dense_mass": cfg.dense_mass,
        "max_tree_depth": cfg.max_tree_depth,
    }
    if init_strategy is not None:
        kernel_kwargs["init_strategy"] = init_strategy

    kernel = NUTS(model, **kernel_kwargs)
    mcmc = MCMC(
        kernel,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        num_chains=cfg.num_chains,
        progress_bar=cfg.progress_bar,
    )

    mcmc.run(rng_key, data, extra_fields=extra_fields)
    samples = mcmc.get_samples()
    summ = numpyro_summary(samples, group_by_chain=False)

    return InferenceResult(
        mcmc=mcmc,
        samples=samples,
        summary=summ,
        config=cfg,
    )


def summarise_samples(samples: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """
    Convert NumPyro posterior samples into a compact scalar summary.

    For vector-valued parameters, entries are flattened and labeled as
    `name[i]`.

    Parameters
    ----------
    samples
        Posterior sample dictionary from `mcmc.get_samples()`.

    Returns
    -------
    dict
        Nested dictionary with mean, sd, median, q05, q95 for each scalar
        parameter entry.
    """
    out: dict[str, dict[str, float]] = {}

    for name, arr in samples.items():
        x = np.asarray(arr)

        if x.ndim == 1:
            out[name] = _summarise_1d(x)
            continue

        flat = x.reshape(x.shape[0], -1)
        for j in range(flat.shape[1]):
            out[f"{name}[{j}]"] = _summarise_1d(flat[:, j])

    return out


def print_summary(summary_dict: Mapping[str, Mapping[str, float]]) -> None:
    """
    Pretty-print a summary dictionary from `summarise_samples(...)`.
    """
    if len(summary_dict) == 0:
        print("(empty summary)")
        return

    header = f"{'parameter':<24} {'mean':>10} {'sd':>10} {'median':>10} {'q05':>10} {'q95':>10}"
    print(header)
    print("-" * len(header))
    for name, stats in summary_dict.items():
        print(
            f"{name:<24} "
            f"{stats['mean']:>10.4f} "
            f"{stats['sd']:>10.4f} "
            f"{stats['median']:>10.4f} "
            f"{stats['q05']:>10.4f} "
            f"{stats['q95']:>10.4f}"
        )


def _summarise_1d(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "median": float(np.median(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q95": float(np.quantile(x, 0.95)),
    }


def _validate_config(config: InferenceConfig) -> None:
    if config.num_chains <= 0:
        raise ValueError("num_chains must be positive")
    if config.num_warmup <= 0:
        raise ValueError("num_warmup must be positive")
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not (0.0 < config.target_accept_prob < 1.0):
        raise ValueError("target_accept_prob must lie in (0, 1)")
    if config.max_tree_depth <= 0:
        raise ValueError("max_tree_depth must be positive")


def _validate_run_inputs(*, model: ModelFn, data: Mapping[str, Any]) -> None:
    if not callable(model):
        raise TypeError("model must be callable")
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping/dictionary-like object")
    if len(data) == 0:
        raise ValueError("data must not be empty")