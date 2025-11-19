from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def _ensure_weights(model_signals: Dict[str, np.ndarray], model_weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalise weights across the provided models, falling back to uniform when absent.
    """
    weights: Dict[str, float] = {}
    provided = {k: float(v) for k, v in (model_weights or {}).items() if v is not None}
    if not provided:
        # Uniform weights across available signals
        n = max(1, len(model_signals))
        return {name: 1.0 / n for name in model_signals}
    # Keep only weights for models that exist in the signal map
    for name, sig in model_signals.items():
        if name in provided:
            weights[name] = provided[name]
    total = float(sum(weights.values()))
    if total <= 0:
        n = max(1, len(weights) or len(model_signals))
        return {name: 1.0 / n for name in (weights or model_signals)}
    return {k: v / total for k, v in weights.items()}


def combine_model_signals(
    model_signals: Dict[str, np.ndarray],
    model_weights: Dict[str, float],
    mode: str = "weighted_sum",
) -> np.ndarray:
    """
    Combine per-model signals (or probabilities) into a single aggregated signal per row.

    - mode \"weighted_sum\": sum(weight_i * signal_i).
    - mode \"vote\": sign of majority vote (ties -> 0).
    """
    if not model_signals:
        raise ValueError("No model signals provided for ensembling")
    lens = {len(v) for v in model_signals.values()}
    if len(lens) != 1:
        raise ValueError("Model signals must all share the same length")
    weight_map = _ensure_weights(model_signals, model_weights)
    mode = (mode or "weighted_sum").lower()
    stacked = np.stack([model_signals[name] for name in model_signals], axis=0)

    if mode == "vote":
        votes = np.sign(stacked)
        tally = np.sum(votes, axis=0)
        return np.sign(tally)

    if mode != "weighted_sum":
        raise ValueError(f"Unsupported combine mode '{mode}'")

    weights = np.array([weight_map.get(name, 0.0) for name in model_signals], dtype=float)
    return np.tensordot(weights, stacked, axes=1)
