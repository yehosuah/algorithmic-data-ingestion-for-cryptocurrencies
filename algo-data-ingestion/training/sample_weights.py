from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

WeightPolicyName = Literal[
    "none",
    "cost_inverse",
    "capacity_proportional",
    "cost_capacity_combo",
]


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    m = np.nanmean(arr)
    if not np.isfinite(m) or m == 0:
        return np.ones_like(arr)
    return arr / m


def compute_sample_weights(
    df: pd.DataFrame,
    policy: WeightPolicyName,
    policy_config: Optional[Dict[str, Any]] = None,
    *,
    cost_col: Optional[str] = None,
    liquidity_col: Optional[str] = None,
) -> np.ndarray:
    """
    Returns a 1D array of weights aligned with df rows.
    """
    cfg = policy_config or {}
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    min_w = float(cfg.get("min_weight", 0.0))
    max_w = float(cfg.get("max_weight", np.inf))
    eps = float(cfg.get("eps", 1e-6))

    if policy == "none":
        return np.ones(n, dtype=float)

    if policy == "cost_inverse":
        col = cost_col or cfg.get("cost_col") or "cost_estimate_bps"
        if col not in df.columns:
            return np.ones(n, dtype=float)
        raw = 1.0 / (np.abs(df[col].astype(float).to_numpy()) + eps)
        w = _normalize(raw)
    elif policy == "capacity_proportional":
        col = liquidity_col or cfg.get("liquidity_col") or "feat_rolling_volume_15m"
        if col not in df.columns:
            return np.ones(n, dtype=float)
        raw = np.abs(df[col].astype(float).to_numpy()) + eps
        w = _normalize(raw)
    elif policy == "cost_capacity_combo":
        cost_c = cost_col or cfg.get("cost_col") or "cost_estimate_bps"
        liq_c = liquidity_col or cfg.get("liquidity_col") or "feat_rolling_volume_15m"
        if cost_c not in df.columns or liq_c not in df.columns:
            return np.ones(n, dtype=float)
        cost_raw = np.abs(df[cost_c].astype(float).to_numpy()) + eps
        liq_raw = np.abs(df[liq_c].astype(float).to_numpy()) + eps
        cost_norm = (cost_raw - cost_raw.min()) / (cost_raw.max() - cost_raw.min() + eps)
        liq_norm = (liq_raw - liq_raw.min()) / (liq_raw.max() - liq_raw.min() + eps)
        cost_score = 1.0 / (cost_norm + eps)
        alpha = float(cfg.get("alpha_capacity", 0.5))
        w = alpha * liq_norm + (1 - alpha) * cost_score
        w = _normalize(w)
    else:
        raise ValueError(f"Unsupported weight policy {policy}")

    w = np.clip(w, min_w if np.isfinite(min_w) else None, max_w if np.isfinite(max_w) else None)
    return w.astype(float)
