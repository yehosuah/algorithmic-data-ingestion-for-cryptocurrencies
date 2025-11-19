from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

SamplingPolicyName = Literal[
    "uniform",
    "vol_weighted",
    "liquidity_weighted",
    "regime_balanced",
]


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 0.0, np.inf)
    s = probs.sum()
    if s <= 0:
        return np.full_like(probs, 1.0 / len(probs))
    return probs / s


def apply_sampling_policy(
    df: pd.DataFrame,
    policy: SamplingPolicyName,
    policy_config: Dict[str, Any] | None = None,
    regime_col: Optional[str] = None,
    vol_feature_col: Optional[str] = None,
    liquidity_feature_col: Optional[str] = None,
) -> np.ndarray:
    """
    Return a boolean mask of rows to keep under the specified sampling policy.

    - `uniform`: keep all rows or a random uniform subset (target_fraction).
    - `vol_weighted`: higher keep-probability for higher volatility feature.
    - `liquidity_weighted`: higher keep-probability for higher liquidity feature.
    - `regime_balanced`: oversample/undersample to balance regimes.
    """
    cfg = policy_config or {}
    n = len(df)
    if n == 0:
        return np.array([], dtype=bool)
    if policy == "uniform":
        frac = float(cfg.get("target_fraction", 1.0))
        frac = min(max(frac, 0.0), 1.0)
        if frac >= 0.999:
            return np.ones(n, dtype=bool)
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        return rng.random(n) < frac

    if policy == "vol_weighted":
        col = vol_feature_col or cfg.get("vol_feature_col")
        if col is None or col not in df.columns:
            raise KeyError("vol_feature_col is required for vol_weighted sampling")
        vol = df[col].astype(float).to_numpy()
        if cfg.get("log_scale", False):
            vol = np.log1p(np.clip(vol, 0, None))
        probs = _normalize_probs(vol - np.nanmin(vol) + 1e-6)
        frac = float(cfg.get("target_fraction", 1.0))
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        keep = rng.random(n) < (probs / probs.max()) * frac
        return keep

    if policy == "liquidity_weighted":
        col = liquidity_feature_col or cfg.get("liquidity_feature_col")
        if col is None or col not in df.columns:
            raise KeyError("liquidity_feature_col is required for liquidity_weighted sampling")
        liq = df[col].astype(float).to_numpy()
        probs = _normalize_probs(liq - np.nanmin(liq) + 1e-6)
        frac = float(cfg.get("target_fraction", 1.0))
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        keep = rng.random(n) < (probs / probs.max()) * frac
        return keep

    if policy == "regime_balanced":
        if not regime_col or regime_col not in df.columns:
            raise KeyError("regime_col is required for regime_balanced sampling")
        min_per = int(cfg.get("min_samples_per_regime", 0))
        max_per = int(cfg.get("max_samples_per_regime", 0)) or None
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        masks = []
        for _, g in df.groupby(regime_col, sort=False):
            idx = g.index.to_numpy()
            if len(idx) == 0:
                continue
            if max_per is not None and len(idx) > max_per:
                chosen = rng.choice(idx, size=max_per, replace=False)
            elif len(idx) < min_per:
                chosen = rng.choice(idx, size=min_per, replace=True)
            else:
                chosen = idx
            masks.append(chosen)
        if not masks:
            return np.zeros(n, dtype=bool)
        keep_idx = np.concatenate(masks)
        idx = pd.Index(df.index)
        return idx.isin(keep_idx).astype(bool)

    raise ValueError(f"Unsupported sampling policy: {policy}")
