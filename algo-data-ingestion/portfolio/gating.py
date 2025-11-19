from __future__ import annotations

from typing import Dict, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd


ThresholdConfig = Dict[str, object]


def _resolve_thresholds_for_row(
    threshold_config: ThresholdConfig,
    regime_value: Optional[object],
) -> MutableMapping[str, float]:
    """
    Merge global + regime-level overrides for a single row.
    """
    base: MutableMapping[str, float] = {}
    global_cfg = threshold_config.get("global", {})
    if isinstance(global_cfg, Mapping):
        base.update({k: float(v) for k, v in global_cfg.items() if v is not None})
    regime_cfg = {}
    if regime_value is not None:
        by_regime = threshold_config.get("by_regime") or {}
        if isinstance(by_regime, Mapping):
            key = str(regime_value)
            if key in by_regime and isinstance(by_regime[key], Mapping):
                regime_cfg = by_regime[key]
    if isinstance(regime_cfg, Mapping):
        for k, v in regime_cfg.items():
            if v is None:
                continue
            try:
                base[k] = float(v)
            except (TypeError, ValueError):
                continue
    return base


def apply_thresholds_to_probs(
    probs: np.ndarray,
    df: pd.DataFrame,
    threshold_config: ThresholdConfig,
    regime_col: str | None = None,
    gate_mask: Optional[pd.Series] = None,
) -> np.ndarray:
    """
    Convert calibrated probabilities into trading signals (+1, 0, -1).

    The function mirrors the dual-threshold, min-hold convention already used
    by the dry-run trading service:
        - entry_long / entry_short open positions
        - exit_long / exit_short close positions
        - optional min_hold_bars keeps a position active before exits engage
        - long_only forces shorts off
        - gate_mask (e.g., manifest gate) zeroes signals when False
    """
    probs = np.asarray(probs, dtype=float)
    if len(probs) != len(df):
        raise ValueError("Probability array and dataframe must have the same length")

    long_only = bool(threshold_config.get("long_only", False))
    min_hold = int(max(1, threshold_config.get("min_hold_bars", 1)))
    if gate_mask is not None:
        gate_mask = gate_mask.reindex(df.index) if isinstance(gate_mask, pd.Series) else pd.Series(gate_mask)
        gate_mask = gate_mask.fillna(False).astype(bool)

    signals = np.zeros(len(probs), dtype=int)
    current = 0
    hold_remaining = 0

    regime_series: Optional[pd.Series] = None
    if regime_col and regime_col in df.columns:
        regime_series = df[regime_col]

    for i, prob in enumerate(probs):
        regime_val = regime_series.iloc[i] if regime_series is not None else None
        thr_cfg = _resolve_thresholds_for_row(threshold_config, regime_val)
        entry_long = float(thr_cfg.get("entry_long", 0.5))
        exit_long = float(thr_cfg.get("exit_long", entry_long))
        entry_short = float(thr_cfg.get("entry_short", 1.0 - entry_long))
        exit_short = float(thr_cfg.get("exit_short", 1.0 - exit_long))

        if gate_mask is not None and not bool(gate_mask.iloc[i]):
            target = 0
        else:
            target = current
            if current == 1:
                if hold_remaining <= 0 and prob < exit_long:
                    target = 0
            elif current == -1:
                if hold_remaining <= 0 and prob > exit_short:
                    target = 0
            else:
                if prob >= entry_long:
                    target = 1
                elif not long_only and prob <= entry_short:
                    target = -1

            if target != current:
                hold_remaining = min_hold - 1
            else:
                hold_remaining = max(0, hold_remaining - 1)

        current = target
        signals[i] = current

    return signals
