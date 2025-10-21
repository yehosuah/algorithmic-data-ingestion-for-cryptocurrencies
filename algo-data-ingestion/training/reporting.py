from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_FLOAT_DEFAULT = 0.0
_INT_DEFAULT = 0
_BOOL_DEFAULT = False


FieldSpec = Tuple[type, Any, bool]


STANDARD_KPI_SPECS: Dict[str, FieldSpec] = {
    "mean_pnl": (float, _FLOAT_DEFAULT, False),
    "std_pnl": (float, _FLOAT_DEFAULT, False),
    "sharpe": (float, _FLOAT_DEFAULT, False),
    "max_drawdown": (float, _FLOAT_DEFAULT, False),
    "final_equity": (float, 1.0, False),
    "avg_turnover": (float, _FLOAT_DEFAULT, False),
    "total_turnover": (float, _FLOAT_DEFAULT, False),
    "toggle_count": (int, _INT_DEFAULT, False),
    "selected_threshold": (float, np.nan, True),
    "criterion": (str, "final_equity", False),
    "cost_bps": (float, 0.0, False),
    "spread_scale": (float, 0.0, False),
    "slippage_bps": (float, 0.0, False),
    "long_only": (bool, _BOOL_DEFAULT, False),
    "min_hold_bars": (int, 1, False),
    "min_total_turnover": (float, 0.0, False),
    "max_total_turnover": (float, None, True),
}


def _coerce_value(value: Any, caster: type, default: Any, allow_none: bool) -> Any:
    if value is None:
        return None if allow_none else default
    try:
        if caster is bool:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "t"}
            return bool(value)
        if caster is float:
            return float(value)
        if caster is int:
            if isinstance(value, bool):
                return int(value)
            return int(float(value))
        if caster is str:
            return str(value)
        return value
    except (TypeError, ValueError):
        return default


def ensure_kpi_schema(
    report: Mapping[str, Any],
    *,
    overrides: Optional[Mapping[str, Any]] = None,
    extra_fields: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Normalize a reporting dictionary to the standard KPI schema.
    - Coerces numeric types to floats/ints.
    - Injects defaults for missing KPI fields.
    - Adds `kpi_schema_version` marker for downstream consumers.
    """
    normalized: Dict[str, Any] = dict(report)
    if overrides:
        normalized.update(overrides)
    specs = STANDARD_KPI_SPECS.copy()
    if extra_fields:
        for field in extra_fields:
            specs.setdefault(field, (float, np.nan, True))

    for key, (caster, default, allow_none) in specs.items():
        if key in normalized:
            normalized[key] = _coerce_value(normalized.get(key), caster, default, allow_none)
        else:
            normalized[key] = default if not callable(default) else default()

    normalized["kpi_schema_version"] = 1
    return normalized


def social_signal_audit(
    df: pd.DataFrame,
    *,
    min_daily_coverage: float = 0.80,
    min_minute_spike_share: float = 0.0005,
) -> Dict[str, Optional[float] | bool | List[str] | Dict[str, float]]:
    """
    Evaluate RSS coverage metrics to determine whether RSS-derived features are reliable.
    Returns metadata describing coverage and whether the RSS feature set should be used.
    """
    result: Dict[str, Optional[float] | bool | List[str] | Dict[str, float]] = {
        "min_daily_coverage": float(min_daily_coverage),
        "min_minute_spike_share": float(min_minute_spike_share),
        "daily_coverage": None,
        "minute_spike_share": None,
        "passed": True,
        "reasons": [],
        "minute_indicator_column": None,
        "minute_spike_share_candidates": {},
    }

    if "rss_has_signal" in df.columns:
        daily_cov = pd.to_numeric(df["rss_has_signal"], errors="coerce").fillna(0.0).mean()
        result["daily_coverage"] = float(daily_cov)
        if daily_cov < min_daily_coverage:
            result["passed"] = False
            result["reasons"].append("rss_has_signal_below_threshold")
    else:
        result["passed"] = False
        result["reasons"].append("rss_has_signal_missing")

    candidate_shares: Dict[str, float] = {}
    indicator_candidates: Sequence[Tuple[str, float]] = (
        ("rss_spike_proximity_flag", 0.5),
        ("rss_spike_halo", 0.0),
        ("rss_spike_trailing_15", 0.0),
        ("rss_spike_leading_15", 0.0),
        ("rss_spike_decay_fast", 1e-6),
        ("rss_spike_presence", 0.0),
        ("rss_spike_decay_long", 1e-6),
        ("rss_spike_decay", 1e-6),
        ("rss_spike_active", 0.0),
        ("rss_count_minute", 0.0),
    )
    best_col: Optional[str] = None
    best_share: Optional[float] = None
    for col, cutoff in indicator_candidates:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        share = float((series > cutoff).mean())
        candidate_shares[col] = share
        if best_share is None or share > best_share:
            best_share = share
            best_col = col

    if best_share is None:
        result["passed"] = False
        result["reasons"].append("rss_minute_indicator_missing")
    else:
        result["minute_spike_share"] = best_share
        result["minute_indicator_column"] = best_col
        if best_share < min_minute_spike_share:
            result["passed"] = False
            result["reasons"].append("rss_minute_spike_share_below_threshold")
    result["minute_spike_share_candidates"] = candidate_shares

    result["fallback_to_no_rss"] = not result["passed"]
    return result
