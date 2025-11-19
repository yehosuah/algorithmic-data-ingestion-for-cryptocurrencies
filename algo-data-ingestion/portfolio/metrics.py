from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _annualize_sharpe(returns: np.ndarray, bars_per_day: float = 24 * 60) -> float:
    if returns.size == 0:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns) + 1e-12)
    return mean / std * np.sqrt(252 * bars_per_day)


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    equity = np.nan_to_num(equity, nan=0.0, posinf=np.inf, neginf=-np.inf)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    dd = np.nan_to_num(dd, nan=0.0, posinf=np.inf)
    return float(np.nanmax(dd))


def compute_portfolio_metrics(
    returns: np.ndarray,
    positions: pd.DataFrame,
    df: pd.DataFrame,
    regimes: Optional[pd.Series] = None,
) -> Dict[str, object]:
    """
    Compute portfolio-level diagnostics from a returns time series and per-row positions.

    The function keeps conventions consistent with training.metrics.summary_stats while
    exposing richer breakdowns (symbol and regime attribution).
    """
    returns = np.asarray(returns, dtype=float)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    # Cumulative P&L path (additive) to avoid overflow in long horizons
    eq = 1.0 + np.cumsum(returns) if returns.size else np.array([], dtype=float)
    gross_by_ts = positions.groupby("timestamp")["position"].apply(lambda s: np.abs(s).sum()) if len(positions) else pd.Series(dtype=float)
    net_by_ts = positions.groupby("timestamp")["position"].sum() if len(positions) else pd.Series(dtype=float)
    turnover_by_ts = positions.groupby("timestamp")["turnover"].sum() if "turnover" in positions else pd.Series(dtype=float)

    per_symbol_pnl = {}
    if "symbol" in positions and "pnl" in positions:
        per_symbol_pnl = positions.groupby("symbol")["pnl"].sum().to_dict()

    per_regime_pnl = {}
    if regimes is not None and "pnl" in positions and "row" in positions:
        aligned = regimes.reindex(df.index)
        pos_with_regime = positions.join(aligned.rename("regime"), on="row")
        per_regime_pnl = pos_with_regime.groupby("regime")["pnl"].sum().to_dict()

    trade_count = int((positions["turnover"] > 0).sum()) if "turnover" in positions else 0
    fraction_time_in_position = float((positions["position"].abs() > 0).mean()) if len(positions) else 0.0

    metrics: Dict[str, object] = {
        "pnl_net": float(np.sum(returns)),
        "sharpe": _annualize_sharpe(returns),
        "max_drawdown": _max_drawdown(eq),
        "turnover": float(turnover_by_ts.mean()) if len(turnover_by_ts) else 0.0,
        "avg_gross_exposure": float(gross_by_ts.mean()) if len(gross_by_ts) else 0.0,
        "avg_net_exposure": float(net_by_ts.mean()) if len(net_by_ts) else 0.0,
        "per_symbol_pnl": per_symbol_pnl,
        "per_regime_pnl": per_regime_pnl,
        "sample_count": int(len(returns)),
        "trade_count": trade_count,
        "fraction_time_in_position": fraction_time_in_position,
    }
    if returns.size:
        metrics["final_equity"] = float(eq[-1])
    return metrics
