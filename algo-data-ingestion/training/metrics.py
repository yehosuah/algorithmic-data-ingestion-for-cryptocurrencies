from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional


def equity_curve(
    ret_next: pd.Series,
    prob: pd.Series,
    threshold: float,
    cost_bps: float = 5.0,
    *,
    spread_series: Optional[pd.Series] = None,
    spread_scale: float = 0.0,
    slippage_bps: float = 0.0,
    long_only: bool = False,
) -> pd.DataFrame:
    """
    Simple thresholded strategy:
      - position at t is decided with prob at t-1 (one-bar lag)
      - pos ∈ {-1,0,1}; apply symmetric threshold around 0.5
      - PnL_t = pos_{t-1} * ret_next_t - cost_bps * turnover_t
    """
    proba = prob.astype(float).values
    ret = ret_next.astype(float).values
    thr = float(threshold)

    pos = np.zeros_like(proba)
    if long_only:
        pos[proba >= thr] = 1
    else:
        pos[proba >= thr] = 1
        pos[proba <= 1.0 - thr] = -1

    pos_lag = np.roll(pos, 1)
    pos_lag[0] = 0
    turnover = np.abs(pos - pos_lag)
    # Base costs in returns terms
    base_cost = ((cost_bps + slippage_bps) / 1e4) * turnover
    spread_cost = 0.0
    if spread_series is not None and spread_scale != 0.0:
        try:
            ss = spread_series.astype(float).values
            # Align length by padding or trimming
            if len(ss) != len(turnover):
                if len(ss) > len(turnover):
                    ss = ss[-len(turnover):]
                else:
                    pad = np.full(len(turnover) - len(ss), ss[-1] if len(ss) else 0.0)
                    ss = np.concatenate([ss, pad])
            spread_cost = spread_scale * ss * turnover
        except Exception:
            spread_cost = 0.0
    cost = base_cost + spread_cost

    pnl = pos_lag * ret - cost
    eq = np.cumprod(1.0 + pnl)

    return pd.DataFrame({
        "pos": pos,
        "pos_lag": pos_lag,
        "turnover": turnover,
        "pnl": pnl,
        "equity": eq,
    })


def summary_stats(df: pd.DataFrame) -> Dict[str, float]:
    pnl = df["pnl"].values
    eq = df["equity"].values
    ret = pnl
    mean = float(np.mean(ret))
    std = float(np.std(ret) + 1e-12)
    sharpe = mean / std * np.sqrt(252*24*60)  # rough annualization for 1m
    max_dd = float(np.max(np.maximum.accumulate(eq) - eq))
    return {
        "mean_pnl": mean,
        "std_pnl": std,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": float(eq[-1]) if len(eq) else 1.0,
        "avg_turnover": float(np.mean(df["turnover"])) if "turnover" in df.columns else 0.0,
        "total_turnover": float(np.sum(df["turnover"])) if "turnover" in df.columns else 0.0,
    }
