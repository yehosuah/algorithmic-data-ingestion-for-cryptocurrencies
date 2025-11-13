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
    gate_mask: Optional[pd.Series] = None,
    min_hold_bars: int = 1,
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

    signal = np.zeros_like(proba)
    if long_only:
        signal[proba >= thr] = 1
    else:
        signal[proba >= thr] = 1
        signal[proba <= 1.0 - thr] = -1

    if gate_mask is not None:
        try:
            gm = gate_mask.astype(bool).to_numpy()
        except AttributeError:
            gm = np.asarray(gate_mask, dtype=bool)
        if len(gm) != len(signal):
            if len(gm) > len(signal):
                gm = gm[-len(signal):]
            else:
                gm = np.concatenate([gm, np.zeros(len(signal) - len(gm), dtype=bool)])
        signal = signal.copy()
        signal[~gm] = 0

    current = 0
    hold_remaining = 0
    min_hold = int(max(1, min_hold_bars))
    pos = np.zeros_like(signal)
    for i in range(len(signal)):
        target = signal[i]
        if current == 0:
            if target != 0:
                current = target
                hold_remaining = min_hold - 1
        else:
            if hold_remaining > 0:
                hold_remaining -= 1
            else:
                if target != current:
                    current = target
                    hold_remaining = min_hold - 1 if current != 0 else 0
        pos[i] = current

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
    pnl = np.clip(pnl, -0.95, 10.0)
    log_equity = np.cumsum(np.log1p(pnl))
    log_equity = np.clip(log_equity, -50.0, 10.0)
    eq = np.exp(log_equity)
    eq = np.clip(eq, 1e-6, 1e6)

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
    toggle_count = 0
    if "turnover" in df.columns:
        toggle_count = int(np.count_nonzero(df["turnover"].values > 0.0))
    return {
        "mean_pnl": mean,
        "std_pnl": std,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": float(eq[-1]) if len(eq) else 1.0,
        "avg_turnover": float(np.mean(df["turnover"])) if "turnover" in df.columns else 0.0,
        "total_turnover": float(np.sum(df["turnover"])) if "turnover" in df.columns else 0.0,
        "toggle_count": toggle_count,
    }
