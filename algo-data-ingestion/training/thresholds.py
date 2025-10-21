from __future__ import annotations
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from .metrics import equity_curve, summary_stats
from .reporting import ensure_kpi_schema


def select_prob_threshold(
    ret_next: pd.Series,
    prob: pd.Series,
    *,
    cost_bps: float = 5.0,
    grid: Optional[np.ndarray] = None,
    criterion: str = "final_equity",
    spread_series: Optional[pd.Series] = None,
    spread_scale: float = 0.0,
    slippage_bps: float = 0.0,
    long_only: bool = False,
    gate_mask: Optional[pd.Series] = None,
    min_hold_bars: int = 1,
    min_total_turnover: float = 0.0,
    max_total_turnover: Optional[float] = None,
) -> Tuple[float, Dict]:
    """
    Search probability threshold p* maximizing PnL/Sharpe on provided series.

    - Symmetric long/short: long if p>=p*, short if p<=1-p*.
    - `criterion` in {"final_equity", "sharpe"}.
    """
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.55, 0.90, 15),
            np.linspace(0.905, 0.995, 10),
            np.array([0.9975, 0.9985, 0.9990, 0.9995, 0.99975, 0.9999, 0.99995, 0.99999]),
        ])

    best_thr = float(grid[0])
    best_val = -np.inf
    best_report = {}

    for thr in grid:
        eq = equity_curve(
            ret_next,
            prob,
            threshold=float(thr),
            cost_bps=cost_bps,
            spread_series=spread_series,
            spread_scale=spread_scale,
            slippage_bps=slippage_bps,
            long_only=long_only,
            gate_mask=gate_mask,
            min_hold_bars=min_hold_bars,
        )
        rep = summary_stats(eq)
        turn_val = rep.get("total_turnover", np.nan)
        if not np.isfinite(turn_val) or not np.isfinite(rep.get("final_equity", np.nan)):
            continue
        if rep.get("total_turnover", 0.0) < float(min_total_turnover):
            continue
        if max_total_turnover is not None and rep.get("total_turnover", 0.0) > float(max_total_turnover):
            continue
        val = rep["final_equity"] if criterion == "final_equity" else rep["sharpe"]
        if val > best_val:
            best_val = val
            best_thr = float(thr)
            best_report = rep

    best_report = dict(best_report)
    best_report.update({
        "selected_threshold": best_thr,
        "criterion": criterion,
        "cost_bps": float(cost_bps),
        "spread_scale": float(spread_scale),
        "slippage_bps": float(slippage_bps),
        "long_only": bool(long_only),
        "min_hold_bars": int(max(1, min_hold_bars)),
        "min_total_turnover": float(min_total_turnover),
        "max_total_turnover": float(max_total_turnover) if max_total_turnover is not None else None,
    })
    best_report = ensure_kpi_schema(best_report)
    return best_thr, best_report
