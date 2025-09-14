from __future__ import annotations
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from .metrics import equity_curve, summary_stats


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
) -> Tuple[float, Dict]:
    """
    Search probability threshold p* maximizing PnL/Sharpe on provided series.

    - Symmetric long/short: long if p>=p*, short if p<=1-p*.
    - `criterion` in {"final_equity", "sharpe"}.
    """
    if grid is None:
        grid = np.linspace(0.55, 0.80, 14)

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
        )
        rep = summary_stats(eq)
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
    })
    return best_thr, best_report
