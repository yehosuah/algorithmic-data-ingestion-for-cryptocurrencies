from __future__ import annotations

from typing import Any

import pandas as pd

# Fallback parameters when microstructure fields are missing
DEFAULT_SPREAD_BPS = 5.0
DEFAULT_FEE_BPS = 1.0
DEFAULT_SLIPPAGE_BPS = 2.0


def estimate_transaction_cost_bps(row: pd.Series) -> float:
    """
    Estimate round-trip cost in basis points for a unit notional trade at time t.
    Uses spread (if available) + fixed fee bps + slippage proxy from volume/liquidity.
    """
    spread_bps = None
    if "spread_bps" in row:
        spread_bps = float(row.get("spread_bps"))
    elif "feat_spread_bps" in row:
        spread_bps = float(row.get("feat_spread_bps"))

    fee_bps = DEFAULT_FEE_BPS
    if "fees_bps" in row:
        try:
            fee_bps = float(row.get("fees_bps"))
        except Exception:
            fee_bps = DEFAULT_FEE_BPS

    slippage_bps = DEFAULT_SLIPPAGE_BPS
    if "feat_liquidity_metric_raw" in row and pd.notna(row.get("feat_liquidity_metric_raw")):
        liq = float(row.get("feat_liquidity_metric_raw"))
        # simple inverse liquidity rule: lower liquidity -> higher slippage
        slippage_bps = float(DEFAULT_SLIPPAGE_BPS * (1.0 + (1.0 / (abs(liq) + 1e-6))))

    spread_component = spread_bps if spread_bps is not None else DEFAULT_SPREAD_BPS
    total_bps = spread_component + fee_bps + slippage_bps
    return float(total_bps)


__all__ = ["estimate_transaction_cost_bps"]
