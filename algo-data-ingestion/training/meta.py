from __future__ import annotations
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd


def rolling_vol(logret: pd.Series, window: int = 50) -> pd.Series:
    return logret.rolling(window, min_periods=window//2).std().fillna(method="bfill")


def triple_barrier_events(
    close: pd.Series,
    *,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_hold: int = 60,
    vol: Optional[pd.Series] = None,
    pt_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate triple-barrier events. Uses relative returns on close.
    - pt/sl are applied to cumulative return threshold derived from volatility.
    - max_hold is time barrier in bars.
    Returns DataFrame with columns: t_end, ret, label (1 success, 0 fail)
    """
    c = close.astype(float).values
    ts = pd.to_datetime(close.index if isinstance(close.index, pd.DatetimeIndex) else pd.to_datetime(close.index, unit='s'), utc=True, errors='ignore')
    if not isinstance(ts, pd.DatetimeIndex):
        # fallback to sequential index if timestamps not provided
        ts = pd.DatetimeIndex(pd.to_datetime(np.arange(len(c)), unit='s'))

    logret = np.diff(np.log(np.clip(c, 1e-12, None)), prepend=np.log(np.clip(c[0],1e-12,None)))
    if vol is None:
        # Percent-based PT/SL when vol not provided
        vol = None
        use_percent = True
        pt_pct = float(pt_pct) if pt_pct is not None else 0.005  # 50 bps default
        sl_pct = float(sl_pct) if sl_pct is not None else 0.005
    else:
        vol = vol.reindex(ts).fillna(method="bfill").fillna(method="ffill")
        use_percent = False

    n = len(c)
    t_end = np.empty(n, dtype=object)
    meta_ret = np.zeros(n, dtype=np.float64)
    label = np.zeros(n, dtype=np.int64)

    for i in range(n):
        # dynamic thresholds based on local vol
        if use_percent:
            pt = pt_pct
            sl = sl_pct
        else:
            v = float(vol.iloc[i]) if i < len(vol) else float(np.nan)
            if not np.isfinite(v) or v <= 0:
                v = float(np.nanmean(vol.values)) if np.isfinite(np.nanmean(vol.values)) else 1e-3
            pt = pt_mult * v
            sl = sl_mult * v
        base = c[i]
        j_max = min(n - 1, i + max_hold)
        outcome = 0
        ret_end = 0.0
        j_hit = i
        for j in range(i + 1, j_max + 1):
            rr = (c[j] - base) / base
            if rr >= pt:
                outcome = 1
                ret_end = rr
                j_hit = j
                break
            if rr <= -sl:
                outcome = 0
                ret_end = rr
                j_hit = j
                break
            # time barrier if reaches j_max without hit
            if j == j_max:
                outcome = 1 if rr > 0 else 0
                ret_end = rr
                j_hit = j
        t_end[i] = ts[j_hit]
        meta_ret[i] = ret_end
        label[i] = outcome

    return pd.DataFrame({
        "t_end": pd.to_datetime(t_end, utc=True),
        "ret": meta_ret,
        "label": label,
    }, index=ts)
