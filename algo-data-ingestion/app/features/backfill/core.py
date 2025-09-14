from __future__ import annotations
from datetime import date
from typing import Iterable, Optional, Tuple, List
import os, glob
import pandas as pd
from app.ingestion_service.config import settings
from app.features.factory.market_factory import build_market_features
from app.features.store.redis_store import get_store

def _safe_symbol_part(s: str) -> str:
    return s.replace("/", "-").replace(":", "-")

def _date_in_range_from_dt_folder(path: str, start: date, end: date) -> bool:
    try:
        dt_str = os.path.basename(os.path.dirname(path)).split("=", 1)[1]
        d = date.fromisoformat(dt_str)
        return start <= d <= end
    except Exception:
        return False

async def backfill_market(
    exchange: str,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    limit_files: Optional[int] = None,
    nx: bool = False,  # if you later add idempotent writes
) -> Tuple[int, int, int]:
    """
    Returns: (files_scanned, rows_in, feature_rows_written)
    """
    base = settings.MARKET_PATH
    sym_part = _safe_symbol_part(symbol)
    pattern = os.path.join(base, f"exchange={exchange}", f"symbol={sym_part}", "dt=*", "*.parquet")
    files = [p for p in sorted(glob.glob(pattern)) if _date_in_range_from_dt_folder(p, start, end)]
    if limit_files:
        files = files[:limit_files]

    store = get_store()
    files_scanned, rows_in, written = 0, 0, 0

    for p in files:
        df = pd.read_parquet(p)
        if df.empty or "timestamp" not in df.columns:
            files_scanned += 1
            continue
        s = df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(s):
            df["timestamp"] = s.dt.tz_convert("UTC") if getattr(s.dt, "tz", None) is not None else s.dt.tz_localize("UTC")
        else:
            df["timestamp"] = pd.to_datetime(s, utc=True)

        # Ensure required context columns
        for col, val in (("symbol", symbol), ("exchange", exchange), ("timeframe", timeframe)):
            if col not in df.columns:
                df[col] = val

        feats = build_market_features(df)
        rows_in += len(df)
        if feats.empty:
            files_scanned += 1
            continue

        payload_cols = [c for c in ("ret_1", "rsi_14", "hl_spread", "oi_obv") if c in feats.columns]
        items: List[dict] = []
        for _, r in feats.iterrows():
            payload = {c: float(r[c]) for c in payload_cols if pd.notna(r[c])}
            if not payload:
                continue
            items.append({
                "domain": "market",
                "symbol": str(r["symbol"]),
                "timeframe": str(r["timeframe"]),
                "ts": r["timestamp"],
                "payload": payload,
                # "ttl": None,  # optional override per item
                # if you add idempotent support in store: "nx": nx
            })
        if items:
            await store.batch_write(items)
            written += len(items)
        files_scanned += 1

    return files_scanned, rows_in, written
