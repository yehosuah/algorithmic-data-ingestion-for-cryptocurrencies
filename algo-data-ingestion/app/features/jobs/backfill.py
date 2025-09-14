# app/features/jobs/backfill.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta, date
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd

from app.features.store.redis_store import get_store, RedisFeatureStore
from app.features.factory.market_factory import build_market_features
from app.features.ingestion.ccxt_client import CCXTClient
from app.features.backfill.core import backfill_market as core_backfill_market

logger = logging.getLogger(__name__)


def timeframe_to_seconds(tf: str) -> int:
    tf = tf.strip().lower()
    n = int(tf[:-1])
    u = tf[-1]
    if u == "m":
        return n * 60
    if u == "h":
        return n * 3600
    if u == "d":
        return n * 86400
    raise ValueError(f"Unsupported timeframe: {tf}")


def floor_epoch(ts: int, step: int) -> int:
    return ts - (ts % step)


@dataclass
class BackfillPlan:
    symbol: str
    timeframe: str
    exchange: str
    expected_ts: List[int]
    missing_ts: List[int]


async def plan_missing_market_keys(
    store: RedisFeatureStore,
    *,
    symbol: str,
    timeframe: str,
    lookback_minutes: int,
) -> BackfillPlan:
    step = timeframe_to_seconds(timeframe)
    now = int(datetime.now(tz=timezone.utc).timestamp())
    end = floor_epoch(now, step)
    start = end - lookback_minutes * 60

    expected: List[int] = []
    t = end
    while t >= start:
        expected.append(t)
        t -= step
    expected.reverse()

    # batch_read expects tuples: (domain, symbol, timeframe, ts)
    queries: List[Tuple[str, str, str, int]] = [("market", symbol, timeframe, ts) for ts in expected]
    vals = await store.batch_read(queries)

    missing = [ts for ts, payload in zip(expected, vals) if payload is None]

    return BackfillPlan(
        symbol=symbol,
        timeframe=timeframe,
        exchange="",  # exchange is not part of the key; caller knows which one to fetch from
        expected_ts=expected,
        missing_ts=missing,
    )


async def build_and_write_market_features(
    store: RedisFeatureStore,
    df: pd.DataFrame,
) -> int:
    """Build features from OHLCV and write to Redis."""
    if df is None or df.empty:
        return 0

    feats = build_market_features(df)
    if feats is None or feats.empty:
        return 0

    payload_cols = [c for c in ["ret_1", "rsi_14", "hl_spread", "oi_obv"] if c in feats.columns]
    items: List[Dict[str, Any]] = []
    for _, r in feats.iterrows():
        payload = {c: r[c] for c in payload_cols if pd.notna(r[c])}
        if not payload:
            continue
        items.append(
            {
                "domain": "market",
                "symbol": str(r["symbol"]),
                "timeframe": str(r["timeframe"]),
                "ts": r["timestamp"],  # tz-aware or epoch ok; store will normalize
                "payload": payload,
            }
        )
    if not items:
        return 0
    await store.batch_write(items)
    return len(items)


async def backfill_market_once(
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    lookback_minutes: int = 120,
) -> Dict[str, Any]:
    """One-shot backfill for a single (exchange, symbol, timeframe)."""
    store = get_store()
    plan = await plan_missing_market_keys(store, symbol=symbol, timeframe=timeframe, lookback_minutes=lookback_minutes)

    written = 0
    if plan.missing_ts:
        # Fetch a window that covers missing points (ccxt will return a range)
        limit = max(50, min(2000, int(lookback_minutes * 60 / timeframe_to_seconds(timeframe)) + 5))
        client = CCXTClient(exchange_name=exchange)
        try:
            df = await client.fetch_historical(symbol=symbol, timeframe=timeframe, since=None, limit=limit)
        finally:
            await client.aclose()

        # Filter to the timestamps we care about (align by floored second)
        step = timeframe_to_seconds(timeframe)

        if not df.empty and "timestamp" in df.columns:
            df = df.copy()
            # Ensure tz-aware UTC
            if not str(df["timestamp"].dtype).endswith("UTC]"):
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            # Keep only rows whose aligned epoch is in missing set
            mset = set(plan.missing_ts)
            df["_aligned"] = df["timestamp"].apply(lambda x: floor_epoch(int(pd.Timestamp(x).timestamp()), step))
            df = df[df["_aligned"].isin(mset)]
            df = df.drop(columns=["_aligned"])
            written = await build_and_write_market_features(store, df)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange": exchange,
        "expected": len(plan.expected_ts),
        "missing_before": len(plan.missing_ts),
        "written": written,
    }


async def backfill_market_parquet(
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    limit_files: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Parquet-driven backfill that reuses the shared core implementation.

    Returns:
        dict with keys: exchange, symbol, timeframe, files_scanned, rows_in, written
    """
    files_scanned, rows_in, written = await core_backfill_market(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        limit_files=limit_files,
    )
    return {
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "files_scanned": files_scanned,
        "rows_in": rows_in,
        "written": written,
    }


async def ttl_sweep_once(
    pattern: str = "features:market:*",
    ttl_default: Optional[int] = None,
    max_keys: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Ensure keys under `pattern` have TTL. If ttl_default is provided, set where missing.
    """
    store = get_store()
    client = store._ensure_client()  # intentionally using the client for scan/ttl
    scanned = 0
    fixed = 0

    # Use SCAN to avoid blocking Redis
    async for key in client.scan_iter(match=pattern, count=1000):
        scanned += 1
        ttl = await client.ttl(key)  # -1 no expire, -2 missing, >=0 seconds
        if ttl == -1 and ttl_default and ttl_default > 0:
            ok = await client.expire(key, ttl_default)
            if ok:
                fixed += 1
        if max_keys and scanned >= max_keys:
            break

    return {"pattern": pattern, "scanned": scanned, "ttl_set": fixed}

__all__ = (
    "BackfillPlan",
    "plan_missing_market_keys",
    "build_and_write_market_features",
    "backfill_market_once",
    "backfill_market_parquet",
    "ttl_sweep_once",
    "timeframe_to_seconds",
    "floor_epoch",
)
