#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import time
from typing import Optional, List
import pandas as pd

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.adapters.ccxt_adapter import CCXTAdapter
from app.ingestion_service.utils import write_to_parquet
from app.ingestion_service.config import settings
from app.common.time_norm import add_dt_partition


def _to_epoch_ms(dt_s: str) -> int:
    try:
        if dt_s.isdigit():
            v = int(dt_s)
            # assume seconds if < 1e12
            return v if v >= 1_000_000_000_000 else v * 1000
    except Exception:
        pass
    dt = pd.to_datetime(dt_s, utc=True)
    if pd.isna(dt):
        raise ValueError(f"Could not parse datetime: {dt_s}")
    return int(dt.value // 1_000_000)


def _tf_seconds(tf: str) -> int:
    tf = tf.strip().lower()
    n = int(tf[:-1])
    u = tf[-1]
    return n * (60 if u == 'm' else 3600 if u == 'h' else 86400 if u == 'd' else 60)


async def run(exchange: str, symbol: str, timeframe: str, start_ms: int, end_ms: int, limit: int, sleep_sec: float) -> int:
    adapter = CCXTAdapter(exchange)
    written_rows = 0
    try:
        since = start_ms
        step_ms = _tf_seconds(timeframe) * 1000 * max(1, limit - 1)
        while since <= end_ms:
            df = await adapter.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=limit)
            if df is None or df.empty:
                since += step_ms
                await asyncio.sleep(sleep_sec)
                continue
            # keep only within [start,end]
            df = df[(df['timestamp'] >= pd.to_datetime(start_ms, unit='ms', utc=True)) & (df['timestamp'] <= pd.to_datetime(end_ms, unit='ms', utc=True))]
            if df.empty:
                since += step_ms
                await asyncio.sleep(sleep_sec)
                continue
            # Ensure single-day writes to satisfy writer invariant (one dt per write)
            df2 = df.copy()
            add_dt_partition(df2, ts_col="timestamp")
            unique_dt = sorted([d for d in df2["dt"].dropna().unique().tolist() if d])
            for d in unique_dt:
                day_df = df2[df2["dt"] == d].copy()
                if day_df.empty:
                    continue
                write_to_parquet(day_df, settings.MARKET_PATH, {"exchange": exchange, "symbol": symbol})
                written_rows += len(day_df)
            # advance since to last + 1 step
            last_ts_ms = int(df['timestamp'].max().value // 1_000_000)
            since = max(since + 1, last_ts_ms + _tf_seconds(timeframe) * 1000)
            await asyncio.sleep(sleep_sec)
    finally:
        try:
            await adapter.client.close()
        except Exception:
            pass
    return written_rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Backfill OHLCV to Parquet over a date range using CCXT")
    ap.add_argument("--exchange", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--start", required=True, help="ISO8601 or epoch seconds/milliseconds (UTC)")
    ap.add_argument("--end", required=True, help="ISO8601 or epoch seconds/milliseconds (UTC)")
    ap.add_argument("--limit", type=int, default=1000, help="Candles per request (exchange-dependent)")
    ap.add_argument("--sleep-sec", type=float, default=0.2, help="Sleep between requests to respect rate limits")
    args = ap.parse_args(argv)

    start_ms = _to_epoch_ms(args.start)
    end_ms = _to_epoch_ms(args.end)
    if start_ms > end_ms:
        raise SystemExit("start must be <= end")

    written = asyncio.run(run(args.exchange, args.symbol, args.timeframe, start_ms, end_ms, args.limit, args.sleep_sec))
    print(f"Backfill complete: rows_written={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
