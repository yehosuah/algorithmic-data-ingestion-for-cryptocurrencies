# app/features/backfill/runner.py
from __future__ import annotations

import argparse
import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd

from app.features.ingestion.ccxt_client import CCXTClient
from app.features.factory.market_factory import build_market_features
from training.feature_eng import augment_market_features
from app.features.store.redis_store import get_store
from app.common.time_norm import to_utc_dt


TF_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}

PAYLOAD_COLS = [
    "ret_1",
    "rsi_14",
    "hl_spread",
    "hl_spread_z",
    "rvol_20",
    "sym_spread_ratio",
    "sym_rvol_ratio",
    "sym_liquidity_rank",
    "oi_obv",
]  # keep payload compact but gate-ready


def _parse_time(s: str) -> int:
    """Accepts ISO8601 or epoch seconds; returns epoch seconds (UTC)."""
    s = (s or "").strip()
    if s.isdigit():
        return int(s)
    # ISO8601
    dt = pd.to_datetime(s, utc=True)
    if pd.isna(dt):
        raise ValueError(f"Could not parse time: {s}")
    return int(dt.view("int64") // 1_000_000_000)


@dataclass
class Plan:
    exchange: str
    symbol: str
    timeframe: str
    start: int
    end: int
    batch_size: int   # number of candles per fetch


async def _fetch_chunk(
    client: CCXTClient,
    symbol: str,
    timeframe: str,
    since_s: int,
    limit: int,
) -> pd.DataFrame:
    """
    Call client's historical fetch. Many CCXT methods accept 'since' in ms.
    Our CCXTClient wrapper in this repo handles the conversion internally,
    so we pass epoch seconds and let it normalize. If your wrapper expects ms,
    change to since=since_s * 1000.
    """
    df = await client.fetch_historical(
        symbol=symbol,
        timeframe=timeframe,
        since=since_s,       # if needed switch to since_s * 1000
        limit=limit,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure utc-aware timestamp
    if "timestamp" in df.columns:
        s = df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(s):
            df["timestamp"] = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
        else:
            df["timestamp"] = pd.to_datetime(s, utc=True)
    else:
        raise ValueError("Historical fetch missing 'timestamp' column")

    # Add identity columns if not present
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    if "timeframe" not in df.columns:
        df["timeframe"] = timeframe

    return df


def _mk_items_for_store(feats: pd.DataFrame) -> List[dict]:
    """Convert engineered features -> Redis batch items."""
    if feats.empty:
        return []

    # keep only numeric payloads and drop NaN/Inf (store None when missing)
    rows = []
    for _, r in feats.iterrows():
        payload = {}
        for c in PAYLOAD_COLS:
            if c in feats.columns:
                v = r[c]
                if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    payload[c] = float(v)
                else:
                    payload[c] = None

        rows.append({
            "domain": "market",
            "symbol": str(r["symbol"]),
            "timeframe": str(r["timeframe"]),
            "ts": r["timestamp"],   # tz-aware UTC; store writer will normalize to epoch
            "payload": payload,
        })
    return rows


async def _write_features(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    feats = build_market_features(df)
    feats = augment_market_features(feats, inplace=False)
    if feats.empty:
        return 0

    items = _mk_items_for_store(feats)
    if not items:
        return 0

    store = get_store()
    await store.batch_write(items)
    return len(items)


async def run(plan: Plan) -> Tuple[int, int]:
    """
    Execute backfill:
      - fetch OHLCV in chunks
      - compute features
      - write to Redis
    Returns: (candles_ingested, features_written)
    """
    tf_sec = TF_SECONDS.get(plan.timeframe)
    if not tf_sec:
        raise ValueError(f"Unsupported timeframe: {plan.timeframe}")

    candles_ingested = 0
    features_written = 0

    client = CCXTClient(exchange_name=plan.exchange)
    try:
        current = plan.start
        while current <= plan.end:
            df = await _fetch_chunk(
                client,
                plan.symbol,
                plan.timeframe,
                since_s=current,
                limit=plan.batch_size,
            )
            if df.empty:
                # advance one step to avoid infinite loop
                current += tf_sec * plan.batch_size
                continue

            # Ensure we don't overshoot the end bound
            df = df[df["timestamp"].dt.tz_convert("UTC").astype("int64") // 1_000_000_000 <= plan.end]

            candles_ingested += len(df)
            features_written += await _write_features(df)

            # advance by batch span using the last timestamp we got
            last_ts_s = int(df["timestamp"].max().to_pydatetime().replace(tzinfo=timezone.utc).timestamp())
            # move to the next candle after last
            current = max(current + 1, last_ts_s + tf_sec)

    finally:
        await client.aclose()

    return candles_ingested, features_written


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Backfill market features into Redis")
    p.add_argument("--exchange", required=True, help="e.g. binance")
    p.add_argument("--symbol", required=True, help="e.g. BTC/USDT")
    p.add_argument("--timeframe", required=True, help="e.g. 1m, 5m, 1h")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", help="ISO8601 or epoch seconds (UTC)")
    p.add_argument("--end", help="ISO8601 or epoch seconds (UTC)")
    group.add_argument("--lookback-minutes", type=int, help="Backfill N minutes ending now")
    p.add_argument("--batch-size", type=int, default=500, help="Candles per fetch")

    args = p.parse_args(argv)

    if args.lookback_minutes:
        end = int(datetime.now(tz=timezone.utc).timestamp())
        start = end - args.lookback_minutes * 60
    else:
        if not args.end:
            raise SystemExit("--end is required when --lookback-minutes is not set")
        start = _parse_time(args.start)
        end = _parse_time(args.end)
        if start > end:
            raise SystemExit("start must be <= end")

    plan = Plan(
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=start,
        end=end,
        batch_size=args.batch_size,
    )

    candles, feats = asyncio.run(run(plan))
    print(f"Backfill complete: candles_ingested={candles} features_written={feats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
