#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import json
from typing import Optional, List
import pandas as pd
import numpy as np
import fsspec

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.ingestion_service.config import settings
from app.features.factory.market_factory import build_market_features


def _sanitize_part(val: str) -> str:
    """Mirror partition sanitization used by the writer.
    Replaces '/' with '-' and spaces with '_'.
    """
    return str(val).replace("/", "-").replace(" ", "_")


def _fs_and_root(path: str):
    opts = {}
    if settings.FSSPEC_STORAGE_OPTIONS:
        try:
            opts = json.loads(settings.FSSPEC_STORAGE_OPTIONS)
        except Exception:
            opts = {}
    return fsspec.core.url_to_fs(path, **opts)


def _glob(fs, root: str, pattern: str) -> List[str]:
    try:
        return fs.glob(os.path.join(root, pattern))
    except Exception:
        return []


def load_ohlcv(exchange: str, symbol: str, timeframe: Optional[str] = None) -> pd.DataFrame:
    base = settings.MARKET_PATH
    fs, root = _fs_and_root(base)
    # Read all files for (exchange, symbol)
    sym_part = _sanitize_part(symbol)
    parts = _glob(fs, root, f"exchange={exchange}/symbol={sym_part}/**/*.parquet")
    if not parts:
        # Fallback to legacy/unsanitized symbol partition just in case
        parts = _glob(fs, root, f"exchange={exchange}/symbol={symbol}/**/*.parquet")
    if not parts:
        raise SystemExit(f"No Parquet found under {base}/exchange={exchange}/symbol={symbol}")
    dfs = []
    for p in parts:
        try:
            with fs.open(p, "rb") as f:
                dfs.append(pd.read_parquet(f))
        except Exception:
            continue
    if not dfs:
        raise SystemExit("No readable Parquet files found")
    df = pd.concat(dfs, ignore_index=True)
    # Basic columns expected: timestamp, open, high, low, close, volume
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Optional: filter timeframe if present
    if timeframe and "timeframe" in df.columns:
        try:
            df = df[df["timeframe"].astype(str) == str(timeframe)].copy()
        except Exception:
            pass
    return df.sort_values("timestamp").reset_index(drop=True)


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_next"] = out["close"].pct_change().shift(-1)
    out["y_dir"] = np.where(out["ret_next"] > 0, 1, 0)
    # drop last row without next label
    return out.iloc[:-1].copy()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build market training dataset (features + labels)")
    ap.add_argument("--exchange", required=True)
    ap.add_argument("--symbol", required=True, help="e.g. BTC/USDT")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--out", default="datasets/market.parquet")
    ap.add_argument("--start-date", default=None, help="Optional ISO date to filter from (e.g., 2025-01-01)")
    ap.add_argument("--end-date", default=None, help="Optional ISO date to filter to (e.g., 2025-12-31)")
    args = ap.parse_args(argv)

    raw = load_ohlcv(args.exchange, args.symbol, timeframe=args.timeframe)
    # De-duplicate on timestamp to avoid accidental double-writes across partitions
    if not raw.empty and 'timestamp' in raw.columns:
        raw = raw.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    if args.start_date:
        raw = raw[raw['timestamp'] >= pd.to_datetime(args.start_date, utc=True)]
    if args.end_date:
        raw = raw[raw['timestamp'] <= pd.to_datetime(args.end_date, utc=True)]
    # Ensure identifiers
    for col, val in (("symbol", args.symbol), ("exchange", args.exchange), ("timeframe", args.timeframe)):
        if col not in raw.columns:
            raw[col] = val

    feats = build_market_features(raw)
    merged = feats.merge(raw[["timestamp", "close"]], on="timestamp", how="left")
    ds = build_labels(merged)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ds.to_parquet(args.out, index=False)
    print(f"Wrote dataset: {args.out} rows={len(ds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
