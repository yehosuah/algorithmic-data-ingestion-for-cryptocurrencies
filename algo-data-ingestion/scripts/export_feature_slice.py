#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.features.factory.market_factory import build_market_features  # noqa: E402
from training.feature_eng import augment_market_features  # noqa: E402
from training.infer import score_base_with_manifest  # noqa: E402


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export a scheduler-style feature slice (with base probabilities) for parity checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-lake-root", default="data_lake/market", help="Root directory for market parquet data.")
    ap.add_argument("--models-root", default="models", help="Root directory containing manifest bundles.")
    ap.add_argument("--base-manifest", default="base_xgb_cost_spread", help="Manifest directory to use for scoring.")
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT", help="Comma-separated symbols to export.")
    ap.add_argument("--timeframe", default="1m", help="Timeframe string (matches scheduler job config).")
    ap.add_argument("--history-minutes", type=int, default=1440, help="Total minutes of OHLCV history to load.")
    ap.add_argument("--lookback-minutes", type=int, default=360, help="Lookback window used for scoring payloads.")
    ap.add_argument("--output", default="/tmp/features_debug.parquet", help="Destination parquet path.")
    ap.add_argument("--now", help="Optional ISO timestamp override for the evaluation watermark.")
    return ap.parse_args(list(argv) if argv is not None else None)


def _safe_symbol_path(symbol: str) -> str:
    return symbol.replace("/", "-").replace(":", "_")


def _load_recent_ohlcv(
    data_root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    now: datetime,
    history_minutes: int,
) -> pd.DataFrame:
    data_dir = data_root / f"exchange={exchange}" / f"symbol={_safe_symbol_path(symbol)}"
    if not data_dir.exists():
        return pd.DataFrame()
    cutoff = now - timedelta(minutes=history_minutes)
    earliest_dt = (cutoff - timedelta(days=1)).date()
    frames: List[pd.DataFrame] = []
    for dt_dir in sorted(p for p in data_dir.glob("dt=*") if p.is_dir()):
        try:
            dt_value = datetime.strptime(dt_dir.name.split("=", 1)[1], "%Y-%m-%d").date()
        except Exception:
            continue
        if dt_value < earliest_dt or dt_value > now.date():
            continue
        for pq_path in sorted(dt_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(pq_path)
            except Exception:
                continue
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if "timeframe" in df.columns:
        df = df[df["timeframe"] == timeframe]
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = df[(df["timestamp"] >= cutoff) & (df["timestamp"] <= now)]
    return df.reset_index(drop=True)


def _build_feature_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        return pd.DataFrame()
    feats = build_market_features(ohlcv)
    feats = augment_market_features(feats, inplace=False)
    return feats.sort_values("timestamp").reset_index(drop=True)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    now = (
        datetime.fromisoformat(args.now).astimezone(timezone.utc)
        if args.now
        else datetime.now(timezone.utc)
    )
    symbols = [chunk.strip() for chunk in args.symbols.split(",") if chunk.strip()]
    if not symbols:
        raise SystemExit("At least one symbol must be provided via --symbols")

    data_root = Path(args.data_lake_root).expanduser().resolve()
    models_root = Path(args.models_root).expanduser().resolve()
    manifest_dir = models_root / args.base_manifest
    if not manifest_dir.exists():
        raise SystemExit(f"Manifest directory not found: {manifest_dir}")

    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        ohlcv = _load_recent_ohlcv(
            data_root,
            exchange="binance",
            symbol=symbol,
            timeframe=args.timeframe,
            now=now,
            history_minutes=int(args.history_minutes),
        )
        if ohlcv.empty:
            continue
        features = _build_feature_frame(ohlcv)
        if features.empty:
            continue
        scored = score_base_with_manifest(
            features,
            manifest_dir,
            mode="inference",
            model_label=args.base_manifest,
            update_metrics=False,
        )
        scored["symbol"] = symbol
        frames.append(scored)

    if not frames:
        raise SystemExit("No feature rows were produced; verify data lake contents.")

    combined = pd.concat(frames, ignore_index=True)
    cutoff = now - timedelta(minutes=args.lookback_minutes)
    if "timestamp" in combined.columns:
        combined = combined[combined["timestamp"] >= cutoff].reset_index(drop=True)
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(
        json.dumps(
            {
                "output": str(out_path),
                "rows": int(len(combined)),
                "symbols": sorted({sym for sym in combined["symbol"].dropna().unique()}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
