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
from app.scheduler.main import _ensure_required_features  # noqa: E402


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
    # Avoid an unbounded filesystem walk by probing for dt partitions we actually need.
    # Supported layouts:
    # - exchange=.../symbol=.../dt=YYYY-MM-DD/*.parquet
    # - exchange=.../symbol=.../year=YYYY/month=MM/day=D/dt=YYYY-MM-DD/*.parquet
    dt_values = []
    day = earliest_dt
    while day <= now.date():
        dt_values.append(day)
        day += timedelta(days=1)

    def _candidate_dt_dirs(dt_value) -> List[Path]:
        date_str = dt_value.isoformat()
        # Some historical dumps used non-zero-padded month folders (month=8 vs month=08).
        month_variants = {str(dt_value.month), f"{dt_value.month:02d}"}
        dirs = [data_dir / f"dt={date_str}"]
        for month in month_variants:
            dirs.append(
                data_dir
                / f"year={dt_value.year}"
                / f"month={month}"
                / f"day={dt_value.day}"
                / f"dt={date_str}"
            )
        return dirs

    required_columns = ["timestamp", "open", "high", "low", "close", "volume", "symbol", "exchange", "timeframe"]

    def _file_time(path: Path) -> datetime:
        """
        Prefer the `part-<epoch_ms>.parquet` convention used by our market ingesters.
        Falls back to filesystem mtime when parsing fails.
        """
        stem = path.name
        if stem.startswith("part-") and stem.endswith(".parquet"):
            raw = stem[len("part-") : -len(".parquet")]
            try:
                ms = int(raw)
            except Exception:
                ms = None
            if ms:
                try:
                    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
                except Exception:
                    pass
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return now

    # The ingesters write many overlapping snapshots (e.g., limit=360 bars). To cover a
    # multi-day window efficiently, sample snapshot files at a coarse cadence so the
    # union of their 360-bar windows tiles the requested history.
    spacing_minutes = 360
    if history_minutes < spacing_minutes:
        spacing_minutes = max(60, int(history_minutes))

    candidates: List[Path] = []
    for dt_value in sorted(dt_values, reverse=True):
        for dt_dir in _candidate_dt_dirs(dt_value):
            if not dt_dir.exists() or not dt_dir.is_dir():
                continue
            candidates.extend(dt_dir.glob("*.parquet"))

    if not candidates:
        return pd.DataFrame()

    candidates = sorted(candidates, key=_file_time, reverse=True)
    selected: List[Path] = []
    next_cutoff = now
    for pq_path in candidates:
        ts_file = _file_time(pq_path)
        if ts_file > next_cutoff:
            continue
        selected.append(pq_path)
        next_cutoff = ts_file - timedelta(minutes=spacing_minutes)
        if next_cutoff <= cutoff:
            break

    for pq_path in selected:
        try:
            df = pd.read_parquet(pq_path, columns=required_columns)
        except Exception:
            try:
                df = pd.read_parquet(pq_path)
            except Exception:
                continue
        if df is None or df.empty or "timestamp" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if "timeframe" in df.columns:
            df = df[df["timeframe"] == timeframe]
        if df.empty:
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
    # Keep raw columns needed by FEATURE_REGISTRY + label/regime helpers (scheduler parity).
    raw_cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in ohlcv.columns]
    if raw_cols:
        raw = ohlcv[raw_cols].copy()
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
        feats = feats.merge(raw, on="timestamp", how="left")
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

    required_features: List[str] = []
    feature_list_path = manifest_dir / "feature_list.json"
    if feature_list_path.exists():
        try:
            required_features = json.loads(feature_list_path.read_text()) or []
        except Exception:
            required_features = []

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
        features["symbol"] = symbol
        frames.append(features)

    if not frames:
        raise SystemExit("No feature rows were produced; verify data lake contents.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    if required_features:
        combined = _ensure_required_features(
            combined,
            required_features,
            job_id="export_feature_slice",
            timeframe=args.timeframe,
        )

    scored = score_base_with_manifest(
        combined,
        manifest_dir,
        mode="inference",
        model_label=args.base_manifest,
        update_metrics=False,
    )
    cutoff = now - timedelta(minutes=args.lookback_minutes)
    if "timestamp" in scored.columns:
        scored = scored[scored["timestamp"] >= cutoff].reset_index(drop=True)
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out_path, index=False)
    print(
        json.dumps(
            {
                "output": str(out_path),
                "rows": int(len(scored)),
                "symbols": sorted({sym for sym in scored["symbol"].dropna().unique()}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
