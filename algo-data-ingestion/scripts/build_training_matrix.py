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
from urllib.parse import urlparse

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


def load_market(exchange: str, symbol: str, timeframe: Optional[str] = None) -> pd.DataFrame:
    base = settings.MARKET_PATH
    fs, root = _fs_and_root(base)
    sym_part = _sanitize_part(symbol)
    parts = _glob(fs, root, f"exchange={exchange}/symbol={sym_part}/**/*.parquet")
    if not parts:
        # Fallback to legacy/unsanitized symbol partition
        parts = _glob(fs, root, f"exchange={exchange}/symbol={symbol}/**/*.parquet")
    if not parts:
        raise SystemExit("No market Parquet found")
    dfs = []
    for p in parts:
        try:
            with fs.open(p, "rb") as f:
                dfs.append(pd.read_parquet(f))
        except Exception:
            continue
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Optional timeframe filter if column exists
    if timeframe and "timeframe" in df.columns:
        try:
            df = df[df["timeframe"].astype(str) == str(timeframe)].copy()
        except Exception:
            pass
    return df.sort_values("timestamp").reset_index(drop=True)


def resample_agg(df: pd.DataFrame, ts_col: str, timeframe: str, on_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], utc=True)
    d = d.set_index(ts_col)
    feats = {}
    if "sentiment_score" in d.columns:
        feats["sentiment_mean"] = d["sentiment_score"].resample(timeframe).mean()
    feats["count"] = d[on_cols[0]].resample(timeframe).count()
    out = pd.DataFrame(feats)
    out.index.name = "timestamp"
    return out.reset_index()


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_next"] = out["close"].pct_change().shift(-1)
    out["y_dir"] = np.where(out["ret_next"] > 0, 1, 0)
    return out.iloc[:-1].copy()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build integrated training matrix (market + social/news aggregates)")
    ap.add_argument("--exchange", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", default="1min", help="pandas offset like 1min, 5min, 1h")
    ap.add_argument("--include-reddit", action="store_true")
    ap.add_argument("--include-rss", action="store_true")
    ap.add_argument("--out", default="datasets/training_matrix.parquet")
    ap.add_argument("--start-date", default=None, help="Optional ISO date to filter from")
    ap.add_argument("--end-date", default=None, help="Optional ISO date to filter to")
    args = ap.parse_args(argv)

    # Market
    mkt = load_market(args.exchange, args.symbol)
    for col, val in (("symbol", args.symbol), ("exchange", args.exchange), ("timeframe", args.timeframe)):
        if col not in mkt.columns:
            mkt[col] = val
    mkt["timestamp"] = pd.to_datetime(mkt["timestamp"], utc=True)
    mkt = mkt.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    feats = build_market_features(mkt)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
    if args.start_date:
        feats = feats[feats["timestamp"] >= pd.to_datetime(args.start_date, utc=True)]
    if args.end_date:
        feats = feats[feats["timestamp"] <= pd.to_datetime(args.end_date, utc=True)]
    base = feats.merge(mkt[["timestamp", "close"]], on="timestamp", how="left")

    # Aggregates
    agg_frames = []
    if args.include_reddit:
        base_social = settings.SOCIAL_PATH.rstrip("/")
        fs, root = _fs_and_root(base_social)
        parts = _glob(fs, root, "reddit/**/*.parquet")
        dfs = []
        for p in parts[:200]:
            try:
                with fs.open(p, "rb") as f:
                    dfs.append(pd.read_parquet(f))
            except Exception:
                continue
        if dfs:
            r = pd.concat(dfs, ignore_index=True)
            # unify text column presence
            if "ts" in r.columns:
                r["ts"] = pd.to_datetime(r["ts"], utc=True)
            else:
                r["ts"] = pd.to_datetime(r.get("timestamp"), utc=True)
            r_agg = resample_agg(r.rename(columns={"ts": "timestamp"}), "timestamp", args.timeframe, ["id" if "id" in r.columns else "title"])
            # enforce tz-aware and optional window
            r_agg["timestamp"] = pd.to_datetime(r_agg["timestamp"], utc=True)
            if args.start_date:
                r_agg = r_agg[r_agg["timestamp"] >= pd.to_datetime(args.start_date, utc=True)]
            if args.end_date:
                r_agg = r_agg[r_agg["timestamp"] <= pd.to_datetime(args.end_date, utc=True)]
            r_agg.columns = ["timestamp", "reddit_sent_mean", "reddit_count"]
            agg_frames.append(r_agg)

    if args.include_rss:
        base_news = settings.NEWS_PATH.rstrip("/")
        fs, root = _fs_and_root(base_news)
        parts = _glob(fs, root, "rss/**/*.parquet")
        dfs = []
        for p in parts[:200]:
            try:
                with fs.open(p, "rb") as f:
                    dfs.append(pd.read_parquet(f))
            except Exception:
                continue
        if dfs:
            n = pd.concat(dfs, ignore_index=True)
            n["published_at"] = pd.to_datetime(n["published_at"], utc=True, errors="coerce")
            # create a sentiment_score if missing (zero)
            if "sentiment_score" not in n.columns:
                n["sentiment_score"] = 0.0
            n_agg = resample_agg(n.rename(columns={"published_at": "timestamp"}), "timestamp", args.timeframe, ["id"])
            # enforce tz-aware and optional window
            n_agg["timestamp"] = pd.to_datetime(n_agg["timestamp"], utc=True)
            if args.start_date:
                n_agg = n_agg[n_agg["timestamp"] >= pd.to_datetime(args.start_date, utc=True)]
            if args.end_date:
                n_agg = n_agg[n_agg["timestamp"] <= pd.to_datetime(args.end_date, utc=True)]
            n_agg.columns = ["timestamp", "rss_sent_mean", "rss_count"]
            agg_frames.append(n_agg)

    X = base.copy()
    for agg in agg_frames:
        X = X.merge(agg, on="timestamp", how="left")
    X[[c for c in X.columns if c.endswith("_count")]] = X[[c for c in X.columns if c.endswith("_count")]].fillna(0)
    X[[c for c in X.columns if c.endswith("_sent_mean")]] = X[[c for c in X.columns if c.endswith("_sent_mean")]].fillna(0.0)

    Y = build_labels(X)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Y.to_parquet(args.out, index=False)
    print(f"Wrote training matrix: {args.out} rows={len(Y)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
