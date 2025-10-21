#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import fsspec
import numpy as np
import pandas as pd

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.ingestion_service.config import settings
from training.data import load_parquet_dataset
from training.feature_eng import augment_market_features
from training.infer import load_base_predictor, predict_base, load_tcn_predictor, predict_tcn


def _parse_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Unable to parse date '{value}'")
    return ts


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


def _resample_agg(df: pd.DataFrame, ts_col: str, timeframe: str, count_col: Optional[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "sentiment_mean", "count"])
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], utc=True)
    d = d.set_index(ts_col)
    feats = {}
    if "sentiment_score" in d.columns:
        feats["sentiment_mean"] = d["sentiment_score"].resample(timeframe).mean()
    if count_col and count_col in d.columns:
        feats["count"] = d[count_col].resample(timeframe).count()
    else:
        feats["count"] = d.resample(timeframe).size()
    out = pd.DataFrame(feats)
    out.index.name = "timestamp"
    return out.reset_index()


def _filter_paths_by_date(paths: Iterable[str], start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> List[str]:
    filt: List[str] = []
    for p in paths:
        if start is None and end is None:
            filt.append(p)
            continue
        match = re.search(r"dt=(\d{4}-\d{2}-\d{2})", p)
        if match is None:
            filt.append(p)
            continue
        dt = pd.to_datetime(match.group(1), utc=True)
        if start is not None and dt < start.tz_convert("UTC").normalize():
            continue
        if end is not None and dt > end.tz_convert("UTC").normalize():
            continue
        filt.append(p)
    return filt


def load_rss_aggregates(
    timeframe: str,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    base_news = settings.NEWS_PATH.rstrip("/")
    fs, root = _fs_and_root(base_news)
    parts = sorted(_glob(fs, root, "rss/**/*.parquet"))
    if not parts:
        return pd.DataFrame(columns=["timestamp", "rss_sent_mean", "rss_count"])
    filtered = _filter_paths_by_date(parts, start, end)
    if max_files is not None and max_files > 0:
        filtered = filtered[:max_files]
    dfs: List[pd.DataFrame] = []
    for path in filtered:
        try:
            with fs.open(path, "rb") as f:
                dfs.append(pd.read_parquet(f))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(columns=["timestamp", "rss_sent_mean", "rss_count"])
    n = pd.concat(dfs, ignore_index=True)
    if "published_at" in n.columns:
        ts_series = n["published_at"]
    elif "timestamp" in n.columns:
        ts_series = n["timestamp"]
    else:
        ts_series = pd.Series(pd.NaT, index=n.index, dtype="datetime64[ns]")
    n["published_at"] = pd.to_datetime(ts_series, utc=True, errors="coerce")
    if "sentiment_score" not in n.columns:
        n["sentiment_score"] = 0.0
    count_col = "id" if "id" in n.columns else ("title" if "title" in n.columns else None)
    agg = _resample_agg(n, "published_at", timeframe, count_col)
    agg.columns = ["timestamp", "rss_sent_mean", "rss_count"]
    return agg


def ensure_horizon_labels(df: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    if "close" not in df.columns:
        raise ValueError("Dataset requires 'close' column to compute horizon returns")
    horizon = max(1, int(horizon))
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[f"ret_next_{horizon}"] = df["close"].pct_change(horizon).shift(-horizon)
    df["ret_next"] = df[f"ret_next_{horizon}"]
    df["y_dir"] = (df["ret_next"] > 0).astype(int)
    df = df.dropna(subset=["ret_next"]).reset_index(drop=True)
    return df


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build blender-ready matrix with market, RSS, base, and TCN features (no gates).")
    ap.add_argument("--source", default="datasets/market_btcusdt_1m_2024_2025.parquet", help="Input market feature parquet")
    ap.add_argument("--out", default="datasets/blender_matrix_year.parquet", help="Output parquet path for blender matrix")
    ap.add_argument("--base-dir", default="models/base_xgb_h120_calmon_spread005", help="Directory containing base XGB artifacts")
    ap.add_argument("--tcn-dir", default="models/tcn_h120_calmon_relaxed", help="Directory containing TCN artifacts")
    ap.add_argument("--timeframe", default="1min", help="Resample timeframe for aggregates (default 1min)")
    ap.add_argument("--start-date", default=None, help="Optional ISO date to filter from (e.g., 2024-01-01)")
    ap.add_argument("--end-date", default=None, help="Optional ISO date to filter to (inclusive)")
    ap.add_argument("--horizon", type=int, default=120, help="Return horizon used for labels (default 120)")
    ap.add_argument("--tcn-stride", type=int, default=30, help="Stride for TCN prediction windows")
    ap.add_argument("--rss-max-files", type=int, default=None, help="Optional cap on RSS parquet files to load")
    ap.add_argument("--include-reddit", action="store_true", help="If set, include Reddit aggregates when available")
    ap.add_argument("--reddit-max-files", type=int, default=None, help="Optional cap on Reddit parquet files to load")
    ap.set_defaults(include_rss=True)
    ap.add_argument("--no-rss", dest="include_rss", action="store_false", help="Disable RSS aggregates")
    args = ap.parse_args(argv)

    start_ts = _parse_date(args.start_date)
    end_ts = _parse_date(args.end_date)

    df = load_parquet_dataset(args.source, drop_duplicates=True)
    if "timestamp" not in df.columns:
        raise ValueError("Input dataset must contain a 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if start_ts is not None:
        df = df[df["timestamp"] >= start_ts]
    if end_ts is not None:
        df = df[df["timestamp"] <= end_ts]
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise SystemExit("No rows after applying date filters – nothing to build.")

    df = augment_market_features(df)
    df = ensure_horizon_labels(df, horizon=args.horizon)

    agg_frames: List[pd.DataFrame] = []
    if args.include_rss:
        rss = load_rss_aggregates(
            args.timeframe,
            start=start_ts,
            end=end_ts,
            max_files=args.rss_max_files,
        )
        if not rss.empty:
            rss["timestamp"] = pd.to_datetime(rss["timestamp"], utc=True)
            agg_frames.append(rss)

    if args.include_reddit:
        base_social = settings.SOCIAL_PATH.rstrip("/")
        fs, root = _fs_and_root(base_social)
        parts = sorted(_glob(fs, root, "reddit/**/*.parquet"))
        if parts:
            filtered = _filter_paths_by_date(parts, start_ts, end_ts)
            if args.reddit_max_files is not None and args.reddit_max_files > 0:
                filtered = filtered[:args.reddit_max_files]
            dfs: List[pd.DataFrame] = []
            for path in filtered:
                try:
                    with fs.open(path, "rb") as f:
                        dfs.append(pd.read_parquet(f))
                except Exception:
                    continue
            if dfs:
                r = pd.concat(dfs, ignore_index=True)
                ts_col = "ts" if "ts" in r.columns else "timestamp"
                r[ts_col] = pd.to_datetime(r[ts_col], utc=True, errors="coerce")
                count_col = "id" if "id" in r.columns else ("title" if "title" in r.columns else None)
                r_agg = _resample_agg(r.rename(columns={ts_col: "timestamp"}), "timestamp", args.timeframe, count_col)
                r_agg.columns = ["timestamp", "reddit_sent_mean", "reddit_count"]
                r_agg["timestamp"] = pd.to_datetime(r_agg["timestamp"], utc=True)
                agg_frames.append(r_agg)

    has_rss = any("rss_count" in frame.columns for frame in agg_frames)
    for frame in agg_frames:
        df = df.merge(frame, on="timestamp", how="left")

    if has_rss:
        if "rss_count" in df.columns:
            df = df.rename(columns={"rss_count": "rss_count_minute"})
        else:
            df["rss_count_minute"] = 0.0
        if "rss_sent_mean" in df.columns:
            df = df.rename(columns={"rss_sent_mean": "rss_sent_mean_minute"})
        else:
            df["rss_sent_mean_minute"] = 0.0
        df["rss_count_minute"] = df["rss_count_minute"].fillna(0.0)
        df["rss_sent_mean_minute"] = df["rss_sent_mean_minute"].fillna(0.0)
        df["rss_sent_mean_signal"] = df["rss_sent_mean_minute"].where(df["rss_count_minute"] > 0)
        df["__rss_date__"] = df["timestamp"].dt.floor("D")
        daily = (
            df.groupby("__rss_date__")
            .agg(
                rss_daily_count_total=("rss_count_minute", "sum"),
                rss_daily_sent_mean=("rss_sent_mean_signal", "mean"),
                minutes_per_day=("timestamp", "count"),
            )
            .reset_index()
        )
        daily["rss_daily_count_avg"] = daily["rss_daily_count_total"] / daily["minutes_per_day"].replace(0, np.nan)
        daily["rss_has_signal"] = (daily["rss_daily_count_total"] > 0).astype(float)
        df = df.merge(daily, on="__rss_date__", how="left")
        df["rss_count"] = df["rss_daily_count_avg"].fillna(0.0)
        df["rss_sent_mean"] = df["rss_daily_sent_mean"].fillna(0.0)
        df["rss_has_signal"] = df["rss_has_signal"].fillna(0.0)
        drop_cols = ["__rss_date__", "rss_sent_mean_signal"]
        drop_cols.extend([c for c in ("minutes_per_day",) if c in df.columns])
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # Build richer intraday RSS features so the blender can access signals even
        # when the minute-level spikes fall between TCN prediction timestamps.
        if "symbol" in df.columns:
            group = df.groupby("symbol", sort=False)
            rollsum_5 = (
                group["rss_count_minute"]
                .rolling(window=5, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            rollsum_15 = (
                group["rss_count_minute"]
                .rolling(window=15, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            rollmax_30 = (
                group["rss_count_minute"]
                .rolling(window=30, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )
            sent_ewm = group["rss_sent_mean_minute"].transform(lambda s: s.ewm(span=15, adjust=False).mean())
        else:
            rollsum_5 = df["rss_count_minute"].rolling(window=5, min_periods=1).sum()
            rollsum_15 = df["rss_count_minute"].rolling(window=15, min_periods=1).sum()
            rollmax_30 = df["rss_count_minute"].rolling(window=30, min_periods=1).max()
            sent_ewm = df["rss_sent_mean_minute"].ewm(span=15, adjust=False).mean()

        df["rss_count_minute_rollsum_5"] = rollsum_5.fillna(0.0)
        df["rss_count_minute_rollsum_15"] = rollsum_15.fillna(0.0)
        rollmax_30 = rollmax_30.fillna(0.0)
        df["rss_spike_active"] = (rollmax_30 > 0.0).astype(float)

        if "symbol" in df.columns:
            df["rss_spike_decay"] = (
                df.groupby("symbol", sort=False)["rss_spike_active"]
                .transform(lambda s: s.ewm(span=30, adjust=False).mean())
                .fillna(0.0)
            )
            df["rss_spike_decay_long"] = (
                df.groupby("symbol", sort=False)["rss_spike_active"]
                .transform(lambda s: s.ewm(span=60, adjust=False).mean())
                .fillna(0.0)
            )
            df["rss_sent_mean_minute_ewm"] = sent_ewm.fillna(0.0)
            df["rss_sent_mean_minute_delta"] = (
                df.groupby("symbol", sort=False)["rss_sent_mean_minute"]
                .diff()
                .fillna(0.0)
            )
        else:
            df["rss_spike_decay"] = df["rss_spike_active"].ewm(span=30, adjust=False).mean().fillna(0.0)
            df["rss_spike_decay_long"] = df["rss_spike_active"].ewm(span=60, adjust=False).mean().fillna(0.0)
            df["rss_sent_mean_minute_ewm"] = sent_ewm.fillna(0.0)
            df["rss_sent_mean_minute_delta"] = df["rss_sent_mean_minute"].diff().fillna(0.0)

        df["rss_spike_presence"] = (df["rss_spike_decay_long"] > 1e-6).astype(float)

        def _compute_spike_windows(spike_series: pd.Series) -> pd.DataFrame:
            arr = spike_series.to_numpy(dtype=float)
            n = len(arr)
            since = np.full(n, np.nan, dtype=float)
            until = np.full(n, np.nan, dtype=float)
            streak = np.zeros(n, dtype=float)
            last_idx = -1
            running = 0
            for i in range(n):
                if arr[i] > 0.0:
                    last_idx = i
                    running += 1
                    streak[i] = running
                    since[i] = 0.0
                else:
                    running = 0
                    streak[i] = 0.0
                    if last_idx != -1:
                        since[i] = float(i - last_idx)
            next_idx = -1
            for i in range(n - 1, -1, -1):
                if arr[i] > 0.0:
                    next_idx = i
                    until[i] = 0.0
                elif next_idx != -1:
                    until[i] = float(next_idx - i)
            return pd.DataFrame({
                "rss_minutes_since_spike": since,
                "rss_minutes_to_next_spike": until,
                "rss_spike_streak": streak,
            }, index=spike_series.index)

        if "symbol" in df.columns:
            time_feats = (
                df.groupby("symbol", sort=False)["rss_spike_active"]
                .apply(_compute_spike_windows)
            )
            time_feats.index = time_feats.index.droplevel(0)
        else:
            time_feats = _compute_spike_windows(df["rss_spike_active"])

        for col in ("rss_minutes_since_spike", "rss_minutes_to_next_spike", "rss_spike_streak"):
            df[col] = pd.to_numeric(time_feats[col], errors="coerce").reindex(df.index).fillna(0.0)

        df["rss_sent_minute_gap"] = (df["rss_sent_mean_minute"] - df["rss_sent_mean"]).fillna(0.0)
        if "symbol" in df.columns:
            df["rss_sent_minute_gap_ewm"] = (
                df.groupby("symbol", sort=False)["rss_sent_minute_gap"]
                .transform(lambda s: s.ewm(span=15, adjust=False).mean())
                .fillna(0.0)
            )
        else:
            df["rss_sent_minute_gap_ewm"] = df["rss_sent_minute_gap"].ewm(span=15, adjust=False).mean().fillna(0.0)

        if "symbol" in df.columns:
            df["rss_spike_velocity"] = (
                df.groupby("symbol", sort=False)["rss_spike_decay"]
                .diff()
                .fillna(0.0)
            )
        else:
            df["rss_spike_velocity"] = df["rss_spike_decay"].diff().fillna(0.0)
        df["rss_count_minute_log1p"] = np.log1p(df["rss_count_minute"].clip(lower=0.0))
        df["rss_sent_mean_minute_abs"] = df["rss_sent_mean_minute"].abs().fillna(0.0)

        if "symbol" in df.columns:
            trailing_15 = (
                df.groupby("symbol", sort=False)["rss_spike_active"]
                .transform(lambda s: s.rolling(window=15, min_periods=1).max())
                .fillna(0.0)
            )
            leading_15 = (
                df.groupby("symbol", sort=False)["rss_spike_active"]
                .transform(lambda s: s.iloc[::-1].rolling(window=15, min_periods=1).max().iloc[::-1])
                .fillna(0.0)
            )
            decay_fast = (
                df.groupby("symbol", sort=False)["rss_spike_active"]
                .transform(lambda s: s.ewm(span=10, adjust=False).mean())
                .fillna(0.0)
            )
        else:
            trailing_15 = df["rss_spike_active"].rolling(window=15, min_periods=1).max().fillna(0.0)
            leading_15 = df["rss_spike_active"].iloc[::-1].rolling(window=15, min_periods=1).max().iloc[::-1].fillna(0.0)
            decay_fast = df["rss_spike_active"].ewm(span=10, adjust=False).mean().fillna(0.0)

        df["rss_spike_trailing_15"] = trailing_15
        df["rss_spike_leading_15"] = leading_15
        df["rss_spike_decay_fast"] = decay_fast
        df["rss_spike_halo"] = np.maximum(trailing_15, leading_15).clip(0.0, 1.0)

        proximity = pd.concat(
            [
                df["rss_minutes_since_spike"].replace(0.0, np.nan),
                df["rss_minutes_to_next_spike"].replace(0.0, np.nan),
            ],
            axis=1,
        ).min(axis=1, skipna=True)
        proximity = proximity.fillna(np.inf)
        df["rss_spike_proximity"] = np.exp(-proximity / 5.0)
        df.loc[np.isinf(proximity), "rss_spike_proximity"] = 0.0
        df["rss_spike_proximity"] = df["rss_spike_proximity"].clip(0.0, 1.0)
        df["rss_spike_proximity_flag"] = (proximity <= 15).astype(float)
        df.loc[np.isinf(proximity), "rss_spike_proximity_flag"] = 0.0

    count_cols = [c for c in df.columns if c.endswith("_count") and c != "rss_count"]
    sent_cols = [c for c in df.columns if c.endswith("_sent_mean") and c != "rss_sent_mean"]
    if count_cols:
        df[count_cols] = df[count_cols].fillna(0)
    if sent_cols:
        df[sent_cols] = df[sent_cols].fillna(0.0)

    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    base_dir = Path(args.base_dir)
    calib_base, feat_cols = load_base_predictor(base_dir)
    base_prob = predict_base(df, calib_base, feat_cols)
    df["base_prob"] = base_prob.values

    tcn_dir = Path(args.tcn_dir)
    model_tcn, calib_tcn, series_cols, scaler, window = load_tcn_predictor(tcn_dir)
    tcn_df = predict_tcn(
        df,
        model_tcn,
        calib_tcn,
        series_cols,
        scaler,
        window,
        stride=max(1, int(args.tcn_stride)),
    )
    if not tcn_df.empty:
        tcn_df["timestamp"] = pd.to_datetime(tcn_df["timestamp"], utc=True)
        df = df.merge(tcn_df, on="timestamp", how="left")
    else:
        df["tcn_prob"] = np.nan

    df = df.sort_values("timestamp").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    summary = {
        "rows": int(len(df)),
        "start": df["timestamp"].min().isoformat(),
        "end": df["timestamp"].max().isoformat(),
        "base_prob_mean": float(df["base_prob"].mean()),
        "tcn_prob_mean": float(df["tcn_prob"].dropna().mean()) if "tcn_prob" in df.columns else None,
    }
    print(json.dumps({"output": str(out_path), "summary": summary, "columns": df.columns.tolist()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
