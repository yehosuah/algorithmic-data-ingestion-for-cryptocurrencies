#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from training.data import load_parquet_dataset, sanitize_market_dataset
from training.feature_eng import augment_market_features


TARGET_COLUMNS = ("hl_spread", "hl_spread_z", "rvol_20", "base_prob")


def _load_frame(path: Path, *, expect_augmented: bool) -> pd.DataFrame:
    df = load_parquet_dataset(path, drop_duplicates=False)
    if not expect_augmented:
        df = augment_market_features(df)
    return df


def _summary(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return stats


def _compare(train_stats: Dict[str, Dict[str, float]], live_stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    diff: Dict[str, Dict[str, float]] = {}
    for col in sorted(set(train_stats) & set(live_stats)):
        train = train_stats[col]
        live = live_stats[col]
        diff[col] = {
            "mean_delta": float(live["mean"] - train["mean"]),
            "std_ratio": float(live["std"] / (train["std"] + 1e-12)),
        }
    return diff


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare training vs live feature summary statistics.")
    ap.add_argument("--train", default="datasets/market_multi_3symbol_1m.parquet", help="Training parquet path.")
    ap.add_argument("--live", default="/tmp/features_debug.parquet", help="Live feature parquet path.")
    ap.add_argument("--expect-live-augmented", action="store_true", help="Skip augmentation for live frame.")
    ap.add_argument("--out", help="Optional JSON file to write the comparison payload.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    train_path = Path(args.train)
    live_path = Path(args.live)
    if not train_path.exists():
        raise SystemExit(f"Training dataset not found: {train_path}")
    if not live_path.exists():
        raise SystemExit(f"Live feature parquet not found: {live_path}")

    train_df = _load_frame(train_path, expect_augmented=False)
    train_df = sanitize_market_dataset(train_df, verbose=True)
    live_df = _load_frame(live_path, expect_augmented=bool(args.expect_live_augmented))
    live_df = sanitize_market_dataset(live_df)

    train_stats = _summary(train_df, TARGET_COLUMNS)
    live_stats = _summary(live_df, TARGET_COLUMNS)
    comparison = _compare(train_stats, live_stats)
    payload = {
        "train_path": str(train_path),
        "live_path": str(live_path),
        "train_rows": int(len(train_df)),
        "live_rows": int(len(live_df)),
        "train_stats": train_stats,
        "live_stats": live_stats,
        "comparison": comparison,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
