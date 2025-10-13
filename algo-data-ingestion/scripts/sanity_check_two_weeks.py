#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import subprocess as sp
from datetime import datetime, timedelta, timezone
from pathlib import Path


def run(cmd: list[str]) -> int:
    print("\n$", " ".join(cmd))
    return sp.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity check: 2-week market backfill + RSS to Parquet + dataset builds")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--rss", nargs="*", default=[
        "https://news.google.com/rss/search?q=bitcoin",
        "https://feeds.feedburner.com/CoinDesk",
    ])
    args = ap.parse_args()

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=14)
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    # 1) Market backfill (2 weeks)
    rc = run([
        "python", "scripts/backfill_ccxt_parquet.py",
        "--exchange", args.exchange,
        "--symbol", args.symbol,
        "--timeframe", args.timeframe,
        "--start", start_s,
        "--end", end_s,
        "--limit", "1000",
    ])
    if rc != 0:
        print("Backfill failed")
        return rc

    # 2) Build market dataset for the same window
    out_ds = Path("datasets/market_two_weeks.parquet")
    out_ds.parent.mkdir(parents=True, exist_ok=True)
    rc = run([
        "python", "scripts/build_market_dataset.py",
        "--exchange", args.exchange,
        "--symbol", args.symbol,
        "--timeframe", args.timeframe,
        "--start-date", start_s,
        "--end-date", end_s,
        "--out", str(out_ds),
    ])
    if rc != 0:
        print("Market dataset build failed")
        return rc

    # 3) RSS snapshot(s) then training matrix with RSS aggregates
    for feed in args.rss:
        run([
            "python", "scripts/rss_to_parquet.py",
            "--feed", feed,
            "--limit", "200",
            "--start-date", start_s,
            "--end-date", end_s,
        ])

    out_matrix = Path("datasets/training_matrix_two_weeks.parquet")
    rc = run([
        "python", "scripts/build_training_matrix.py",
        "--exchange", args.exchange,
        "--symbol", args.symbol,
        "--timeframe", "1min",
        "--include-rss",
        "--out", str(out_matrix),
    ])
    if rc != 0:
        print("Training matrix build failed")
        return rc

    print("\nSanity check completed.")
    print(f"Market dataset: {out_ds} -> exists={out_ds.exists()}")
    print(f"Training matrix: {out_matrix} -> exists={out_matrix.exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
