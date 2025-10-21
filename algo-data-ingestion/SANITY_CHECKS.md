# Sanity Checks and Optional Improvements

_Last updated: 2025-10-21 02:50 UTC_

This document summarizes quick validation steps for a 2‑week backfill (market + RSS) and tracks optional improvements to reference during iteration.

## 2‑Week Backfill (Market + RSS)

Run everything from repo root with your Python 3.11 environment.

### One‑shot orchestrator
```bash
python scripts/sanity_check_two_weeks.py \
  --exchange binance \
  --symbol BTC/USDT \
  --timeframe 1m \
  --rss https://news.google.com/rss/search?q=bitcoin https://feeds.feedburner.com/CoinDesk
```
What it does:
- Backfills ~2 weeks of OHLCV to Parquet using CCXT (writes under `MARKET_PATH`).
- Builds a curated dataset (features + labels) for the same window at `datasets/market_two_weeks.parquet`.
- Fetches RSS snapshots (filtered by the 2‑week window) and writes to `NEWS_PATH/rss/...`.
- Builds a training matrix (market + RSS aggregates) at `datasets/training_matrix_two_weeks.parquet`.

### Manual steps (if you prefer)
1) Market backfill
```bash
python scripts/backfill_ccxt_parquet.py \
  --exchange binance \
  --symbol BTC/USDT \
  --timeframe 1m \
  --start 2025-08-01 \
  --end 2025-08-15 \
  --limit 1000
```
2) Market dataset build (same window)
```bash
python scripts/build_market_dataset.py \
  --exchange binance \
  --symbol BTC/USDT \
  --timeframe 1m \
  --start-date 2025-08-01 \
  --end-date 2025-08-15 \
  --out datasets/market_two_weeks.parquet
```
3) RSS → Parquet (same window, multiple feeds OK)
```bash
python scripts/rss_to_parquet.py --feed https://news.google.com/rss/search?q=bitcoin --start-date 2025-08-01 --end-date 2025-08-15
python scripts/rss_to_parquet.py --feed https://feeds.feedburner.com/CoinDesk --start-date 2025-08-01 --end-date 2025-08-15
```
4) Training matrix (market + RSS aggregates)
```bash
python scripts/build_training_matrix.py \
  --exchange binance \
  --symbol BTC/USDT \
  --timeframe 1min \
  --include-rss \
  --out datasets/training_matrix_two_weeks.parquet
```

## Optional Improvements (Backlog)

Data & Features
- Multi‑symbol, multi‑timeframe coverage (BTC/USDT, ETH/USDT; 1m + 5m).
- Extend feature set (higher‑order returns, regime features, realized volatility variants, microstructure if L2 is added).
- Social/news: expand RSS sources and add Twitter keys; ensure minute spikes stay ≥5e-4 so the blender’s RSS audit passes (`scripts/build_blender_matrix.py` emits coverage stats).
- On‑chain: add Glassnode metrics (with keys), align to bar closes.

ML & Evaluation
- Walk‑forward cross‑validation across multiple windows, purging and embargoing data.
- Model zoo: gradient boosting, calibrated probabilities, monotonic constraints, temporal ensembling.
- PnL‑centric validation with the relaxed gate artifacts (`models/base_xgb_h120_calmon_spread0`, `models/tcn_h120_calmon_relaxed`, `models/blender_h120_v6`); regression-test via `scripts/report_shortlist.py`.
- Experiment tracking (MLflow/W&B) and reproducible pipelines.

Serving & Ops
- Real‑time scoring path that mirrors training transformations (avoid skew).
- Feature monitoring: drift detection, data availability SLAs.
- Hardening: retries/circuit breakers, backpressure on ingest, structured logging.
