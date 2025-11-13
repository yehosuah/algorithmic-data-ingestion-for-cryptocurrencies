# Sanity Checks and Optional Improvements

_Last updated: 2025-11-13 04:43 UTC_

> Update 2025-11-13: Added the sanitizer/parity workflow (multi-symbol parquet, `export_feature_slice.py`, `compare_feature_stats.py`) plus the symbol-gate generator so these checks now align with the gates that scheduler/trading enforce.

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

### Multi-symbol sanitizer + gate config
Use the combined BTC/ETH/SOL parquet when validating relaxed gates:
```bash
python3 - <<'PY'
from training.data import load_parquet_dataset, sanitize_market_dataset
df = load_parquet_dataset("raw/market_multi_3symbol_1m.parquet", drop_duplicates=False)
clean = sanitize_market_dataset(df, verbose=True)
clean.to_parquet("datasets/market_multi_3symbol_1m.parquet", index=False)
PY
python scripts/compute_symbol_gate_config.py \
  --data datasets/market_multi_3symbol_1m.parquet \
  --out release/symbol_gates/market_multi_3symbol_1m.json
```
Check the generated JSON into `release/symbol_gates/` so CI + manifests inherit the same `hl_spread`, `rvol`, and liquidity caps.

## Trading Dry Run Validation

With manifests refreshed and the Docker stack running, validate the scheduler → Redis → trading loop:
1. Ensure `INFER_JOBS` is populated (see `.env.example`) and restart `scheduler` so it reloads manifests from `MODELS_ROOT`.
2. Watch scheduler metrics and logs:
   ```bash
   curl -s http://localhost:9002/metrics | grep scheduler_decision
   docker compose logs -f scheduler | grep decision
   ```
3. Confirm Redis queue health:
   ```bash
   redis-cli -u redis://localhost:6379/0 llen trading:decisions
   ```
   The value should oscillate around 0 as the trading service drains decisions.
4. Inspect trading metrics:
   ```bash
   curl -s http://localhost:9010/metrics | egrep 'trading_trade_attempts_total|trading_position_active'
   ```
   Run for a few minutes and ensure counters advance while positions toggle as expected.
5. Audit persisted state:
   ```bash
   python scripts/verify_trading_redis.py
   ```
   Review the `trading:positions` hash and `trading:audit` stream for gate/trade entries.
6. Grafana dashboards `scheduler-overview` and `trading-overview` visualise queue depth, coverage, trade attempts, and dry-run P&L; keep Prometheus alert rules green throughout the exercise.
7. Export a parity slice and compare it with the sanitized training parquet before loosening gates:
   ```bash
   python scripts/export_feature_slice.py \
     --data-lake-root data_lake/market \
     --base-manifest base_xgb_cost_spread \
     --symbols BTC/USDT,ETH/USDT,SOL/USDT \
     --output /tmp/features_debug.parquet
   python scripts/compare_feature_stats.py \
     --train datasets/market_multi_3symbol_1m.parquet \
     --live /tmp/features_debug.parquet \
     --out release/calibration/latest/feature_parity.json
   ```
   Attach the resulting JSON to your run log so reviewers can see live vs training `hl_spread`, `hl_spread_z`, `rvol_20`, and `base_prob` drift.

## Optional Improvements (Backlog)

Data & Features
- Multi‑symbol, multi‑timeframe coverage (BTC/USDT, ETH/USDT; 1m + 5m).
- Extend feature set (higher‑order returns, regime features, realized volatility variants, microstructure if L2 is added).
- Keep `release/symbol_gates` current by re-running `scripts/compute_symbol_gate_config.py` whenever the sanitized multi-symbol parquet refreshes so manifests, scheduler, and `TRADING_MODELS` stay in sync.
- Social/news: expand RSS sources and add Twitter keys; ensure minute spikes stay ≥5e-4 so the blender’s RSS audit passes (`scripts/build_blender_matrix.py` emits coverage stats). The `/ingest/news` endpoint now uses `fetch_news_rss_once` to persist RSS/API payloads into `NEWS_PATH`, so live feeds can be mirrored in the sanity run.
- On‑chain: add Glassnode metrics (with keys), align to bar closes.

ML & Evaluation
- Walk-forward cross-validation across multiple windows, purging and embargoing data.
- Model zoo: gradient boosting, calibrated probabilities, monotonic constraints, temporal ensembling.
- PnL-centric validation with the relaxed gate artifacts (`models/base_xgb_h120_calmon_spread0`, `models/tcn_h120_calmon_relaxed`, `models/blender_h120_v6`); regression-test via `scripts/report_shortlist.py` and keep `tests/regression` (manifest gating + shortlist) green in CI.
- Leverage the refreshed `scripts/run_oos_eval.py --family {base_xgb,tcn,blender}` interface to generate consistent forward replay diagnostics across all manifests (the CI guardrail wraps the TCN family; blender support keeps ensemble audits in sync).
- Forward gate audit: replay `datasets/blender_matrix_2025-10_to-2025-11_with_preds.parquet` through `models/oos_replay_summary_latest.json` to confirm deployable masks retain coverage (base logs 12 gate hits, blender sustains ≈15.8 % coverage, and the relaxed TCN manifests now clear the guardrail with `gate_coverage 4.73e-4/7.71e-4/4.23e-4` for horizons 60/120/180); keep the archived zero-coverage snapshot for regression.
- Experiment tracking (MLflow/W&B) and reproducible pipelines.
- Capture `gate_smoothing_stride` from blender reports (default 30) and use the stride‑1 sandbox runs (`models/blender_h120_gate_test`, `blender_h120_stride1`, `blender_h120_stride1_v2`) to benchmark turnover ceilings before relaxing manifests further.

Serving & Ops
- Real-time scoring path that mirrors training transformations (avoid skew).
- Feature monitoring: drift detection, data availability SLAs; hook `app/monitoring/model_metrics.py` gauges (`model_gate_coverage_ratio`, `model_rss_minute_spike_share`, `model_probability_sigma`) into dashboards and alert when thresholds (from manifests) are breached.
- Persist feature parity diffs from `scripts/export_feature_slice.py` + `scripts/compare_feature_stats.py` in `release/calibration/latest` for every rehearsal so gate changes cite concrete drift metrics.
- Hardening: retries/circuit breakers, backpressure on ingest, structured logging.
- CI hygiene: `.github/workflows/ci.yml` now runs ingestion service E2E tests, KPI regressions, and a forward replay guardrail that calls `scripts/run_oos_eval.py --family tcn --stride 30` for h60/h120/h180 (fails if `gate_fraction < 5e-4` or `final_equity < 1.2`); extend it with environment-specific smoke checks as needed.
- Exercise the stride-aware batching in `training/infer.predict_tcn` during staging runs so smaller strides do not exhaust memory when evaluating new gates.
- Trading dry-run ops: document how `app/trading/state.py` swaps between file/Redis/Postgres backends, keep `scripts/verify_trading_redis.py` in the runbook, and add smoke tests for `tests/trading/test_service.py` when tweaking queue formats.
