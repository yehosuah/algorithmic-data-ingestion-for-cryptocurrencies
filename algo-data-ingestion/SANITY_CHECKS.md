# Sanity Checks and Optional Improvements

_Last updated: 2025-11-30 18:55 UTC_

> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-19: Dry-run defaults now point to the promoted portfolio sweep bundle (`experiments/perf_sweeps/medium_xgb_low_cost/portfolio_final/models/final_xgb_primary`) via `configs/deployment_portfolio_contract.yaml` + `configs/dry_run/infer_jobs_portfolio_policy.yaml`; added checks to keep `TRADING_MODELS` and mounts aligned with the contract while watching queue/backlog guardrails.

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
1. Run a quick trigger preflight before booting containers so dry-runs do not start with zero coverage: `python scripts/trigger_preflight.py --contract configs/canonical_training_contract_market_multi_3symbol_1m.yaml --policy configs/final_trigger_policy.yaml --model-dir experiments/perf_sweeps/medium_xgb_low_cost/portfolio_final/models/final_xgb_primary --max-rows 5000`.
1. Ensure `./experiments/perf_sweeps` is mounted (compose already does) and the promoted bundle exists at `experiments/perf_sweeps/medium_xgb_low_cost/portfolio_final/models/final_xgb_primary` (or whatever path your deployment contract references).
1. Point `INFER_JOBS` to `configs/dry_run/infer_jobs_portfolio_policy.yaml` (or inline JSON) and restart `scheduler` so it reloads manifests from `MODELS_ROOT`; keep `TRADING_MODELS` aligned with the contract (`xgb_primary` on ETH/USDT, 1m by default, guard fields for spread/SL/TP/max hold).
1. Watch scheduler metrics and logs:
   ```bash
   curl -s http://localhost:9002/metrics | grep scheduler_decision
   docker compose logs -f scheduler | grep decision
   ```
1. Confirm Redis queue health:
   ```bash
   redis-cli -u redis://localhost:6379/0 llen trading:decisions
   ```
   The value should oscillate around 0 as the trading service drains decisions.
1. Spot-check Redis feature payloads to ensure augmented fields are present (`hl_spread_z`, `rvol_20`, `sym_spread_ratio`, `sym_rvol_ratio`, `sym_liquidity_rank`) and that `close`/`price` columns are populated on the slices feeding inference; if missing, rerun ingest/backfill before trusting the run.
1. Watch backlog safety:
   ```bash
   curl -s http://localhost:9010/metrics | grep trading_decision_queue_depth
   ```
   Keep depth near 0; if it creeps, tune `DECISION_PAYLOAD_ITEMS` (or per-job `max_decision_items`) and `TRADING_QUEUE_POLL_TIMEOUT`. After long downtime, ensure `TRADING_LAST_TS_GRACE_BARS` cleared stale last-processed timestamps (trading startup logs will note any reset).
1. Inspect trading metrics:
   ```bash
   curl -s http://localhost:9010/metrics | egrep 'trading_trade_attempts_total|trading_position_active'
   ```
   Run for a few minutes and ensure counters advance while positions toggle as expected.
1. Validate the new invariants:
   ```bash
   curl -s http://localhost:9010/metrics | egrep 'trading_safe_mode_latched|trading_intent_ledger_state_total|trading_risk_blocked_total|deadlock_action_taken_total|trading_reconcile_runs_total'
   ```
   Safe mode should read 0 during healthy dry-runs, intent ledger counts should increase for each order lifecycle, reconciliation successes should increment every interval, and deadlock action counters should remain flat unless you simulate a coverage stall.
1. Confirm audit provenance:
   ```bash
   python scripts/verify_trading_redis.py --show-audit --limit 3
   ```
   Each entry should include `audit_source`, `audit_run_id`, `audit_seq`, and the HMAC digest; record the run_id in dry-run notes.
1. Audit persisted state:
   ```bash
   python scripts/verify_trading_redis.py
   ```
   Review the `trading:positions` hash and `trading:audit` stream for gate/trade entries.
1. Grafana dashboards `scheduler-overview` and `trading-overview` visualise queue depth, coverage, trade attempts, and dry-run P&L; keep Prometheus alert rules green throughout the exercise.
1. Export a parity slice and compare it with the sanitized training parquet before loosening gates:
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
9. Run the live distribution audit on sampled probabilities to catch drift/collapse before changing gates:
   ```bash
   python3 scripts/probability_distribution_audit.py \
     --samples logs/probability_samples \
     --fold-logits models/base_xgb_h120_calmon_spread0/fold_logits.parquet \
     --fold-column prob_calibrated \
     --features /tmp/features_debug.parquet \
     --hourly-dir release/calibration/latest/live_prob_hourly \
     --summary-out release/calibration/latest/distribution_audit.json
   ```
   Keep `distribution_audit.json` with the run log and alert if collapse/saturation is flagged for any model/prob/timeframe bucket.

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
- Probability drift probes: keep `PROB_SAMPLE_ENABLED=1` so `logs/probability_samples/<model>_<prob>.jsonl` stays fresh, then archive the diagnostics generated by `scripts/plot_probability_distributions.py` + `scripts/run_calibrator_check.py` whenever coverage drops or parity diffs widen.
- Persist feature parity diffs from `scripts/export_feature_slice.py` + `scripts/compare_feature_stats.py` in `release/calibration/latest` for every rehearsal so gate changes cite concrete drift metrics.
- Hardening: retries/circuit breakers, backpressure on ingest, structured logging.
- CI hygiene: `.github/workflows/ci.yml` now runs ingestion service E2E tests, KPI regressions, and a forward replay guardrail that calls `scripts/run_oos_eval.py --family tcn --stride 30` for h60/h120/h180 (fails if `gate_fraction < 5e-4` or `final_equity < 1.2`); extend it with environment-specific smoke checks as needed.
- Exercise the stride-aware batching in `training/infer.predict_tcn` during staging runs so smaller strides do not exhaust memory when evaluating new gates.
- Trading dry-run ops: document how `app/trading/state.py` swaps between file/Redis/Postgres backends, keep `scripts/verify_trading_redis.py` in the runbook, and add smoke tests for `tests/trading/test_service.py` when tweaking queue formats.
