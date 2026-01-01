# Algo Data Ingestion (Docker App)

_Last updated: 2026-01-01 04:19 UTC_

> Update 2026-01-01: Extended the trigger optimizer sweeps with `--disable-prob-exits` + entry filters (`--entry-rsi-min`/`--entry-macd-min`), added profit-fix search-space presets + the latest 48h docker alignment/forensics bundles, and refreshed stage-0/compose defaults.
> Update 2025-12-30: Added price-monitor exits (stop/take-profit/profit-trailing/max-hold) so positions can close even when decision payloads stall, introduced optional entry filters (`entry_rsi_min`/`entry_macd_min`) + `disable_prob_exits` to reduce churn, and refreshed stage-0 sizing/risk limits (capital 200, equity_fraction 0.33, vol-scaled stops with a hard cap).
> Update 2025-12-19: Added a dry-run profit forensics loop (`scripts/extract_container_logs.py` → `analysis/trading_log_forensics.py` → `analysis/market_trade_alignment.py` with `RUNBOOK_DRY_RUN_PROFIT.md`), tightened stage-0 sizing to equity-fraction compounding (capital 100 USDT, base notionals 20/15/12, `max_total_notional=80`, compounding step 5) with per-symbol trigger overrides and longer holds, and refreshed compose/env defaults to mirror the new thresholds while keeping reports/logs out of Docker build context.
> Update 2025-12-17: Added a single-command live-readiness check plus post-launch diagnostics (`analysis/live_readiness_check`, `analysis/acceptance_trade_proof`, `analysis/exit_attribution_report`, `analysis/project_stance_snapshot`) and hardened scheduler/trading with loss-guarding, queue-age pruning, quote-aware exits, and the `INFER_APPLY_CALIBRATION` override so experiments stay aligned with production.

> Update 2025-11-30 22:29 UTC: Stage-0 runtime risk overrides cut the post-exit cooldown to 1 minute and the post-loss cooldown to 5 minutes, the dry-run compose wired BTC/ETH/SOL as primary entries with 300 USDT notional + 10 bps spread guards (no `TRADING_SHADOW_SYMBOLS`), and the bulky parquet datasets were dropped from git—regenerate them via the sanity scripts when you need fresh snapshots. Superseded by the 2025-12-19 equity-fraction sizing drop.
> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-19: Added sampling/weighting knobs to the time-series CV + random-search lane (`--sampling-policy/--weight-policy`, configs under `configs/sampling_*.yaml` + `configs/weights_cost_capacity.yaml`), threaded sample weights through DeepLOB/TCN/Transformer training, and shipped a portfolio performance-sweep harness (`portfolio/run_perf_sweeps.py`). The promoted sweep (`medium_xgb_low_cost`) is codified in `configs/deployment_portfolio_contract.yaml` plus `configs/dry_run/infer_jobs_portfolio_policy.yaml`, with Docker now mounting `experiments/perf_sweeps` so scheduler/trading can load the `xgb_primary` artifact during dry-runs.
>
> Update 2025-11-17: Added a time-series CV + random hyperparameter search lane (`training/time_series_cv.py`, `training/run_hparam_search.py`, `configs/cv_config.yaml`, `configs/hparam_spaces.yaml`), promoted the winning configs to `configs/best_model_configs.{yaml,json}`, and wired the sequence builders + TCN/Transformer trainers with stride-aware batching and P&L-aware early stopping so stride-1 experiments stay memory-safe while monitoring deployable Sharpe/coverage.
>
> Update 2025-11-16: Wired probability sampling into the scheduler/ingestion paths, added the distribution audit + recalibration workflow (`scripts/probability_distribution_audit.py`, refreshed `refresh_calibration.py`), capped enqueued decisions per job (`DECISION_PAYLOAD_ITEMS`/`max_decision_items`) with a queue-depth gauge, and added a restart grace (`TRADING_LAST_TS_GRACE_BARS`) so trading clears stale timestamps after downtime.

End-to-end ingestion for **market**, **on-chain**, **social**, and **news** data with a Redis feature store, admin backfills/TTL sweeps, a scheduler, and monitoring (Prometheus + Grafana).

- **API:** FastAPI (`ingestion-api`)
- **Store:** Redis (+ redis_exporter)
- **Scheduler:** APScheduler job runner (calls admin endpoints)
- **Monitoring:** Prometheus (scrapes API, scheduler, Redis), Grafana dashboards
- **Parquet sink:** Data lake under `/app/data_lake/...`

---

## TL;DR

1) Copy env and set a token:
```bash
cp .env.example .env
# edit .env -> set ADMIN_TOKEN to a strong random string
```

2) Boot (build lean image without heavy ML deps):
```bash
docker compose up -d --build
```

3) Quick checks:
- API health:
  ```bash
  curl -s http://localhost:8000/health
  ```
- API metrics:
  ```bash
  curl -s http://localhost:8000/metrics | head
  ```
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default: admin/admin)
- Trading metrics exporter:
  ```bash
  curl -s http://localhost:9010/metrics | grep trading_trade_attempts_total
  ```

### Trigger optimizer + preflight
1. Run a sweep (quick example):
   ```bash
   python3 -m analysis.trigger_optimizer \
     --contract configs/canonical_training_contract_market_multi_3symbol_1m.yaml \
     --search-space configs/trigger_search_space_quick.yaml \
     --model-dir models/base_xgb_h120_calmon_spread0 \
     --max-rows 20000 \
     --output-dir experiments/trigger_sweeps/eth_quick \
     --promote-best
   ```
   This writes ranked results to `experiments/trigger_sweeps/eth_quick/results.csv` and exports primary/conservative/aggressive to `configs/final_trigger_policy.yaml` (with `meta.active_policy`).
   - Optional runtime-alignment knobs: `--disable-prob-exits` and entry filters (`--entry-rsi-min`/`--entry-macd-min`) to match `TRADING_MODELS` churn-reduction settings.
   - If you pass `--spread-column hl_spread`, the optimizer converts the ratio to bps automatically for spread guards.
   - Search-space presets: `configs/trigger_search_space_profit_fix_*.yaml`, `configs/trigger_search_space_btc_probexits_recent.yaml`.
2. Preflight coverage/trades before starting services:
   ```bash
   python3 scripts/trigger_preflight.py \
     --contract configs/canonical_training_contract_market_multi_3symbol_1m.yaml \
     --model-dir models/base_xgb_h120_calmon_spread0 \
     --max-rows 5000
   ```
   Exits non-zero if predicted coverage or trade-count proxy is near zero.
3. Start services: `docker compose up -d`
   - Model paths in the deployment contract resolve relative to `MODELS_ROOT` (compose sets `/opt/models`). Base artifacts are copied into the image from `models/` (see `Dockerfile`), and optional sweep bundles can still be mounted under `/opt/models/perf_sweeps`.

> Notes
> - La imagen se construye “lean” por defecto (sin instalar `torch`/`transformers`) para acelerar el build. Si necesitas ML dentro del contenedor, puedes construir con `--build-arg INSTALL_ML=1`.
> - Sin llaves API, social/news y on-chain pueden devolver `no_data` o errores 401 del proveedor. Con llaves, funcionará scraping real.
> - Update 2025-11-05: Scheduler inference jobs now publish trading decisions to Redis, the new `trading` service consumes them with dry-run order execution, and Prometheus/Grafana ship with a `trading-overview.json` dashboard plus alert coverage for the new metrics.

---

## Services (docker-compose)

- **ingestion-api** – FastAPI app exposing ingest, features, admin; exposes `/health` and `/metrics`.
- **redis** – Feature store.
- **redis-exporter** – Redis metrics for Prometheus.
- **scheduler** – Calls admin endpoints on cron (market backfill, TTL sweep); exposes `/metrics` on port 9002.
- **prometheus** – Scrapes API, scheduler, redis-exporter.
- **grafana** – Dashboards in `monitoring/grafana/dashboards`.
- **trading** – Consumes `trading:decisions`, enforces manifest/portfolio gates, emits Prometheus metrics, and persists state/audit trails.

Feature payloads are now enriched throughout the stack: `ingestion-api` builds augmented market features during ingest/backfill, writes them to Redis with `hl_spread_z`/`rvol_20`/liquidity ranks included, and the scheduler attaches `close`/`price` columns and backfills any missing manifest features before scoring so gates/trigger sweeps see real prices rather than zeros.

---

## Trading Dry Run

The scheduler now orchestrates three job types:
- Market maintenance (backfills + TTL sweeps).
- Live ingest loops (`MARKET_INGEST_JOBS`) that keep Redis feature TTL topped up.
- **Inference jobs (`INFER_JOBS`)** that replay the latest parquet window, score base/TCN manifests, and enqueue trading payloads to Redis (`DECISION_QUEUE_KEY`, default `trading:decisions`).

Each payload carries manifest metadata, side, and probability, so the trading worker can:
- Stream and dedupe decisions from Redis (`DECISION_QUEUE_URL`), caching artifacts via `training.infer.load_manifest_artifacts` and honoring each manifest’s `apply_calibration` flag (override with `INFER_APPLY_CALIBRATION` when you need deterministic logits in dry runs).
- Backfill missing manifest/labeled columns (regimes, `directional_15m`, `net_return_15m`, etc.) on the fly so the live model always sees the shape it was trained on even when the parquet window is short.
- Apply min-hold and stale-position guards per `TRADING_MODELS`, using spread checks before routing orders with the CCXT adapter. Set `TRADING_DRY_RUN=0` when moving to live execution.
- Persist per-symbol state with the configured backend (`file`, `redis`, `postgres`) and mirror audit events (`gate_toggle`, `trade`) to Redis streams or Postgres tables.
- Expose Prometheus counters/gauges (`trading_trade_attempts_total`, `trading_trade_notional_total`, `trading_gate_toggles_total`, `trading_position_active`, `trading_realized_pnl_total`) on `TRADING_METRICS_PORT` (default 9010).
- Trim per-job decision payloads to the freshest `DECISION_PAYLOAD_ITEMS` (default 3; override per job with `max_decision_items`) so Redis does not accumulate stale signals; monitor `trading_decision_queue_depth{queue="trading:decisions"}` to ensure the consumer keeps up.

Loss-streak protection: the runtime risk limits now expose a `loss_guard` block (default: 3 consecutive losses, 90-minute cooldown, notional scaled by 50%) and the driver emits `loss_guard` metrics/audit entries whenever it rejects a decision.

Decision deduping: `TRADING_LAST_TS_GRACE_BARS` lets trading accept decisions a few bars older than the stored timestamp so short restarts keep running, while `TRADING_DECISION_MAX_AGE_SECONDS` (240 in stage-0) skips stale payloads and keeps the queue healthy instead of reprocessing old signals.

Quote-aware exits: the executor now fetches live bid/ask quotes (with order-book fallbacks) so `decide_bar` gets fresh spreads and computes expected PnL/net before hitting a `pnl_block` skip—spread estimates from quotes only apply when we can trust live prices.

Trading also clears stale dedupe timestamps after restarts when downtime exceeds `TRADING_LAST_TS_GRACE_BARS` bars (default 3) so fresh decisions are not dropped due to old state; increase the grace if using long bar intervals.

The default dry-run compose now loads `configs/dry_run/infer_jobs_portfolio_policy.yaml` and scores the baked-in `xgb_primary` bundle (`models/base_xgb_h120_calmon_spread0` → `/opt/models/base_xgb_h120_calmon_spread0` inside containers). Trading is configured for BTC/ETH/SOL with tight spreads (`max_spread_bps=8`), per-symbol stop-loss/profit-trailing/max-hold guards (see compose `TRADING_MODELS`), and optional churn-reduction knobs (`disable_prob_exits`, `entry_rsi_min`/`entry_macd_min`). Stage-0 risk limits live in `configs/runtime_overrides/risk_limits_stage_0.yaml` (`capital=200`, `equity_fraction=0.33`, `max_total_notional=80`, cooldowns 2/5, vol-scaled stops); note `symbols.BTC/USDT.trigger_overrides.entry_threshold=1.0` currently suppresses BTC entries unless you lower it.

Trading trigger guards are centralized: `TRADING_MODELS` entries now accept `max_spread_bps`, `stop_loss_pct`, `take_profit_pct`, and `max_hold_minutes`, and the live loop shares `app/trading/decision.py::decide_bar` with the offline optimizer so spread/hold/SL/TP checks behave identically in sweeps and production.

> Tip: Let `scheduler` and `trading` run for at least an hour during rehearsals without restarting them—the extra runway keeps the `rvol_20` rolling window stable and gives `trading_trade_attempts_total` / `trading_gate_toggles_total` time to move off zero. Capture start/end counter values in your dry-run log.

Useful helpers:
- `python scripts/verify_trading_redis.py` inspects the Redis position hash and audit stream.
- `python live/decision_logic.py --deployment-contract configs/deployment_portfolio_contract.yaml --dummy-features-path /tmp/live_slice.parquet --output-path /tmp/targets.json` smoke-tests the portfolio policy over a live-like slice without touching Redis.
- Grafana dashboard `monitoring/grafana/dashboards/trading-overview.json` surfaces queue depth, gate coverage, and trade attempt metrics; alert rules now include stale decision queue detection.
- Dry-run profit loop: follow `RUNBOOK_DRY_RUN_PROFIT.md` to capture container evidence (`scripts/extract_container_logs.py`), run forensics (`analysis.trading_log_forensics`), and align trades vs OHLCV (`analysis.market_trade_alignment`) after each rehearsal.

> **Recovery plan:** Whenever live coverage or PnL drifts from the training reports, follow the authoritative checklist in `docs/live_trading_recovery_plan.md`. Treat that document as the source of truth for remediation steps before adjusting manifests or redeploying.

---

## Live Deployment Hardening

- **Multi-symbol rollout bundle** – `configs/deployment_portfolio_contract.yaml` and `configs/dry_run/infer_jobs_portfolio_policy.yaml` now encode the BTC/ETH/SOL ladder, per-symbol `policy_id` overrides, `shadow_mode` flags, and `model_key` routing so scheduler/trading stay in lockstep. Compose wires the overrides via `TRADING_MODELS`, `TRADING_SHADOW_SYMBOLS`, `TRADING_RISK_LIMITS_PATH`, and `TRADING_DEADLOCK_POLICY_PATH`; the stage ladder (`configs/live_launch_ladder.yaml`) emits env bundles under `configs/runtime_overrides/stage_*.yaml`, and the default dry-run env keeps `TRADING_SHADOW_SYMBOLS=[]` so every rehearsal runs these three symbols as primary instead of shadow.
- **Signed audit & provenance** – Trading now requires `TRADING_AUDIT_HMAC_KEY` (plus optional `TRADING_AUDIT_RUN_ID`/`TRADING_AUDIT_HOST_ID`) and writes `audit_source`/`audit_run_id`/`audit_seq` metadata inside every event so `analysis.validate_deployment_contract` can assert observability requirements before go-live. Redis/Postgres/file backends all compute the HMAC digest.
- **Intent ledger + reconciliation** – `TRADING_INTENT_LEDGER_BACKEND=redis`, `TRADING_INTENT_LEDGER_REDIS_URL`, and the new `IntentLedger` class dedupe order intents, track lifecycle status (`pending_submit`→`filled`/`canceled`/`error`), and feed metrics (`trading_intent_ledger_state_total`). A periodic reconcile loop compares internal state vs exchange truth and latches safe-mode until a healthy streak succeeds; emissions land in the audit stream under `reconciliation`.
- **Runtime risk engine** – `app/trading/risk.py::assess_and_adjust_order` consumes `configs/portfolio_risk_limits.yaml` (or the stage overrides) to block/clip orders based on turnover, exposure, drawdown, order cadence, cooldowns, and per-symbol spread/notional/qty rules. Stage-0 overrides now run equity-fraction sizing (`capital=200`, `equity_fraction=0.33`, `max_equity_fraction=0.35`, `compounding_step_usd=5`, `max_total_notional=80`) and define stop-loss shaping (`min_stop_loss_pct`, `hard_stop_loss_pct`, `vol_stop_rvol_mult`) so tight stops scale with realized volatility but never exceed a hard cap. Audit payloads include `risk_block_reason`/`risk_clip_reasons`, and Prometheus gets `trading_risk_blocked_total`/`trading_risk_clipped_total`.
- **Loss guard & aged decisions** – The stage risk limits ship a `loss_guard` block (3-loss cooldown with optional notional downscaling) plus `TRADING_LAST_TS_GRACE_BARS`/`TRADING_DECISION_MAX_AGE_SECONDS` so repeated losses raise `loss_guard` audits and stale queue items are silently skipped; `TRADING_STATE_BACKEND`/`TRADING_AUDIT_BACKEND` keep persistence consistent when stage bundles move between file/redis clients.
- **Deadlock policy automation** – `app/trading/deadlock.py`, the new `analysis.*launch_stage*` CLIs, and the Prometheus suite (`trading_deadlock_*`, `deadlock_action_taken_total`) monitor per-symbol coverage windows, execute staged mitigations (prob gate adjustments, policy switches, safe-mode), and log actions/audits with policy hashes.
- **Observability & controls** – Kill-/safe-mode env vars (`TRADING_KILL_SWITCH`, `TRADING_SAFE_MODE`) are enforced inside `app/trading/decision.py`; trading only enters when both are clear or we’re exiting safely. Metrics now include decision coverage, skip/dedup/risk block counters, orders-per-hour, turnover/drawdown gauges, safe-mode latch state, and reconciliation health. Grafana’s `trading-overview.json` dashboard plots the additional telemetry, and alert rules reference the same invariants listed in the deployment contract.

Guardrail CLIs:
- `python -m analysis.validate_deployment_contract --contract configs/deployment_portfolio_contract.yaml` now verifies symbol/policy/model mapping, `TRADING_MODELS` parity, kill/safe env wiring, audit fields/counters, and the presence of live risk limits.
- `python -m analysis.apply_launch_stage --stage stage_N --ladder configs/live_launch_ladder.yaml --contract configs/deployment_portfolio_contract.yaml` emits runtime overrides (risk limits, deadlock policy, env bundle) for the requested stage, while `analysis.evaluate_launch_stage` and `analysis.rollback_to_stage` publish readiness/rollback artifacts under `reports/`.

Promotion checklist:
1. Export the current stage bundle (`configs/runtime_overrides/stage_N.yaml`) and set `TRADING_AUDIT_HMAC_KEY`, `TRADING_INTENT_LEDGER_*`, `TRADING_KILL_SWITCH`, and `TRADING_SAFE_MODE` before starting trading.
2. Run `analysis.preflight_coverage`, `analysis.shadow_readiness`, and the deployment contract validator; review the Markdown/JSON reports (coverage, deadlock health, symbol readiness) under `reports/`.
3. Start trading with the exported env, leave kill/safe-mode asserted until reconciliation succeeds, and monitor Prometheus counters (`deadlock_action_taken_total`, `trading_deadlock_coverage_ratio`, `trading_safe_mode_latched`, `trading_risk_blocked_total`, `trading_reconcile_runs_total`) plus audit logs for provenance.

### Operational readiness reports

- **Live readiness check** – `python3 -m analysis.live_readiness_check --deployment-contract configs/deployment_portfolio_contract.yaml` chains contract validation, coverage/shadow/promotion preflights, and optional stage evaluation, writes GO/NO-GO Markdown+JSON bundles under `reports/live_readiness/`, links the underlying artifacts so every stage promotion has a single status artifact, and lets `analysis.preflight_coverage` warn when the manifest gate thresholds differ from the portfolio risk limits so operators spot config skew before promoting a stage.
- **Acceptance trade proof** – `analysis/acceptance_trade_proof.py` reads the audit log, filters executed sell exits, and writes Markdown/JSON pairs (e.g., `reports/acceptance_trade_proof_latest.*`) that summarize per-symbol counts/PnL so ops can prove the last trades honored guardrails.
- **Exit attribution + stance snapshot** – `analysis/exit_attribution_report.py` tallies exit reasons, churn, and PnL fractions while `analysis/project_stance_snapshot.py` snapshots the deployment contract, runtime env vars, risk limits, and deadlock policy so review decks cite the exact configuration used for the current ladder.

---

## Feature Parity & Gate Calibration

- Heavy parquet snapshots (`datasets/market_multi_*`, `datasets/blender_*`, etc.) are no longer tracked in git; regenerate them locally via the backfill/build scripts before running sweeps or parity checks.
- `datasets/market_multi_3symbol_1m.parquet` (≈1.19 M rows across BTC/ETH/SOL) now powers the relaxed gate. Run `training/data.sanitize_market_dataset` before retraining so duplicates/outliers are removed and symbol-level liquidity columns stay stable.
- Export a scheduler-style slice straight from the data lake plus manifests:
  ```bash
  python scripts/export_feature_slice.py \
    --data-lake-root data_lake/market \
    --base-manifest base_xgb_cost_spread \
    --symbols BTC/USDT,ETH/USDT,SOL/USDT \
    --output /tmp/features_debug.parquet
  ```
- Compare training vs live distributions to catch parity drift (`hl_spread`, `hl_spread_z`, `rvol_20`, `base_prob`) before changing gates:
  ```bash
  python scripts/compare_feature_stats.py \
    --train datasets/market_multi_3symbol_1m.parquet \
    --live /tmp/features_debug.parquet \
    --out release/calibration/latest/feature_parity.json
  ```
- Whenever the dataset refreshes, regenerate per-symbol gate caps so manifests, scheduler jobs, and `TRADING_MODELS` stay in lockstep:
  ```bash
  python scripts/compute_symbol_gate_config.py \
    --data datasets/market_multi_3symbol_1m.parquet \
    --out release/symbol_gates/market_multi_3symbol_1m.json
  ```
- `app/features/factory/market_factory.py` now drops OHLCV rows whose `(high-low)/close` exceeds 1 %, preventing hl_spread/rvol explosions before augmentation—mirror that expectation when debugging sparse exchanges.

## Environment

Create `.env` at the repo root.

### `.env.example`
```bash
# Core
ADMIN_TOKEN=change_me_please

# API host/port
INGEST_HOST=0.0.0.0
INGEST_PORT=8000

# CORS (JSON list)
CORS_ORIGINS=["*"]

# Exchange config (optional)
EXCHANGE_NAME=binance
EXCHANGE_API_KEY=
EXCHANGE_API_SECRET=

# Feature TTL in seconds (optional; admin TTL sweep can enforce)
FEATURE_TTL_SEC=3600

# Background backfill loop (inside ingestion-api; optional)
BACKFILL_ENABLED=0
BACKFILL_EXCHANGE=binance
BACKFILL_SYMBOLS=BTC/USDT
BACKFILL_TIMEFRAMES=1m
BACKFILL_LOOKBACK_MIN=15
BACKFILL_INTERVAL_SEC=300

# Background TTL sweep loop (inside ingestion-api; optional)
TTL_SWEEP_ENABLED=0
TTL_SWEEP_INTERVAL_SEC=900

# Scheduler
API_BASE_URL=http://ingestion-api:8000
RUN_ON_START=1
SCHED_TZ=UTC
SCHED_METRICS_PORT=9002
MARKET_JOBS=[{"exchange":"binance","symbol":"BTC/USDT","timeframe":"1m","lookback_minutes":15,"cron":"*/5 * * * *"}]
TTL_SWEEP_CRON=*/15 * * * *
TTL_SWEEP_PATTERN=features:market:*
TTL_SWEEP_TTL=3600

# External keys (Phase 3)
GLASSNODE_API_KEY=
TWITTER_BEARER_TOKEN=
NEWS_API_KEY=
# Trading service
TRADING_DRY_RUN=1
TRADING_MODELS=[{"model":"xgb_primary","symbol":"ETH/USDT","exchange":"binance","timeframe":"1m","order_notional":45.0,"max_spread_bps":8,"stop_loss_pct":0.008,"take_profit_pct":null,"profit_trailing_start_pct":0.004,"profit_trailing_stop_pct":0.002,"max_hold_minutes":240,"shadow_mode":false,"policy_id":"primary","min_hold_bars_override":15,"disable_prob_exits":true,"entry_rsi_min":50.0},{"model":"xgb_primary","symbol":"BTC/USDT","exchange":"binance","timeframe":"1m","order_notional":25.0,"max_spread_bps":8,"stop_loss_pct":0.005,"take_profit_pct":null,"profit_trailing_start_pct":null,"profit_trailing_stop_pct":null,"max_hold_minutes":90,"shadow_mode":false,"policy_id":"conservative","min_hold_bars_override":15,"disable_prob_exits":true,"entry_macd_min":0.0},{"model":"xgb_primary","symbol":"SOL/USDT","exchange":"binance","timeframe":"1m","order_notional":30.0,"max_spread_bps":8,"stop_loss_pct":0.005,"take_profit_pct":null,"profit_trailing_start_pct":null,"profit_trailing_stop_pct":null,"max_hold_minutes":90,"shadow_mode":false,"policy_id":"conservative","min_hold_bars_override":15,"disable_prob_exits":true,"entry_rsi_min":45.0}]
TRADING_SHADOW_SYMBOLS=[]
TRADING_AUDIT_HMAC_KEY=changeme_in_prod
TRADING_INTENT_LEDGER_BACKEND=redis
TRADING_INTENT_LEDGER_REDIS_URL=redis://redis:6379/0
TRADING_DEPLOYMENT_CONTRACT=configs/deployment_portfolio_contract.yaml
TRADING_RISK_LIMITS_PATH=configs/runtime_overrides/risk_limits_stage_0.yaml
TRADING_DEADLOCK_POLICY_PATH=configs/runtime_overrides/deadlock_policy_stage_0.yaml
TRADING_PRICE_MONITOR_INTERVAL_SECONDS=10
TRADING_KILL_SWITCH=1
TRADING_SAFE_MODE=1
TRADING_SAFE_MODE_ALLOW_EXITS=1
```

---

## API Surface

### Service health & docs
- `GET /health` → `{"status":"ok"}`
- `GET /metrics` → Prometheus exposition
- `GET /openapi.json` / `GET /docs`

### Ingest
- `POST /ingest/market/{exchange}`  
  **Body:** `{"symbol":"BTC/USDT","granularity":"1m","limit":100}`  
  Writes OHLCV features into Redis and Parquet.
- `POST /ingest/onchain/{source}` (e.g. `glassnode`, `covalent`)  
  **Body (examples):**  
  `{"symbol":"BTC","metric":"active_addresses","days":1}`  
  `{"chain_id":1,"address":"0x0000000000000000000000000000000000000000"}`
- `POST /ingest/social/{platform}` (e.g. `twitter`)  
  **Body:** `{"query":"bitcoin","since":null,"until":null,"max_results":5}`
- `POST /ingest/news`  
  **Body:**  
  API: `{"source_type":"api","category":"business"}`  
  RSS: `{"source_type":"rss","feed_url":"https://..."}` 

**Typical success:**
```json
{"status":"ok","path":"/app/data_lake/.../part-*.parquet","features_written":123}
```

**No keys / no results:**
```json
{"status":"no_data","path":null,"features_written":0}
```

**Upstream error (example):**
```json
{"detail":"ingest_onchain failed: Client error '401 Unauthorized' for url '...'"}
```

### Feature retrieval (point lookups)
- `GET /ingest/features/market`  
  `symbol=BTC/USDT&timeframe=1m&ts=1724140800&ts=...`
- `GET /ingest/features/onchain`  
  `symbol=BTC&metric=active_addresses&ts=...`
- `GET /ingest/features/social`  
  `topic=twitter&timeframe=1m&ts=...`

### Feature retrieval (range)
- `GET /ingest/features/market/range`  
  `symbol=BTC/USDT&timeframe=1m&start={epoch}&end={epoch}&limit=100`
- `GET /ingest/features/social/range`  
  `topic=twitter&timeframe=1m&start={epoch}&end={epoch}&limit=100`
- `GET /ingest/features/onchain/range`  
  `symbol=BTC&metric=active_addresses&start={epoch}&end={epoch}&limit=100`

### Admin (requires `X-Admin-Token`)
- `POST /ingest/admin/backfill/market`  
  `exchange=binance&symbol=BTC/USDT&timeframe=1m&lookback_minutes=15`  
  → `{"symbol":"BTC/USDT","timeframe":"1m","exchange":"binance","expected":N,"missing_before":M,"written":W}`
- `POST /ingest/admin/features/ttl-sweep`  
  `pattern=features:market:*&ttl_default=3600&max_keys=1000`  
  → `{"pattern":"...","scanned":N,"ttl_set":M}`

---

## Feature Storage Layout (Redis)

- **Per-point keys:**  
  `features:{domain}:{id_or_symbol}:{timeframe}:{ts_epoch}`
  - `features:market:BTC-USDT:1m:1724140800`
  - `features:social:twitter:1m:1724140800`
  - `features:onchain:BTC:active_addresses:1724100000`
- **Index keys:**  
  `features:{domain}:{id_or_symbol}:{timeframe}:_idx`  
  Used for range queries and TTL sweep traversal.

> TTL can be enforced with `FEATURE_TTL_SEC` or via the admin TTL sweep’s `ttl_default`.

---

## Scheduler

**Behavior**
- On boot (when `RUN_ON_START=1`): executes one backfill per `MARKET_JOBS` and a TTL sweep.
- On schedule: runs backfills via cron in `MARKET_JOBS` and TTL sweeps via `TTL_SWEEP_CRON`.

**Env (key vars)**
- `API_BASE_URL` (default internal: `http://ingestion-api:8000`)
- `RUN_ON_START`, `SCHED_TZ`, `SCHED_METRICS_PORT`
- `MARKET_JOBS` e.g.  
  `[{"exchange":"binance","symbol":"BTC/USDT","timeframe":"1m","lookback_minutes":15,"cron":"*/5 * * * *"}]`
- `TTL_SWEEP_CRON`, `TTL_SWEEP_PATTERN`, `TTL_SWEEP_TTL`
- `DECISION_PAYLOAD_ITEMS` caps the number of decision payloads pushed per job run (default 3); override per job with `max_decision_items` inside `INFER_JOBS` when queues get noisy.

**Metrics**
- Exposed on port `9002` (host-mapped in compose):  
  ```bash
  curl -s http://localhost:9002/metrics | head
  ```

**Manual smoke (inside scheduler container)**
```bash
docker compose exec scheduler sh -lc 'curl -s http://ingestion-api:8000/health'

docker compose exec scheduler sh -lc \
  'curl -s -X POST -H "X-Admin-Token: $ADMIN_TOKEN" \
  "http://ingestion-api:8000/ingest/admin/backfill/market?exchange=binance&symbol=BTC%2FUSDT&timeframe=1m&lookback_minutes=1"'

docker compose exec scheduler sh -lc \
  'curl -s -X POST -H "X-Admin-Token: $ADMIN_TOKEN" \
  "http://ingestion-api:8000/ingest/admin/features/ttl-sweep?pattern=features%3Amarket%3A%2A&ttl_default=3600&max_keys=25"'
```

---

## Monitoring

### Prometheus
- Scrapes:
  - `ingestion-api` → `http://ingestion-api:8000/metrics`
  - `scheduler` → `http://scheduler:9002/metrics`
  - `redis-exporter` → `http://redis-exporter:9121/metrics`
- Config: `monitoring/prometheus/prometheus.yml`

Sanity:
```bash
docker compose exec prometheus sh -lc \
  'wget -qO- http://localhost:9090/api/v1/targets | jq ".data.activeTargets[].labels.job" | sort -u'
```

### Grafana
- Access: http://localhost:3000 (admin/admin on first run)
- Datasource: Prometheus at `http://prometheus:9090`
- Import dashboards from: `monitoring/grafana/dashboards/*.json`

Common metrics:
- API: `service_info{service="raw-data-ingestion"}` + feature store counters/histograms
- Scheduler: APScheduler & process metrics on `:9002`
- Redis: via redis_exporter
- Model gates: `model_gate_coverage_ratio{model,mode}`, `model_rss_minute_spike_share`, and `model_probability_sigma` expose manifest coverage, RSS health, and probability variance; matching `*_threshold` gauges set from manifests so Prometheus alerts (`monitoring/alert.rules.yml`) fire when coverage or sigma fall below guardrails.
- Trading queue health: `trading_decision_queue_depth{queue="trading:decisions"}` alarms when scheduler producers outpace the consumer—tune `DECISION_PAYLOAD_ITEMS`/`max_decision_items` or `TRADING_QUEUE_POLL_TIMEOUT` accordingly.

### Probability Distribution Debugging
- `app.monitoring.probability_sampler` captures pre-gate probability samples for every scheduler/API batch and appends them to `logs/probability_samples/<model>_<prob>.jsonl`.  
  Configure via:
  - `PROB_SAMPLE_ENABLED` (default `1`)
  - `PROB_SAMPLE_ROOT` (default `logs/probability_samples`)
  - `PROB_SAMPLE_MAX_ROWS` (bounded rows per batch, default `512`)
  - `PROB_SAMPLE_REDIS_URL`, `PROB_SAMPLE_REDIS_STREAM`, `PROB_SAMPLE_REDIS_MAXLEN` when the feed should also mirror into Redis Streams.
- Build a rolling drift baseline and stratified KPIs from the live sampler (versus training logits and the last N days of live data):
  ```bash
  python3 scripts/probability_distribution_audit.py \
    --samples logs/probability_samples \
    --fold-logits models/base_xgb_h120_calmon_spread0/fold_logits.parquet \
    --fold-column prob_calibrated \
    --features /tmp/features_debug.parquet \
    --out-parquet release/calibration/latest/live_prob_samples.parquet \
    --hourly-dir release/calibration/latest/live_prob_hourly \
    --summary-out release/calibration/latest/distribution_audit.json
  ```
  The script tags each sample with session/regime/symbol buckets, emits KS/PSI/Wasserstein + collapse/saturation flags, and can also generate stress digests (e.g., `distribution_audit_stress.json`) when fed perturbed slices.
- Plot live vs training distributions (fold logits) and drop the summaries into `release/calibration/latest/`:
  ```bash
  python3 scripts/plot_probability_distributions.py \
    --samples logs/probability_samples/tcn_h120_calmon_relaxed_tcn_prob.jsonl \
    --model-dir models/tcn_h120_calmon_relaxed \
    --prob-column tcn_prob \
    --out release/calibration/latest/live_vs_fold_tcn_prob.png \
    --summary-out release/calibration/latest/live_vs_fold_tcn_prob.json
  ```
- Re-run calibrators on a fresh feature slice to ensure probability sigma hasn't collapsed:
  ```bash
  python3 scripts/run_calibrator_check.py \
    --live /tmp/features_debug.parquet \
    --base-model models/base_xgb_h120_calmon_spread0 \
    --tcn-model models/tcn_h120_calmon_relaxed \
    --tcn-stride 2 \
    --summary-out release/calibration/latest/calibrator_health.json
  ```
  The JSON reports mean/std/quantiles for calibrated and uncalibrated probabilities; alert when the observed σ falls below the manifest's `prob_sigma_guardrail`.
  Use `scripts/refresh_calibration.py` to re-fit calibrators on live slices—the script now re-scores the raw booster before fitting to avoid double-scaling clipped probabilities.

---

## Example Calls (host → API)

**Health & OpenAPI**
```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/openapi.json | jq '.paths | keys'
```

**Market ingest**
```bash
curl -s -X POST "http://localhost:8000/ingest/market/binance" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC/USDT","granularity":"1m","limit":2}'
```

**On-chain (Glassnode)**
```bash
curl -s -X POST "http://localhost:8000/ingest/onchain/glassnode" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC","metric":"active_addresses","days":1}'
```

**Social (Twitter)**
```bash
curl -s -X POST "http://localhost:8000/ingest/social/twitter" \
  -H "Content-Type: application/json" \
  -d '{"query":"bitcoin","max_results":5}'
```

**News**
```bash
curl -s -X POST "http://localhost:8000/ingest/news" \
  -H "Content-Type: application/json" \
  -d '{"source_type":"api","category":"business"}'
```
> The endpoint normalises RSS/API payloads with `fetch_news_rss_once`/`fetch_news_api`, partitions by `dt` + `source`, writes Parquet under `NEWS_PATH`, and mirrors the rows into Redis when the schema check passes.

**Admin: backfill & TTL sweep**
```bash
curl -s -X POST -H "X-Admin-Token: $ADMIN_TOKEN" \
  "http://localhost:8000/ingest/admin/backfill/market?exchange=binance&symbol=BTC%2FUSDT&timeframe=1m&lookback_minutes=15"

curl -s -X POST -H "X-Admin-Token: $ADMIN_TOKEN" \
  "http://localhost:8000/ingest/admin/features/ttl-sweep?pattern=features%3Amarket%3A%2A&ttl_default=3600&max_keys=100"
```

**Feature retrieval (range)**
```bash
# Market
curl -s "http://localhost:8000/ingest/features/market/range?symbol=BTC/USDT&timeframe=1m&start=1724140800&end=1724142000&limit=10"

# Social
curl -s "http://localhost:8000/ingest/features/social/range?topic=twitter&timeframe=1m&start=1724140800&end=1724142000&limit=10"

# Onchain
curl -s "http://localhost:8000/ingest/features/onchain/range?symbol=BTC&metric=active_addresses&start=1724140800&end=1724142000&limit=10"
```

---

## Tests

- `pytest tests/ingestion_service` covers async route flows (market/on-chain/social ingest, feature retrieval, metrics) with fakeredis and in-memory stores.
- `pytest tests/regression` guards KPIs by asserting manifests stay aligned with their `report.json` files and that `scripts/report_shortlist.py` still elevates the Calmon baseline.
- CI (`.github/workflows/ci.yml`) provisions Python 3.11 + CPU PyTorch, installs requirements, and runs the regression and `tests/training` suites on pushes/PRs.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ingestion_service -q
pytest tests/regression -q
```

---

## Troubleshooting

- **Scheduler “All connection attempts failed” on boot** – Likely a race; scheduler retries. Ensure `API_BASE_URL=http://ingestion-api:8000`.
- **404 on admin** – Use `/ingest/admin/...` prefix and include `X-Admin-Token`.
- **401 from Glassnode** – Expected without `GLASSNODE_API_KEY`. Social/news may return `no_data` when unauthenticated.
- **Weird `jq`/zsh errors** – Avoid inline comments in multi-line commands; prefer one command per line. Use `\curl` to bypass aliases.

---

## Directory Map

- `app/ingestion_service/main.py` – FastAPI app (health, metrics mount, lifespan).
- `app/ingestion_service/routes.py` – Ingest + features + admin endpoints.
- `app/ingestion_service/schemas.py` – Request/response models.
- `app/ingestion_service/utils.py` – Metrics registry and helpers.
- `app/features/store/redis_store.py` – Feature store + Prometheus metrics.
- `app/features/ingestion/*_client.py` – External clients (market/onchain/social/news).
- `app/features/jobs/backfill.py` – Admin backfill & TTL sweep.
- `app/scheduler/main.py` – APScheduler runner.
- `monitoring/prometheus/prometheus.yml` – Prometheus scrape config.
- `monitoring/grafana/dashboards/*.json` – Grafana dashboards.

---

## Security

Admin endpoints require `X-Admin-Token`. Keep `.env` out of version control, use a strong random token, and rotate if shared across environments.

---

## Roadmap to Phase 3

- Plug in real keys (Glassnode, Twitter/X, News API), enable production fetches.
- Add tests for the news route.
- Expand scheduler jobs to on-chain/social/news once keys are present.
- Optional: retention policies (TTL per domain), richer Grafana dashboards.

---

## Data Scraping (via Docker Compose)

El archivo `docker-compose.yml` ya incluye un servicio `scheduler` con trabajos de backfill configurados para:

- `binance BTC/USDT 1m` cada 5 minutos (lookback 60 min)
- `binance BTC/USDT 5m` cada 15 minutos (lookback 360 min)
- `binance ETH/USDT 1m` cada 5 minutos (lookback 60 min)

Además, un trabajo de TTL sweep cada 15 minutos para aplicar expiraciones en Redis (`TTL_SWEEP_*`).

Cómo corre el scraping:
- Al iniciar el stack (`docker compose up -d --build`), el scheduler espera a que el API esté disponible y ejecuta los trabajos una vez (por `RUN_ON_START=1`) y luego según el cron.
- El API expone los endpoints admin bajo `/ingest/admin/*` protegidos por `X-Admin-Token` (tomado de `.env`).
- Los datos crudos (OHLCV normalizado) se escriben en Parquet bajo `./data_lake/market/...` y las features a Redis.

Verificación rápida de backfill:
```bash
# Logs del scheduler
docker compose logs -f scheduler

# Archivos parquet generados (en el host)
find data_lake/market -type f -name '*.parquet' | head

# Claves de features en Redis (dentro del contenedor redis)
docker compose exec redis redis-cli --scan --pattern 'features:market:*' | head
```

Habilitar ML en la imagen (opcional):
```bash
docker compose build --build-arg INSTALL_ML=1 ingestion-api
docker compose up -d ingestion-api
```

Añadir más trabajos de mercado:
- Edita `docker-compose.yml`, variable de entorno `MARKET_JOBS` del `scheduler` y agrega entradas JSON con `{exchange, symbol, timeframe, lookback_minutes, cron}`.

---

## Parquet Ingest (scheduler)

Además de los backfills hacia Redis, el `scheduler` puede ejecutar ingestiones periódicas que escriben OHLCV normalizado a Parquet en el data lake usando el endpoint `POST /ingest/market/{exchange}`.

- Variable: `MARKET_INGEST_JOBS` (JSON list)
- Esquema por item: `{ "exchange": "binance", "symbol": "BTC/USDT", "timeframe": "1m", "limit": 500, "cron": "*/10 * * * *" }`

Ejemplo en `docker-compose.yml`:
```
MARKET_INGEST_JOBS=[
  {"exchange":"binance","symbol":"BTC/USDT","timeframe":"1m","limit":500,"cron":"*/10 * * * *"},
  {"exchange":"binance","symbol":"ETH/USDT","timeframe":"1m","limit":500,"cron":"*/10 * * * *"}
]
```

Verificación rápida:
```bash
# Logs del scheduler
docker compose logs -f scheduler

# Nuevos archivos parquet en el host
find data_lake/market -type f -name '*.parquet' | head
```

---

## ML Inference (Opt-in)

Para habilitar inferencia de sentimiento con modelos reales (DistilBERT por defecto) y el endpoint `/ml/sentiment/predict`:

1) Construir la imagen del API con ML:
   - En `docker-compose.yml` cambiar `INSTALL_ML: 0` a `INSTALL_ML: 1` para `ingestion-api`.
2) Activar los flags en entorno:
   - `ML_SENTIMENT_ENABLED=1`
   - `SENTIMENT_MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english`
   - `ML_MAX_WORKERS=4` (opcional)
   - `HF_HOME=/app/.cache/huggingface` (cache de modelos)
   - (opcional) `SOCIAL_SENTIMENT_ENRICH=1` para enriquecer ingest social con `sentiment_label` y `sentiment_score`.
3) (Recomendado) Montar un volumen para cache de modelos:
```
volumes:
  - hf-cache:/app/.cache/huggingface
```
4) Reconstruir y levantar:
```
docker compose build ingestion-api
docker compose up -d ingestion-api
```

Probar inferencia:
```
curl -s -X POST http://localhost:8000/ml/sentiment/predict \
  -H 'Content-Type: application/json' \
  -d '{"texts":["btc to the moon","market looks bad"]}' | jq
```

Métricas:
- `ml_infer_requests_total{model=...}`
- `ml_infer_duration_seconds{model=...}`
- `ml_infer_errors_total{model=...,type=...}`

---

## Dataset Builders

The repo includes simple scripts to produce training datasets from the data lake.

1) Market dataset (features + labels)
```bash
python scripts/build_market_dataset.py \
  --exchange binance \
  --symbol BTC/USDT \
  --timeframe 1m \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --out datasets/market_btcusdt_1m_2025.parquet
```
This reads Parquet under `MARKET_PATH`, computes our standard market features, and adds labels (next‑bar return and direction).

2) RSS → Parquet (one shot)
```bash
python scripts/rss_to_parquet.py --feed https://news.google.com/rss/search?q=bitcoin
```
Writes normalized RSS entries to `NEWS_PATH/rss/...` using the Parquet writer (supports S3/GCS if configured).

3) Training matrix (market + aggregates from RSS/Reddit)
```bash
python scripts/build_training_matrix.py \
  --exchange binance \
  --symbol BTC/USDT \
  --timeframe 1min \
  --include-rss \
  --out datasets/training_matrix_btcusdt_1m.parquet
```
This merges market features with aggregated counts and mean sentiment for RSS/Reddit (if present), and builds labels.

Notes
- Scripts read from the data lake paths configured in env (local or S3/GCS). For S3/GCS, ensure credentials/env are set as described above.
- The training matrix script expects that some RSS or Reddit Parquet exists; you can generate it via the `rss_to_parquet.py` script or schedule ingest jobs.

## Time-Series CV & Hyperparameter Search

- Random-search sweeps now use the time-based splits defined in `configs/cv_config.yaml` (expanding window; 15D validation, 1D gap). Example (TCN):
  ```bash
  python -m training.run_hparam_search \
    --model tcn \
    --contract configs/canonical_training_contract_market_multi_3symbol_1m.yaml \
    --cv-config configs/cv_config.yaml \
    --hparam-space configs/hparam_spaces.yaml \
    --n-trials 32 \
    --output-dir experiments/hparam_search/tcn \
    --horizon 2 --seq-stride 10 --max-rows 800000 \
    --cost-bps 5 --min-hold-bars 1
  ```
  Sequence models accept `--seq-stride` to thin windows so stride‑1 experiments stay memory-safe.
- Every trial writes JSON + CSV under `experiments/hparam_search/<model>/`; promote the winners into reusable configs with:
  ```bash
  python -m training.promote_best_configs \
    --search-root experiments/hparam_search \
    --min-sharpe 0.0 --top-n 1 \
    --output configs/best_model_configs.yaml
  ```
  The repo already includes the promoted configs (`xgb_trial_010`, `tcn_trial_011`, `transformer_trial_023`) mirrored in both YAML/JSON.
- Sequence trainers (TCN/Transformer) now monitor deployable Sharpe during training when `val_returns` is provided, applying `cost_bps`, `long_only`, and `min_hold_bars` in early stopping/selection instead of pure loss.

## Modeling Snapshot (Nov 2025)

- `scripts/build_blender_matrix.py` now ships `datasets/blender_matrix_2025-09_to_2025-11_oos.parquet` (84 839 rows, 2025‑09‑01 ➜ 2025‑10‑29) built straight from the live feature pipeline: market features, RSS aggregations, refreshed base/TCN predictions, and the production label definition.
- `scripts/refresh_calibration.py` consumes that matrix, fits post-hoc calibrators, and emits charts + JSON into `release/calibration/<stamp>/`. The latest refresh (split 65/35 train/validation) keeps `base_xgb_h120_calmon_spread005` on Platt scaling (Brier 6.998e‑4 → 6.996e‑4, ECE 1.04e‑4 → 1.02e‑4), lifts `tcn_h120_calmon_relaxed` onto an isotonic/identity blend after the stride‑2 replay (Brier 5.75e‑2 → 5.74e‑2, ECE 1.65e‑2 → 1.25e‑2), and leaves the blender at identity because the calibrated replay already matches the OOS reference.
- Each manifest now carries its calibrated inference path automatically: loading a predictor attaches `models/<manifest>/calibration/<prob_col>.json + .joblib`, and inference/training scripts apply those mappings before computing gates or downstream KPIs.
- The blender manifest was re-aligned with the calibrated probabilities — `prob_gate_min` (training+inference) and `threshold.txt` both move to **0.55**, keeping gate coverage around 19 % while tightening the dual-threshold filter for the trading service.
- Forward validation metrics, calibration plots, and gate/threshold recommendations live in `release/calibration/2025-11-oos_stride2/` alongside the machine-readable `calibration_summary.json` for dashboards/reporting.

See `docs/calibration.md` for the step-by-step refresh workflow and `docs/oos_gate_runbook.md` for the end-to-end OOS + gate-coverage playbook.

---

## Object Storage (S3/GCS) via fsspec (Opt-in)

The Parquet writer supports any fsspec backend. To write directly to object storage:

1) Optional deps (already in the Docker image): `s3fs`, `gcsfs`.
2) Point your data lake paths to S3 or GCS and set credentials.

S3 example
```bash
# .env (host) or docker-compose env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
DATA_LAKE_PATH=s3://my-bucket/algo-data-lake
MARKET_PATH=s3://my-bucket/algo-data-lake/market
# Optional advanced fsspec options (JSON string)
FSSPEC_STORAGE_OPTIONS='{"client_kwargs": {"region_name": "us-east-1"}}'
```

GCS example
```bash
# Mount your service account JSON in the container
# docker-compose.yml
#   volumes:
#     - ./secrets:/secrets:ro

# .env
GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-sa.json
DATA_LAKE_PATH=gs://my-bucket/algo-data-lake
MARKET_PATH=gs://my-bucket/algo-data-lake/market
```

Notes
- Use absolute paths for the base `DATA_LAKE_PATH` and domain paths.
- `FSSPEC_STORAGE_OPTIONS` (JSON) is passed to fsspec’s `url_to_fs` for advanced configuration.
- For S3, IAM roles or instance profiles also work (omit explicit keys).

Storage check endpoint (admin)
```bash
# Resolve backends and run a write/delete probe (local/S3/GCS depending on env)
curl -s -H "X-Admin-Token: $ADMIN_TOKEN" \
  "http://localhost:8000/ingest/admin/storage/check?probe=1&domain=market" | jq
```
