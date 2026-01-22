# Algo Data Ingestion – Comprehensive Project Dossier

_Last updated: 2026-01-22 17:38 UTC_

> Update 2026-01-22: Documented the scheduler data lake cleanup job (retention-based partition pruning) with `DATA_LAKE_RETENTION_DAYS`, `DATA_LAKE_CLEANUP_CRON`, and `DATA_LAKE_CLEANUP_ROOTS`.
> Update 2025-12-30: Added quote-based price monitoring for exits (`TRADING_PRICE_MONITOR_INTERVAL_SECONDS`), vol-aware stop shaping (`min_stop_loss_pct`/`hard_stop_loss_pct`/`vol_stop_rvol_mult`), and optional entry/exit churn controls (`entry_rsi_min`/`entry_macd_min`, `disable_prob_exits`); refreshed stage-0 sizing to capital 200 with `equity_fraction=0.33`.
> Update 2025-12-19: Added a dry-run profit forensics workflow (`scripts/extract_container_logs.py` → `analysis/trading_log_forensics.py` → `analysis/market_trade_alignment.py` with `RUNBOOK_DRY_RUN_PROFIT.md`), switched stage-0 sizing to equity-fraction compounding (capital 100, base notionals 20/15/12, `max_total_notional=80`, `compounding_step_usd=5`) with per-symbol trigger overrides and longer holds, and refreshed compose/env defaults accordingly.
> Update 2025-12-17: Added a live-readiness check, new audit diagnostics (`analysis/acceptance_trade_proof`, `analysis/exit_attribution_report`, `analysis/project_stance_snapshot`), and scheduler/trading hardening (loss guard, queue-aging, quote-aware exits, `INFER_APPLY_CALIBRATION` override) so the deployment contract stay aligned with production.

> Update 2025-11-30 22:29 UTC: Stage-0 runtime risk overrides set a 1-minute cooldown after exits and 5 minutes after losses, the dry-run compose promoted BTC/ETH/SOL to full primary entries (300 USDT notional, `max_spread_bps=10`, empty `TRADING_SHADOW_SYMBOLS`), and the bulky parquet datasets were dropped from git so they’re regenerated via the existing backfill/sanitize scripts. Superseded by the 2025-12-19 equity-fraction sizing drop.
> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-19: Introduced sampling + weighting controls in the time-series CV pipeline (`--sampling-policy/--weight-policy`, configs in `configs/sampling_*.yaml` and `configs/weights_cost_capacity.yaml`), piping sample weights through DeepLOB/TCN/Transformer training and the random-search objective. Added a portfolio performance-sweep harness (`portfolio/run_perf_sweeps.py` + `configs/perf_sweep_scenarios.yaml`) and promoted the XGB-only winner (`medium_xgb_low_cost`, 100 trades, Sharpe 278, ≈51 % time-in-position) into `configs/deployment_portfolio_contract.yaml` and `configs/dry_run/infer_jobs_portfolio_policy.yaml`, with Docker mounting `experiments/perf_sweeps` so scheduler/trading can load `xgb_primary` during dry-runs.
>

### 0. 2025-12-19 Profit forensics & bounded sizing
- **Dry-run profit workflow** – New runbook (`RUNBOOK_DRY_RUN_PROFIT.md`) chains container evidence extraction (`scripts/extract_container_logs.py`), audit forensics (`analysis/trading_log_forensics.py`), and market alignment (`analysis/market_trade_alignment.py`) into `reports/log_forensics/...` so each rehearsal ships with PnL, exit-attribution, and regret metrics.
- **Equity-fraction sizing** – Stage-0 risk overrides now run `sizing_mode=equity_fraction` with `capital=200`, `max_total_notional=80`, `equity_fraction=0.33`, `max_equity_fraction=0.35`, `compounding_step_usd=5`, per-symbol caps, and per-symbol trigger overrides, plus `cooldown_minutes_after_exit: 2` / `cooldown_minutes_after_loss: 5`. Stops are now shaped via `min_stop_loss_pct`/`hard_stop_loss_pct`/`vol_stop_rvol_mult`.
- **Env/compose parity** – `docker-compose.yml` and `configs/deployment_portfolio_contract.yaml` now point `xgb_primary` at the baked-in bundle `models/base_xgb_h120_calmon_spread0`; the trading defaults add optional churn controls (`disable_prob_exits`, entry filters) and a quote-based exit monitor (`TRADING_PRICE_MONITOR_INTERVAL_SECONDS`) while `.dockerignore` excludes reports/logs/experiments from build context to keep images lean.

### 0. 2025-12-17 Readiness & Trading Resilience
- **Live-readiness + audit diagnostics** – `analysis.live_readiness_check` chains contract validation, coverage/shadow/preflight checks, and optional ladder evaluation into one GO/NO-GO command while writing Markdown/JSON bundles under `reports/live_readiness/` and letting `analysis.preflight_coverage` warn when manifest gate thresholds diverge from portfolio risk limits. `analysis/acceptance_trade_proof`, `analysis/exit_attribution_report`, and `analysis/project_stance_snapshot` now surface executed exits, exit-attribution stats, and the exact contract/env/risk posture for the current ladder so every promotion can cite a documented stance.
- **Scheduler/inference resilience** – The scheduler’s inference lane now auto-computes missing manifest columns (special features, regimes, label lookbacks), honors manifest `apply_calibration` via the `INFER_APPLY_CALIBRATION` override, and respects the rolling special feature set (`directional_15m`, `cost_adjusted_15m`, `vol_regime`, `liquidity_regime`, `spread_regime`, `event_flag`).
- **Trading guardrails** – Runtime risk limits support `loss_guard` (three losses + 90-minute cooldown with optional notional downscale), processor logs richer exit contexts (expected/net PnL, quote sources), executor quote pulls, and the stage bundle couples `TRADING_LAST_TS_GRACE_BARS`, `TRADING_DECISION_MAX_AGE_SECONDS`, `TRADING_STATE_BACKEND`, and `TRADING_AUDIT_BACKEND` so stale queue items are dropped instead of reprocessed.

### 0. 2025-11-30 Live-Hardening Highlights
- **Launch ladder + overrides** – `configs/live_launch_ladder.yaml`, the new `analysis.apply_launch_stage`/`analysis.evaluate_launch_stage`/`analysis.rollback_to_stage` CLIs, and `configs/runtime_overrides/stage_*.yaml` coordinate BTC/ETH/SOL rollout bundles (per-symbol policy/shadow overrides, runtime risk/deadlock policies, env exports, reports).
- **Deployment contract enforcement** – `analysis.validate_deployment_contract` now checks kill/safe env wiring, audit field/counter requirements, risk limit coverage, symbol/policy/model mappings, and parity with `TRADING_MODELS` + `symbol_model_key`, failing fast when invariants drift.
- **Runtime checks** – Trading enforces HMAC-signed audit logs, Redis intent ledger dedupe (with Prometheus counters), reconciliation safe-mode latching, runtime risk clipping, kill/safe-mode gating, and programmable deadlock policies that emit metrics/actions/audits per contract requirements.
- **Observability** – Prometheus/Grafana ingest decision coverage, skip/dedup/risk block counters, turnover/drawdown gauges, safe-mode latch state, reconciliation success, deadlock coverage/actions, and intent-ledger states to mirror the deployment contract’s observability section.
- **Stage-0 guard tweak** – The runtime risk overrides now run equity-fraction sizing (`capital=200`, `equity_fraction=0.33`, `max_equity_fraction=0.35`, `compounding_step_usd=5`, `max_total_notional=80`) with per-symbol trigger overrides, stop shaping, and `cooldown_minutes_after_exit: 2` / `cooldown_minutes_after_loss: 5` while the dry-run compose wires BTC/ETH/SOL as full primary entries so rehearsals track the intended live posture.

### 0. 2025-11-29 Refresh Highlights
- **Trigger optimizer + preflight** – New CLI (`analysis/trigger_optimizer.py`) sweeps entry/exit/hold/SL/TP/spread guardrails via `configs/trigger_search_space*.yaml`, promotes the winner to `configs/final_trigger_policy.yaml`, and pairs with `scripts/trigger_preflight.py` to fail fast when coverage or trade-count proxy drops before spinning up services.
- **Shared trading decision logic** – Trading now routes through `app/trading/decision.py::decide_bar`, applying spread/hold/SL/TP (plus optional profit-trailing) checks consistently across live loops and offline sweeps; `TRADING_MODELS` accepts guard fields (max spread, stop loss, take profit, max hold, min hold overrides) plus optional churn controls (`disable_prob_exits`, entry filters). Decision payloads now carry price/spread fields so exits respect execution context.
- **Feature enrichment + inference resilience** – Market ingest/backfill/scheduler compute augmented features inline (adds `hl_spread_z`, `rvol_20`, liquidity ranks), push None-safe payloads to Redis, attach `close`/`price` before scoring, and backfill missing manifest features on the fly. Manifest loaders respect `apply_calibration` flags with safer temp-file reads, and deployment contracts now resolve model paths under `MODELS_ROOT=/opt/models` (`perf_sweeps/...` bundles are mounted there in Docker).

### 0. 2025-11-17 Refresh Highlights
- **Hyperparameter search pipeline** – Random-search CLI now runs expanding-window splits (15D validation, 1D gap by default) and logs results under `experiments/hparam_search/<model>/results.csv` + per-trial JSON. `training/promote_best_configs.py` distills the current leaders (`xgb_trial_010`, `tcn_trial_011`, `transformer_trial_023`) into versioned configs for downstream scripts.
- **Deployability-aware sequence training** – TCN/Transformer training loops now accept stride-aware sequence builders, clip gradients with configurable ceilings, and early-stop on equity/Sharpe computed with `cost_bps`, `long_only`, and `min_hold_bars` when validation returns are provided, keeping stride-1 experiments memory-safe while tracking live-like KPIs.
- **Dataset contract resolution** – Canonical contracts now prefer project-root-relative paths before falling back to contract-local resolution, reducing surprises when configs live under `configs/` but datasets are stored at repo root or in `data/`.
- **Symbol-aware gate + monitoring (previous refresh)** – `datasets/market_multi_3symbol_1m.parquet` and `release/symbol_gates/market_multi_3symbol_1m.json` keep manifests, scheduler jobs, and `TRADING_MODELS` aligned on `hl_spread`/`rvol` caps; parity diffs land in `release/calibration/latest` and Grafana panels continue to track queue depth, gate toggles, and parity drift.

---

## 1. Executive Summary
- **Mission**: Deliver an end-to-end ingestion and modeling platform that converts high-frequency market, social, and news data into actionable trading signals and deploys them through a monitored production stack.
- **Scope**: Covers data acquisition (API ingest, backfills), feature engineering, supervised learning (XGBoost baseline, Temporal Convolutional Networks), ensemble/meta-label strategies, backtesting & diagnostics, and deployment readiness (FastAPI service, Redis store, scheduler, monitoring).
- **Current State**: Data pipelines and infrastructure are operational with async route coverage under `tests/ingestion_service` and manifest regression checks in CI. The relaxed-gate Horizon-120 XGB (`final_equity 4.48`), Calmon TCN suite (`final_equity 1.28/3.62/1.85` for horizons 60/120/180), and elastic-net blender (`final_equity 4.48`) all clear 5 bps costs after the latest retrains. The Oct 1–Oct 28 2025 forward replay (40 201 rows) confirms deployable masks now fire across the stack: the base manifest posts 12 gate hits (8 toggles, `final_equity 1.2336`, `gate_coverage 2.99e-4`), the TCN relaxed gates finally cross the coverage floor (`gate_coverage 4.73e-4/7.71e-4/4.23e-4`, `final_equity 1.03/1.94/1.01` at inference for h60/h120/h180), and the eased blender manifest sustains `gate_coverage 15.8 %` (`toggle_count 6 346`). Blender stride‑1 sandbox runs (`gate_coverage ≈0.2 %`, 134 toggles) still bound turnover expectations, stride-aware batching in inference keeps relaxed experiments production-safe, and the new scheduler-driven inference lane feeds a Redis decision queue that the trading dry-run service consumes with Prometheus/Grafana coverage. Scheduler inference now auto-computes missing manifest/regime/lookback columns, respects the manifest `apply_calibration` flag (override via `INFER_APPLY_CALIBRATION`), and trading logs richer exit contexts (expected/net PnL, quote source, `loss_guard` events) so dry runs mirror production instrumentation.
- **Live readiness**: Deployment contracts, launch ladder CLIs, runtime risk/intent-ledger/deadlock modules, and the expanded audit/metrics footprint enforce kill-/safe-mode policies, signed audit provenance, reconciliation safe-mode latching, and coverage-focused deadlock mitigations before promoting BTC/ETH/SOL to live trading. `analysis.live_readiness_check` now produces GO/NO-GO Markdown and JSON bundles while `analysis/acceptance_trade_proof`, `analysis/exit_attribution_report`, and `analysis/project_stance_snapshot` document the executed exits, exit reasons, and current stage posture for every promotion; dry-run evidence + forensics (`scripts/extract_container_logs.py`, `analysis/trading_log_forensics.py`, `analysis/market_trade_alignment.py`) now add PnL/regret artifacts under `reports/log_forensics/` for each rehearsal.
- **Portfolio sweeps**: XGB-only scenarios in `configs/perf_sweep_scenarios.yaml` crowned the `medium_xgb_low_cost` run (Sharpe 278, 100 trades, ≈51 % time-in-position; ETH+SOL exposure) as the primary policy. Artifacts live under `experiments/perf_sweeps/medium_xgb_low_cost/portfolio_final` with policies mirrored to `configs/final_portfolio_policies.yaml` and deployment wiring via `configs/deployment_portfolio_contract.yaml` + `configs/dry_run/infer_jobs_portfolio_policy.yaml`; Docker mounts `experiments/perf_sweeps` so scheduler/trading can load `xgb_primary` during dry-runs.
- **Immediate Goal**: Harden the forward replay guardrails (now part of CI for the TCN suite), ship deployability runbooks that capture the widened manifests, and decide on rollout sequencing (base + blender vs full trio) once monitoring confirms the new coverage floor persists.

---

## 2. Repository Map & Responsibilities
| Path | Description |
|------|-------------|
| `app/` | FastAPI ingestion service, adapters for exchanges/news/social, feature storage, scheduler entrypoints. |
| `scripts/` | CLI workflows: backfills (`backfill_ccxt_parquet.py`), dataset builders, sanity checks, and all training scripts (`train_base_gbdt.py`, `train_tcn.py`, `train_blender.py`, `train_meta_label.py`). |
| `training/` | Shared ML utilities: data loading, feature augmentation, model wrappers, calibration, metrics, walk-forward split, TCN architecture, meta-label helpers. |
| `datasets/` | Curated offline parquet datasets for modeling (market-only, market+RSS, temporal subsets). |
| `data_lake/` | Raw ingested data partitions (market, news, etc.) for inspection or rebuilds. |
| `models/` | Serialized artifacts from latest experiments (baseline GBDT, TCN, etc.). |
| `monitoring/` | Prometheus/Grafana configurations for production observability. |
| `notebooks/` | Exploratory research notebooks (`starter_training.ipynb`) showcasing training workflows. |
| `structure_exports/` | Auto-generated JSON snapshots of the repo layout used for documentation/tests. |
| `analyzer.py` | Stub maintained for legacy imports in compliance tests. |
| `tests/` | Unit tests for adapters, data lake modules, features, etc. |
| Docs (`README.md`, `TRAINING_WALKTHROUGH.md`, `TRAINING_STATUS.md`, `PROJECT_OVERVIEW.md`) | Developer onboarding, step-by-step workflows, experiment logs, strategic overview (this file). |

---

## 3. Data Ingestion & Infrastructure
### 3.1 Services (Docker Compose)
- **ingestion-api**: FastAPI app (`app/ingestion_service/main.py`) that pre-warms CCXT/News/Social/Onchain clients, mounts a custom `/metrics`, and lazily loads HuggingFace sentiment models when `ML_SENTIMENT_ENABLED=1`.
- **redis** + **redis-exporter**: Redis feature store with metrics exporter; persistent volumes include `redis-data` and the optional HuggingFace cache (`hf-cache`).
- **scheduler**: APScheduler worker calling admin endpoints (`MARKET_JOBS`, `MARKET_INGEST_JOBS`, TTL sweeps, data lake cleanup) sourced from env vars; publishes metrics on port `9002` and now owns manifest-driven inference jobs that enqueue decisions to Redis.
- **trading**: Async consumer of the decision queue (`trading:decisions` by default) that enforces manifest gates, evaluates runtime risk controls, dedupes intents via Redis locks, reconciles state vs exchange, executes deadlock policies, persists HMAC-signed audit logs, and exports Prometheus metrics on `TRADING_METRICS_PORT`.
- **prometheus** & **grafana**: Monitoring stack fronting the ingestion API, scheduler, Redis exporter, and custom dashboards under `monitoring/grafana`.

### 3.2 Configuration (`app/ingestion_service/config.py`)
- Loads env via `.env`, covering exchange/news/social/on-chain keys, Redis connection info (`REDIS_URL`, `FEATURE_NAMESPACE`, TTL), scheduler cadence, and admin token defaults.
- Data lake roots are configurable (`MARKET_PATH`, `ONCHAIN_PATH`, `SOCIAL_PATH`, `NEWS_PATH`) alongside toggles for `BACKFILL_*` loops, TTL sweeps (`TTL_SWEEP_*`), and retention-based cleanup (`DATA_LAKE_RETENTION_DAYS`, `DATA_LAKE_CLEANUP_CRON`, `DATA_LAKE_CLEANUP_ROOTS`).
- ML and storage extras: `ML_SENTIMENT_ENABLED`, `SENTIMENT_MODEL_ID`, `HF_HOME`, `ML_MAX_WORKERS`, and optional `FSSPEC_STORAGE_OPTIONS` for remote parquet targets.

### 3.3 Adapters & Features
- **Market** (`app/features/factory/market_factory.py`): Builds `ret_1`, `logret_1`, EMAs, MACD, RSI, `hl_spread`, `oi_obv`; `app/features/pipelines/market_pipeline.py` and `app/features/jobs/backfill.py` push the engineered payloads into Redis with TTL/metrics.
- **Social** (`app/features/ingestion/social_client.py` + `app/adapters/sentiment_adapter.py`): Async Twitter/Reddit fetchers with tenacity retries and optional sentiment enrichment before writing parquet.
- **News** (`app/features/ingestion/news_client.py` + `app/adapters/news_adapter.py`): API/RSS ingestion with schema guards, normalized partitions under `data_lake/news`; `fetch_news_rss_once` powers both the FastAPI route and CLI, and the `/ingest/news` endpoint now persists Parquet partitions (`dt` + `source`) and mirrors rows into Redis (covered by `tests/ingestion_service/test_routes.py::test_post_news_rss_success`).
- **On-chain** (`app/features/ingestion/onchain_client.py` + `app/adapters/onchain_adapter.py`): Glassnode/Covalent surfaces into `data_lake/onchain` (populates once API keys are supplied) while returning schema-stable frames on failure.
- **Feature Store** (`app/features/store/redis_store.py`): Async Redis cache with Prometheus counters/histograms reused by ingestion endpoints, backfill jobs, and scheduler TTL sweeps.
- **Inline enrichment**: Market ingest/backfill now run `training.feature_eng.augment_market_features` inline, pushing `hl_spread_z`, `rvol_20`, liquidity ranks, and None-safe payloads to Redis while attaching `close`/`price` columns for downstream inference/parity checks.

### 3.4 Scheduler Inference & Trading Dry Run
- `app/scheduler/main.py` now parses `INFER_JOBS`, loads deployable manifests via `training.infer`, replays a rolling parquet window, and enqueues decision payloads to Redis (`DECISION_QUEUE_KEY`, default `trading:decisions`). It attaches `close`/`price` columns to the feature frame, backfills missing manifest features (computing registered ones when possible), and respects manifest `apply_calibration` flags when loading predictors. Metrics track gate coverage per model/symbol via `observe_gate_coverage` plus new counters for queue throughput.
- `app/trading/service.py` consumes that queue, keeps per-symbol state (`app/trading/state.py`) with file/Redis/Postgres backends, logs audit events (`app/trading/audit.py`), and routes orders through the CCXT adapter with spread guardrails and dry-run toggles. The shared `app/trading/decision.py::decide_bar` applies entry/exit/hold/SL/TP/spread checks consistently across live loops and offline sweeps; `TRADING_MODELS` entries accept guard fields plus optional churn controls, and the trading worker can optionally enforce quote-based exits on an interval via `TRADING_PRICE_MONITOR_INTERVAL_SECONDS`.
- Prometheus integration ships with dedicated metrics (`trading_trade_attempts_total`, `trading_trade_notional_total`, `trading_gate_toggles_total`, `trading_position_active`, `trading_realized_pnl_total`), surfaced on port 9010 and wiring into `monitoring/grafana/dashboards/trading-overview.json` plus new alert rules for stale queues/state drift.
- Helper CLI `scripts/verify_trading_redis.py` inspects the Redis-backed state/audit artefacts so ops can verify dry-run behaviour without hitting the database.

---

## 4. Dataset Pipeline
> Heavy parquet snapshots (market_multi, blender, rss variants) were trimmed from git in this drop; rerun the backfill/build helpers below to recreate them locally whenever you need refreshed inputs.
### 4.1 Acquisition
- `scripts/backfill_ccxt_parquet.py`: Backfills OHLCV data for specified exchange/symbol/timeframe into `data_lake/market/exchange=...` partition.
- `scripts/rss_to_parquet.py`: Pulls RSS feeds over a time window, stores in `data_lake/news/rss`.

### 4.2 Curation
- `scripts/build_market_dataset.py`: Reads market parquet partitions, merges engineered features, and produces labeled dataset (timestamp-aligned, with `ret_next`, `y_dir`).
- `scripts/build_training_matrix.py`: Legacy helper that combines market features with coarse RSS aggregates for focused validation windows.
- `scripts/build_blender_matrix.py`: Generates the year-wide RSS-enriched matrix with intraday spike features, probability momentum (`prob_diff`, `*_mom_1`), relaxed-gate masks, and summary stats (`..._rss_latest_stats.json`). The current build covers 2024-09-01 ➜ 2025-10-26 (606 121 rows) at `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet` (mirrored as `..._2025-10_rss_latest.parquet`), while the forward replay snapshot `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (40 201 rows) carries model probabilities for Oct 1 – Oct 28 2025 audits.

### 4.3 Sanity Checks
- `scripts/sanity_check_two_weeks.py`: Orchestrated run that backfills market data, builds datasets, and ensures RSS pipeline completeness for a two-week window.

### 4.4 Redis Backfill & Admin APIs
- `app/features/backfill/core.py` & `runner.py`: Reuse parquet partitions to repopulate Redis feature keys for specific symbols/timeframes.
- FastAPI admin routes (`app/ingestion_service/routes.py`) expose market/social/news/on-chain ingest endpoints plus TTL sweeps, mirroring what the scheduler automates.

---

## 5. Feature Engineering
### 5.1 Baseline Features
- Market features include returns, log returns, rolling volatility (`rvol_*`), MACD ensemble, RSI, `hl_spread`, and On-Balance Volume.

### 5.2 Augmented Features (`training/feature_eng.py`)
- Adds derivative signals: absolute/squared returns, rolling means & stds, z-scores (e.g., `ret_z_20`, `rvol_z_50`), MACD histogram, EMA ratios, spread z-score, OBV differentials.
- Utilized by both XGB and TCN training pipelines to enrich signal space without rebuilding datasets.

### 5.3 Sliding Windows (`training/data.py`)
- Supports configurable window length and stride.
- Handles per-window standardization and global scaling (StandardScaler) for TCN inputs.
- Provides `select_market_features` for tabular models and `ensure_labels` to enforce `ret_next`/`y_dir` availability.

### 5.4 Feature Store Integration
- `app/features/pipelines/market_pipeline.py` batches engineered rows into Redis via `RedisFeatureStore`, preserving version tags and TTLs.
- Scheduler TTL sweeps (`app/features/jobs/backfill.py` + env `TTL_SWEEP_*`) keep cached feature namespaces fresh and observable.

---

## 6. Modeling Components
### 6.1 Walk-Forward Framework (`training/walkforward.py`)
- Time-based cross-validation with purging and embargo support.
- Ensures no leakage between training and validation folds.

### 6.2 Baseline XGBoost (`scripts/train_base_gbdt.py` / `training/model.py`)
- CLI defaults mirror the relaxed Calmon gate (`--max-spread-z 0.25`, `--max-rvol20 2e-4`, no probability filter) while manifests persist the deployable inference mask (`hl_spread ≤ 0.0007`, `hl_spread_z ≤ -0.25`, `rvol_20 ≤ 8e-5`, `prob ≥ 0.72`, `min_hold 10`).
- Reports (`report.json`) capture calendar-month diagnostics, spread stress tests (`spread_scale` grid), and RSS coverage audits; KPI payloads are normalised through `training/reporting.ensure_kpi_schema`.
- Artifacts include booster JSON, calibrator, feature list, threshold, manifest, and gate coverage replay CSV. Current relaxed run: `final_equity 4.48`, Sharpe 108, `gate_fraction 9.4 %` on the 2024–2025 feed.
- `scripts/report_shortlist.py` surfaces deployable candidates by scanning reports and enforcing equity/turnover/RSS criteria, helping reviewers validate the baseline alongside TCN/blender outputs.
- Regression guardrails (`tests/regression/test_manifest_gating.py`) ensure every manifest’s `gate_config` and threshold stays in lockstep with its `report.json`, catching drift during CI.

### 6.3 Temporal Convolutional Network (`scripts/train_tcn.py` / `training/tcn_model.py`)
- TinyTCN architecture with residual TemporalBlocks, dropout, and adaptive pooling; stride-aware window builder reduces sample counts for long horizons.
- Outputs include `tcn.pt`, scaler/preprocess bundle, calibrator, manifest, threshold, `fold_logits.parquet`, and monthly probability σ diagnostics.
- Relaxed Calmon runs yield post-cost profitability:
  - `tcn_h60_calmon_relaxed`: `final_equity 1.054`, 67 toggles, Sharpe 16.6, threshold 0.55.
  - `tcn_h120_calmon_relaxed`: `final_equity 1.331`, 180 toggles, Sharpe 24.9, threshold 0.65.
  - `tcn_h180_calmon_relaxed`: `final_equity 1.190`, 48 toggles, Sharpe 29.6, threshold 0.575.
- `models/oos_replay_summary_latest.json` and `models/tcn_gate_replay_summary.json` log gate behaviour for audit, ensuring live inference adheres to the deployable mask (the previous `...oct_nov_2025.json` snapshot remains for regression comparisons).
- October 2025 forward replay (`models/oos_replay_summary_latest.json`, 40 201 rows) now shows deployable masks live across the TCN suite: h120 registers 31 gate hits (62 toggles, `gate_coverage 7.71e-4`, `final_equity 1.94`), while h60 and h180 land shallower but non-zero coverage floors (`gate_coverage 4.73e-4`/`4.23e-4`, `final_equity 1.03`/`1.01`). The archived zero-coverage snapshot (`...oct_nov_2025.json`) remains for regression contrast.

### 6.4 Blender (`scripts/train_blender.py` / `training/blender.py`)
- Elastic-net logistic regression (StandardScaler + LogisticRegressionCV) over probability momentum, RSS spike features, and regime fields from the blender matrix.
- Threshold search enforces turnover guards and records RSS audits. `models/blender_h120_v6` now posts `final_equity 4.48`, Sharpe 206.8, 4 809 toggles at threshold 0.5 with the relaxed training gate, and the deployable manifest (`prob ≥ 0.5`, `rvol_20 ≤ 5e-4`, `min_hold 10`) sustains ≈15.8 % coverage (6 346 toggles) on the Oct 2025 replay. Stride‑1 sandbox variants (`blender_h120_stride1_v2`) retain 4.48 equity with 134 toggles under a 0.5133 probability gate, providing turnover bounds.
- CLI adds `--class-weight {balanced,none}` (default balanced) and treats `--calibration-cv <=1` as “no calibration”, matching the production stack’s preference for deterministic logits.
- Feature inventories and RSS coverage stats live alongside manifests so ops knows when to fall back to no-RSS feature sets.

### 6.5 Meta-Label (`scripts/train_meta_label.py` / `training/meta.py`)
- Triple-barrier labeling with flexible volatility targets feeds a logistic meta filter. The refreshed script supports the relaxed gate defaults, stride control, and shares KPI schema with the rest of the stack.
- Current attempts remain exploratory (probability collapse on narrow validation windows); production rollout is gated on extending the blender matrix and recovering dynamic range before fitting.

### 6.6 Inference Utilities (`training/infer.py`)
- Centralises manifest handling with `load_manifest_artifacts`/`apply_manifest_gates`, ensuring inference reads gate predicates, probability columns, and thresholds straight from each artifact bundle.
- Publishes Prometheus gauges (`model_gate_coverage_ratio`, `model_rss_minute_spike_share`, `model_probability_sigma`, plus their `_threshold` counterparts) so live scoring surfaces coverage drift, RSS health, and probability variance guardrails alongside predictions.
- Provides helpers (`score_base_with_manifest`) that attach gate decisions to scored batches, aligning regression tests and the production API.

---

## 7. Experimental Ledger
### 7.1 XGBoost Experiments
- **Calmon relaxed (current)** – `models/base_xgb_h120_calmon_spread0` retrained on the 2024–2025 feed delivers `final_equity 4.48`, `gate_fraction 9.4 %`, and Sharpe 108 after costs. Monthly diagnostics confirm stability and expose the coverage replay used for live gating.
- **Cost stress (`spread_scale` sweep)** – Variants `{0.0, 0.05, 0.1, 0.2}` preserve the same equity/turnover envelope, demonstrating resilience to 20 % spread inflation.
- **Gate replay** – `live_gate_coverage.csv` keeps the deployable mask within ±1.63× of the baseline coverage, ensuring turnover budgets hold when the strict inference gate is enforced.
- **Forward replay** – `models/oos_replay_summary_latest.json` (Oct 1 → Oct 28 2025, 40 201 rows) retains 4.48 equity under the relaxed training gate; the deployable manifest now fires 12 gate hits (8 trades, `final_equity 1.2336`, `gate_coverage 2.99e-4`), up from the zero-coverage `...oct_nov_2025.json` snapshot kept for regression.

### 7.2 TCN Experiments
- **Relaxed Calmon suite** – Horizons 60/120/180 all clear 5 bps costs with tight turnover guards (≤200 toggles). Probability variance guardrails remain above the 0.03 threshold, signalling healthy calibration after loosening training gates.
- **OOS gate audits** – `models/oos_replay_summary_latest.json` and `tcn_gate_replay_summary.json` confirm the deployable TCN masks still collapse coverage (<0.001 %) even after the base/blender retune, informing live expectations and highlighting that retraining can focus on the relaxed gate while keeping deployable safety nets.
- **Inference batching** – `training/infer.predict_tcn` now processes batches sized by stride, so shrinking stride for deployability experiments no longer exhausts GPU/CPU memory.

### 7.3 Blender Experiments
- **v5 (baseline)** – First elastic-net attempt showed modest gains but relied on sparse RSS spikes, limiting deployment appetite.
- **v6 (current)** – With the expanded matrix (`build_blender_matrix.py` intraday features + probability momentum) the logistic stack reaches `final_equity 4.48`, Sharpe 206.8, 4 809 toggles at threshold 0.5. `rss_audit` passes with daily coverage 99.5 % and minute spike share 0.991, and the deployable manifest now gates inference at `prob ≥ 0.5`, `rvol_20 ≤ 5e-4`, `min_hold 10` while the report captures the relaxed gate metadata (stride-aware smoothing is explored in the stride‑1 sandbox lineage).
- **Stride‑1 sandbox** – `models/blender_h120_gate_test`, `blender_h120_stride1`, and `blender_h120_stride1_v2` collapse the smoothing window to one bar, keeping relaxed equity at 4.48 while pushing gate share into the 52–70 % range; they provide an upper bound for turnover before finalising production manifests.
- **Forward replay** – `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (40 201 rows, Oct 1 → Oct 28 2025) and `models/oos_replay_summary_latest.json` capture how the training-vs-inference gates behave on the latest window; the retuned manifest delivers ≈15.8 % deployable coverage (6 346 toggles) versus the archived zero-coverage `...oct_nov_2025.json` snapshot.

### 7.4 Diagnostics & Tooling
- `training/reporting.ensure_kpi_schema` standardises KPI payloads; `scripts/report_shortlist.py` ranks deployable models (base, TCN, blender) under consistent criteria.
- Threshold diagnostics (`diagnostic_final_equity`) and RSS audits are embedded across all reports to make regression testing and CI validation straightforward. `tests/regression/test_report_shortlist.py` executes the shortlist CLI in CI to guarantee the relaxed Calmon baseline still satisfies the deployable filters.

---

## 8. Deployment Readiness
- **Model Artifacts**: Stored under `models/`; paired `training/infer.py` loaders guarantee live compatibility (feature list, calibrators, scalers).
- **Feature Store**: Redis (`app/features/store/redis_store.py`) backed by metrics and TTL sweeps; scheduler/admin endpoints keep namespaces fresh.
- **Monitoring & Metrics**: Prometheus scrapes ingestion API, scheduler, Redis exporter, feature store, and ML inference counters. New gauges under `app/monitoring/model_metrics.py` expose gate coverage, RSS minute spike share, and probability σ alongside manifest-configured thresholds; Grafana dashboards + alert rules (`monitoring/alert.rules.yml`) raise warnings when coverage or variance drift below guardrails, and the latest rule (`TCNGateCoverageUnexpected`) notifies when deployable TCN coverage exceeds the historic zero baseline so operators can validate widened manifests quickly.
- **Trading controls**: `app/trading/service.py` now enforces kill-/safe-mode envs, runtime risk gating (turnover/exposure/cooldown/per-symbol limits), Redis intent ledger dedupe, reconciliation safe-mode latching, programmable deadlock policies, richer audit payloads (expected/net PnL, quote source, `loss_guard` hits), and HMAC-signed logging. Stage-0 overrides now run equity-fraction sizing (capital 200, `max_total_notional=80`, `equity_fraction=0.33`, `max_equity_fraction=0.35`, `compounding_step_usd=5`), per-symbol trigger overrides, stop shaping, and `cooldown_minutes_after_exit: 2` / `cooldown_minutes_after_loss: 5` while the dry-run compose keeps BTC/ETH/SOL as primary symbols (`TRADING_SHADOW_SYMBOLS=[]`). Reports from `analysis.*` CLIs (including `analysis.live_readiness_check`, `analysis.acceptance_trade_proof`, `analysis.exit_attribution_report`, `analysis.project_stance_snapshot`, plus the forensics/alignment chain) and Prometheus metrics (`trading_deadlock_*`, `trading_safe_mode_latched`, `trading_risk_blocked_total`, `trading_intent_ledger_state_total`, `trading_reconcile_runs_total`) mirror the deployment contract's live invariants.
- **ML Sentiment Endpoint**: `/ml/sentiment/predict` (`app/ingestion_service/ml_routes.py`) serves HuggingFace pipelines when enabled, respecting `ML_MAX_WORKERS` and publishing latency stats.
- **Parquet Writer**: `app/ingestion_service/utils.py` validates schemas, adds dt partitions, and supports remote backends via `FSSPEC_STORAGE_OPTIONS`.
- **Tests**: `tests/ingestion_service` exercises async routes end-to-end with fakeredis; `tests/regression` keeps manifests in sync with `report.json` and enforces shortlist criteria; `.github/workflows/ci.yml` now front-loads a TCN forward replay guardrail (runs `scripts/run_oos_eval.py --family tcn --stride 30` for h60/h120/h180 and fails when deployable coverage <5e-4 or `final_equity < 1.2`) before executing training and ingestion suites. Modeling utilities (`training/`) still need deeper unit coverage.

---

## 9. Dependencies & Environment
- Python 3.12 via local `.venv` or Docker; packaging metadata lives in `setup.py` / `algo_data_ingestion.egg-info`.
- `requirements.txt` (extra index for CPU PyTorch) is the source of truth for both services and training.
- Async/infra stack: `fastapi`, `uvicorn[standard]`, `httpx`, `tenacity`, `redis>=5`, `pydantic-settings`, `APScheduler`, `feedparser`, `tweepy`, `prometheus-client`, plus `fakeredis` for isolated route tests.
- ML stack: `pandas`, `numpy`, `scikit-learn`, `xgboost 3.x`, `joblib`, optional `torch` + `transformers` for sentiment, plus `numba` for feature speed-ups.
- Storage tools: `fsspec` with optional `s3fs`/`gcsfs` and HuggingFace cache control via `HF_HOME`.
- Recent change: Reinstalled `numpy==1.26.4` to resolve typing/import errors during TCN runs.

---

## 10. Risk & Mitigation
| Risk | Impact | Mitigation |
|------|--------|------------|
| Deployable gate drift (prob/vol/spread masks) | Turnover explodes or drops to zero in production | Replay `live_gate_coverage.csv` each retrain, run the Oct 2025 replay (`models/oos_replay_summary_latest.json`) in CI, mirror predicates in inference adapters, and alert on coverage outside ±2× historical band. |
| RSS coverage collapse | Blender loses signal, equity regresses | Track `rss_audit minute_spike_share` and fall back to the no-RSS feature set when coverage <5e-4; expand feed roster and monitoring. |
| Probability variance collapse | Thresholds become unstable; blender/meta stack unusable | Enforce the `prob_sigma_guardrail` and halt deployment if monthly σ <0.03; re-run relaxed gate retrains or adjust stride/windows. |
| Artifact/config skew between training and live | Live scoring diverges from reports | Consume manifests for feature lists + gate configs, add regression tests that pipe historical data through inference adapters, and version control manifests. |
| Large binary artifacts bloating repo | Slow CI and clone times | Use Git LFS or artifact storage for heavy parquet/torch files once deployment pipeline is in place; prune outdated experiment folders. |

---

## 11. Backlog & Action Items
### 11.1 Validation & Monitoring
- [x] Extend out-of-sample replays into Oct–Nov 2025 to confirm relaxed-gate robustness across new regimes (`models/oos_replay_summary_latest.json` keeps the retuned snapshot; `...oct_nov_2025.json` remains for comparison).
- [ ] Monitor the widened TCN inference gates via the CI guardrail and adjust thresholds only if Oct–Nov 2025 coverage drops back below the 5e-4 floor while respecting turnover budgets.
- [ ] Automate coverage drift alerts using manifest gate baselines (spread/rvol/prob/min-hold) and `live_gate_coverage.csv` comparisons.
- [ ] Wire RSS audit metrics into monitoring so the blender auto-falls back when minute spike share <5e-4.

### 11.2 Modeling Enhancements
- [ ] Explore alternative loss functions or monotonic constraints for the base XGB while preserving the relaxed gate.
- [ ] Sweep TCN channels/dilations and evaluate shorter strides for high-volatility months without exceeding the 200-toggle guardrail.
- [ ] Resume meta-label experiments once an extended blender matrix is available; enforce ≥1.2 equity and ≥20 toggles before considering deployment.

### 11.3 Tooling & Automation
- [x] Integrate `scripts/report_shortlist.py` into CI to flag KPI regressions automatically (`tests/regression/test_report_shortlist.py`).
- [x] Add manifest drift regression checks (`tests/regression/test_manifest_gating.py`) to the GitHub Actions workflow.
- [ ] Package dataset + model builds in a reproducible orchestration script (Makefile/`invoke`) for retrains.
- [ ] Add regression tests that replay historical data through `training/infer.py` using the manifest gates.

### 11.4 Documentation & Ops
- [ ] Keep `TRAINING_STATUS.md` and `TRAINING_WALKTHROUGH.md` updated after each major run.
- [ ] Document the live inference path (feature fetch, gating, threshold application) with references to manifest fields.
- [ ] Plan Git LFS or external artifact storage to keep repository size manageable as more experiments accumulate.

---

## 12. Reference Commands
- **Build RSS-enriched blender matrix**
  ```bash
  python scripts/build_blender_matrix.py \
    --source datasets/market_btcusdt_1m_2024_2025.parquet \
    --out datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet \
    --base-dir models/base_xgb_h120_calmon_spread0 \
    --tcn-dir models/tcn_h120_calmon_relaxed \
    --tcn-stride 30 --include-reddit
  ```
- **Train base XGB (Calmon relaxed)**
  ```bash
  python scripts/train_base_gbdt.py \
    --data datasets/market_btcusdt_1m_2024_2025.parquet \
    --out models/base_xgb_h120_calmon_spread0 \
    --fold-scheme calendar_month --n-folds 6 \
    --cost-bps 5 --max-spread-z 0.25 --max-rvol20 2e-4
  ```
- **Train TCN (Calmon relaxed)**
  ```bash
  python scripts/train_tcn.py \
    --data datasets/market_btcusdt_1m_2024_2025.parquet \
    --out models/tcn_h120_calmon_relaxed \
    --window 192 --stride 30 --channels 64,64 \
    --epochs 10 --batch-size 256 --lr 5e-4 --class-weight 2.0 \
    --cost-bps 5 --max-spread-z 0.25 --max-rvol20 2e-4
  ```
- **Train elastic-net blender**
  ```bash
  python scripts/train_blender.py \
    --matrix datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet \
    --base-dir models/base_xgb_h120_calmon_spread0 \
    --tcn-dir models/tcn_h120_calmon_relaxed \
    --out models/blender_h120_v6 \
    --cost-bps 5 --tcn-stride 30 \
    --max-total-turnover 10000 --min-toggle-count 2
  ```
- **Compile shortlist for review**
  ```bash
  python scripts/report_shortlist.py --models-root models --out models/report_shortlist.json
  ```
- **Run forward replay guardrail / diagnostics**
  ```bash
  python scripts/run_oos_eval.py \
    --family {base_xgb|tcn|blender} \
    --data datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet \
    --model-dir models/<model_name> \
    --align-gates --stride 30 --window 192 --channels 64,64
  ```
  Use `--family blender` with the refreshed CLI to audit logistic manifests alongside base/TCN runs; the CI guardrail wraps `--family tcn` for horizons 60/120/180 and enforces the 5e-4 coverage floor.

---

## 13. Supporting Documentation
- `README.md`: Core setup, docker usage, monitoring stack overview.
- `TRAINING_WALKTHROUGH.md`: Step-by-step base + blender training instructions.
- `IMPLEMENT_WITH_YOUR_DATASETS.md`: How to tailor pipeline for new datasets.
- `SANITY_CHECKS.md`: Backfill + dataset validation flow.
- `TRAINING_STATUS.md`: Rolling summary of recent experiments and next steps.
- `notebooks/starter_training.ipynb`: Exploratory notebook for feature augmentation and walk-forward inspection.
- `.github/workflows/ci.yml`: GitHub Actions pipeline executing regression (manifest + shortlist) and training suites on pushes/PRs.

---

## 14. Glossary
- **OOF**: Out-of-fold predictions from walk-forward validation, used for unbiased performance assessment.
- **PnL**: Profit and Loss, computed via `equity_curve` function using thresholded probabilities.
- **Spread Scaling**: Additional transaction cost proportional to observed bid-ask spread proxies (`hl_spread`).
- **TCN**: Temporal Convolutional Network, leveraging causal convolutions to model sequential dependencies.
- **Blender**: Model that combines base and TCN probabilities with exogenous features to improve decision quality.
- **Meta-label**: Secondary classifier determining whether to act on base signals, filtering noise via triple-barrier events.

---

## 15. Contact & Ownership
- **Modeling**: ML research/engineering team responsible for `training/` and `scripts/train_*` workflows.
- **Ingestion Ops**: Platform/infrastructure team managing `app/`, Docker services, and monitoring stack.
- **Documentation/Data**: Shared responsibility; updates reflected in `TRAINING_STATUS.md` and this dossier after significant changes.

---

_This document is intended to be the single source of truth for technical stakeholders. Update after each major experiment, infrastructure change, or milestone._
