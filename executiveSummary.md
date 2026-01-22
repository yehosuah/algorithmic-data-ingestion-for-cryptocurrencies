# Executive Summary: Technical Architecture and Trading Pipeline

_Last updated: 2026-01-22 17:38 UTC_

This document is the onboarding map for every technical component that turns market data into deployable trading decisions in this repo. It explains the financial technologies, the modeling stack, and the production trading environment, and it ends with a technical debrief meant to ground new contributors quickly.

---

## 1) What this project does (end-to-end)
- Acquire multi-source market intelligence (exchange OHLCV, order books, quotes, on-chain, social, news).
- Engineer features, build labeled datasets, and train multiple model families.
- Calibrate probabilities, enforce gates and thresholds, and generate decisions.
- Route decisions into a live-like trading service with risk controls, audit trails, and monitoring.
- Operate a staged launch ladder with readiness checks, deadlock handling, and forensic reporting.

Core outputs include the Redis feature store, Parquet data lake partitions, model artifacts and manifests, deployment contracts and runtime overrides, trading audit/state logs, and readiness or forensics reports used for promotions.

---

## 2) System diagram (logical flow)
```text
[CCXT/Exchange] ---> [Ingestion API] ---> [Redis Feature Store] ---> [Scheduler]
        |                  |                    |                    |
        v                  v                    v                    v
   [Order Books]     [Parquet Data Lake]     [TTL Sweeps]     [Inference Jobs]
                                                                  |
                                                                  v
                                                     [Decision Queue (Redis)]
                                                                  |
                                                                  v
                                                     [Trading Service + Risk]
                                                                  |
                                                                  v
                                                         [CCXT/Exchange]

[Training + Analysis] <--- [Data Lake + Reports] ---> [Models/Calibrators/Manifests]
                                  |
                                  v
                       [Prometheus + Grafana + Alerts]
```

This is the logical flow; in practice these components run as docker-compose services with environment-driven configuration and stage-specific overrides.

**Deployment view (docker-compose + env wiring)**
- Services: `ingestion-api`, `scheduler`, `redis`, `redis-exporter`, `trading`, `prometheus`, `grafana`.
- Default ports: ingestion `8000`, scheduler metrics `9002`, trading metrics `9010`, Prometheus `9090`, Grafana `3000`, Redis exporter `9121`.
- Core wiring uses environment variables to bind services and policies:
  - Ingestion API: `ADMIN_TOKEN`, `EXCHANGE_NAME`, `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET`, `REDIS_URL`, `FEATURE_TTL_SEC`, `MARKET_PATH`, `NEWS_PATH`, `ONCHAIN_PATH`, `SOCIAL_PATH`, `ML_SENTIMENT_ENABLED`, `SENTIMENT_MODEL_ID`.
  - Scheduler: `API_BASE_URL`, `MARKET_JOBS`, `MARKET_INGEST_JOBS`, `TTL_SWEEP_CRON`, `DECISION_PAYLOAD_ITEMS`, `INFER_JOBS`, `DECISION_QUEUE_URL`, `DECISION_QUEUE_KEY`, `DATA_LAKE_RETENTION_DAYS`, `DATA_LAKE_CLEANUP_CRON`, `DATA_LAKE_CLEANUP_ROOTS`.
  - Trading: `TRADING_DRY_RUN`, `TRADING_MODELS`, `TRADING_DEPLOYMENT_CONTRACT`, `TRADING_RISK_LIMITS_PATH`, `TRADING_DEADLOCK_POLICY_PATH`, `DECISION_QUEUE_URL`, `DECISION_QUEUE_KEY`, `TRADING_INTENT_LEDGER_BACKEND`, `TRADING_INTENT_LEDGER_REDIS_URL`, `TRADING_AUDIT_HMAC_KEY`, `TRADING_KILL_SWITCH`, `TRADING_SAFE_MODE`, `TRADING_DECISION_MAX_AGE_SECONDS`, `TRADING_LAST_TS_GRACE_BARS`.
  - Model paths resolve via `MODELS_ROOT` (default `/opt/models` inside containers).

```text
ingestion-api (8000) --API_BASE_URL--> scheduler
scheduler --DECISION_QUEUE_KEY--> redis
trading --DECISION_QUEUE_URL/INTENT_LEDGER--> redis
prometheus -> ingestion-api/scheduler/redis-exporter/trading
grafana -> prometheus
```

---

## 3) Core technology stack (by layer)
**Languages and runtime**
- Python 3.12 for services, pipelines, training, and analysis.
- Docker and docker-compose for repeatable local and staged deployments.

**APIs and services**
- FastAPI + Uvicorn for ingestion, feature retrieval, and admin operations.
- APScheduler for cron-style backfills, TTL sweeps, ingest loops, and inference jobs.
- A standalone trading service that consumes decisions, enforces gates, and manages risk.

**Data and storage**
- Redis as the feature store, decision queue, audit stream, and intent ledger backend.
- Parquet data lake (pyarrow + pandas) for raw and derived data snapshots.
- Optional object storage via fsspec (S3/GCS) and optional Postgres for state/audit.

**Market connectivity**
- CCXT (REST) and CCXT Pro (websocket) for exchange data and execution flows.
- Exchange configuration supports spot or futures (default is USDT-margined futures).

**ML and analytics**
- pandas, numpy, scikit-learn, xgboost, joblib for tabular modeling.
- PyTorch-based sequence models (TCN, Transformer, DeepLOB) for temporal inference.
- Calibration utilities and probability diagnostics baked into training and inference.
- Opt-in NLP via HuggingFace transformers and sentence-transformers.

**Monitoring**
- Prometheus metrics, Redis exporter, and Grafana dashboards for telemetry.
- Alert rules tied to model coverage, probability variance, and trading invariants.

**Backtesting and research**
- VectorBT, TA-Lib, Optuna, and notebooks for rapid strategy research.

**Supporting libraries (glue and infra)**
- pydantic + pydantic-settings for configuration and schema validation.
- httpx + tenacity for async HTTP with retries; websockets for streaming.
- psycopg for Postgres connectivity, PyYAML for config parsing.
- numba for accelerated feature kernels; matplotlib for diagnostic plots.
- feedparser for RSS ingestion; tweepy + ratelimit for social ingestion.

---

## 4) Data inputs and financial data domains
**Market data (primary)**
- OHLCV, order books, and quote snapshots are pulled from exchanges via CCXT.
- Symbols include BTC/USDT, ETH/USDT, SOL/USDT for the multi-symbol ladder.
- Timeframes are typically 1m and 5m with rolling windows for replay and inference.

**On-chain data**
- Providers include Glassnode and Covalent for chain metrics.
- Features capture deltas, rolling stats, drawdowns, volatility proxies, and z-scores.

**Social data**
- Twitter/X and Reddit ingestion with retry, rate-limit, and optional sentiment enrichment.
- Text embeddings are available when sentence-transformers is enabled.

**News data**
- News API and RSS ingestion with schema validation and partitioned storage.
- News entries can be mirrored into Redis for consistent feature availability.

All external sources are key-based; missing keys or empty sources return no-data results but keep schemas stable, which is important for downstream pipelines.

---

## 5) Ingestion system and storage topology
**FastAPI ingestion service**
- `/ingest/*` endpoints pull market, on-chain, social, and news data.
- `/ingest/features/*` endpoints serve point and range lookups.
- `/ingest/admin/*` endpoints drive backfills and TTL sweeps (token-protected).
- Metrics are exposed on `/metrics` for Prometheus scraping.

**Scheduler (APScheduler)**
- Runs backfills, TTL sweeps, and continuous ingest loops on cron schedules.
- Owns the manifest-driven inference lane that scores data and emits decisions.
- Can cap decision payloads per job to avoid queue bloat.

**Feature store (Redis)**
- Per-point keys are indexed for range queries and scheduled TTL sweeps.
- Redis streams and hashes support audit, decision, and state workflows.

**Data lake (Parquet)**
- Market, news, and on-chain data are partitioned by domain and time.
- Local or remote (S3/GCS) targets are supported via fsspec configs.
- Retention-based cleanup prunes old partitions on a scheduler cron to keep storage bounded.

---

## 6) Feature engineering (financial signal generation)
**Market features**
- Returns and log returns, rolling volatility (`rvol_*`), spread metrics (`hl_spread`, `hl_spread_z`).
- Trend and momentum features (SMA/EMA, MACD, RSI, OBV, VWAP, CCI, ADX, MFI).
- Liquidity ranks and normalization for spread and volatility regimes.

**Order book features**
- Imbalance, spread, and depth features computed with numba-accelerated kernels.
- Used both in research scripts and in live inference where available.

**On-chain features**
- Rolling mean/std, percent changes, drawdowns, and volatility proxies.
- Whale flow flags and quantile signals for regime tagging.

**Social and news features**
- Counts, sentiment labels/scores, and optional text embeddings.
- Aggregations align to market timeframes for joined datasets.

**Regime and augmentation**
- Regime flags (volatility, liquidity, spread, event) feed gating and filtering.
- Inline enrichment in ingestion and scheduler paths ensures live parity.

---

## 7) Labeling and dataset construction
**Labels**
- Directional and return-based labels for supervised learning.
- Triple-barrier labels for meta-labeling and trade filtering.

**Dataset builders**
- Market-only datasets for base models and sequence models.
- Blender matrices that join market features, RSS/social aggregates, and model predictions.
- Training matrices that align market and exogenous features for experiments.

**Sanity and parity**
- Sanitize datasets to remove duplicates and outliers before training.
- Feature parity checks compare live slices against training distributions.
- Per-symbol gate caps derived from multi-symbol data keep constraints aligned.

---

## 8) Model families and training pipeline
**Model families**
- XGBoost baseline for fast, explainable probability outputs.
- TCN and Transformer models for temporal structure and longer-horizon signal.
- DeepLOB variants for order book oriented sequence modeling.
- Blender models that combine base and TCN probabilities with exogenous signals.
- Meta-labeling models that filter or validate base signals.
- Regime blender and stacking models for portfolio aggregation.

**Training mechanics**
- Time-series CV with embargo gaps to reduce leakage.
- Randomized hyperparameter search with shared config spaces.
- Sampling and weighting policies to balance regimes and costs.
- Walk-forward and forward replay evaluation to enforce deployability.

**Artifacts**
- Model weights, feature lists, thresholds, and calibration bundles.
- Manifests that encode gates for training vs inference.
- Reports that standardize KPI schemas, coverage, and cost diagnostics.

---

## 9) Calibration, gating, and probability hygiene
- Post-hoc calibration via Platt scaling and isotonic blends when needed.
- Manifests encode training and inference gates for probability, spread, volatility, and hold time.
- Trigger optimizer sweeps tune entry/exit/hold/SL/TP guardrails before deployment.
- Preflight coverage checks fail fast when gates collapse or trade counts approach zero.
- Probability drift monitoring uses KS/PSI/Wasserstein, sigma guardrails, and live samplers.
- Inference can override calibration behavior for deterministic dry-run comparisons.

---

## 10) Portfolio policy and optimization
**Portfolio tooling**
- Performance sweeps across policy scenarios with strict turnover and cost controls.
- Selection workflows for best policies and final promotion into deployment configs.
- Ensemble and gating utilities to combine model outputs.

**Portfolio controls**
- Policies and risk limits live in YAML configs and are referenced by deployment contracts.
- Symbol mapping ties model keys and policy IDs to each traded pair.

---

## 11) Inference and decision generation
- Scheduler loads manifests and replay windows to score features consistently.
- Missing features are backfilled or computed to preserve model input parity.
- Gates and thresholds are applied before decisions are enqueued.
- Decision payloads include symbol, side, probability, timestamps, and manifest metadata.
- Payload volume is capped per job to reduce stale queue pressure.

---

## 12) Trading service (execution + controls)
**Signal generation path (how a trade is created)**
- Scheduler loads the deployment manifest and pulls the latest feature window from the data lake and Redis.
- Features are augmented and backfilled to match the manifest feature list, then probabilities are scored.
- Calibrators are applied (or overridden in dry runs), and manifest gates are evaluated.
- A trade signal exists only when the inference gate passes and the probability exceeds the threshold.
- The decision payload is published to the Redis queue with symbol, side, prob, timestamps, and manifest metadata.
- The trading service consumes the queue, dedupes, checks age/grace windows, and runs the shared decision logic.
- Orders are submitted only if the runtime risk engine allows entry and trading guardrails pass.

**Deployed model posture (current state)**
- The deployment contract maps `xgb_primary` to BTC/ETH/SOL for the current ladder.
- Manifests define the exact feature list, calibration path, and inference gates.
- The trading layer adds a second gate via `TRADING_MODELS` and runtime risk limits.
- TCN, blender, and meta-label models exist for research and portfolio sweeps but are only active when the contract promotes them.

**Execution**
- CCXT adapter handles order routing in dry-run or live mode.
- Quote-aware exits use bid/ask updates for spread-accurate decisions.
- Decision logic is shared with offline optimizers for parity.

**Risk engine**
- Limits include leverage, exposure, turnover, drawdown, and daily loss caps.
- Per-symbol caps, cooldowns, and minimum trade notionals are enforced.
- Spread and volatility gates block entries during unfavorable conditions.
- Data staleness checks halt entry when clocks or features fall behind.

**Order and position management**
- Stop-loss, take-profit, profit trailing, max-hold, and min-hold guards.
- Optional churn control via entry filters and disabling probability exits.
- Decision age limits and restart grace windows keep queues healthy.

**State, audit, and reconciliation**
- State backends include file, Redis, or Postgres.
- Audits are HMAC-signed with provenance fields for traceability.
- Intent ledger dedupes and reconciles orders; failures latch safe mode.
- Deadlock policies adjust gates or enter safe mode when coverage stalls.

**Strengths of the trading architecture**
- Manifest-driven gating and runtime risk gates are both enforced, reducing silent drift.
- Decision logic is shared across offline optimization and live execution, limiting parity gaps.
- Risk controls are explicit, audited, and tied to coverage and exposure invariants.
- Observability is built in: coverage, probability variance, and queue health are first-class signals.

**Weaknesses and known risks**
- Gate coverage can collapse in new regimes; when coverage goes to zero, trading halts by design.
- External data availability (news/social/on-chain) can degrade blender-style signals.
- Inference is only as good as feature parity; drift can reduce signal quality without obvious errors.
- Execution quality depends on exchange liquidity and CCXT behavior; live slippage still needs validation.

**What we have proven vs what remains**
- Proven: end-to-end ingest -> feature store -> inference -> queue -> trading flow in dry-run and replay.
- Proven: manifest gating, calibration, and preflight coverage checks catch low-signal conditions.
- Proven: risk engine, deadlock policies, and audit trails behave deterministically with runbooks.
- Still to prove: sustained live performance after fees/slippage across regime shifts.
- Still to prove: long-horizon stability of gate coverage under real-time exchange noise.
- Still to prove: resilience under production outages (exchange, Redis, scheduler) with live capital.

**Model evidence snapshot (strengths, weaknesses, proof)**
| Model family | Deploy status | Evidence / strengths | Known gaps / risks | Proof artifacts |
| --- | --- | --- | --- | --- |
| Base XGB (`xgb_primary`) | Deployed in contract | Stable tabular baseline; calibrated probabilities; gate coverage proven in replays | Coverage can collapse in new regimes; live slippage still unvalidated | `models/base_xgb_h120_calmon_spread0/report.json`, `models/oos_replay_summary_latest.json` |
| TCN (h60/h120/h180) | Candidate (not default) | Temporal structure; forward replay shows non-zero coverage after retune | Coverage remains thin; needs promotion criteria and live validation | `models/tcn_*_calmon_relaxed/report.json`, `models/tcn_gate_replay_summary.json` |
| Blender (elastic-net) | Candidate (portfolio sweeps) | High coverage when RSS signals present; strong replay metrics | Dependent on exogenous data quality; risk of live data sparsity | `models/blender_h120_v6/report.json`, `datasets/blender_matrix_*_with_preds.parquet` |
| Meta-label | Exploratory | Designed to filter false positives and reduce churn | Probability collapse in small windows; not promoted | `models/*meta*`, `reports/` |
| Portfolio policy (XGB-only) | Promoted | Policy sweeps provide ranked scenarios and constraints | Portfolio behavior still tied to base model coverage | `experiments/perf_sweeps/*/portfolio_final/*.json`, `configs/deployment_portfolio_contract.yaml` |

---

## 13) Monitoring, diagnostics, and runbooks
**Metrics**
- Model coverage, probability sigma, and RSS coverage live as gauges.
- Trading emits attempts, notional, PnL, risk blocks, and deadlock counters.
- Queue depth and scheduler metrics expose throughput and backpressure.

**Dashboards and alerts**
- Grafana dashboards cover ingestion, scheduler, feature store, and trading flows.
- Prometheus alert rules align with deployment contract invariants.

**Readiness and forensics**
- Live readiness checks produce GO/NO-GO bundles.
- Acceptance trade proofs, exit attribution, and stance snapshots capture live posture.
- Dry-run profit forensics align trades with market data for approval trails.

---

## 14) Deployment and configuration contracts
- docker-compose defines ingestion, scheduler, Redis, trading, and monitoring services.
- `.env` and runtime overrides control environment-specific behavior.
- Deployment contracts tie datasets, models, policies, and risk limits together.
- Launch ladder stages generate stage-specific risk and deadlock policies.
- Apply, evaluate, and rollback workflows keep staged promotions deterministic.

---

## 15) Testing and CI
- Adapter, ingestion service, feature store, scheduler, and trading risk tests.
- Regression tests enforce manifest alignment and shortlist criteria.
- Forward replay guardrails enforce minimum coverage and equity thresholds.
- CI installs dependencies and runs targeted suites to catch drift.

---

## 16) Local research and prototyping scripts
- `Main.py` streams OHLCV from CCXT Pro and is a lightweight live feed example.
- `HistoricalDataTest.py` fetches OHLCV, computes TA indicators, runs VectorBT backtests, and includes Optuna sweeps plus order book imbalance streaming.
- These scripts are research-grade utilities and are not part of the production stack.

---

## 17) Where to read next (canonical docs)
- `algo-data-ingestion/README.md` for service setup, APIs, and monitoring.
- `algo-data-ingestion/PROJECT_OVERVIEW.md` for the full technical dossier.
- `algo-data-ingestion/TRAINING_WALKTHROUGH.md` for end-to-end retraining.
- `algo-data-ingestion/RUNBOOK_LIVE_LAUNCH.md` and `algo-data-ingestion/RUNBOOK_DRY_RUN_PROFIT.md` for ops.
- `algo-data-ingestion/docs/` for calibration, deployability, and recovery playbooks.

---

## 18) Debrief (current technical posture)
This project is a full-stack trading platform that treats data integrity, model gating, and runtime risk control as the core contract between research and live execution. The ingestion system standardizes multi-source market intelligence into a shared feature store and Parquet data lake. Training then produces model artifacts and manifests that encode both the features used and the gates that are allowed in production. Those manifests are the bridge between modeling and deployment, and the scheduler plus trading service are designed to enforce them without drift.

Operationally, the system is built to minimize silent failures. Gate coverage, probability variance, and queue health are first-class metrics; when any of these degrade, the deadlock policies, safe mode latching, and readiness checks are meant to stop promotion or pause entries before capital is exposed. The trading service enforces kill and safe mode flags, captures HMAC-signed audit trails, and reconciles intent state with exchange reality, which gives this stack traceability and rollback safety during staged launches.

The most important invariants for new contributors are: feature parity between training and inference, manifest-driven gating, and risk limit alignment across configs. Changes should always be validated with preflight coverage checks and forward replays, then observed under dry-run instrumentation before progressing to the live ladder. If those steps are preserved, this system offers a repeatable path from research signals to controlled, observable trading execution.

**Strengths**
- Strong separation between training artifacts and deployed runtime via manifests and contracts.
- Two-stage gating (model gates + runtime risk) reduces accidental exposure.
- Audited decision trails and reconciliation logic support forensic and rollback workflows.

**Weaknesses**
- Coverage is sensitive to regime changes; guards can halt trading when signals dry up.
- Dependence on external data sources can weaken exogenous features in production.
- True execution quality is only known after sustained live runs.

**Proven in this repo**
- End-to-end dry-run trading with deterministic audits and readiness checks.
- Forward replays and coverage diagnostics that catch gate collapse.
- Risk engine behavior and deadlock policies under staged ladder workflows.

**Left to prove**
- Live execution stability under real slippage, latency, and exchange outages.
- Long-horizon profitability and coverage stability across new market regimes.
- Robustness of exogenous (news/social/on-chain) signals in live conditions.
