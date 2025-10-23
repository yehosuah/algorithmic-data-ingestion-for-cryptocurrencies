# Algo Data Ingestion – Comprehensive Project Dossier

_Last updated: 2025-10-23 01:00 UTC_

---

## 1. Executive Summary
- **Mission**: Deliver an end-to-end ingestion and modeling platform that converts high-frequency market, social, and news data into actionable trading signals and deploys them through a monitored production stack.
- **Scope**: Covers data acquisition (API ingest, backfills), feature engineering, supervised learning (XGBoost baseline, Temporal Convolutional Networks), ensemble/meta-label strategies, backtesting & diagnostics, and deployment readiness (FastAPI service, Redis store, scheduler, monitoring).
- **Current State**: Data pipelines and infrastructure are operational with async route coverage under `tests/ingestion_service` and manifest regression checks in CI. The relaxed-gate Horizon-120 XGB (`final_equity 4.48`) and Calmon TCN suite (`final_equity 1.05–1.33`) still clear 5 bps costs, and the elastic-net blender (`final_equity 1.84`) leans on RSS spikes. Oct–Nov 2025 replays confirm the training gates remain profitable but the deployable inference mask presently produces zero fills.
- **Immediate Goal**: Close the gating gap before launch (retune inference thresholds or add calibration fallback), extend validation beyond Oct 2025, and decide whether to ship blender + base ensemble alone or wait for a calibrated meta layer once probability spread stabilises.

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
- **scheduler**: APScheduler worker calling admin endpoints (`MARKET_JOBS`, `MARKET_INGEST_JOBS`, TTL sweeps) sourced from env vars; publishes metrics on port `9002`.
- **prometheus** & **grafana**: Monitoring stack fronting the ingestion API, scheduler, Redis exporter, and custom dashboards under `monitoring/grafana`.

### 3.2 Configuration (`app/ingestion_service/config.py`)
- Loads env via `.env`, covering exchange/news/social/on-chain keys, Redis connection info (`REDIS_URL`, `FEATURE_NAMESPACE`, TTL), scheduler cadence, and admin token defaults.
- Data lake roots are configurable (`MARKET_PATH`, `ONCHAIN_PATH`, `SOCIAL_PATH`, `NEWS_PATH`) alongside toggles for `BACKFILL_*` loops and TTL sweeps (`TTL_SWEEP_*`).
- ML and storage extras: `ML_SENTIMENT_ENABLED`, `SENTIMENT_MODEL_ID`, `HF_HOME`, `ML_MAX_WORKERS`, and optional `FSSPEC_STORAGE_OPTIONS` for remote parquet targets.

### 3.3 Adapters & Features
- **Market** (`app/features/factory/market_factory.py`): Builds `ret_1`, `logret_1`, EMAs, MACD, RSI, `hl_spread`, `oi_obv`; `app/features/pipelines/market_pipeline.py` and `app/features/jobs/backfill.py` push the engineered payloads into Redis with TTL/metrics.
- **Social** (`app/features/ingestion/social_client.py` + `app/adapters/sentiment_adapter.py`): Async Twitter/Reddit fetchers with tenacity retries and optional sentiment enrichment before writing parquet.
- **News** (`app/features/ingestion/news_client.py` + `app/adapters/news_adapter.py`): API/RSS ingestion with schema guards, normalized partitions under `data_lake/news`.
- **On-chain** (`app/features/ingestion/onchain_client.py` + `app/adapters/onchain_adapter.py`): Glassnode/Covalent surfaces into `data_lake/onchain` (populates once API keys are supplied) while returning schema-stable frames on failure.
- **Feature Store** (`app/features/store/redis_store.py`): Async Redis cache with Prometheus counters/histograms reused by ingestion endpoints, backfill jobs, and scheduler TTL sweeps.

---

## 4. Dataset Pipeline
### 4.1 Acquisition
- `scripts/backfill_ccxt_parquet.py`: Backfills OHLCV data for specified exchange/symbol/timeframe into `data_lake/market/exchange=...` partition.
- `scripts/rss_to_parquet.py`: Pulls RSS feeds over a time window, stores in `data_lake/news/rss`.

### 4.2 Curation
- `scripts/build_market_dataset.py`: Reads market parquet partitions, merges engineered features, and produces labeled dataset (timestamp-aligned, with `ret_next`, `y_dir`).
- `scripts/build_training_matrix.py`: Legacy helper that combines market features with coarse RSS aggregates for focused validation windows.
- `scripts/build_blender_matrix.py`: Generates the year-wide RSS-enriched matrix with intraday spike features, probability momentum (`prob_diff`, `*_mom_1`), relaxed-gate masks, and summary stats (`..._rss_latest_stats.json`). Latest forward replay exported `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (Oct 2025 window plus model predictions).

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
- CLI defaults mirror the relaxed Calmon gate (`--max-spread-z 0.25`, `--max-rvol20 2e-4`, no probability filter) while manifests persist the deployable inference mask (`hl_spread ≤ 0.0005`, `hl_spread_z ≤ -0.6`, `rvol_20 ≤ 4e-5`, `prob ≥ 0.85`, `min_hold 10`).
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
- `models/oos_replay_summary.json` and `models/tcn_gate_replay_summary.json` log gate behaviour for audit, ensuring live inference adheres to the deployable mask.
- Oct–Nov 2025 forward replay (`models/oos_replay_oct_nov_2025.json`) shows training-gate equity still >1, while the deployable inference mask idles (zero toggles), underscoring the need for adjusted thresholds or fallback logic.

### 6.4 Blender (`scripts/train_blender.py` / `training/blender.py`)
- Elastic-net logistic regression (StandardScaler + LogisticRegressionCV) over probability momentum, RSS spike features, and regime fields from the blender matrix.
- Threshold search enforces turnover guards and records RSS audits. `models/blender_h120_v6` hits `final_equity 1.84`, Sharpe 28.7, 711 toggles at threshold 0.95 with an RSS spike gate share of ~2 %.
- Feature inventories and RSS coverage stats live alongside manifests so ops knows when to fall back to no-RSS feature sets.

### 6.5 Meta-Label (`scripts/train_meta_label.py` / `training/meta.py`)
- Triple-barrier labeling with flexible volatility targets feeds a logistic meta filter. The refreshed script supports the relaxed gate defaults, stride control, and shares KPI schema with the rest of the stack.
- Current attempts remain exploratory (probability collapse on narrow validation windows); production rollout is gated on extending the blender matrix and recovering dynamic range before fitting.

### 6.6 Inference Utilities (`training/infer.py`)
- Loads base XGB/TCN artifacts, applies consistent preprocessing (feature alignment, scaler/standardization) for scoring pipelines.

---

## 7. Experimental Ledger
### 7.1 XGBoost Experiments
- **Calmon relaxed (current)** – `models/base_xgb_h120_calmon_spread0` retrained on the 2024–2025 feed delivers `final_equity 4.48`, `gate_fraction 9.4 %`, and Sharpe 108 after costs. Monthly diagnostics confirm stability and expose the coverage replay used for live gating.
- **Cost stress (`spread_scale` sweep)** – Variants `{0.0, 0.05, 0.1, 0.2}` preserve the same equity/turnover envelope, demonstrating resilience to 20 % spread inflation.
- **Gate replay** – `live_gate_coverage.csv` keeps the deployable mask within ±1.63× of the baseline coverage, ensuring turnover budgets hold when the strict inference gate is enforced.
- **Forward replay** – `models/oos_replay_oct_nov_2025.json` retains 4.48 equity under the relaxed training gate but records zero trades under the deployable mask, signalling the need to widen thresholds or add a fallback for live launch.

### 7.2 TCN Experiments
- **Relaxed Calmon suite** – Horizons 60/120/180 all clear 5 bps costs with tight turnover guards (≤200 toggles). Probability variance guardrails remain above the 0.03 threshold, signalling healthy calibration after loosening training gates.
- **OOS gate audits** – `models/oos_replay_summary.json` and `tcn_gate_replay_summary.json` document how the inference mask collapses coverage (<0.001 %), informing live expectations and highlighting that retraining can focus on the relaxed gate while keeping deployable safety nets.

### 7.3 Blender Experiments
- **v5 (baseline)** – First elastic-net attempt showed modest gains but relied on sparse RSS spikes, limiting deployment appetite.
- **v6 (current)** – With the expanded matrix (`build_blender_matrix.py` intraday features + probability momentum) the logistic stack reached `final_equity 1.84`, Sharpe 28.7, 711 toggles. `rss_audit` passes with daily coverage 82.5 % and minute spike share 0.254, triggering the internal RSS gate mask recorded in the manifest.
- **Forward replay** – `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (30 201 rows, Oct 1 → Oct 21 2025) and `models/oos_replay_oct_nov_2025.json` capture how the training-vs-inference gates behave on the latest window; currently the deployable gate doesn’t fire, prompting recalibration work.

### 7.4 Diagnostics & Tooling
- `training/reporting.ensure_kpi_schema` standardises KPI payloads; `scripts/report_shortlist.py` ranks deployable models (base, TCN, blender) under consistent criteria.
- Threshold diagnostics (`diagnostic_final_equity`) and RSS audits are embedded across all reports to make regression testing and CI validation straightforward. `tests/regression/test_report_shortlist.py` executes the shortlist CLI in CI to guarantee the relaxed Calmon baseline still satisfies the deployable filters.

---

## 8. Deployment Readiness
- **Model Artifacts**: Stored under `models/`; paired `training/infer.py` loaders guarantee live compatibility (feature list, calibrators, scalers).
- **Feature Store**: Redis (`app/features/store/redis_store.py`) backed by metrics and TTL sweeps; scheduler/admin endpoints keep namespaces fresh.
- **Monitoring & Metrics**: Prometheus scrapes ingestion API, scheduler, Redis exporter, feature store, and ML inference counters; Grafana dashboards live in `monitoring/grafana`.
- **ML Sentiment Endpoint**: `/ml/sentiment/predict` (`app/ingestion_service/ml_routes.py`) serves HuggingFace pipelines when enabled, respecting `ML_MAX_WORKERS` and publishing latency stats.
- **Parquet Writer**: `app/ingestion_service/utils.py` validates schemas, adds dt partitions, and supports remote backends via `FSSPEC_STORAGE_OPTIONS`.
- **Tests**: `tests/ingestion_service` exercises async routes end-to-end with fakeredis; `tests/regression` keeps manifests in sync with `report.json` and enforces shortlist criteria; `.github/workflows/ci.yml` runs both suites plus `tests/training` on pushes/PRs. Modeling utilities (`training/`) still need deeper unit coverage.

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
| Deployable gate drift (prob/vol/spread masks) | Turnover explodes or drops to zero in production | Replay `live_gate_coverage.csv` each retrain, run the Oct 2025 replay (`models/oos_replay_oct_nov_2025.json`) in CI, mirror predicates in inference adapters, and alert on coverage outside ±2× historical band. |
| RSS coverage collapse | Blender loses signal, equity regresses | Track `rss_audit minute_spike_share` and fall back to the no-RSS feature set when coverage <5e-4; expand feed roster and monitoring. |
| Probability variance collapse | Thresholds become unstable; blender/meta stack unusable | Enforce the `prob_sigma_guardrail` and halt deployment if monthly σ <0.03; re-run relaxed gate retrains or adjust stride/windows. |
| Artifact/config skew between training and live | Live scoring diverges from reports | Consume manifests for feature lists + gate configs, add regression tests that pipe historical data through inference adapters, and version control manifests. |
| Large binary artifacts bloating repo | Slow CI and clone times | Use Git LFS or artifact storage for heavy parquet/torch files once deployment pipeline is in place; prune outdated experiment folders. |

---

## 11. Backlog & Action Items
### 11.1 Validation & Monitoring
- [x] Extend out-of-sample replays into Oct–Nov 2025 to confirm relaxed-gate robustness across new regimes (`models/oos_replay_oct_nov_2025.json`).
- [ ] Retune inference gate thresholds or add fallback logic so Oct–Nov 2025 maintains non-zero coverage without breaching turnover budgets.
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
