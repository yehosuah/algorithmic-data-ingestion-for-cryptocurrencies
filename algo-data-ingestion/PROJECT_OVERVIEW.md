# Algo Data Ingestion – Comprehensive Project Dossier

_Last updated: 2025-09-29 02:05 UTC_

---

## 1. Executive Summary
- **Mission**: Deliver an end-to-end ingestion and modeling platform that converts high-frequency market, social, and news data into actionable trading signals and deploys them through a monitored production stack.
- **Scope**: Covers data acquisition (API ingest, backfills), feature engineering, supervised learning (XGBoost baseline, Temporal Convolutional Networks), ensemble/meta-label strategies, backtesting & diagnostics, and deployment readiness (FastAPI service, Redis store, scheduler, monitoring).
- **Current State**: Data pipelines and infrastructure are operational. Core models are trained but fail to clear realistic cost thresholds. Focus is on improving per-model profitability before stacking and meta-labeling.
- **Immediate Goal**: Achieve post-cost positive equity (>1.0) for both the XGB baseline and TCN models on out-of-fold evaluations with 5 bps transaction costs. Once achieved, proceed to blender and meta-label training.

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
- `scripts/build_training_matrix.py`: Combines market dataset with aggregated RSS/Reddit features, creating matrices for blender/meta-label training.

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
- Configurable via CLI for hyperparameters, costs, spread adjustments, long-only toggles.
- Implements isotonic calibration (`sklearn.calibration.CalibratedClassifierCV`).
- Persists artifacts: booster JSON, calibrator joblib, feature list, threshold, evaluation report.
- Current default parameters (post-tuning) around depth 6, 1200 estimators, `scale_pos_weight` auto-calculated.
- Logging improvements include auto class-weight detection and parameter override echoing.
- CLI exposes calendar-based folds (`--fold-scheme`), cost knobs (`--spread-scale`, `--slippage-bps`), and optional `--long-only` gating for post-training evaluation.

### 6.3 Temporal Convolutional Network (`scripts/train_tcn.py` / `training/tcn_model.py`)
- Custom TinyTCN with residual TemporalBlock layers, dropout, adaptive pooling, and fully connected head.
- Training config (epochs, lr, batch size, weight decay, class weighting) + optional progress callbacks for epoch logging.
- Sliding window generation uses stride to reduce sample count for long histories.
- Outputs: PyTorch state dict (`tcn.pt`), preprocessing bundle (`tcn_preproc.joblib`), calibrator (`tcn_calibrator.joblib`), metadata, threshold, report.
- CLI parameters include `--stride`, `--series-cols`, `--fold-scheme`, and class weighting to tune sample coverage vs compute.

### 6.4 Blender (`scripts/train_blender.py` / `training/blender.py`)
- Logistic regression pipeline on features `[base_prob, tcn_prob, rss_count, rss_sent_mean, reddit_count, reddit_sent_mean, rvol_*]`.
- Currently on hold until base models exceed cost thresholds.

### 6.5 Meta-Label (`scripts/train_meta_label.py` / `training/meta.py`)
- Triple-barrier event labeling (`pt_mult`, `sl_mult`, `max_hold`).
- Logistic meta model filters trades, evaluating threshold grids to mask predictions below confidence.
- Artifacts: `meta_model.joblib`, feature list, threshold, report.

### 6.6 Inference Utilities (`training/infer.py`)
- Loads base XGB/TCN artifacts, applies consistent preprocessing (feature alignment, scaler/standardization) for scoring pipelines.

---

## 7. Experimental Ledger
### 7.1 XGBoost Experiments
- **Baseline (No Augment)**: Negative equity under costs, moderate OOF AUC (~0.53–0.54).
- **Augmented Features (`models/base_xgb_tuned_features_*`)**:
  - Zero-cost equity >1.15 (profitable), but 5 bps cost reduces to ~0.95.
- **Depth 4, Regularized (`models/base_xgb_depth4_cost`)**:
  - Slight improvement in threshold (0.905) and drawdown; still sub-1 equity.
- **Spread-aware penalties**:
  - Using `hl_spread_z` with large scaling (0.5) resulted in catastrophic losses (final equity ≈ 0) because z-score magnitudes convert to enormous effective costs.

### 7.2 TCN Experiments
- **Baseline (stride 1)**: Training too slow, epoch loss stuck near 0.693, calibration failed due to NaNs.
- **Improved pipeline**: Added feature augmentation, stride parameter, and class weighting.
- **Stride 5, 48-length windows (`models/tcn_tuned*`)**: Achieved positive zero-cost equity (1.02) but negative after costs. OOF count reduced to ~153k windows.

### 7.3 Diagnostics
- Probability threshold grid script (Step 3) shows equity climbs gradually and plateaus near 0.9 thresholds, confirming the need to lower costs or improve signal quality.
- Spread statistics: `hl_spread` averages ≈0.00068; `hl_spread_z` averages ≈0.79 (95% ~1.99). Cost scaling must respect these magnitudes.

---

## 8. Deployment Readiness
- **Model Artifacts**: Stored under `models/`; paired `training/infer.py` loaders guarantee live compatibility (feature list, calibrators, scalers).
- **Feature Store**: Redis (`app/features/store/redis_store.py`) backed by metrics and TTL sweeps; scheduler/admin endpoints keep namespaces fresh.
- **Monitoring & Metrics**: Prometheus scrapes ingestion API, scheduler, Redis exporter, feature store, and ML inference counters; Grafana dashboards live in `monitoring/grafana`.
- **ML Sentiment Endpoint**: `/ml/sentiment/predict` (`app/ingestion_service/ml_routes.py`) serves HuggingFace pipelines when enabled, respecting `ML_MAX_WORKERS` and publishing latency stats.
- **Parquet Writer**: `app/ingestion_service/utils.py` validates schemas, adds dt partitions, and supports remote backends via `FSSPEC_STORAGE_OPTIONS`.
- **Tests**: `pytest` modules cover adapters, time normalization, feature store logic; modeling utilities (`training/`) still need broader coverage.

---

## 9. Dependencies & Environment
- Python 3.12 via local `.venv` or Docker; packaging metadata lives in `setup.py` / `algo_data_ingestion.egg-info`.
- `requirements.txt` (extra index for CPU PyTorch) is the source of truth for both services and training.
- Async/infra stack: `fastapi`, `uvicorn[standard]`, `httpx`, `tenacity`, `redis>=5`, `pydantic-settings`, `APScheduler`, `feedparser`, `tweepy`, `prometheus-client`.
- ML stack: `pandas`, `numpy`, `scikit-learn`, `xgboost 3.x`, `joblib`, optional `torch` + `transformers` for sentiment, plus `numba` for feature speed-ups.
- Storage tools: `fsspec` with optional `s3fs`/`gcsfs` and HuggingFace cache control via `HF_HOME`.
- Recent change: Reinstalled `numpy==1.26.4` to resolve typing/import errors during TCN runs.

---

## 10. Risk & Mitigation
| Risk | Impact | Mitigation |
|------|--------|------------|
| Models fail to beat transaction costs | Strategy unusable in production | Enhance features, adjust cost modeling, explore alternate targets (classification/regression), and incorporate gating/meta filters. |
| Cost modeling mis-specified (e.g., spread scaling) | Wildly inaccurate performance estimates | Use realistic spread metrics (raw ratios), cap penalties, perform scenario analysis. |
| Overfitting due to calibration & narrow threshold bands | False sense of profitability | Expand threshold grid, evaluate per-fold metrics, add validation months, run out-of-sample tests. |
| Lack of extensive testing | Regression risk before deployment | Expand unit/integration tests around new feature engineering and modeling code. |
| Live data drift | Model underperformance post-deployment | Incorporate monitoring dashboards, drift detectors, and periodic retraining workflows. |

---

## 11. Backlog & Action Items
### 11.1 Modeling
- [ ] XGB: Continue hyperparameter sweeps, tune `scale_pos_weight`, try alternative objectives (`binary:logitraw` + calibration).
- [ ] XGB: Implement trade gating based on `hl_spread_z`/vol regime before thresholding.
- [ ] TCN: Increase channels/windows, experiment with dilation patterns, incorporate dropout scheduling or batch norm.
- [ ] TCN: Evaluate no per-window standardization or hybrid normalization to preserve slow trends.
- [ ] TCN: Implement simple early stopping (monitor validation loss) to save compute.

### 11.2 Ensemble & Meta
- [ ] Prepare blender dataset once base models profitable; rerun `scripts/train_blender.py`.
- [ ] Revisit meta-label training with updated base probabilities.
- [ ] Evaluate ensemble/multi-model weighting strategies beyond logistic blending (e.g., gradient boosting on meta features).

### 11.3 Infrastructure & Ops
- [ ] Add CI checks for training scripts (linting, smoke tests).
- [ ] Extend Prometheus/Grafana dashboards for model metrics (PnL, Sharpe, turnover, coverage).
- [ ] Document inference pipeline for live deployment (feature extraction, model loading, threshold application).
- [ ] Build reproducible training workflow (Makefile or `invoke` tasks) to ensure consistent runs.

### 11.4 Documentation & Testing
- [ ] Expand unit tests for `training/feature_eng.py`, `training/thresholds.py`, and TCN utilities.
- [ ] Update `TRAINING_WALKTHROUGH.md` with new commands (stride option, augmented features) and pitfalls (spread scaling).
- [ ] Maintain `TRAINING_STATUS.md` after each major experiment to track deltas and avoid rework.

---

## 12. Reference Commands
- **Base GBDT (current best zero-cost)**
  ```bash
  source .venv/bin/activate
  python scripts/train_base_gbdt.py \
    --data datasets/market_btcusdt_1m_2024_2025.parquet \
    --out models/base_xgb_tuned_features_nocost \
    --cost-bps 0 --slippage-bps 0 --spread-scale 0 \
    --xgb-n-estimators 1200 --xgb-learning-rate 0.03 \
    --xgb-max-depth 6 --xgb-min-child-weight 5 \
    --xgb-subsample 0.9 --xgb-colsample-bytree 0.9 \
    --xgb-gamma 0.1 --xgb-reg-lambda 2.0 --xgb-reg-alpha 0.1 \
    --auto-scale-pos-weight
  ```
- **TCN (stride 5)**
  ```bash
  source .venv/bin/activate
  python scripts/train_tcn.py \
    --data datasets/market_btcusdt_1m_2024_2025.parquet \
    --out models/tcn_tuned \
    --window 48 --stride 5 --channels 48,48 \
    --epochs 12 --batch-size 512 --lr 7e-4 \
    --dropout 0.0 --weight-decay 1e-5 --class-weight 1.1 \
    --n-folds 6 --embargo-minutes 60 \
    --series-cols ret_1,logret_1,rvol_5,rvol_20,macd,macd_signal_9,rsi_14,hl_spread,oi_obv,ret_mean_5,ret_mean_20,ret_std_20,ret_z_20,macd_hist,macd_hist_abs,rvol_ratio,rvol_delta,rvol_z_50,rsi_centered,hl_spread_z,obv_diff,obv_z_50 \
    --cost-bps 5 --spread-scale 0 --fold-scheme even
  ```
- **Threshold Diagnostics**
  ```bash
  source .venv/bin/activate
  python - <<'PY'
  # (See Section 7.3 for full script.)
  PY
  ```
- **Blender (post-cost positive base & TCN)**
  ```bash
  source .venv/bin/activate
  python scripts/train_blender.py \
    --data datasets/training_matrix_months_2025-08-09.parquet \
    --base-dir models/base_xgb_tuned_features_cost \
    --tcn-dir models/tcn_tuned \
    --out models/blender_tuned \
    --cost-bps 5 --spread-scale 0 --slippage-bps 0
  ```
- **Meta-label Logistic Filter**
  ```bash
  source .venv/bin/activate
  python scripts/train_meta_label.py \
    --data datasets/training_matrix_months_2025-08-09.parquet \
    --base-dir models/base_xgb_tuned_features_cost \
    --tcn-dir models/tcn_tuned \
    --out models/meta \
    --pt-mult 2.0 --sl-mult 2.0 --max-hold 60 \
    --primary-prob-column base_prob --cost-bps 5
  ```

---

## 13. Supporting Documentation
- `README.md`: Core setup, docker usage, monitoring stack overview.
- `TRAINING_WALKTHROUGH.md`: Step-by-step base + blender training instructions.
- `IMPLEMENT_WITH_YOUR_DATASETS.md`: How to tailor pipeline for new datasets.
- `SANITY_CHECKS.md`: Backfill + dataset validation flow.
- `TRAINING_STATUS.md`: Rolling summary of recent experiments and next steps.
- `notebooks/starter_training.ipynb`: Exploratory notebook for feature augmentation and walk-forward inspection.

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
