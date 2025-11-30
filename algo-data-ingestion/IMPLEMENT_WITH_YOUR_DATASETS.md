# Walkthrough: Implement with Your Datasets

_Last updated: 2025-11-30 18:55 UTC_

> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-17: Added the time-series CV + random-search lane (`training/run_hparam_search.py`, `configs/cv_config.yaml`, `configs/hparam_spaces.yaml`) plus the promoted configs (`configs/best_model_configs.{yaml,json}`) so custom datasets can reuse/extend the shared sweeps. The sanitizer/parity workflow remains as before.

This plan mirrors the refreshed Calmon stack. Adapt the paths/parameters to your own instruments once you have equivalent market + RSS coverage.

## Reference Artifacts
- Market history: `datasets/market_btcusdt_1m_2024_2025.parquet` (2024-01-01 ➜ 2025-10-27, 959 039 bars).
- RSS-enriched blender matrix: `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet` (606 121 rows, mirrored as `..._2025-10_rss_latest.parquet`).
- Baseline models: `models/base_xgb_h120_calmon_spread0`, `models/tcn_h120_calmon_relaxed`
- Forward replay snapshot: `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` + `models/oos_replay_summary_latest.json` (Oct 1–Oct 28 deployable vs relaxed gate comparison; `...oct_nov_2025.json` kept for regression).

## Step 0 – Sanitize Multi-Symbol Feed & Gate Config
- Load your raw parquet (single or multi-symbol) via `training.data.load_parquet_dataset(..., drop_duplicates=False)` and run it through `training.data.sanitize_market_dataset` to drop duplicate (timestamp, symbol) rows, clamp log-return outliers, and seed per-symbol rolling volatility columns. Persist the cleaned file as `datasets/<name>.parquet`.
- Generate symbol-aware caps that scheduler/trading will share with training/inference:
  ```bash
  python scripts/compute_symbol_gate_config.py \
    --data datasets/<name>.parquet \
    --out release/symbol_gates/<name>.json
  ```
  The CLI writes `training` vs `inference` keys with `hl_spread`, `rvol`, spread/vol ratios, and liquidity ranks; keep the JSON in git so any retrain with the same dataset stem auto-loads it.

## Step 1 – Feature Engineering
1. **Market dataset**: reuse `scripts/build_market_dataset.py` to generate your symbol’s feature parquet (ensures consistent `ret_next` and `y_dir` labels).
2. **RSS aggregation**: ingest feeds via `scripts/rss_to_parquet.py` (or your own collectors). The blender builder depends on daily coverage ≥80 % and minute spike share ≥5e-4.
3. **Blender matrix**: run `scripts/build_blender_matrix.py`, pointing `--base-dir` / `--tcn-dir` at the calibrated models to backfill probabilities and engineered RSS features. Capture forward audit windows (e.g., Oct 2025) into a dedicated matrix with predictions so you can compare training vs inference gates later.
   - The stride you select (or that the builder infers) doubles as the blender gate smoothing window and will be persisted as `gate_smoothing_stride` in the resulting report.

## Step 2 – Base Learner
- Train with relaxed gate defaults:
  ```bash
  python scripts/train_base_gbdt.py \
    --data <your_market_dataset.parquet> \
    --out models/base_xgb_h120_calmon_spread0_yoursymbol \
    --fold-scheme calendar_month --n-folds 6 \
    --cost-bps 5 --max-spread-z 0.25 --max-rvol20 2e-4 \
    --symbol-gate-config <release/symbol_gates/your_dataset.json>
  ```
- Validate the RSS audit and monthly diagnostics in `report.json`, then review deployable gates within the generated manifest.
- When a matching gate file lives under `release/symbol_gates/` (same filename stem as your dataset), the CLI auto-loads it; passing the flag keeps deployable training/inference caps aligned when you deviate from the default dataset name.

### Step 2b – Time-Series CV + Hyperparameter Search (optional)
- Use the shared CV schema (`configs/cv_config.yaml`, expanding window; 15D validation, 1D gap) and search spaces (`configs/hparam_spaces.yaml`) to sweep your dataset:
  ```bash
  python -m training.run_hparam_search \
    --model tcn \
    --contract configs/canonical_training_contract_market_multi_3symbol_1m.yaml \
    --cv-config configs/cv_config.yaml \
    --hparam-space configs/hparam_spaces.yaml \
    --n-trials 24 \
    --output-dir experiments/hparam_search/tcn_<yoursuffix> \
    --horizon 2 --seq-stride 10 --max-rows 800000 \
    --cost-bps 5 --min-hold-bars 1
  ```
  Sequence models accept `seq_stride` to thin windows; TCN/Transformer early-stop on Sharpe computed with `cost_bps`/`min_hold_bars` when `val_returns` are supplied.
- Promote best configs into a reusable YAML/JSON bundle with:
  ```bash
  python -m training.promote_best_configs \
    --search-root experiments/hparam_search \
    --min-sharpe 0.0 --top-n 1 \
    --output configs/best_model_configs.yaml
  ```
  The repo ships with the latest promoted configs (`xgb_trial_010`, `tcn_trial_011`, `transformer_trial_023`) if you want a starting point.

## Step 3 – Temporal Model
- Clone the TCN run with horizon tuned to your strategy (120 bars by default). Adjust `--window`, `--channels`, and `--stride` to match volatility profile while keeping turnover ≤200.
- Store the manifest + fold logits to allow recalibration without retraining from scratch.

## Step 4 – Blender / Ensemble
- Feed the RSS-enriched matrix into `scripts/train_blender.py` with an elastic-net sweep. Confirm:
  - RSS audit `passed = true`
  - Threshold respects turnover guardrails
  - `report.json` lists meaningful feature weights (probability momentum, RSS spikes, regime features)
- Adjust `--class-weight` (`balanced` vs `none`) to match your label imbalance; manifests exported from the recipe gate inference at `prob ≥ 0.5`, `rvol_20 ≤ 5e-4`, `min_hold 10` to retain coverage on forward windows.
- Inspect the reported `gate_smoothing_stride` (defaults to your stride) and keep sandbox stride‑1 runs handy to understand turnover ceilings before loosening gates.

## Step 5 – Meta Filter (optional)
- Once the base + TCN + blender produce non-degenerate probabilities on your data, experiment with `scripts/train_meta_label.py`.
- Adjust the triple-barrier params to your instrument’s volatility; require ≥1.2 final equity and ≥20 toggles before promoting the meta gate.

## Step 6 – Deployment Readiness
- Use `scripts/report_shortlist.py` to summarise candidates.
- Replay live coverage with the manifest gates (`live_gate_coverage.csv` pattern) and integrate `training/infer.py::score_base_with_manifest` into your inference stack so Prometheus gauges (`model_gate_coverage_ratio`, `model_rss_minute_spike_share`, `model_probability_sigma`) surface in monitoring.
- Monitor RSS coverage and probability variance to trigger fallbacks if data quality dips below thresholds captured in the reports.
- Reproduce the guardrail locally with `scripts/run_oos_eval.py --family {base_xgb,tcn,blender}` to generate forward replay diagnostics; enforce `gate_fraction ≥ 5e-4` and `final_equity ≥ 1.2` for deployable horizons before promoting manifests.
- Run regression guardrails before deployment: `pytest tests/regression -q` keeps manifests aligned with reports and verifies the shortlist; `pytest tests/ingestion_service -q` exercises the async API.
- Inspect your forward replay equivalent of `models/oos_replay_summary_latest.json`; aim for at least the current baseline (base: 12 gate hits, `final_equity 1.2336`, gate coverage 2.99e-4; TCN h60/h120/h180: gate coverage 4.73e-4/7.71e-4/4.23e-4 with toggles 4/62/2; blender: ≈15.8 % coverage with 6 346 toggles). If deployable masks fall back to zero, widen thresholds or stage a fallback gate prior to launch (keep an archived zero-coverage snapshot like `...oct_nov_2025.json` for regression).
- Ensure your inference path uses the updated stride-aware batching in `training/infer.predict_tcn` so experiments with smaller strides do not overload memory.
- Before modifying gate thresholds, export a scheduler-style slice and compare it with the sanitized training parquet:
  ```bash
  python scripts/export_feature_slice.py --output /tmp/features_debug.parquet
  python scripts/compare_feature_stats.py \
    --train datasets/<name>.parquet \
    --live /tmp/features_debug.parquet \
    --out release/calibration/latest/<name>_parity.json
  ```
  Attach the JSON to your rollout ticket so reviewers can see live vs training drift across `hl_spread`, `hl_spread_z`, `rvol_20`, and `base_prob`.

## Step 7 – Scheduler & Trading Dry Run
- Populate `INFER_JOBS` with your symbols/timeframes and manifest names (base/TCN optional). Pair each job with a `lookback_minutes` window large enough to reconstruct features plus a `history_minutes` margin for warm-up.
- Update `TRADING_MODELS` to reflect your manifolds, sizing strategy (`order_amount` or `order_notional`), and optional hold overrides. Paths are resolved relative to `MODELS_ROOT` (defaults to the repo `models/` mount).
- Spin up the Docker stack and monitor:
  ```bash
  curl -s http://localhost:9002/metrics | grep scheduler_decision
  curl -s http://localhost:9010/metrics | egrep 'trading_trade_attempts_total|trading_position_active'
  python scripts/verify_trading_redis.py
  ```
  Verify the Redis queue drains quickly, trading metrics increment, and audit/state logs reflect your symbol set.
- Decide on the persistence backend:
  - File (`TRADING_STATE_BACKEND=file`) keeps `data_lake/trading/state.json` updated on disk (default).
  - Redis (`TRADING_STATE_BACKEND=redis`) mirrors state to `TRADING_STATE_REDIS_HASH`; useful when multiple workers share a queue.
  - Postgres (`TRADING_STATE_BACKEND=postgres`) materialises `trading_positions` / `trading_audit_events` tables with automatic schema creation; supply DSNs via env.
- Keep Grafana dashboards (`scheduler-overview`, `trading-overview`) open during the dry run and capture deviations in your deployment notes. Flip `TRADING_DRY_RUN=0` only after rehearsing the checklist with ops/compliance.
