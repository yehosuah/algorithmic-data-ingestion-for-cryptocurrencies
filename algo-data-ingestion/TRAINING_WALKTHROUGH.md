# Training Walkthrough (Base · TCN · Blender)

_Last updated: 2025-11-17 14:38 UTC_

> Update 2025-11-17: Layered in the time-series CV + random-search lane (`training/time_series_cv.py`, `training/run_hparam_search.py`) with shared search/cv configs (`configs/hparam_spaces.yaml`, `configs/cv_config.yaml`), promoted the current winners (`configs/best_model_configs.{yaml,json}`), and added stride-aware sequence builders plus P&L-aware early stopping for TCN/Transformer to keep stride-1 sweeps memory-safe while monitoring deployable Sharpe.

This guide walks through the refreshed modeling stack: relaxed-gate retrains for the horizon-120 XGBoost baseline, the Calmon TCN suite, and the elastic-net blender that now clears 5 bps costs with RSS enrichment.

## Prerequisites
- Python 3.12 virtualenv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `python -m pip install -r requirements.txt`
- Ensure the year-wide market parquet and the latest RSS matrix are present:
  - `datasets/market_btcusdt_1m_2024_2025.parquet` (2024-01-01 ➜ 2025-10-27, 959 039 bars).
  - `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet` (alias `..._2025-10_rss_latest.parquet`, 606 121 rows covering 2024-09-01 ➜ 2025-10-26).
  - Optional forward replay matrix for gate audits: `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (40 201 rows, 2025-10-01 ➜ 2025-10-28 22:00).
- Multi-symbol relaxed gate bundle (BTC/ETH/SOL) if you plan to retrain on the combined feed:
  - `datasets/market_multi_3symbol_1m.parquet` (≈1.19 M rows spanning 2024-01-01 ➜ 2025-11-04).
  - `release/symbol_gates/market_multi_3symbol_1m.json` stores per-symbol spread/vol caps; `scripts/train_base_gbdt.py` now auto-loads a gate config whose filename matches the dataset stem (e.g., `market_multi_3symbol_1m.parquet` → `release/symbol_gates/market_multi_3symbol_1m.json`).

## 0b. Time-Series CV + Hyperparameter Search
- Default splits live in `configs/cv_config.yaml` (expanding window, 15D validation, 1D gap). Run random-search sweeps with the shared search spaces in `configs/hparam_spaces.yaml`:
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
  Sequence builders now accept `seq_stride` so stride‑1 sweeps stay memory-safe; TCN/Transformer loops monitor Sharpe computed with `cost_bps`/`min_hold_bars` when `val_returns` are supplied, improving deployability alignment vs. loss-only early stopping.
- Each sweep writes per-trial JSON + `results.csv`; collect leaders into configs with:
  ```bash
  python -m training.promote_best_configs \
    --search-root experiments/hparam_search \
    --min-sharpe 0.0 --top-n 1 \
    --output configs/best_model_configs.yaml
  ```
  The current promoted configs are already checked in (`xgb_trial_010`, `tcn_trial_011`, `transformer_trial_023`), mirrored in YAML/JSON for scripting.

## 0. Prepare Multi-Symbol Feed & Gates
- Sanitize fresh parquet pulls before training so duplicate (timestamp, symbol) rows and price outliers do not blow up `hl_spread`/`rvol`:
  ```bash
  python3 - <<'PY'
  from pathlib import Path
  from training.data import load_parquet_dataset, sanitize_market_dataset

  raw = load_parquet_dataset("raw/market_multi_3symbol_1m.parquet", drop_duplicates=False)
  clean = sanitize_market_dataset(raw, verbose=True)
  Path("datasets").mkdir(exist_ok=True)
  clean.to_parquet("datasets/market_multi_3symbol_1m.parquet", index=False)
  PY
  ```
- Generate symbol-aware caps straight from the sanitized parquet; manifests, scheduler jobs, and `TRADING_MODELS` all consume the JSON payload:
  ```bash
  python scripts/compute_symbol_gate_config.py \
    --data datasets/market_multi_3symbol_1m.parquet \
    --out release/symbol_gates/market_multi_3symbol_1m.json
  ```
- Keep the JSON alongside the dataset—`scripts/train_base_gbdt.py` auto-loads a config whose filename matches the dataset stem (override with `--symbol-gate-config` when needed) so BTC/ETH/SOL inherit consistent `hl_spread`, `rvol`, and liquidity ranks through training, inference, and trading.

## 1. Generate/Refresh Feature Matrices
The relaxed gate relies on augmented features and RSS spikes engineered by the new builder.
```bash
python scripts/build_blender_matrix.py \
  --source datasets/market_btcusdt_1m_2024_2025.parquet \
  --out datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet \
  --base-dir models/base_xgb_h120_calmon_spread0 \
  --tcn-dir models/tcn_h120_calmon_relaxed \
  --tcn-stride 30 --include-reddit --timeframe 1min
```
- Emits a JSON summary (`..._stats.json`) with window start/end and probability means for quick sanity checks.
- Persists minute/day RSS features, probability momentum (`prob_diff`, `*_mom_1`), and gating fields aligned to the relaxed training mask.
- The matrix cadence (set by `--tcn-stride` or inferred from timestamps) now controls the blender gate smoothing window, so choose the stride you expect to honour at inference.
- For forward audits we snapshot `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (Oct 1–Oct 20 2025, probabilities included); regenerate by rerunning the builder with the desired window and output path.

## 2. Train Horizon-120 XGBoost (Relaxed Gate)
```bash
python scripts/train_base_gbdt.py \
  --data datasets/market_btcusdt_1m_2024_2025.parquet \
  --out models/base_xgb_h120_calmon_spread0 \
  --fold-scheme calendar_month --n-folds 6 --embargo-minutes 60 \
  --cost-bps 5 --max-spread-z 0.25 --max-rvol20 2e-4 \
  --threshold-criterion final_equity --diagnostic-thresholds 0.5,0.55,0.6,0.65 \
  --calmon-gate
```
- Swap `--data` for `datasets/market_multi_3symbol_1m.parquet` when retraining across BTC/ETH/SOL; the CLI automatically applies `release/symbol_gates/market_multi_3symbol_1m.json`, but you can pass it explicitly via `--symbol-gate-config` to keep the manifest caps identical to scheduler inference.
- Artifacts: booster (`model.json`), calibrator, feature list, threshold, manifest (`gates.training` vs `gates.inference`), `report.json`; the deployable mask now keeps `prob ≥ 0.2`, `min_hold 10`, and `long_only` while leaving spread/rvol enforcement to the trading layer.
- `report.json` captures monthly diagnostics, RSS audits, and spread stress-test metadata.

## 3. Train Calmon TCN Variants
```bash
python scripts/train_tcn.py \
  --data datasets/market_btcusdt_1m_2024_2025.parquet \
  --out models/tcn_h120_calmon_relaxed \
  --window 192 --stride 30 --channels 64,64 \
  --epochs 10 --batch-size 256 --lr 5e-4 --dropout 0.1 \
  --class-weight 2.0 --n-folds 4 --embargo-minutes 60 \
  --cost-bps 5 --max-spread-z 0.25 --max-rvol20 2e-4 \
  --min-total-turnover 4 --max-total-turnover 200 \
  --horizon 120 --diagnostic-thresholds 0.55,0.6,0.65,0.7
```
- Produces `tcn.pt`, scaler/preprocess bundle, calibrator, manifest, threshold, `fold_logits.parquet`, and `report.json` with the probability variance guardrail.
- The refreshed runs land `final_equity 1.28/3.62/1.85` (h60/h120/h180) with inference gates now pared back to `prob ≥ 0.25` (min-hold 10, long-only); spread and volatility guards remain in manifests for training diagnostics but live enforcement happens in the trading service.
- Repeat with `--horizon 60` / `--horizon 180` to populate the alternate manifests (`models/tcn_h60_calmon_relaxed`, `models/tcn_h180_calmon_relaxed`), and feed the artifacts to the OOS runner described below for guardrail checks.

## 4. Train Elastic-Net Blender
```bash
python scripts/train_blender.py \
  --matrix datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet \
  --base-dir models/base_xgb_h120_calmon_spread0 \
  --tcn-dir models/tcn_h120_calmon_relaxed \
  --out models/blender_h120_v6 \
  --cost-bps 5 --tcn-stride 30 \
  --max-total-turnover 10000 --min-toggle-count 2 \
  --l1-ratio-grid 0.15 0.35 0.55 0.75 0.9
```
- Builds a StandardScaler + LogisticRegressionCV pipeline, sweeps thresholds, and records RSS audits, feature usage, and elastic-net weights.
- Result: `final_equity 4.48`, Sharpe 206.8, 4 809 toggles at threshold 0.5 on the relaxed training gate; the deployable manifest (`prob ≥ 0.5`, `rvol_20 ≤ 5e-4`, `min_hold 10`) sustains ≈15.8 % coverage (6 346 toggles) on the Oct 2025 replay.
- Gate masks can be smoothed over the stride detected above (defaults to the TCN stride); sandboxed stride‑1 runs (`models/blender_h120_stride1_v2`) remove smoothing to benchmark turnover ceilings while keeping equity at 4.48.
- CLI adds `--class-weight {balanced,none}` (default balanced) and treats `--calibration-cv <= 1` as “no calibration”; pair the refreshed manifests with `scripts/run_oos_eval.py --family blender` to replay forward windows under consistent guardrails.

## 5. Compile Deployment Shortlist
```bash
python scripts/report_shortlist.py \
  --models-root models \
  --min-equity 1.05 --min-turnover 10 --min-sharpe 0 \
  --out models/report_shortlist.json
```
- Reads every `report.json`, enforces KPI schema, filters by RSS audit pass/fail, and ranks candidates (base, TCN, blender).
- The shortlist feeds deployment review or CI checks.

## 6. Forward Replay & Gate Audit
- Review `models/oos_replay_summary_latest.json` (Oct 1–Oct 28 window, 40 201 rows) alongside the forward matrix to compare training-vs-inference gate metrics across base, TCN, and blender models. The helper script `scripts/run_oos_eval.py --family {base_xgb,tcn,blender}` now powers this replay (and the CI guardrail).
- The deployable manifests all fire: base logs 12 gate hits (8 trades, `final_equity 1.2336`, `gate_coverage 2.99e-4`), TCN h60/h120/h180 clear the 5e-4 floor (`gate_coverage 4.73e-4/7.71e-4/4.23e-4`, toggles 4/62/2), and the eased blender manifest produces 6 346 toggles (`gate_coverage 15.8 %`), while the stride‑1 sandbox variant offers a low-smoothing comparison (134 toggles, `gate_coverage 0.204 %`).
- Wire the replay into monitoring by feeding batches through `training/infer.py::score_base_with_manifest`; the helper updates Prometheus gauges (`model_gate_coverage_ratio`, `model_rss_minute_spike_share`, `model_probability_sigma`) so coverage drift and probability variance show up on dashboards.
- `training/infer.predict_tcn` continues to process inference in stride-aware batches, preventing memory spikes when experimenting with smaller strides (e.g., the stride‑1 blender tests).

## 7. Run Regression Suites
- Ensure manifests stay aligned with their reports and the shortlist keeps surfacing deployable candidates:
  ```bash
  pytest tests/regression -q
  ```
- Exercise async ingestion routes end-to-end with fakeredis before packaging artifacts:
  ```bash
  pytest tests/ingestion_service -q
  ```
- Mirror the CI guardrail locally when adjusting TCN manifests:
  ```bash
  for horizon in h60 h120 h180; do
    python scripts/run_oos_eval.py \
      --family tcn \
      --data datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet \
      --model-dir models/tcn_${horizon}_calmon_relaxed \
      --align-gates --stride 30 --window 192 --channels 64,64 \
      > /tmp/tcn_${horizon}_replay.json
  done
  ```
  Guardrail expectations: `gate_fraction ≥ 5e-4` and `final_equity ≥ 1.2` per horizon.

## 8. Optional – Meta Label Refresh
- Meta gating still lacks a stable decision surface; rerun `scripts/train_meta_label.py` only after extending the blender matrix to newer months and confirming the new deployable coverage plateaus via the guardrail.
- Leverage the existing manifests for deployable gates in live inference until a calibrated meta filter clears the ≥1.2 equity hurdle without compromising the 5e-4 coverage floor.

## 9. Wire Into Scheduler + Trading Dry Run
- Copy refreshed manifests and reports to the models directory mounted by Docker (`MODELS_ROOT`). The scheduler reads them when parsing `INFER_JOBS`.
- Configure `INFER_JOBS` (env JSON list) with exchange/symbol/timeframe, lookback/history windows, and manifest names. Example:
  ```json
  [{"exchange":"binance","symbol":"ETH/USDT","timeframe":"1m","lookback_minutes":1440,
    "history_minutes":2880,"base_model":"base_xgb_h120_calmon_spread0",
    "tcn_model":"tcn_h120_calmon_relaxed","stride":30,"queue_key":"trading:decisions",
    "cron":"*/1 * * * *"}]
  ```
- Bring up `docker compose up scheduler trading` (or the full stack). The scheduler publishes decisions to Redis, and the trading service consumes them, enforcing min-hold/coverage constraints before issuing dry-run orders via the CCXT adapter.
- Observe metrics:
  ```bash
  curl -s http://localhost:9002/metrics | grep scheduler_decision
  curl -s http://localhost:9010/metrics | egrep 'trading_trade_attempts_total|trading_position_active'
  ```
  Grafana dashboards `ingestion-overview`, `scheduler-overview`, and `trading-overview` ship pre-wired panels for queue depth, gate coverage, and audit throughput.
- Use `python scripts/verify_trading_redis.py` to inspect the Redis-backed trading state (`trading:positions`) and audit stream (`trading:audit`) during the dry run. Flip `TRADING_DRY_RUN=0` only after compliance/ops sign-off and update the runbook with observed gate coverage + P&L metrics.

## 10. Feature Parity & Live Gate Drift
- Export the same scheduler slice you just dry-ran so you can diff it against the sanitized training parquet:
  ```bash
  python scripts/export_feature_slice.py \
    --data-lake-root data_lake/market \
    --base-manifest base_xgb_cost_spread \
    --symbols BTC/USDT,ETH/USDT,SOL/USDT \
    --output /tmp/features_debug.parquet
  ```
- Compare live vs training stats and persist the JSON summary alongside the manifest bundle:
  ```bash
  python scripts/compare_feature_stats.py \
    --train datasets/market_multi_3symbol_1m.parquet \
    --live /tmp/features_debug.parquet \
    --out release/calibration/latest/feature_parity.json
  ```
- Treat the parity file + Prometheus `model_*` gauges as the final gate drift check before promoting a manifest or trading config change. Any widening of `hl_spread`/`rvol` thresholds must cite both the sanitizer output and this comparison payload.
