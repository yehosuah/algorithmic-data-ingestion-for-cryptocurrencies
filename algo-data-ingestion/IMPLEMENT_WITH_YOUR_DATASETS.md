# Walkthrough: Implement with Your Datasets

_Last updated: 2025-10-23 01:00 UTC_

This plan mirrors the refreshed Calmon stack. Adapt the paths/parameters to your own instruments once you have equivalent market + RSS coverage.

## Reference Artifacts
- Market history: `datasets/market_btcusdt_1m_2024_2025.parquet`
- RSS-enriched blender matrix: `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet`
- Baseline models: `models/base_xgb_h120_calmon_spread0`, `models/tcn_h120_calmon_relaxed`
- Forward replay snapshot: `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` + `models/oos_replay_oct_nov_2025.json` (training vs inference gate comparison)

## Step 1 – Feature Engineering
1. **Market dataset**: reuse `scripts/build_market_dataset.py` to generate your symbol’s feature parquet (ensures consistent `ret_next` and `y_dir` labels).
2. **RSS aggregation**: ingest feeds via `scripts/rss_to_parquet.py` (or your own collectors). The blender builder depends on daily coverage ≥80 % and minute spike share ≥5e-4.
3. **Blender matrix**: run `scripts/build_blender_matrix.py`, pointing `--base-dir` / `--tcn-dir` at the calibrated models to backfill probabilities and engineered RSS features. Capture forward audit windows (e.g., Oct 2025) into a dedicated matrix with predictions so you can compare training vs inference gates later.

## Step 2 – Base Learner
- Train with relaxed gate defaults:
  ```bash
  python scripts/train_base_gbdt.py \
    --data <your_market_dataset.parquet> \
    --out models/base_xgb_h120_calmon_spread0_yoursymbol \
    --fold-scheme calendar_month --n-folds 6 \
    --cost-bps 5 --max-spread-z 0.25 --max-rvol20 2e-4
  ```
- Validate the RSS audit and monthly diagnostics in `report.json`, then review deployable gates within the generated manifest.

## Step 3 – Temporal Model
- Clone the TCN run with horizon tuned to your strategy (120 bars by default). Adjust `--window`, `--channels`, and `--stride` to match volatility profile while keeping turnover ≤200.
- Store the manifest + fold logits to allow recalibration without retraining from scratch.

## Step 4 – Blender / Ensemble
- Feed the RSS-enriched matrix into `scripts/train_blender.py` with an elastic-net sweep. Confirm:
  - RSS audit `passed = true`
  - Threshold respects turnover guardrails
  - `report.json` lists meaningful feature weights (probability momentum, RSS spikes, regime features)

## Step 5 – Meta Filter (optional)
- Once the base + TCN + blender produce non-degenerate probabilities on your data, experiment with `scripts/train_meta_label.py`.
- Adjust the triple-barrier params to your instrument’s volatility; require ≥1.2 final equity and ≥20 toggles before promoting the meta gate.

## Step 6 – Deployment Readiness
- Use `scripts/report_shortlist.py` to summarise candidates.
- Replay live coverage with the manifest gates (`live_gate_coverage.csv` pattern) and integrate the gating logic into your inference stack.
- Monitor RSS coverage and probability variance to trigger fallbacks if data quality dips below thresholds captured in the reports.
- Run regression guardrails before deployment: `pytest tests/regression -q` keeps manifests aligned with reports and verifies the shortlist; `pytest tests/ingestion_service -q` exercises the async API.
- Inspect your forward replay equivalent of `models/oos_replay_oct_nov_2025.json`; if the deployable mask drops to zero coverage, widen thresholds or add a fallback gate prior to launch.
