# Subtask 3 – Blender Refresh

_Last updated: 2025-11-29 14:33 UTC_

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-13: Folded in the sanitizer + symbol-gate workflow and the feature parity helpers (`export_feature_slice.py`, `compare_feature_stats.py`) so the blender progress log stays aligned with the gates/metrics enforced downstream.

## Run
```
python scripts/build_blender_matrix.py \
  --source datasets/market_btcusdt_1m_2024_2025.parquet \
  --out datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet \
  --base-dir models/base_xgb_h120_calmon_spread0 \
  --tcn-dir models/tcn_h120_calmon_relaxed \
  --tcn-stride 30 --include-reddit

python scripts/train_blender.py \
  --matrix datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet \
  --base-dir models/base_xgb_h120_calmon_spread0 \
  --tcn-dir models/tcn_h120_calmon_relaxed \
  --out models/blender_h120_v6 \
  --cost-bps 5 --tcn-stride 30 \
  --max-total-turnover 10000 --min-toggle-count 2 \
  --l1-ratio-grid 0.15 0.35 0.55 0.75 0.9
```

## Result (`models/blender_h120_v6/report.json`)
- `final_equity` **4.482**
- `sharpe` **206.8**
- `total_turnover` **4 809** at threshold 0.5
- RSS audit passed (daily coverage 0.995, minute spike share 0.991); deployable manifest: `prob ≥ 0.5`, `rvol_20 ≤ 5e-4`, `min_hold 10`.

## Findings
- Intraday RSS spike features and probability momentum restored signal diversity; elastic-net fits converge across the l1 ratio grid with identical metrics.
- KPI schema + shortlist tooling (`training/reporting.ensure_kpi_schema`, `scripts/report_shortlist.py`) make regression checks straightforward.
- Forward audit artifact `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (40 201 rows) is captured for gate analysis; regenerate by re-running the builder over the target window before packaging.
- Oct 2025 forward replay (`models/oos_replay_summary_latest.json`) now fires 6 346 deployable trades (`gate_fraction ≈ 0.158`), confirming the eased manifest stays aligned with the retuned base thresholds.
- Smoothing follows the stride specified in training; sandbox runs (`models/blender_h120_stride1_v2`) collapse it to 1 bar, dropping gate share to ≈0.2 % (134 toggles) while preserving equity, mapping the turnover ceiling for production manifests.
- Scheduler inference emits blender decisions alongside base/TCN; check `scheduler_decision_messages_enqueued_total` and the trading dashboard when experimenting with stride or smoothing tweaks so the queue volume mirrors the 6 346 deployable trades.
- Feature parity proof now lives in `release/calibration/latest/blender_parity.json` (generated via `scripts/export_feature_slice.py` + `scripts/compare_feature_stats.py --train datasets/market_multi_3symbol_1m.parquet --live /tmp/features_debug.parquet`); refresh it whenever stride/thresholds move so ops know live `hl_spread`, `hl_spread_z`, `rvol_20`, and `base_prob` drift stays within tolerance.

## Follow-ups
1. Keep the deployable gate aligned with the base manifest so Oct–Nov 2025 maintains ≈15.8 % coverage and turnover within ±25 % of the 4 809-toggle baseline; use the stride‑1 sandbox runs as the upper bound when adjusting smoothing.
2. Expose RSS audit metrics, gate share, and `gate_smoothing_stride` to monitoring; implement automatic fallback to a no-RSS blender when coverage drops below thresholds.
3. Document inference feature pipeline (StandardScaler + selected columns) to mirror training behaviour in production.
4. Validate the trading dry run captures faux P&L for blender decisions (`trading_realized_pnl_total`) and that audit stream entries enumerate gate toggles when smoothing changes.
