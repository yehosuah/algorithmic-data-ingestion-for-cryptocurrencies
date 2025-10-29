# Subtask 3 – Blender Refresh

_Last updated: 2025-10-29 15:53 UTC_

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
- `final_equity` **1.837**
- `sharpe` **28.7**
- `total_turnover` **711** at threshold 0.95
- RSS audit passed (daily coverage 0.825, minute spike share 0.254); RSS spike gate threshold 0.08 on `rss_spike_decay_fast`.

## Findings
- Intraday RSS spike features and probability momentum restored signal diversity; elastic-net fits converge across the l1 ratio grid with identical metrics.
- KPI schema + shortlist tooling (`training/reporting.ensure_kpi_schema`, `scripts/report_shortlist.py`) make regression checks straightforward.
- Forward audit artifact `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` is captured for gate analysis; regenerate by re-running the builder over the target window before packaging.
- Oct 2025 forward replay (`models/oos_replay_summary_latest.json`) now fires 5 870 deployable trades (`gate_fraction ≈ 0.162`), confirming the eased manifest stays aligned with the retuned base thresholds.

## Follow-ups
1. Retune blender gating so Oct–Nov 2025 delivers equity ≥1.2 with turnover within ±25 % of baseline instead of the current zero-coverage outcome.
2. Expose RSS audit metrics and gate shares to monitoring; implement automatic fallback to a no-RSS blender when coverage drops below thresholds.
3. Document inference feature pipeline (StandardScaler + selected columns) to mirror training behaviour in production.
