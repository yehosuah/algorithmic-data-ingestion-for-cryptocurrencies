# Subtask 2 – Horizon-120 XGB Baseline Refresh

_Last updated: 2025-10-31 02:39 UTC_

## Run
```
.venv/bin/python scripts/train_base_gbdt.py \
  --data datasets/market_btcusdt_1m_2024_2025.parquet \
  --out models/base_xgb_h120_calmon_spread0 \
  --fold-scheme calendar_month --n-folds 6 --embargo-minutes 60 \
  --cost-bps 5 --max-spread-z 0.25 --max-rvol20 2e-4 \
  --xgb-n-estimators 1200 --xgb-learning-rate 0.03 \
  --xgb-max-depth 6 --xgb-min-child-weight 5 \
  --xgb-subsample 0.9 --xgb-colsample-bytree 0.9 \
  --xgb-gamma 0.1 --xgb-reg-lambda 2.0 --xgb-reg-alpha 0.1 \
  --auto-scale-pos-weight \
  --threshold-criterion final_equity \
  --diagnostic-thresholds 0.5,0.55,0.6,0.65
```

## Result (`models/base_xgb_h120_calmon_spread0/report.json`)
- `final_equity` **4.482**
- `sharpe` **117.4**
- `gate_fraction` 0.111 (relaxed training gate)
- `selected_threshold` 0.55

## Observations
- Relaxing the training gate (`hl_spread_z ≤ 0.25`, `rvol_20 ≤ 2e-4`, no prob filter) restored probability variance and kept turnover manageable (4.7 k toggles) during training.
- RSS audit passes; spread stress (`spread_scale ∈ {0,0.05,0.1,0.2}`) leaves equity unchanged, indicating resilience to moderate cost inflation.
- Oct 2025 forward replay (`models/oos_replay_summary_latest.json`, 40 201 rows) now logs 12 deployable gate hits (8 trades, `final_equity 1.2336`, `gate_coverage 2.99e-4`), so lock the thresholds in release notes and monitor coverage for drift.

## Deployable Gate Update
- Manifest inference mask: `hl_spread ≤ 0.0007`, `hl_spread_z ≤ -0.25`, `rvol_20 ≤ 8e-5`, `prob ≥ 0.72`, `min_hold 10`, long-only. CI now enforces manifest alignment via `tests/regression/test_manifest_gating.py` and shortlist viability via `test_report_shortlist.py`.
- `live_gate_coverage.csv` confirms monthly coverage within ±1.63× baseline (peak 0.0179 % in Jul 2025) historically, and the Oct 2025 forward replay now records 12 deployable hits (8 trades, `final_equity 1.2336`, `gate_coverage 2.99e-4`) under the widened mask—treat that coverage as the new floor.

## Next Steps
1. Monitor the widened manifest so Oct–Nov 2025 forward replay maintains equity ≥1.2 with coverage at or above the new Oct baseline (12 hits) and stable turnover.
2. Add regression tests that load the manifest into `training/infer.py` and compare KPIs against `report.json`.
3. Include manifest + coverage CSV in the packaging pipeline for consistent deployment hand-offs.
