# Subtask 2 – Horizon-120 XGB Baseline

_Last updated: 2025-11-13 04:43 UTC_

> Update 2025-11-13: Folded in the sanitized multi-symbol feed + symbol-gate generator and the feature parity helpers (`export_feature_slice.py`, `compare_feature_stats.py`) so the base retrain mirrors the gates/metrics enforced downstream in scheduler + trading.

## Goal
Retrain the base XGBoost classifier on the 2024–2025 minute feed using the relaxed Calmon gate so post-cost equity exceeds 1.0 at 5 bps while preserving a deployable inference mask.

## Command
```bash
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
- Swap `--data` for the sanitized multi-symbol feed (`datasets/market_multi_3symbol_1m.parquet`, produced via `training.data.sanitize_market_dataset`) and keep the matching gate JSON in lockstep by running `scripts/compute_symbol_gate_config.py --data datasets/market_multi_3symbol_1m.parquet --out release/symbol_gates/market_multi_3symbol_1m.json`; the CLI auto-loads configs whose filenames match the dataset stem.

## Result (`models/base_xgb_h120_calmon_spread0/report.json`)
- `final_equity`: **4.482**
- `sharpe`: 117.4
- `total_turnover`: 4 668 (relaxed gate)
- `selected_threshold`: 0.55
- RSS audit: daily coverage 1.00, minute spike share 9.92e-1 (pass)

## Deployable Gate Snapshot
- Manifest inference mask now enforces `prob ≥ 0.2`, `min_hold 10`, `long_only`; spread and volatility limits are enforced when trades are evaluated.
- Monthly coverage replay (see `live_gate_coverage.csv`) remains within ±1.63× baseline, keeping live turnover <0.02 % of bars.
- Oct 2025 forward replay (`models/oos_replay_summary_latest.json`, 40 201 rows) logs 12 deployable gate hits (8 trades, `final_equity 1.2336`, `gate_coverage 2.99e-4`) under the retuned manifest—keep the thresholds under review as you expand to new months.
- CI now runs `tests/regression/test_manifest_gating.py` and `test_report_shortlist.py` to keep manifests aligned with reports and highlight KPI regressions automatically.
- Scheduler inference jobs publish these base decisions to Redis for the trading dry run; ensure `scheduler_decision_messages_enqueued_total` and `trading_trade_attempts_total` track new threshold experiments.

## Next Steps
1. Monitor deployable coverage via `model_gate_coverage_ratio` and iterate thresholds if Oct–Nov 2025 drops below the new floor.
2. Integrate manifest gates and threshold into `training/infer.py` regression tests so CI catches drift (use `score_base_with_manifest`).
3. Export coverage alert thresholds and RSS audit metadata alongside the artifact bundle.
4. During dry-run rehearsals, confirm Redis queue depth and audit logs mirror the 12 deployable base trades before promoting threshold adjustments.
5. Before tagging a release, run the manifest/report parity sweep (`python - <<'PY' ...` helper or `pytest tests/regression/test_manifest_gating.py`) so `prob_gate_min`, `hl_spread_z_max`, and `rvol20_max` stay locked to the validated report values.
6. When training updates shift gate targets, overwrite manifests by re-exporting from the refreshed `report.json` artifacts (never hand-edit inference gates) and repeat the parity + regression checks to document the new thresholds.
7. Attach feature parity output from `python scripts/export_feature_slice.py --output /tmp/features_debug.parquet` + `python scripts/compare_feature_stats.py --train datasets/market_multi_3symbol_1m.parquet --live /tmp/features_debug.parquet --out release/calibration/latest/base_parity.json` whenever widening gates so reviewers have concrete `hl_spread`, `hl_spread_z`, `rvol_20`, and `base_prob` drift data.
