# Subtask 2 – Horizon-120 XGB Baseline

_Last updated: 2025-10-23 01:00 UTC_

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

## Result (`models/base_xgb_h120_calmon_spread0/report.json`)
- `final_equity`: **4.482**
- `sharpe`: 108.3
- `total_turnover`: 3 648 (relaxed gate)
- `selected_threshold`: 0.55
- RSS audit: daily coverage 0.852, minute spike share 6.5e-4 (pass)

## Deployable Gate Snapshot
- Manifest inference mask: `hl_spread ≤ 0.0005`, `hl_spread_z ≤ -0.6`, `rvol_20 ≤ 4e-5`, `prob ≥ 0.85`, `min_hold 10`, long-only.
- Monthly coverage replay (see `live_gate_coverage.csv`) remains within ±1.63× baseline, keeping live turnover <0.02 % of bars.
- Oct 2025 forward replay (`models/oos_replay_oct_nov_2025.json`) retained 4.48 equity under the relaxed gate but delivered zero toggles under the deployable mask—thresholds/fallback logic must be adjusted before deployment.
- CI now runs `tests/regression/test_manifest_gating.py` and `test_report_shortlist.py` to keep manifests aligned with reports and highlight KPI regressions automatically.

## Next Steps
1. Retune the inference mask so Oct–Nov 2025 delivers non-zero coverage and equity ≥1.2 without blowing turnover budgets.
2. Integrate manifest gates and threshold into `training/infer.py` regression tests so CI catches drift.
3. Export coverage alert thresholds and RSS audit metadata alongside the artifact bundle.
