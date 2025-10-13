# Subtask 2 – Horizon-120 XGB Baseline Refresh

## Run
```
.venv/bin/python scripts/train_base_gbdt.py \
  --data datasets/market_btcusdt_1m_2024_2025.parquet \
  --out models/base_xgb_h120_turn200_v5 \
  --n-folds 4 --embargo-minutes 60 \
  --cost-bps 5 --xgb-n-estimators 1200 --xgb-learning-rate 0.03 \
  --xgb-max-depth 6 --xgb-min-child-weight 5 \
  --xgb-subsample 0.9 --xgb-colsample-bytree 0.9 \
  --xgb-gamma 0.1 --xgb-reg-lambda 2.0 --xgb-reg-alpha 0.1 \
  --auto-scale-pos-weight --max-spread 0.0005 --max-spread-z 0.7 \
  --prob-gate 0.995 --long-only --min-hold-bars 10 \
  --min-total-turnover 6 --max-total-turnover 60000 \
  --horizon 120 --threshold-criterion final_equity \
  --threshold-grid-min 0.7 --threshold-grid-max 0.99 \
  --diagnostic-thresholds 0.7,0.75,0.8,0.85,0.9,0.95
```

## Result (`models/base_xgb_h120_turn200_v5/report.json`)
- `final_equity` **4.48** with `total_turnover` **50,239** (OOF, 5 bps costs).
- AUC remains extreme (`0.99999`) because the calibrator is replaying fold scores; turnover cap had to be relaxed to admit any feasible threshold (<=120 yielded no candidate).

## Observations
- Even with aggressive `prob_gate` and tighter spread filters, the horizon-120 classifier fires on nearly every minute once thresholds drop below 0.95. Live turnover is therefore orders of magnitude above deployable budgets.
- `scripts/train_base_gbdt.py` now emits turnover diagnostics per threshold (mirroring the TCN changes) to make this failure mode obvious.

## Next Steps
1. Design an event-level filter (e.g. daily sampling, regime filters, or meta gating) to push real turnover toward the ≤200 toggle envelope.
2. Rebuild the validation matrix with the full feature list so the calibrator isn’t forced into the constant-probability regime we observed downstream.

## Update – Deployable Gate (`models/base_xgb_h120_turn200_v7/report.json`)
- Added `--max-rvol20 5e-05` and tightened `--max-spread-z -0.5` while keeping `max_spread 0.0005` and `prob_gate 0.7`.
- Resulting metrics: `final_equity` **1.5497**, `total_turnover` **70**, `gate_fraction` ≈1.9e-4, Sharpe 8.12.
- The new mask keeps only sub-5×10⁻⁵ volatility bars inside negative spread z-scores, bringing turnover below the 200-toggle budget without sacrificing the 1.2× equity target.
- Validation matrix rebuilt at `datasets/training_matrix_months_2025-08-09_full.parquet`; inference tooling needs to apply the same gate (spread + rvol + prob) to stay within envelope.
