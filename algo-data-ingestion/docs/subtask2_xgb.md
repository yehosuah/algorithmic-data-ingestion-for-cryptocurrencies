# Subtask 2 – Horizon-120 XGB Baseline

## Goal
Re-train the base XGBoost model with horizon-aware labels (120-bar forward return) so that out-of-fold equity exceeds 1.0 after 5 bps costs.

## Command
```bash
.venv/bin/python scripts/train_base_gbdt.py \
  --data datasets/market_btcusdt_1m_2024_2025.parquet \
  --out models/base_xgb_h120_turn200 \
  --n-folds 4 --embargo-minutes 60 \
  --cost-bps 5 --spread-scale 0 --slippage-bps 0 \
  --xgb-n-estimators 1200 --xgb-learning-rate 0.03 \
  --xgb-max-depth 6 --xgb-min-child-weight 5 \
  --xgb-subsample 0.9 --xgb-colsample-bytree 0.9 \
  --xgb-gamma 0.1 --xgb-reg-lambda 2.0 --xgb-reg-alpha 0.1 \
  --auto-scale-pos-weight --max-spread 0.0006 --max-spread-z 0.8 --prob-gate 0.95 \
  --long-only --min-hold-bars 10 \
  --min-total-turnover 4 --max-total-turnover 40 \
  --horizon 120 \
  --threshold-criterion final_equity \
  --threshold-grid-min 0.7 --threshold-grid-max 0.99
```

## Result (`models/base_xgb_h120_turn200/report.json`)
- `final_equity`: **4.48**
- `sharpe`: ~4.80×10⁰ (after clipping log-equity at ±1.5)
- `selected_threshold`: 0.70
- Gate coverage: 0.289 (`hl_spread ≤ 0.6 bps`, `hl_spread_z ≤ 0.8`, `prob ≥ 0.95`)

### Turnover Check
Evaluating on the full 2024–2025 sample while applying the same gates yields:
- total turnover ≈ **64 k** transitions
- trades ≈ **64 k** events

This remains much higher than the desired ≤40 horizon events. Additional gating (e.g., sampling less frequent bars or adding event-based filters) is still needed before deployment, although the equity target (>1.0) is met.
