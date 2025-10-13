# Subtask 1 – TCN Turnover Tightening

## Goal
Reduce turnover for the 120-bar horizon TCN model while keeping post-cost equity > 1.0 (5 bps costs).

## Environment
- Python 3.11 virtualenv (`.venv`) rebuilt fresh.
- Key deps: torch 2.3.1, scikit-learn 1.5.2, xgboost 2.1.1.

## Command
```bash
.venv/bin/python scripts/train_tcn.py \
  --data datasets/market_btcusdt_1m_2024_2025.parquet \
  --out models/tcn_cost_h120_turn200 \
  --window 192 --stride 30 --channels 48,48 \
  --epochs 6 --batch-size 512 --lr 5e-4 \
  --dropout 0.1 --weight-decay 1e-5 \
  --class-weight 2.0 \
  --n-folds 3 --embargo-minutes 60 \
  --cost-bps 5 --spread-scale 0 \
  --max-spread 0.0006 --max-spread-z 0.8 \
  --long-only --min-hold-bars 5 \
  --min-total-turnover 4 --max-total-turnover 200 \
  --threshold-criterion final_equity \
  --base-dir models/base_xgb_tuned_features_cost \
  --horizon 120 \
  --diagnostic-thresholds 0.55,0.6,0.65,0.7,0.75,0.8
```

## Result (`models/tcn_cost_h120_turn200/report.json`)
- `final_equity`: **1.0046**
- `sharpe`: 4.18
- `total_turnover`: 8 trades (within ≤200 target)
- `selected_threshold`: 0.675
- Gate coverage: 0.812 (`hl_spread ≤ 0.6 bps`, `hl_spread_z ≤ 0.8`)

## Notes
- Added `--max-total-turnover` support plus equity clipping (see `training/metrics.py`) to prevent exploding equity curves.
- Warnings about older calibrators persist until we retrain the base XGB model with the rebuilt environment (handled in Subtask 2).
