# Subtask 1 – 120-Bar TCN Turnover Control

## Run
```
.venv/bin/python scripts/train_tcn.py \
  --data datasets/market_btcusdt_1m_2024_2025.parquet \
  --out models/tcn_cost_h120_turn200_ls \
  --window 192 --stride 30 --channels 48,48 \
  --epochs 6 --batch-size 512 --lr 5e-4 \
  --dropout 0.1 --weight-decay 1e-5 --class-weight 2.0 \
  --n-folds 3 --embargo-minutes 60 \
  --cost-bps 5 --max-spread 0.00065 --max-spread-z 0.85 \
  --min-total-turnover 10 --max-total-turnover 200 \
  --min-hold-bars 5 --base-dir models/base_xgb_tuned_features_cost \
  --horizon 120 --diagnostic-thresholds 0.55,0.575,0.6,0.625,0.65,0.7
```

## Result (`models/tcn_cost_h120_turn200_ls/report.json`)
- `final_equity` **1.4871** at threshold `0.575`
- `total_turnover` **168** (≤200 target)
- `gate_fraction` 0.839 with spread caps (`hl_spread ≤ 0.65 bps`, `hl_spread_z ≤ 0.85`)
- Strategy now symmetric (`long_only: false`) which halves net exposure while keeping turnover inside guardrails.

## Notes
- Added diagnostic turnover dumps to `scripts/train_tcn.py` for threshold sweeps, which surfaced how aggressively turnover balloons below `p=0.57`.
- Allowing the short leg recovered equity >1.2 without breaching the 200-toggle ceiling; long-only variants plateaued near 1.14 even with looser gates.

## Follow-ups
1. Stress-test the symmetric rule on adjacent horizons (e.g. 90/150 bars) before sign-off.
2. Re-evaluate whether production deployment should keep shorts enabled or reintroduce a filtered long-only variant.
