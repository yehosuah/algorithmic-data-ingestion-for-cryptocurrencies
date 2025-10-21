# Subtask 1 – TCN Turnover Tightening

_Last updated: 2025-10-21 02:50 UTC_

## Goal
Reduce turnover for the 120-bar horizon TCN model while keeping post-cost equity > 1.0 (5 bps costs).

## Environment
- Python 3.11 virtualenv (`.venv`) rebuilt fresh.
- Key deps: torch 2.3.1, scikit-learn 1.5.2, xgboost 2.1.1.

## Command
```bash
.venv/bin/python scripts/train_tcn.py \
  --data datasets/market_btcusdt_1m_2024_2025.parquet \
  --out models/tcn_h120_calmon_relaxed \
  --window 192 --stride 30 --channels 64,64 \
  --epochs 10 --batch-size 256 --lr 5e-4 \
  --dropout 0.1 --weight-decay 1e-5 \
  --class-weight 2.0 \
  --n-folds 4 --embargo-minutes 60 \
  --cost-bps 5 --max-spread-z 0.25 --max-rvol20 2e-4 \
  --min-total-turnover 4 --max-total-turnover 200 \
  --min-hold-bars 5 --long-only 0 \
  --threshold-criterion final_equity \
  --base-dir models/base_xgb_h120_calmon_spread0 \
  --horizon 120 \
  --diagnostic-thresholds 0.55,0.6,0.65,0.675,0.7
```

## Result (`models/tcn_h120_calmon_relaxed/report.json`)
- `final_equity`: **1.331**
- `sharpe`: 24.9
- `total_turnover`: 180 (≤200 target)
- `selected_threshold`: 0.65
- Gate coverage (training): 0.0896 under the relaxed mask (`hl_spread_z ≤ 0.25`, `rvol_20 ≤ 2e-4`); deployable inference gate mirrors the base manifest (`hl_spread ≤ 0.0005`, `hl_spread_z ≤ -0.6`, `rvol_20 ≤ 4e-5`, `prob ≥ 0.85`, `min_hold 10`).

## Notes
- Per-fold logits persist (`fold_logits.parquet`), enabling recalibration without rerunning the network.
- Probability σ guardrail stays above 0.03 across validation months, avoiding the collapse seen with earlier tight gates.

## Next Steps
- Replay Oct–Nov 2025 with the deployable gate to confirm turnover remains under the live budget.
- Evaluate horizon 60 and 180 siblings for ensemble coverage; document selection criteria alongside manifests.
