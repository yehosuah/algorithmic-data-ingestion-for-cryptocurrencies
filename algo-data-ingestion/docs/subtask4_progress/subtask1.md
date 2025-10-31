# Subtask 1 – 120-Bar TCN Turnover Control

_Last updated: 2025-10-31 02:39 UTC_

## Run
```
.venv/bin/python scripts/train_tcn.py \
  --data datasets/market_btcusdt_1m_2024_2025.parquet \
  --out models/tcn_h120_calmon_relaxed \
  --window 192 --stride 30 --channels 64,64 \
  --epochs 10 --batch-size 256 --lr 5e-4 --dropout 0.1 \
  --weight-decay 1e-5 --class-weight 2.0 \
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
- `final_equity` **3.624** (threshold 0.55)
- `total_turnover` **128** (≤200 guardrail)
- `sharpe` **99.8**
- Training gate coverage 0.0839; deployable inference mask uses widened thresholds (`hl_spread ≤ 0.0009`, `hl_spread_z ≤ 0.25`, `rvol_20 ≤ 1.8e-4`, `prob ≥ 0.68`, `min_hold 10`).

## Notes
- Fold logits now persist, making recalibration and diagnostics reproducible without rerunning the network.
- Monthly probability σ remains above 0.03, indicating no variance collapse under the relaxed gate.
- Oct 2025 forward replay (`models/oos_replay_summary_latest.json`) now shows deployable coverage for the h120 model (`gate_hits 31`, `toggle_count 62`, `gate_fraction 7.71e-4`, `final_equity 1.94`); keep the guardrail (`gate_fraction ≥ 5e-4`, `final_equity ≥ 1.2`) in mind when tuning thresholds further.
- `training/infer.predict_tcn` now batches inference by stride, letting us explore stride‑1 gates without exhausting memory.

## Follow-ups
1. Maintain the deployable gate above the 5e-4 floor while iterating on turnover; document fallback behaviour if coverage regresses.
2. Train sibling horizons (60/180) for ensemble coverage and document selection triggers in the manifest.
