# Subtask 1 – 120-Bar TCN Turnover Control

_Last updated: 2025-11-30 18:55 UTC_

> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-13: Folded in the sanitized multi-symbol feed + symbol-gate generator and the parity helpers (`export_feature_slice.py`, `compare_feature_stats.py`) so subtask 1 tracks the same gates/metrics enforced downstream.

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
- Training gate coverage 0.0839; deployable inference mask now keeps only `prob ≥ 0.25`, `min_hold 10`, `long_only`, with spread/volatility guards enforced at execution time.

## Notes
- Fold logits now persist, making recalibration and diagnostics reproducible without rerunning the network.
- Monthly probability σ remains above 0.03, indicating no variance collapse under the relaxed gate.
- Oct 2025 forward replay (`models/oos_replay_summary_latest.json`) now shows deployable coverage for the h120 model (`gate_hits 31`, `toggle_count 62`, `gate_fraction 7.71e-4`, `final_equity 1.94`); keep the guardrail (`gate_fraction ≥ 5e-4`, `final_equity ≥ 1.2`) in mind when tuning thresholds further.
- `training/infer.predict_tcn` now batches inference by stride, letting us explore stride‑1 gates without exhausting memory.
- `app/scheduler/main.py` consumes this manifest via `INFER_JOBS` to publish Redis decisions for the trading dry run; monitor `scheduler_decision_messages_enqueued_total` when experimenting with stride or threshold changes.
- Capture feature parity drift after each tweak via:
  ```bash
  python scripts/export_feature_slice.py --output /tmp/features_debug.parquet
  python scripts/compare_feature_stats.py \
    --train datasets/market_multi_3symbol_1m.parquet \
    --live /tmp/features_debug.parquet \
    --out release/calibration/latest/subtask1_tcn_parity.json
  ```
  so deployability reviewers can see how `hl_spread`, `hl_spread_z`, `rvol_20`, and `base_prob` moved before signing off.

## Follow-ups
1. Maintain the deployable gate above the 5e-4 floor while iterating on turnover; document fallback behaviour if coverage regresses.
2. Train sibling horizons (60/180) for ensemble coverage and document selection triggers in the manifest.
3. Confirm the trading dry run stays stable when this manifest updates (bounded Redis queue, advancing `trading_trade_attempts_total`, audit stream entries for ETH/BTC/SOL lanes).
