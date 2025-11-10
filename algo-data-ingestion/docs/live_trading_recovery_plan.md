# Live Trading Recovery Plan

_Last updated: 2025-11-10 04:13 UTC_

> Update 2025-11-10: Recovery playbook now references the docker-compose trading service, Redis queue tooling (`scripts/verify_trading_redis.py`), and release/20251030 manifest telemetry.

_Source of truth for reconciling training metrics with dry-run behaviour._

## 1. Verify Live Gating & Coverage
- Ensure scheduler/trading containers have reloaded manifests (`docker compose exec scheduler cat /opt/models/<model>/manifest.json`).
- Monitor `model_gate_coverage_ratio` for base + TCN via `curl -s localhost:9002/metrics | rg 'model_gate_coverage_ratio.*(base_xgb|tcn)_h120_calmon_relaxed'`.
- Inspect recent Redis decision payloads/audit logs to confirm gate predicates embedded in the stream.

## 2. Align Manifests With Training Reports
- Diff each `manifest.json` against its paired `report.json`, reconciling `prob_gate_min`, `hl_spread_z_max`, and `rvol20_max`.
- Re-export manifests from the training artifacts when drift is found; avoid manual edits that diverge from validated configs.

## 3. Validate Feature & Data Parity
- Dump a live feature frame sample and compare statistics/columns to the training parquet (especially `hl_spread_z`, `rvol_20`, and derived probabilities).
- Confirm `_load_recent_ohlcv` supplies the same lookback and symbol universe used during training and that z-score windows match.

## 4. Check Probability & Calibration Health
- Log live probability distributions pre-gate and compare to `fold_logits.parquet`; investigate if the calibrator drifts or outputs collapse.
- Reapply calibrators on a fresh batch to verify deterministic behaviour.

## 5. Reproduce Training Metrics on Live Data
- Run the replay tooling on the exact OHLCV window currently feeding dry-run with the deployed manifest.
- Compare simulated entries/exits/PnL against the audit stream results to pinpoint divergence (gating vs execution vs costs).

## 6. Fix, Redeploy, and Guard
- After alignment, rebuild `scheduler` and `trading`, verify coverage > 0 for both models, and watch Prom + audit telemetry.
- Add CI checks that fail when manifest gate coverage or feature parity drifts from the `report.json` expectations.
- Document future gate adjustments in this file before rollout; treat it as the authoritative checklist for regression recovery.
