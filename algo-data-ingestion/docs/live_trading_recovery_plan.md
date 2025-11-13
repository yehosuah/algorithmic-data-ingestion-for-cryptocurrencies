# Live Trading Recovery Plan

_Last updated: 2025-11-13 04:43 UTC_

> Update 2025-11-13: Folded in the sanitizer + symbol-gate workflow plus the feature parity helpers (`export_feature_slice.py`, `compare_feature_stats.py`) so recovery steps cite the same datasets/gates enforced in scheduler + trading.

_Source of truth for reconciling training metrics with dry-run behaviour._

## 1. Verify Live Gating & Coverage
- Ensure scheduler/trading containers have reloaded manifests (`docker compose exec scheduler cat /opt/models/<model>/manifest.json`).
- Monitor `model_gate_coverage_ratio` for base + TCN via `curl -s localhost:9002/metrics | rg 'model_gate_coverage_ratio.*(base_xgb|tcn)_h120_calmon_relaxed'`.
- Inspect recent Redis decision payloads/audit logs to confirm gate predicates embedded in the stream.

## 2. Align Manifests With Training Reports
- Diff each `manifest.json` against its paired `report.json`, reconciling `prob_gate_min`, `hl_spread_z_max`, and `rvol20_max`.
- Re-export manifests from the training artifacts when drift is found; avoid manual edits that diverge from validated configs.
- Confirm the manifest `gates.training`/`gates.inference` sections match the sanitized multi-symbol gate payload (`release/symbol_gates/market_multi_3symbol_1m.json`). If the JSON changed (after re-running `scripts/compute_symbol_gate_config.py`), reload scheduler + trading so inference and execution observe identical caps.

## 3. Validate Feature & Data Parity
- Dump a live feature frame sample and compare statistics/columns to the training parquet (especially `hl_spread_z`, `rvol_20`, and derived probabilities).
- Confirm `_load_recent_ohlcv` supplies the same lookback and symbol universe used during training and that z-score windows match.
- Automate the comparison with the new helpers:
  ```bash
  python scripts/export_feature_slice.py --output /tmp/features_debug.parquet
  python scripts/compare_feature_stats.py \
    --train datasets/market_multi_3symbol_1m.parquet \
    --live /tmp/features_debug.parquet \
    --out release/calibration/latest/live_recovery_parity.json
  ```
  Attach the JSON to the incident ticket so gate changes cite concrete drift in `hl_spread`, `hl_spread_z`, `rvol_20`, and `base_prob`.

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
