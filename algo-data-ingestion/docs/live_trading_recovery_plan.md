# Live Trading Recovery Plan

_Last updated: 2025-12-30 22:59 UTC_

> Update 2025-12-30: Stage-0 now uses equity-fraction sizing with capital 200 and adds vol-aware stop shaping plus optional quote-based price monitoring (`TRADING_PRICE_MONITOR_INTERVAL_SECONDS`) so stop/take-profit/trailing exits can fire even when decision payloads stall.
> Update 2025-11-30 22:29 UTC: Stage-0 runtime risk overrides now enforce a 1-minute cooldown after exits and 5 minutes after losses, the dry-run/live env keeps `TRADING_SHADOW_SYMBOLS=[]` with BTC/ETH/SOL running as the primary policies (300 USDT notional, `max_spread_bps=10`), and heavy parquet snapshots were removed from git so regenerate slices via the parity helpers when debugging.
> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-16: Added queue/backlog safeguards (`DECISION_PAYLOAD_ITEMS`, `trading_decision_queue_depth`, `TRADING_LAST_TS_GRACE_BARS`), folded in the probability sampler + distribution audit pipeline, and clarified calibrator refresh (re-score base booster before fitting) so recovery follows the latest live diagnostics.

_Source of truth for reconciling training metrics with dry-run behaviour._

## 1. Verify Live Gating & Coverage
- If coverage collapses or before restarting services after downtime, run `python scripts/trigger_preflight.py --contract configs/canonical_training_contract_market_multi_3symbol_1m.yaml --policy configs/final_trigger_policy.yaml --model-dir /opt/models/base_xgb_h120_calmon_spread0 --max-rows 5000 --min-coverage 0.01 --min-trades 5` (adjust path if running outside Docker) to catch dead trigger configs early.
- Confirm the exported env keeps BTC/ETH/SOL as primary entries (inspect `TRADING_MODELS` in the active stage bundle and ensure `TRADING_SHADOW_SYMBOLS=[]`); re-run `analysis.apply_launch_stage` if the bundle drifted before opening gates.
- Ensure scheduler/trading containers have reloaded manifests (`docker compose exec scheduler cat /opt/models/<model>/manifest.json`).
- Monitor `model_gate_coverage_ratio` for base + TCN via `curl -s localhost:9002/metrics | rg 'model_gate_coverage_ratio.*(base_xgb|tcn)_h120_calmon_relaxed'`.
- Inspect recent Redis decision payloads/audit logs to confirm gate predicates embedded in the stream.
- Watch queue health while gates are enforced: `curl -s localhost:9010/metrics | rg trading_decision_queue_depth` should stay near 0. The scheduler trims decision batches to `DECISION_PAYLOAD_ITEMS` (or job-level `max_decision_items`) so stale entries do not accumulate.
- After restarts, confirm the trading service cleared stale dedupe offsets when downtime exceeded `TRADING_LAST_TS_GRACE_BARS` (logged on startup) so fresh decisions are not skipped.
- Assert kill/safe switches while diagnosing: set `TRADING_KILL_SWITCH=1`/`TRADING_SAFE_MODE=1`, verify `trading_safe_mode_latched` flips to 1 and only unlatches after reconciliation success. Check `deadlock_action_taken_total` and audit `deadlock_status` payloads if coverage craters for a symbol.

## 2. Align Manifests With Training Reports
- Diff each `manifest.json` against its paired `report.json`, reconciling `prob_gate_min`, `hl_spread_z_max`, and `rvol20_max`.
- Re-export manifests from the training artifacts when drift is found; avoid manual edits that diverge from validated configs.
- Confirm the manifest `gates.training`/`gates.inference` sections match the sanitized multi-symbol gate payload (`release/symbol_gates/market_multi_3symbol_1m.json`). If the JSON changed (after re-running `scripts/compute_symbol_gate_config.py`), reload scheduler + trading so inference and execution observe identical caps.

## 3. Validate Feature & Data Parity
- The repo no longer stores the sanitized parquet snapshots in git, so always regenerate slices via the helpers below when comparing live vs training features.
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
- **Sample the live streams:** `app.monitoring.probability_sampler` now logs every pre-gate probability batch to `logs/probability_samples/<model>_<prob>.jsonl` (override via `PROB_SAMPLE_*` env vars or mirror to Redis Streams). Keep it enabled whenever scheduler/trading runs.
- **Run the stratified drift audit:** convert the live sampler outputs into tagged hourly parquet + JSON guardrails so collapse/saturation by regime/session is surfaced:
  ```bash
  python3 scripts/probability_distribution_audit.py \
    --samples logs/probability_samples \
    --fold-logits models/base_xgb_h120_calmon_spread0/fold_logits.parquet \
    --fold-column prob_calibrated \
    --features /tmp/features_debug.parquet \
    --hourly-dir release/calibration/latest/live_prob_hourly \
    --summary-out release/calibration/latest/distribution_audit.json
  ```
  For stress cases, feed perturbed slices and stash the resulting `distribution_audit_stress.json` alongside incident notes.
- **Compare against training fold logits:** plot the live histogram versus the reference `fold_logits.parquet` bundled with the TCN manifests and stash the chart + summary JSON with the incident artifacts:
  ```bash
  python3 scripts/plot_probability_distributions.py \
    --samples logs/probability_samples/tcn_h120_calmon_relaxed_tcn_prob.jsonl \
    --model-dir models/tcn_h120_calmon_relaxed \
    --prob-column tcn_prob \
    --out release/calibration/latest/live_vs_fold_tcn_prob.png \
    --summary-out release/calibration/latest/live_vs_fold_tcn_prob.json
  ```
  Repeat for `base_prob` (swap `--prob-column` and the sample filename). If the live σ collapses relative to `fold_logits.parquet`, treat it as a feature/calibrator regression instead of widening gates.
- **Re-run calibrators on a fresh slice:** once `/tmp/features_debug.parquet` is refreshed via `scripts/export_feature_slice.py`, apply the deployable calibrators directly and capture the numeric summary:
  ```bash
  python3 scripts/run_calibrator_check.py \
    --live /tmp/features_debug.parquet \
    --base-model models/base_xgb_h120_calmon_spread0 \
    --tcn-model models/tcn_h120_calmon_relaxed \
    --tcn-stride 2 \
    --summary-out release/calibration/latest/calibrator_health.json
  ```
  The script prints/records calibrated vs uncalibrated σ so you can prove whether the post-hoc mapping is behaving or whether the feature feed collapsed before gating.
  When recalibrating with labels, `scripts/refresh_calibration.py` now re-scores the raw booster before fitting to avoid double-scaling clipped probabilities; stash outputs under `release/calibration/live_recalibration_latest/`.

## 5. Reproduce Training Metrics on Live Data
- Run the replay tooling on the exact OHLCV window currently feeding dry-run with the deployed manifest.
- Compare simulated entries/exits/PnL against the audit stream results to pinpoint divergence (gating vs execution vs costs).

## 6. Fix, Redeploy, and Guard
- After alignment, rebuild `scheduler` and `trading`, verify coverage > 0 for both models, and watch Prom + audit telemetry.
- Add CI checks that fail when manifest gate coverage or feature parity drifts from the `report.json` expectations.
- Document future gate adjustments in this file before rollout; treat it as the authoritative checklist for regression recovery.

## 7. Runtime Controls & Launch Ladder
- **Intent ledger health** – `trading_intent_ledger_state_total` should reflect state transitions (`pending_submit`, `submitted`, `filled`, etc.). If dedupe blocks legitimate orders, flush the Redis ledger keys (`${TRADING_INTENT_LEDGER_PREFIX}:*`) after ensuring no open intents remain, then restart trading with `TRADING_KILL_SWITCH=1` to resume exits only.
- **Runtime risk limits** – Inspect audit `risk_block_reason`/`risk_clip_reasons` and Prometheus `trading_risk_blocked_total`/`trading_risk_clipped_total`. Stage-0 now enforces `cooldown_minutes_after_exit=2` and `cooldown_minutes_after_loss=5`; if you see longer gaps, diff `configs/runtime_overrides/risk_limits_stage_*.yaml` against `configs/portfolio_risk_limits.yaml` and re-run `analysis.apply_launch_stage` to restore overrides.
- **Deadlock policy** – Use `analysis.shadow_readiness` to produce a coverage/deadlock report, review `trading_deadlock_*` gauges, and confirm `deadlock_policy.enabled`/`adjust_prob_gate_min` entries match the deployment contract. If policy actions misfire, run `analysis.evaluate_launch_stage` to regenerate overrides and audit logs.
- **Launch ladder rollback** – To revert a problematic stage, set kill/safe env vars, run `python -m analysis.rollback_to_stage --stage stage_0 --ladder configs/live_launch_ladder.yaml --contract configs/deployment_portfolio_contract.yaml`, redeploy env bundle (`configs/runtime_overrides/stage_0.yaml`), and wait for reconciliation success before re-opening entries.
