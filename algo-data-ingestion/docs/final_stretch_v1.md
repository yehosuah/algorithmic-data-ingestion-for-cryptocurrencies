# Final Stretch – Production Checklist (Calmon Stack)

Last updated: 2025-11-05 14:56 UTC

Scope: Align the relaxed-gate Horizon-120 XGB, Calmon TCN suite, and elastic-net blender for a deployable release, with manifest-driven governance and monitoring.

## 1. Pre-Flight
- Freeze `datasets/market_btcusdt_1m_2024_2025.parquet`, `datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet`, and the forward replay matrix `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (record commit hash + SHA in release notes).
- Record `gate_smoothing_stride` from the blender `report.json` (and the stride-1 sandbox metrics) so smoothing changes are captured in release notes.
- Persist monthly gate coverage snapshots for base and TCN (`live_gate_coverage.csv`), forward replay diagnostics (`models/oos_replay_summary_latest.json` with the archived `...oct_nov_2025.json` for regression), and attach to the release package.
- Ensure manifests include deployable gates, thresholds, feature lists, and RSS audits; treat them as the contract between training and inference.
- Keep `.github/workflows/ci.yml` green so manifest gating and shortlist regressions (`tests/regression`) stay enforced ahead of release tagging.
- Dry-run the scheduler `INFER_JOBS` → Redis queue → trading service pipeline, capturing Prometheus metrics (`scheduler_decision_messages_enqueued_total`, `trading_trade_attempts_total`, `trading_position_active`) and Redis audit logs for the release window.

## 2. Base XGBoost (H120 Calmon)
- **Status**: `final_equity 4.48`, Sharpe 117, relaxed gate coverage 9.4 %. Oct 1–Oct 28 2025 replay now logs 12 deployable gate hits (8 trades, `final_equity 1.2336`, `gate_coverage 2.99e-4`) under the simplified manifest (`prob ≥ 0.2`, `min_hold 10`, `long_only`) while spread/rvol guardrails are enforced downstream in the trading service.
- **Actions**
  1. Lock the retuned manifest into the release bundle (document thresholds + coverage floor) and schedule a weekly replay to confirm coverage stays above the minimal target.
  2. Extend regression tests to call `training/infer.py::score_base_with_manifest`, asserting gate alignment against `report.json` and the Prometheus gauges emitted by `app/monitoring/model_metrics.py`.
  3. Export coverage alert configs (±2× baseline) for production monitoring with `model_gate_coverage_ratio` thresholds.

## 3. TCN Suite (H60/H120/H180 Calmon)
- **Status**: All horizons clear 5 bps costs (`final_equity` 1.28/3.62/1.85) with ≤200 toggles and share the widened deployable mask; Oct 2025 replay now records non-zero coverage (`gate_coverage 4.73e-4/7.71e-4/4.23e-4`, toggles 4/62/2), so the focus shifts to keeping the 5e-4 floor intact without breaching turnover guards.
- **Actions**
 1. Keep the widened TCN manifests under the CI guardrail (`gate_fraction ≥ 5e-4`, `final_equity ≥ 1.2`) and document fallback modes should coverage regress.
 2. Bundle `fold_logits.parquet`, calibrator, scaler, and manifests for each horizon; document when to switch horizons.
 3. Integrate probability σ guardrail (alert <0.03) and gate coverage checks into monitoring so TCN drift surfaces alongside base/blender metrics (Prometheus rule `TCNGateCoverageUnexpected` now tracks unexpected spikes).
 4. Exercise the stride-aware batching in `training/infer.predict_tcn` during staging runs so stride experiments (e.g., stride‑1) remain production-safe.

## 4. Elastic-Net Blender (H120)
- **Status**: `final_equity 4.48`, Sharpe 206.8, 4 809 toggles at threshold 0.5 with RSS audit passing at 99 % coverage. Oct 2025 replay fires 6 346 deployable trades (`gate_coverage ≈ 15.8 %`) under the eased manifest (`prob ≥ 0.5`, `rvol_20 ≤ 5e-4`, `min_hold 10`), and stride‑1 sandbox runs (`blender_h120_stride1_v2`) bound turnover at 134 toggles (`gate_coverage ≈ 0.2 %`).
- **Actions**
 1. Verify RSS coverage alerts (minute spike share ≥5e-4, daily coverage ≥0.8). Configure automatic fallback to no-RSS feature set.
 2. Document feature weights, probability momentum, class weight choice, and gating logic so live scoring can match the training pipeline and explain coverage swings.
 3. Keep the blender manifest aligned with the latest base thresholds and confirm Oct 2025 replay maintains equity ≥1.2 with turnover within ±25 % of the current run before sign-off; log any smoothing or stride adjustments in monitoring.

## 5. Meta Label (Deferred)
- Training still collapses due to limited dynamic range. Revisit after extending the blender matrix or widening event definitions. Until then, rely on deterministic manifest gates + blender.

## 6. Release Packaging
- Assemble `/release/<YYYYMMDD>` containing:
  - Manifests + artifacts (`models/base_xgb_h120_calmon_spread0`, `tcn_h{60,120,180}_calmon_relaxed`, `blender_h120_v6`).
  - Coverage CSVs, RSS audits, shortlist (`models/report_shortlist.json`), OOS replay snapshots (`models/oos_replay_summary_latest.json` plus the archived `...oct_nov_2025.json`), and the forward matrix `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`.
  - Documentation snapshot: updated `TRAINING_STATUS.md`, `TRAINING_WALKTHROUGH.md`, this checklist.
- Tag git commit, push artifacts to remote storage (or repo with Git LFS) per ops policy.

## 7. Monitoring Rollout
- Metrics to expose: equity curve, turnover, gate activation rate (`model_gate_coverage_ratio`), RSS coverage (`model_rss_minute_spike_share`), probability σ (`model_probability_sigma`), and the recorded `gate_smoothing_stride`, plus their manifest thresholds.
- Trading-specific metrics: queue depth (Redis `trading:decisions`), Prometheus counters/gauges (`trading_trade_attempts_total`, `trading_trade_notional_total`, `trading_gate_toggles_total`, `trading_position_active`, `trading_realized_pnl_total`), and audit stream freshness.
- Alert thresholds:
  - Gate coverage outside ±2× baseline for two consecutive days.
  - RSS minute spike share <5e-4 or audit `passed=false`.
  - Probability σ <0.03 for any validation window.
  - Trading metrics stuck (no `trading_trade_attempts_total` increments for >15 min, `trading_position_active` stale beyond max hold, or Redis audit stream idle).
- Document remediation runbooks (e.g., rerun relaxed retrain, switch to backup model, disable blender).

## 8. Post-Launch Tasks
- Schedule monthly retrain cadence with automated shortlist generation.
- Expand dataset coverage (additional symbols/timeframes) using the same relaxed gate framework.
- Continue meta-label R&D with the longer matrix and event definitions tuned for Calmon coverage.
- Formalise the trading dry-run retrospective (queue depth, audit volume, faux P&L) and schedule the go-live rehearsal (flip `TRADING_DRY_RUN=0` behind feature flag) once monitoring stays green for seven consecutive days.
