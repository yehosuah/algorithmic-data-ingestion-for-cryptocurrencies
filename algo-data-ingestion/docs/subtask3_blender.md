# Subtask 3 – Elastic-Net Blender Refresh

_Last updated: 2025-11-29 14:33 UTC_

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-13: Folded in the sanitizer + symbol-gate workflow and the feature parity helpers (`export_feature_slice.py`, `compare_feature_stats.py`) so blender retrains reference the same gates/metrics enforced in scheduler + trading.

## Goal
Train an elastic-net logistic blender that combines Calmon relaxed base and TCN probabilities with RSS spike features, clearing 5 bps transaction costs while maintaining actionable turnover.

## Workflow
1. Build the RSS-enriched blender matrix using `scripts/build_blender_matrix.py` (intraday RSS spikes, probability momentum, relaxed gate masks) and capture forward windows (e.g., Oct 2025) into `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`. The stride you supply (or allow the script to infer) now becomes the smoothing window for the blender gate.
2. Run `scripts/train_blender.py` with elastic-net sweep and turnover guards.
3. Review RSS audit (`passed=true`) and feature inventory before promoting the artifact.
4. After fitting, export a scheduler slice (`scripts/export_feature_slice.py`) and compare it to the sanitized multi-symbol parquet via `scripts/compare_feature_stats.py --train datasets/market_multi_3symbol_1m.parquet --live /tmp/features_debug.parquet --out release/calibration/latest/blender_parity.json` so gate adjustments cite concrete drift.

## Command
```bash
python scripts/train_blender.py \
  --matrix datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet \
  --base-dir models/base_xgb_h120_calmon_spread0 \
  --tcn-dir models/tcn_h120_calmon_relaxed \
  --out models/blender_h120_v6 \
  --cost-bps 5 --tcn-stride 30 \
  --max-total-turnover 10000 --min-toggle-count 2 \
  --l1-ratio-grid 0.15 0.35 0.55 0.75 0.9
```

## Result (`models/blender_h120_v6/report.json`)
- `final_equity`: **4.482**
- `sharpe`: 206.8
- `total_turnover`: 4 809
- `selected_threshold`: 0.5
- RSS audit: daily coverage 0.995, minute spike share 0.991 (pass, indicator `rss_spike_presence`)
- Deployable manifest: `prob ≥ 0.5`, `rvol_20 ≤ 5e-4`, `min_hold 10` (long-only).
- Manifest (`gates.inference`): `prob ≥ 0.5`, `rvol_20 ≤ 5e-4`, `min_hold 10`.
- Oct 2025 replay: `gate_fraction ≈ 0.158`, `toggle_count 6 346`, deployable `final_equity 4.48`.

## Notes
- Feature set includes probability momentum (`prob_diff`, `*_mom_1`), RSS spike windows, and volatility deltas. The manifest lists candidate features for transparency.
- Training pipeline now leverages KPI schema normalization (`training/reporting.ensure_kpi_schema`) so reports align with base/TCN outputs.
- The matrix stats JSON (`..._rss_latest_stats.json`) provides sanity checks on probability distributions and RSS coverage before fitting.
- CLI exposes `--class-weight {balanced,none}` and treats `--calibration-cv <= 1` as “no calibration”, matching production’s deterministic requirements. The refreshed `scripts/run_oos_eval.py --family blender` keeps forward replay checks aligned with base/TCN guardrails.
- Oct 2025 replay (`models/oos_replay_summary_latest.json`) confirms the eased manifest now delivers 6 346 deployable trades while maintaining `final_equity 4.48`; keep blender aligned with base thresholds as they evolve.
- Sandbox runs (`models/blender_h120_stride1_v2`) collapse the smoothing window to 1 bar to stress turnover—relaxed equity stays at 4.48 while gate share drops to ≈0.2 % with 134 toggles, giving a ceiling for production manifests when smoothing is reduced.
- Scheduler + trading dry run metrics should mirror this coverage: watch `scheduler_decision_messages_enqueued_total` and `trading_trade_attempts_total` after retraining to ensure redispatched blender decisions match the expected trade volume.
- Keep `release/symbol_gates/market_multi_3symbol_1m.json` refreshed via `scripts/compute_symbol_gate_config.py` so base/TCN gate caps (which feed the blender matrix) stay consistent with the sanitized dataset powering scheduler/trading inference.

## Next Steps
1. Keep blender gating in lockstep with the base manifest so Oct–Nov 2025 maintains equity ≥1.2 with turnover within ±25 % of the 4 809-toggle baseline; use the stride‑1 sandbox results as the turnover ceiling when coverage dips below ≈15.8 %.
2. Integrate RSS audit thresholds into monitoring; automatically switch to a no-RSS fallback when coverage dips below requirements.
3. Document feature preprocessing in inference code to mirror StandardScaler + selected columns from the manifest.
4. Capture Grafana `trading-overview` snapshots during the dry run after each blender refresh to validate faux P&L and gate toggles line up with the replay stats.
5. Each time blender training adjusts gates or thresholds, regenerate the manifest from `report.json` and rerun the manifest/report parity sweep plus `pytest tests/regression/test_manifest_gating.py` before exporting release artifacts.
