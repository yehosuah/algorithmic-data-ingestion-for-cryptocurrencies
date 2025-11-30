# Final Stretch – Snapshot Redirect

_Last updated: 2025-11-30 18:55 UTC_

> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-13: Redirect now highlights the sanitized multi-symbol feed + symbol-gate generator and the parity helpers (`export_feature_slice.py`, `compare_feature_stats.py`) documented in `docs/final_stretch_v1.md`, which the Calmon stack now treats as mandatory before sign-off.

This document has been superseded by `docs/final_stretch_v1.md`, which tracks the current production checklist for the Calmon stack (base XGB, TCN suite, elastic-net blender).

Key highlights from the latest run:
- Base XGB relaxed gate: `final_equity 4.48`, deployable mask now focuses on `prob ≥ 0.2`, `min_hold 10`, `long_only`, with spread/rvol constraints handled in the trading service.
- TCN Calmon suite: horizons 60/120/180 deliver `final_equity 1.05–1.33` with ≤200 toggles and shared manifests.
- Blender H120 v6: elastic-net logistic stack with `final_equity 1.84`, 711 toggles, RSS spike audit passed (daily coverage 82.5 %, minute share 0.254).
- Oct 2025 forward replay (`models/oos_replay_summary_latest.json`, `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`) shows the retuned base/blender manifests restoring minimal coverage (base: 12 gate hits, blender: ≈16 %), while TCN manifests remain idle; the main checklist tracks next steps.
- `.github/workflows/ci.yml` now enforces manifest gating and shortlist regression tests on every push.
- Blender reports now persist `gate_smoothing_stride` (30 by default) and stride‑1 sandbox runs (`models/blender_h120_gate_test`, `blender_h120_stride1`, `blender_h120_stride1_v2`) map the turnover ceiling when smoothing is removed.
- `training/infer.predict_tcn` batches inference by stride, so future retunes can safely explore shorter strides without memory spikes.
- Scheduler inference jobs now feed Redis decisions to the trading dry run (`app/trading/service.py`), with metrics surfaced in `monitoring/grafana/dashboards/trading-overview.json`; see the newer checklist for dry-run rehearsal steps.

Refer to the updated checklist for actionable steps, packaging instructions, and monitoring requirements.
