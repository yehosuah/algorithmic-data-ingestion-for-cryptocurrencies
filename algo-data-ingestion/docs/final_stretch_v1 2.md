# Final Stretch – Snapshot Redirect

Last updated: 2025-11-05 14:56 UTC

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
