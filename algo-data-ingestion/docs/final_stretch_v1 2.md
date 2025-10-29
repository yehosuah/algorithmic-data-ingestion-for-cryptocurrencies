# Final Stretch – Snapshot Redirect

Last updated: 2025-10-29 15:53 UTC

This document has been superseded by `docs/final_stretch_v1.md`, which tracks the current production checklist for the Calmon stack (base XGB, TCN suite, elastic-net blender).

Key highlights from the latest run:
- Base XGB relaxed gate: `final_equity 4.48`, deployable mask (`hl_spread ≤ 0.0007`, `hl_spread_z ≤ -0.25`, `rvol_20 ≤ 8e-5`, `prob ≥ 0.72`).
- TCN Calmon suite: horizons 60/120/180 deliver `final_equity 1.05–1.33` with ≤200 toggles and shared manifests.
- Blender H120 v6: elastic-net logistic stack with `final_equity 1.84`, 711 toggles, RSS spike audit passed (daily coverage 82.5 %, minute share 0.254).
- Oct 2025 forward replay (`models/oos_replay_summary_latest.json`, `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`) shows the retuned base/blender manifests restoring minimal coverage (base: 12 gate hits, blender: ≈16 %), while TCN manifests remain idle; the main checklist tracks next steps.
- `.github/workflows/ci.yml` now enforces manifest gating and shortlist regression tests on every push.

Refer to the updated checklist for actionable steps, packaging instructions, and monitoring requirements.
