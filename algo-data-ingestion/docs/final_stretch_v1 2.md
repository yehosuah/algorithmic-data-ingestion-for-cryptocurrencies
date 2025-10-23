# Final Stretch – Snapshot Redirect

Last updated: 2025-10-23 01:00 UTC

This document has been superseded by `docs/final_stretch_v1.md`, which tracks the current production checklist for the Calmon stack (base XGB, TCN suite, elastic-net blender).

Key highlights from the latest run:
- Base XGB relaxed gate: `final_equity 4.48`, deployable mask (`hl_spread ≤ 0.0005`, `hl_spread_z ≤ -0.6`, `rvol_20 ≤ 4e-5`, `prob ≥ 0.85`).
- TCN Calmon suite: horizons 60/120/180 deliver `final_equity 1.05–1.33` with ≤200 toggles and shared manifests.
- Blender H120 v6: elastic-net logistic stack with `final_equity 1.84`, 711 toggles, RSS spike audit passed (daily coverage 82.5 %, minute share 0.254).
- Oct 2025 forward replay (`models/oos_replay_oct_nov_2025.json`, `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`) showed relaxed gates still profit but deployable masks sit at zero coverage; the main checklist tracks the required gate retune.
- `.github/workflows/ci.yml` now enforces manifest gating and shortlist regression tests on every push.

Refer to the updated checklist for actionable steps, packaging instructions, and monitoring requirements.
