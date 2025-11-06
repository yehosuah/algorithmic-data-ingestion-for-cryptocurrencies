# Oct 2025 Forward Replay – Manifest Refresh (2025-10-27)

_Last updated: 2025-11-05 14:56 UTC_

## Model Refresh Snapshot
- `models/base_xgb_h120_calmon_spread0` (deployable manifest now enforces `prob ≥ 0.2`, `min_hold 10`, `long_only`) registers 12 gate hits on Oct 2025 (`toggle_count 8`, deployable `final_equity 1.2336`, `gate_coverage 0.0002985`) while spread/rvol checks are handled by the trading service.  
- `models/tcn_h120_calmon_relaxed` clears the deployable mask with the simplified inference gate (`prob ≥ 0.25`, `min_hold 10`, `long_only`); the Oct 2025 run logs 31 gate hits (`toggle_count 62`, `gate_coverage 0.0007711`) and `final_equity 1.9356`. Horizons 60/180 now show shallow but non-zero coverage floors (`gate_coverage 4.73e-4`/`4.23e-4`) at the same probability rule.  
- `models/blender_h120_v6` (deployable manifest `prob ≥ 0.5`, `rvol_20 ≤ 5e-4`, `min_hold 10`) delivers 6 346 deployable toggles (`gate_coverage 0.1579`, deployable `final_equity 4.48`) while retaining `final_equity 1.84` under the relaxed gate and now records `gate_smoothing_stride = 30` in `report.json`.  
- RSS audit for the replay window remains healthy (daily coverage 100 %, minute spike share ≈1.0e0), confirming the rebuilt RSS lake.

## Shortlist Status
- `models/report_shortlist.json` still prioritises earlier Calmon variants (`spread{0/0.05/0.1/0.2}`) because the refreshed baseline sits below the `final_equity ≥ 1.05` guardrail after costs. Update the shortlist thresholds or seed it with the new manifest bundle before hand-off.

## Manifest-Gated Oct 1 – Oct 27 Replay
Source: `models/oos_replay_summary_latest.json` (40 201 rows, Oct 1 00:00 ➜ Oct 28 22:00 UTC). Forward matrix: `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (40 201 rows, Oct 1 ➜ Oct 28 22:00 UTC).

- **Base XGB** – relaxed gate coverage 3.64 %; deployable mask now hits 12 bars (`toggle_count 8`, `avg_minutes_between_trades ≈ 10 217`, `gate_coverage 2.99e-4`).  
- **TCN h120** – relaxed gate retains profitability and now fires 62 toggles (`gate_hits 31`, `gate_coverage 7.71e-4`, `avg_minutes_between_trades ≈ 599`).  
- **Blender h120** – relaxed gate yields 92 toggles (average 274 min spacing); deployable manifest fires 6 346 trades (`gate_fraction ≈ 15.8 %`, `avg_minutes_between_trades ≈ 16.9`). Sandbox stride‑1 runs (`models/blender_h120_gate_test`, `blender_h120_stride1`, `blender_h120_stride1_v2`) demonstrate coverage >50 % when smoothing is removed, providing an upper-cap turnover reference.

## Observations & Next Steps
1. Document the retuned base manifest (thresholds + coverage floor) and add a weekly regression replay that fails when `model_gate_coverage_ratio` for the base drops below the new baseline.  
2. Capture the simplified TCN inference gate (`prob ≥ 0.25`, `min_hold 10`, stride 30) in change logs and regression docs so downstream manifests adopt the same floor.  
3. Align shortlist criteria with the refreshed manifest bundle so reviewers see the deployable artifacts that actually ship and include the recorded `gate_smoothing_stride` in release notes.  
4. Ensure CI exercises `training/infer.py::score_base_with_manifest` over the replay window and fails when deployable coverage or probability σ violates guardrails.  
5. Feed this replay window through the scheduler dry run (Redis queue + trading service) and confirm metrics (`scheduler_decision_messages_enqueued_total`, `trading_trade_attempts_total`, `trading_position_active`) reflect the gate counts above.
