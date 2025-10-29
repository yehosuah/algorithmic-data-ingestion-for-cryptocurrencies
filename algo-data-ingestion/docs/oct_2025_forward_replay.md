# Oct 2025 Forward Replay – Manifest Refresh (2025-10-27)

_Last updated: 2025-10-29 15:53 UTC_

## Model Refresh Snapshot
- `models/base_xgb_h120_calmon_spread0` (deployable manifest `hl_spread ≤ 7e-4`, `hl_spread_z ≤ -0.25`, `rvol_20 ≤ 8e-5`, `prob ≥ 0.72`, `min_hold 10`) now registers 12 gate hits on Oct 2025 (`toggle_count 8`, deployable `final_equity 1.23`, `gate_coverage 0.00031`).  
- `models/tcn_h120_calmon_relaxed` shares the same manifest but remains idle out of sample (`gate_hits 0`, `toggle_count 0`, `gate_coverage 0`). Horizons 60/180 behave similarly.  
- `models/blender_h120_v6` (deployable manifest `prob ≥ 0.5`, `rvol_20 ≤ 5e-4`, `min_hold 10`) delivers 5 870 deployable toggles (`gate_coverage 0.162`, deployable `final_equity 4.48`) while retaining `final_equity 1.84` under the relaxed gate.  
- RSS audit for the replay window remains healthy (daily coverage 100 %, minute spike share ≈1.0e0), confirming the rebuilt RSS lake.

## Shortlist Status
- `models/report_shortlist.json` still prioritises earlier Calmon variants (`spread{0/0.05/0.1/0.2}`) because the refreshed baseline sits below the `final_equity ≥ 1.05` guardrail after costs. Update the shortlist thresholds or seed it with the new manifest bundle before hand-off.

## Manifest-Gated Oct 1 – Oct 27 Replay
Source: `models/oos_replay_summary_latest.json` (38 879 rows, Oct 1 00:00 ➜ Oct 27 23:58 UTC). Forward matrix: `datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet` (28 681 rows, Oct 1 ➜ Oct 20 22:00 UTC).

- **Base XGB** – relaxed gate coverage 3.64 %; deployable mask now hits 12 bars (`toggle_count 8`, `avg_minutes_between_trades ≈ 10 217`).  
- **TCN h120** – relaxed gate retains profitability but deployable mask remains idle (`gate_hits 0`, `toggle_count 0`).  
- **Blender h120** – relaxed gate yields 92 toggles (average 274 min spacing); deployable manifest fires 5 870 trades (`gate_fraction ≈ 16.2 %`, `avg_minutes_between_trades ≈ 12.7`).  

## Observations & Next Steps
1. Document the retuned base manifest (thresholds + coverage floor) and add a weekly regression replay that fails when `model_gate_coverage_ratio` for the base drops below the new baseline.  
2. Extend the retune—or provide a fallback—for the TCN suite; zero deployable hits persist despite the widened thresholds.  
3. Align shortlist criteria with the refreshed manifest bundle so reviewers see the deployable artifacts that actually ship.  
4. Ensure CI exercises `training/infer.py::score_base_with_manifest` over the replay window and fails when deployable coverage or probability σ violates guardrails.
