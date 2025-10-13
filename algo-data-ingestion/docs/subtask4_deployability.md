# Subtask 4 – Deployability Check

## Models
- **TCN (120-bar horizon, symmetric)**: `models/tcn_cost_h120_turn200_ls/report.json`
  - `final_equity` **1.4871**, `total_turnover` **168**, threshold `0.575`.
  - Trade gating: `hl_spread ≤ 0.65 bps`, `hl_spread_z ≤ 0.85`; long/short enabled, min-hold 5 bars.
- **XGB Baseline (120-bar horizon, deployable gate)**: `models/base_xgb_h120_turn200_v7/report.json`
  - `final_equity` **1.5497**, `total_turnover` **70**, threshold `0.70`.
  - Trade gating: `hl_spread ≤ 0.5 bps`, `hl_spread_z ≤ -0.5`, `rvol_20 ≤ 5e-5`, `prob_gate ≥ 0.7`, min-hold 10 bars.
- **Blender**: `models/blender_h120_v3/report.json`
  - `final_equity` 0.9985, `total_turnover` 2; current matrix slice still leaves base/TCN probabilities nearly binary, so logistic stacking remains ineffective.
- **Meta-label**: `models/meta_h120_v2/report.json`
  - Barrier labels now populate (pos share ≈0.54), but the neutralised `base_prob` stream keeps the fitted gate flat (`final_equity` 1.00). Meta refresh deferred until blend/base signals regain dynamic range on the validation window.

## Data Artifacts
- Horizon-aware validation matrix rebuilt with augmented features and stored at `datasets/training_matrix_months_2025-08-09_full.parquet`.
  - Columns include the full tree feature bundle, `ret_next_120`, `base_prob`, and persisted `tcn_prob` for downstream scripts.

## Environment Health
- Virtualenv rebuilt with Python 3.11; key packages reinstalled (`torch 2.3.1`, `scikit-learn 1.5.2`, `xgboost 2.1.1`).
- Numerical safeguards in `training/metrics.py` prevent equity blow-up during horizon evaluation.

## Outstanding Work
1. Re-cut the validation matrix with richer RSS/Reddit coverage (current slice has sparse aggregates), then re-run blender/meta so the stacked model clears the ≥1.2 equity bar.
2. Port the XGB gate (spread + rvol + prob masks) into inference tooling so live turnover stays within the 70-toggle envelope.
3. Run forward validation on neighbouring months to confirm the symmetric TCN and gated XGB remain stable outside Aug–Sep 2025.
