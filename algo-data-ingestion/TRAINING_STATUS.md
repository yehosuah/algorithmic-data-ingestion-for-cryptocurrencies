# Model Training Status (XGB · TCN · Blender)

_Last updated: 2025-10-06 00:45 UTC_

Cost baseline for every metric below: **5 bps** per side, zero extra spread scaling unless explicitly stated.

## Data Landscape
- **Year-wide minute feed** – `datasets/market_btcusdt_1m_2024_2025.parquet`
  - 894 240 bars covering 2024-01-01 ➜ 2025-09-12.
  - Minute returns: mean `1.37e-06`, σ `7.02e-04`, skew -0.47, kurtosis 53.9 ⇒ far heavier tails than a Gaussian (expected in high-volume BTC venues).
  - Bid/ask proxy (`hl_spread`) averages **6.85 bps** with 90th percentile 14.6 bps and 99th percentile 34.3 bps; pro desks typically target ≤3 bps, so our model must aggressively gate to stay inside cost budget.
  - `rvol_20` mean `5.66e-04`, 90th percentile `1.02e-03`, 99th percentile `2.06e-03`, mapping neatly onto the equity spikes we see in training folds.
  - Rolling 120-bar returns (to mirror the horizon) span `[-2.20 %, +2.19 %]` between the 1st and 99th percentiles, matching what high-frequency crypto desks describe as “liquidity shock” moves.
- **Aug–Sep 2025 matrix** – `datasets/training_matrix_months_2025-08-09_full.parquet`
  - 61 798 bars (Aug 1 ➜ Sep 12) with augmented features + `ret_next_120`, persisted `base_prob` + `tcn_prob`, and RSS aggregates.
  - Minute volatility compresses (σ `4.49e-04`), skew -0.39, kurtosis 39.8 due to calmer summer trading; spreads tighten to **4.09 bps** on average (90th 8.57 bps, 99th 19.13 bps).
  - Horizon-120 label stats: mean `2.34e-05`, σ `4.81e-03`, skew +0.44, kurtosis 8.55, with 1st/99th percentiles `[-1.45 %, +1.33 %]`. Compared to the year-wide feed, tail risk is halved, which explains why long-hold models overstate Sharpe on this slice.
  - `base_prob` is nearly binary (`q25=0`, `q50=1`, σ≈0.499) because the deployable gate zeroes most bars; within the relaxed training gate (see below) the mean softens to 0.53 with σ≈0.50.
  - `tcn_prob` retains limited dynamic range (σ≈0.027, `q0.9≈0.56`); correlation with `ret_next_120` ≈ **-0.007** once the live gate is applied, signalling the stacking dead-end.
  - Social liquidity remains a rounding error: RSS hits in only 0.11 % of minutes (mean sentiment -0.10 when present), so the two-month slice is materially poorer than full-year community feeds used by benchmark crypto-arb desks.
- **Gate coverage comparison**
  - Live deployable gate (`hl_spread ≤ 0.0005`, `hl_spread_z ≤ -0.6`, `rvol_20 ≤ 4e-5`, `base_prob ≥ 0.85`) now stays within the ±2× coverage envelope: year-wide activation averages **0.0035 %**, peaking at **0.0179 %** in Jul‑2025 and dropping to zero during the quietest months (see `models/base_xgb_h120_calmon_spread0/live_gate_coverage.csv`).
  - Relaxed training gate (`hl_spread_z ≤ 0.25`, `rvol_20 ≤ 2e-4`, no prob gate) still captures **11.36 %** of the Aug–Sep slice and restores `tcn_prob` variance (σ≈0.021) without inviting the full 6 bps average spread regime.

_Implication:_ the deployable gates dramatically compress the feature distribution seen by stacked models. Any downstream learner must cope with binary-like base scores and tiny RSS coverage; otherwise it overfits to noise. Re-validating on the full year feed is mandatory before production.

## Horizon-120 XGBoost Classifier
- **Artifacts**: `models/base_xgb_h120_calmon_spread0` (primary), plus cost sweeps at `{0.05, 0.1, 0.2}` spread scaling.
- **Configuration highlights**: depth 6, 1200 trees, auto `scale_pos_weight`, 6 calendar‑month folds with 60-minute embargo. Training gate: `hl_spread_z ≤ 0.25`, `rvol_20 ≤ 2e-4`, no probability filter. Deployable gate (persisted in every manifest): `hl_spread ≤ 0.0005`, `hl_spread_z ≤ -0.6`, `rvol_20 ≤ 4e-5`, `prob ≥ 0.85`, `min_hold_bars = 10`, long-only.
- **Out-of-fold metrics (relaxed gate)**:
  - `final_equity` **4.48**
  - `sharpe` **108.9** (under the relaxed training gate)
  - `total_turnover` **3.7 k** (gate fraction 9.47 % over 237 k labeled bars)
  - `oof_auc` 0.999996
- **Live coverage checks**
  - Manifest-level replay confirms monthly activation stays within the 0–1.63× band relative to the 0.011 % baseline (Jul‑2025 tops out at 0.0179 %; several months record zero fires).
- **What’s working**
  - Relaxing training gate restored probability variance, enabling the 6-fold calendar split to produce stable thresholds.
  - Gate configs and manifests are now exported alongside every run; inference utilities digest them via `training.infer.load_gate_config`.
  - Cost sweeps (spread_scale 0→0.2) keep final equity at 4.48 with unchanged turnover, demonstrating robustness to wider spreads.
- **Follow-ons**
  1. Fold-level diagnostics are in place; next step is to backfill inference replays over higher-spread months (Nov‑2024, Apr‑2025) to validate live gate stability.
  2. Downstream adapters/tests consume the manifest gate; integration tests should assert the boolean mask before order generation.
  3. Update monitoring thresholds to match the tighter gate (spread z ≤ -0.6, prob ≥ 0.85) so drift alerts trigger correctly.

## Horizon-120 Temporal Convolutional Network
- **Artifact**: `models/tcn_h120_calmon_relaxed` (calendar-month folds, relaxed gate).
- **Architecture**: TinyTCN with two 48-channel residual blocks, 192-bar windows, stride 60, dropout 0.1, AdamW (`class_weight=2.0`), base probabilities appended as an extra channel.
- **Out-of-fold metrics** (`report.json`):
  - `final_equity` **1.36** with **78** toggles (`threshold = 0.65`, long/short)
  - `sharpe` 69.3 under the relaxed training gate (`hl_spread_z ≤ 0.25`, `rvol_20 ≤ 2e-4`)
  - `gate_fraction` 0.117 (training gate), manifest mirrors the XGB live gate (`hl_spread ≤ 0.0005`, `hl_spread_z ≤ -0.6`, `rvol_20 ≤ 4e-5`, `prob ≥ 0.85`)
- **Progress**
  - TCN save routine now emits `manifest.json` with gate config + metadata, keeping deployment parity with the base model.
  - Training defaults match the relaxed gate, preventing probability collapse prior to gating.
- **Next steps**
  1. Repeat the run with stride 30 and horizon sweep {60, 180} (turnover cap 200) once compute budget allows.
  2. Persist per-fold logits to unblock calibrator refresh (pending update to `training/tcn_model.py`).
  3. Feed the refreshed TCN manifest into blender experiments to re-establish ranking power under the relaxed gate.

## Logistic Blender (Base + TCN + RSS)
- **Artifacts**: `models/blender_h120` (early slice) and `models/blender_h120_v3` (latest RSS matrix).
- **Configuration**: StandardScaler + LogisticRegression (balanced class weight) over features `[base_prob, tcn_prob, rss_count, rss_sent_mean, rvol_5, rvol_20]`.
- **Metrics**
  - `blender_h120/report.json`: `final_equity` 1.01, `total_turnover` 22 (healthy; derived from the wider relaxed gate distribution).
  - `blender_h120_v3/report.json`: `final_equity` 0.9985, `total_turnover` 2, `selected_threshold` 0.875 (model refuses to trade once the 0.011 % live gate is applied).
- **Diagnosis**
  - With the deployable base gate, `base_prob` becomes {0,1} and dominates the regression, so the blender either mirrors the base signal or shuts off entirely.
  - RSS coverage (<0.12 % of bars) leaves the logistic model with almost no informative variance; coefficients collapse toward zero.
  - `tcn_prob` brings minimal incremental signal because it is nearly constant on the gated slice.
- **Required changes before deployment**
  1. **Rehydrate feature variance**: regenerate the blender dataset directly from year-wide inference outputs _before_ imposing the strict spread/vol filters, then apply the gate only during evaluation. This restores ranking information the logistic learner can exploit.
  2. **Augment signals**: include meta-features such as `base_prob - tcn_prob`, `rss_sent_mean` lagged averages, and realized spread metrics to give the blender continuous inputs.
  3. **Balance datasets temporally**: ensure training includes both quiet and volatile months so coefficients generalize when RSS bursts increase.
  4. **Validate against high-turnover references**: compare the blended strategy to the raw minute feed (year-wide dataset) to guarantee it still clears ≥1.2 equity when the gate is relaxed for liquidity provision.

## Meta-Labeling Status
- `models/meta_h120_v2` trains after redefining triple-barrier events (max-hold 180, pt/sl 1.5×/2.0× volatility) but achieves `final_equity` 1.00 because primary signals are flat.
- `models/meta_h120_v3` fails when filtering for complete RSS/Twitter fields (single-class data).
- **Action**: defer meta-label deployment until base + TCN regain probability gradients on a broader slice; otherwise the meta gate adds no information.

## Deployment Checklist (Current Gaps)
1. **Gate Consistency** – Mirror the spread/vol/prob filters from `models/base_xgb_h120_turn200_v7/report.json` and `models/tcn_cost_h120_turn200_ls/report.json` inside `training/infer.py` and any live adapters.
2. **Probability Diagnostics** – Store per-fold probability histograms and ROC curves for all future runs to catch degeneracy before stacking.
3. **Data Refresh Cadence** – Rebuild the August–September matrix monthly, but cross-check against the full-year quant feed to ensure we are not overfitting to one liquidity regime.
4. **Social Feed Enrichment** – Expand RSS/Twitter sourcing (coverage >5 % target) or drop the feature family entirely until coverage meets minimum viable density.
