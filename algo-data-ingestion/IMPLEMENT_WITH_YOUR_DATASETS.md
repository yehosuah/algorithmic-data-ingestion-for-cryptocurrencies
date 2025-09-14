# Walkthrough: Implement with Your Datasets

This document captures the end-to-end implementation plan using your datasets. It is actionable and aligned with the repository’s current structure and scripts.

Datasets
- Training (market-only): `datasets/market_btcusdt_1m_2024_2025.parquet`
- Validation (market+RSS): `datasets/training_matrix_months_2025-08-09.parquet`

Step 1: Preprocess
- Ensure timestamp UTC and unique per bar (already enforced in our dataset builders).
- For TCN, construct a sliding window tensor from the market dataset (e.g., last 32 bar deltas/returns; standardize per window).
- For labels: use `y_dir`; optionally create higher-confidence labels (e.g., `|ret_next| > threshold` for positives) to train the meta-filter later.

Step 2: Base Learner (GBDT)
- Train XGBoost classifier on market features (no RSS) with walk-forward CV across 2024→2025 windows.
- Calibrate predictions (Isotonic/Platt) on the backtest folds.
- Save: base model + probability calibrator + feature list + chosen probability threshold.

Step 3: Temporal Edge (Tiny TCN)
- Inputs: short window (e.g., 32) of returns/ohlcv deltas; target `y_dir`.
- Keep small (1–2 residual blocks, kernel size 3–5) to avoid overfit.
- Calibrate outputs (map to probabilities).
- Save: TCN model + scaler + calibrator + chosen threshold.

Step 4: Combine (Stack/Blend)
- Build a blender dataset on the validation month:
  - Features: `base_prob`, `tcn_prob`, `rss_count`, `rss_sent_mean`, and regime features (e.g., `rvol_20`).
  - Target: `y_dir` for the month.
- Train a logistic blender (or shallow tree).
- Threshold selection: choose `p*` that maximizes PnL with costs on the validation period.

Step 5: Meta‑Labeling Filter (optional but recommended)
- Generate events via triple‑barrier (profit‑taking, stop‑loss, time barrier).
- Train logistic meta‑model to predict event success conditioned on base signals and TCN features.
- Use meta‑prob to mask low‑quality trades before applying thresholds.

Step 6: Risk & Execution
- Volatility targeting: position size `s = k / recent_vol` (cap max size).
- Dynamic threshold: `p* = c0 + alpha × (spread/vol)` to avoid trading when costs are high.
- Turnover constraint: suppress flips within N minutes; apply small hysteresis between enter/exit thresholds.

Step 7: Live Path (staged)
- Feature store: read latest features from Redis; compute TCN inputs from recent history buffer.
- Apply base + TCN; optional meta‑filter; blender; threshold to trade.
- If RSS present live, feed RSS features; else fallback to base stack (blender should handle NaN/zero gracefully).

