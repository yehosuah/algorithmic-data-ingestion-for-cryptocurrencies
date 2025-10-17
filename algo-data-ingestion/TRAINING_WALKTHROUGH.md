# Training Walkthrough (Market Baseline + RSS Blender)

This guide explains how to train a production‑ready baseline model on 1‑minute OHLCV features (year‑long dataset) and how to build a validation blender that uses RSS aggregates.

## Datasets

- Training (market‑only):
  - `datasets/market_btcusdt_1m_2024_2025.parquet`
- Validation (market + RSS):
  - `datasets/training_matrix_months_2025-08-09.parquet`

If you need to (re)build these, refer to `SANITY_CHECKS.md` and `README.md`.

## Installation

- Local venv (Python 3.11 recommended):
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 1) Train Base XGBoost (market‑only, walk‑forward + calibration)

Trains an XGBoost classifier on market features to predict `y_dir`. Performs time‑based walk‑forward, calibrates probabilities (isotonic), and selects a probability threshold that maximizes PnL with transaction costs.

```bash
python scripts/train_base_xgb.py \
  --dataset datasets/market_btcusdt_1m_2024_2025.parquet \
  --time-embargo-min 60 \
  --cost-bps 5 \
  --out-dir artifacts/base_xgb
```

Outputs in `artifacts/base_xgb/`:
- `model.json` (XGBoost booster)
- `calibrator.joblib` (isotonic)
- `feature_list.json` (feature names)
- `threshold.json` (chosen prob threshold)
- `report.json` (AUC, PnL, drawdown, Sharpe per fold + overall)

## 2) Train Blender (market + RSS on validation month)

Loads the base model and applies it to the validation matrix (market + RSS) to get `base_prob`. Trains a logistic blender using `[base_prob, rss_count, rss_sent_mean, regime features (if present)]` and chooses a PnL‑optimal threshold.

```bash
python scripts/train_blender.py \
  --matrix datasets/training_matrix_months_2025-08-09.parquet \
  --base-artifacts artifacts/base_xgb \
  --cost-bps 5 \
  --out-dir artifacts/blender
```

Outputs in `artifacts/blender/`:
- `blender.joblib` (logistic model)
- `threshold.json` (prob threshold for trade) 
- `report.json` (PnL metrics on validation month)

## 3) Leak-Proof Evaluation (base_xgb & TCN)

Use `scripts/run_oos_eval.py` to replay walk-forward evaluation with embargoed folds, aligned trade gates, and optional baseline comparisons. The script works for both the tree baseline and the TCN variants.

```bash
python3 scripts/run_oos_eval.py \
  --family base_xgb \
  --data datasets/market_btcusdt_1m_2024_2025.parquet \
  --model-dir models/base_xgb_h120_calmon_spread0 \
  --align-gates \
  --save-oos models/base_xgb_h120_calmon_spread0/oos_eval_latest.parquet \
  --baseline-oos models/base_xgb_h120_calmon_spread0/oos_eval_2025-09.parquet
```

- `--align-gates` copies the inference gate onto the training gate so the reported metrics match the live deployment rules.
- `--baseline-oos` (optional) loads an existing snapshot and reports deltas (MAE on probabilities, gate mismatch rate, etc.) before accepting the new metrics. Omit this flag until you have a trusted snapshot on disk.
- For TCN families swap `--family tcn` and provide the temporal parameters (e.g., `--window`, `--channels`, `--epochs`, `--stride`). Use `--save-fold-logits` if you want the per-fold TCN logits for further analysis.

## 4) Inference (outline)

During live serving:
1. Read latest market features from the feature store (Redis) at time t.
2. Apply base model → `base_prob(t)` (using the same feature list saved in artifacts).
3. If RSS is available, compute current aggregates (last 1‑minute count/sentiment) and pass `[base_prob, rss_count, rss_sent_mean]` to the blender.
4. Compare to threshold; if `p >= p*` go long (or short for symmetric rule), otherwise flat.
5. Apply position sizing and turnover rules.

## 5) Notes

- Walk‑forward and embargo are used to avoid leakage.
- Probability calibration is essential so thresholds reflect actual hit rate.
- RSS is used for the blender on the month window; the base model does not depend on RSS (so it generalizes across the full year).
