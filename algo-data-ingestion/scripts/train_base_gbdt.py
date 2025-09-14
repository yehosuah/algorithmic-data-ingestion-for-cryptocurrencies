#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from training.data import load_parquet_dataset, ensure_labels
from training.walkforward import time_folds
from training.model import extract_features_labels, train_xgb, calibrate, predict_proba, save_artifacts
from training.thresholds import select_prob_threshold
from training.metrics import summary_stats
from sklearn.metrics import roc_auc_score


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train base XGBoost classifier with walk-forward CV and probability calibration")
    ap.add_argument("--data", default="datasets/market_btcusdt_1m_2024_2025.parquet")
    ap.add_argument("--out", default="models/base_xgb")
    ap.add_argument("--n-folds", type=int, default=6)
    ap.add_argument("--embargo-minutes", type=int, default=60)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--spread-col", default="hl_spread", help="Feature column to use as spread proxy for costs")
    ap.add_argument("--spread-scale", type=float, default=0.0, help="Scale factor applied to spread_col for additional costs")
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--fold-scheme", choices=["even", "calendar_month"], default="even")
    args = ap.parse_args(argv)

    df = load_parquet_dataset(args.data)
    df = ensure_labels(df)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Walk-forward evaluation
    oof_prob = np.zeros(len(df))
    oof_mask = np.zeros(len(df), dtype=bool)
    feat_cols_used: List[str] | None = None
    calib_for_artifact = None
    model_for_artifact = None
    last_val_idx = None

    for k, (tr_idx, va_idx) in enumerate(time_folds(df, n_folds=args.n_folds, embargo_minutes=args.embargo_minutes, scheme=args.fold_scheme)):
        fold = k + 1
        tr = df.iloc[tr_idx]
        va = df.iloc[va_idx]

        X_tr, y_tr, feat_cols = extract_features_labels(tr)
        X_va, y_va, _ = extract_features_labels(va)
        feat_cols_used = feat_cols

        booster = train_xgb(
            X_tr, y_tr,
            X_val=X_va, y_val=y_va,
            early_stopping_rounds=int(args.early_stopping_rounds),
        )
        calib_cv = calibrate(booster, X_va, y_va, method="isotonic")
        p_va = predict_proba(calib_cv, X_va)
        oof_prob[va_idx] = p_va
        oof_mask[va_idx] = True

        # Keep last fold artifacts for saving
        model_for_artifact = booster
        calib_for_artifact = calib_cv
        last_val_idx = va_idx

    # Build threshold and report on OOF predictions
    valid_idx = np.where(oof_mask)[0]
    ret_next = df.loc[valid_idx, "ret_next"]
    p = pd.Series(oof_prob[valid_idx], index=ret_next.index)
    spread_series = None
    if args.spread_scale != 0.0 and args.spread_col and args.spread_col in df.columns:
        spread_series = df.loc[valid_idx, args.spread_col]
    thr, rep = select_prob_threshold(
        ret_next,
        p,
        cost_bps=args.cost_bps,
        spread_series=spread_series,
        spread_scale=args.spread_scale,
        slippage_bps=args.slippage_bps,
    )

    # Additional fold-level metrics
    rep_extra: Dict[str, float] = {"oof_count": int(oof_mask.sum()), "oof_auc": float(roc_auc_score(df.loc[valid_idx, "y_dir"], p.values))}
    rep.update(rep_extra)

    # Save artifacts
    out_dir = Path(args.out)
    save_artifacts(out_dir, model_for_artifact, calib_for_artifact, feat_cols_used or [], float(thr), rep)
    print(json.dumps({"out_dir": str(out_dir), "threshold": float(thr), "report": rep}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
