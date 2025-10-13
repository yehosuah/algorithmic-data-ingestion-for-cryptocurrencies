#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, List
import os
import sys

import numpy as np
import pandas as pd
 
# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from training.data import load_parquet_dataset, ensure_labels
from training.model import extract_features_labels
from training.blender import train_blender, save_blender
from training.infer import load_base_predictor, predict_base, load_tcn_predictor, predict_tcn




def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a blender model combining base and TCN predictions with RSS features")
    ap.add_argument("--data", default="datasets/training_matrix_months_2025-08-09.parquet")
    ap.add_argument("--base-dir", default="models/base_xgb")
    ap.add_argument("--tcn-dir", default="models/tcn")
    ap.add_argument("--out", default="models/blender")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--spread-col", default="hl_spread")
    ap.add_argument("--spread-scale", type=float, default=0.0)
    ap.add_argument("--tcn-stride", type=int, default=30, help="Stride used when generating TCN predictions for blender dataset")
    args = ap.parse_args(argv)

    df = load_parquet_dataset(args.data)
    df = ensure_labels(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Base predictions over all rows
    calib_base, feat_cols = load_base_predictor(Path(args.base_dir))
    base_prob = predict_base(df, calib_base, feat_cols)
    df["base_prob"] = base_prob.values

    # TCN predictions (will be shorter due to window)
    model_tcn, calib_tcn, series_cols, scaler, window = load_tcn_predictor(Path(args.tcn_dir))
    tcn_df = predict_tcn(
        df,
        model_tcn,
        calib_tcn,
        series_cols,
        scaler,
        window,
        stride=max(1, int(args.tcn_stride)),
    )

    merged = df.copy()
    merged = merged.merge(tcn_df, on="timestamp", how="left")

    # Keep rows where both prob and labels exist; drop NA tcn_prob for training blender
    bset = merged.dropna(subset=["base_prob", "tcn_prob", "y_dir", "ret_next"]).reset_index(drop=True)

    spread_series = None
    if args.spread_scale != 0.0 and args.spread_col in bset.columns:
        spread_series = bset[args.spread_col]
    pipe, thr, rep, cols = train_blender(
        bset,
        cost_bps=args.cost_bps,
        spread_series=spread_series,
        spread_scale=args.spread_scale,
        slippage_bps=args.slippage_bps,
    )
    out_dir = Path(args.out)
    save_blender(out_dir, pipe, cols, thr, rep)

    print(json.dumps({"out_dir": str(out_dir), "threshold": float(thr), "report": rep, "features": cols}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
