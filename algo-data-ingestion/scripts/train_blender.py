#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
import torch

from training.data import load_parquet_dataset, ensure_labels
from training.model import extract_features_labels
from training.blender import train_blender, save_blender
from training.tcn_model import TinyTCN


def load_base_predictor(base_dir: Path):
    # Load prefit calibrator that wraps the XGB classifier
    calib = joblib.load(base_dir / "calibrator.joblib")
    feat_cols = json.loads((base_dir / "feature_list.json").read_text())
    return calib, feat_cols


def predict_base(df: pd.DataFrame, calib, feat_cols: List[str]) -> pd.Series:
    # Align features to training order; missing columns -> 0
    X = pd.DataFrame(index=df.index)
    for c in feat_cols:
        X[c] = df[c].astype(float) if c in df.columns else 0.0
    p = calib.predict_proba(X.values)[:, 1]
    return pd.Series(p, index=df.index, name="base_prob")


def load_tcn_predictor(tcn_dir: Path):
    meta = json.loads((tcn_dir / "tcn_meta.json").read_text())
    pre = joblib.load(tcn_dir / "tcn_preproc.joblib")
    calib = joblib.load(tcn_dir / "tcn_calibrator.joblib")
    series_cols = pre["series_cols"]
    scaler = pre["scaler"]
    channels = tuple(int(x) for x in meta.get("channels", [32, 32]))
    kernel_size = int(meta.get("kernel_size", 3))
    window = int(meta.get("window", 32))
    model = TinyTCN(n_inputs=len(series_cols), channels=channels, kernel_size=kernel_size)
    state = torch.load(tcn_dir / "tcn.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, calib, series_cols, scaler, window


def predict_tcn(df: pd.DataFrame, model: TinyTCN, calib, series_cols: List[str], scaler, window: int) -> pd.DataFrame:
    # Build sliding windows strictly on provided df
    vals = df[series_cols].astype(float).values
    if scaler is not None:
        try:
            vals = scaler.transform(vals)
        except Exception:
            pass
    n, c = vals.shape
    L = window
    N = n - L
    if N <= 0:
        return pd.DataFrame(columns=["timestamp", "tcn_prob"])  # not enough data
    X = np.empty((N, c, L), dtype=np.float32)
    for i in range(N):
        seg = vals[i:i+L, :].T
        m = seg.mean(axis=1, keepdims=True)
        s = seg.std(axis=1, keepdims=True) + 1e-6
        seg = (seg - m) / s
        X[i] = seg

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).view(-1).cpu().numpy()
    p = calib.predict_proba(logits.reshape(-1, 1))[:, 1]
    ts = pd.to_datetime(df["timestamp"], utc=True).iloc[L:].reset_index(drop=True)
    out = pd.DataFrame({"timestamp": ts, "tcn_prob": p})
    return out


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
    args = ap.parse_args(argv)

    df = load_parquet_dataset(args.data)
    df = ensure_labels(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Base predictions over all rows
    calib_base, feat_cols = load_base_predictor(Path(args.base_dir))
    base_prob = predict_base(df, calib_base, feat_cols)

    # TCN predictions (will be shorter due to window)
    model_tcn, calib_tcn, series_cols, scaler, window = load_tcn_predictor(Path(args.tcn_dir))
    tcn_df = predict_tcn(df, model_tcn, calib_tcn, series_cols, scaler, window)

    merged = df.copy()
    merged["base_prob"] = base_prob.values
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
