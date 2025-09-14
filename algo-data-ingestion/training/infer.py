from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import json
import joblib
import numpy as np
import pandas as pd
import torch

from .tcn_model import TinyTCN


def load_base_predictor(base_dir: Path):
    calib = joblib.load(base_dir / "calibrator.joblib")
    feat_cols = json.loads((base_dir / "feature_list.json").read_text())
    return calib, feat_cols


def predict_base(df: pd.DataFrame, calib, feat_cols: List[str]) -> pd.Series:
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
        seg = vals[i:i + L, :].T
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

