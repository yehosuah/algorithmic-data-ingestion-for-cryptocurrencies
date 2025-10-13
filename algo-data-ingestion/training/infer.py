from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import json
import copy
import joblib
import numpy as np
import pandas as pd
import torch

from .tcn_model import TinyTCN


DEFAULT_GATE_CONFIG: Dict[str, Any] = {
    "spread_column": "hl_spread",
    "prob_column": "base_prob",
    "training": {
        "hl_spread_max": None,
        "hl_spread_z_max": 0.25,
        "rvol20_max": 2e-4,
        "prob_gate_min": None,
        "min_hold_bars": 10,
        "long_only": True,
    },
    "inference": {
        "hl_spread_max": 5e-4,
        "hl_spread_z_max": -0.6,
        "rvol20_max": 4e-5,
        "prob_gate_min": 0.85,
        "min_hold_bars": 10,
        "long_only": True,
    },
}


def load_base_predictor(base_dir: Path):
    feat_cols = json.loads((base_dir / "feature_list.json").read_text())
    calib_path = base_dir / "calibrator.joblib"
    if calib_path.exists():
        calib = joblib.load(calib_path)
    else:
        # Load raw booster model if calibrator absent
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(base_dir / "model.json"))
        calib = xgb.XGBClassifier()
        calib._Booster = booster
        calib._le = None
        calib.n_features_in_ = len(feat_cols)
        calib.classes_ = np.array([0, 1])
    return calib, feat_cols


def load_gate_config(base_dir: Path) -> Dict[str, Any]:
    """
    Load gate configuration stored in a manifest; fall back to defaults when absent.
    """
    manifest_path = base_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            gates = manifest.get("gates")
            if isinstance(gates, dict):
                return gates
        except (json.JSONDecodeError, OSError):
            pass
    return copy.deepcopy(DEFAULT_GATE_CONFIG)


def predict_base(df: pd.DataFrame, calib, feat_cols: List[str]) -> pd.Series:
    X = pd.DataFrame(index=df.index)
    for c in feat_cols:
        X[c] = df[c].astype(float) if c in df.columns else 0.0
    p = calib.predict_proba(X.values)[:, 1]
    return pd.Series(p, index=df.index, name="base_prob")


def compute_gate_mask(
    df: pd.DataFrame,
    gate_config: Optional[Dict[str, Any]] = None,
    *,
    prob: Optional[pd.Series] = None,
    mode: str = "inference",
) -> pd.Series:
    """
    Compute a boolean mask indicating which rows satisfy the configured trade gate.

    Parameters
    ----------
    df:
        Feature frame containing at least the spread/volatility fields referenced by the gate.
    gate_config:
        Dictionary shaped like DEFAULT_GATE_CONFIG; when omitted, defaults are used.
    prob:
        Optional probability series to evaluate the probability gate; defaults to df[prob_column].
    mode:
        Either "inference" or "training" to pick the appropriate sub-gate.
    """
    cfg = gate_config or DEFAULT_GATE_CONFIG
    gate = cfg.get(mode) or {}
    mask = pd.Series(True, index=df.index, dtype=bool)

    spread_col = cfg.get("spread_column")
    if spread_col and gate.get("hl_spread_max") is not None and spread_col in df.columns:
        try:
            mask &= df[spread_col].astype(float) <= float(gate["hl_spread_max"])
        except Exception:
            mask &= False

    if gate.get("hl_spread_z_max") is not None and "hl_spread_z" in df.columns:
        try:
            mask &= df["hl_spread_z"].astype(float) <= float(gate["hl_spread_z_max"])
        except Exception:
            mask &= False

    if gate.get("rvol20_max") is not None and "rvol_20" in df.columns:
        try:
            mask &= df["rvol_20"].astype(float) <= float(gate["rvol20_max"])
        except Exception:
            mask &= False

    prob_col = cfg.get("prob_column", "base_prob")
    prob_threshold = gate.get("prob_gate_min")
    if prob_threshold is not None:
        if prob is None:
            if prob_col not in df.columns:
                raise KeyError(f"Probability column '{prob_col}' required for gate evaluation")
            prob_series = df[prob_col]
        else:
            prob_series = prob
        if not isinstance(prob_series, pd.Series):
            prob_series = pd.Series(prob_series, index=df.index)
        else:
            prob_series = prob_series.reindex(df.index)
        mask &= prob_series.astype(float) >= float(prob_threshold)

    return mask.fillna(False)


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


def predict_tcn(df: pd.DataFrame, model: TinyTCN, calib, series_cols: List[str], scaler, window: int, *, stride: int = 1) -> pd.DataFrame:
    series_df = df[series_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    series_df = series_df.ffill().bfill().fillna(0.0)
    vals = series_df.values
    if scaler is not None:
        try:
            vals = scaler.transform(vals)
        except Exception:
            pass
    n, c = vals.shape
    L = window
    stride = max(1, int(stride))
    starts = list(range(0, max(0, n - L), stride))
    N = len(starts)
    if N == 0:
        return pd.DataFrame(columns=["timestamp", "tcn_prob"])  # not enough data
    X = np.empty((N, c, L), dtype=np.float32)
    ts_idx = []
    for i, start in enumerate(starts):
        seg = vals[start:start + L, :].T
        m = seg.mean(axis=1, keepdims=True)
        s = seg.std(axis=1, keepdims=True) + 1e-6
        seg = (seg - m) / s
        X[i] = seg
        ts_idx.append(start + L)

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).view(-1).cpu().numpy()
    p = calib.predict_proba(logits.reshape(-1, 1))[:, 1]
    ts = pd.to_datetime(df["timestamp"], utc=True).iloc[ts_idx].reset_index(drop=True)
    out = pd.DataFrame({"timestamp": ts, "tcn_prob": p})
    return out
