from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .thresholds import select_prob_threshold


BLENDER_NON_FEAT = {"timestamp", "y_dir", "ret_next", "dt", "symbol", "exchange", "timeframe", "feature_version", "close"}


def build_blender_features(df: pd.DataFrame, *, candidate_cols: List[str] | None = None) -> Tuple[pd.DataFrame, List[str]]:
    if candidate_cols is None:
        candidate_cols = [
            "base_prob", "tcn_prob",
            "rss_count", "rss_sent_mean",
            "reddit_count", "reddit_sent_mean",
            "rvol_5", "rvol_20",
        ]
    cols = [c for c in candidate_cols if c in df.columns]
    if not cols:
        raise ValueError("No blender feature columns found. Provide candidate_cols or ensure columns exist.")
    X = df[cols].astype(float)
    return X, cols


def train_blender(
    df: pd.DataFrame,
    *,
    cost_bps: float = 5.0,
    spread_series: Optional[pd.Series] = None,
    spread_scale: float = 0.0,
    slippage_bps: float = 0.0,
) -> Tuple[Pipeline, float, Dict, List[str]]:
    X, cols = build_blender_features(df)
    y = df["y_dir"].astype(int)
    # Simple logistic with scaling
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X.values, y.values)
    prob = pipe.predict_proba(X.values)[:, 1]
    thr, rep = select_prob_threshold(
        df["ret_next"],
        pd.Series(prob, index=df.index),
        cost_bps=cost_bps,
        spread_series=spread_series,
        spread_scale=spread_scale,
        slippage_bps=slippage_bps,
    )
    return pipe, thr, rep, cols


def save_blender(out_dir: Path, model: Pipeline, feat_cols: List[str], threshold: float, report: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "blender.joblib")
    (out_dir / "blender_features.txt").write_text("\n".join(feat_cols))
    (out_dir / "threshold.txt").write_text(str(float(threshold)))
    (out_dir / "report.json").write_text(__import__("json").dumps(report, indent=2))
