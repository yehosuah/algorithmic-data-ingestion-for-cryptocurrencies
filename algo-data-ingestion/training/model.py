from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb


NON_FEAT = {"timestamp","dt","symbol","exchange","timeframe","feature_version","close","ret_next","y_dir"}


def extract_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    y = df["y_dir"].astype(int)
    feat_cols = [c for c in df.columns if c not in NON_FEAT]
    X = df[feat_cols].astype(float)
    return X, y, feat_cols


def train_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[Dict] = None,
    *,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    early_stopping_rounds: int = 50,
) -> xgb.XGBClassifier:
    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "logloss",
        }
    model = xgb.XGBClassifier(**params)
    if X_val is not None and y_val is not None and len(X_val) > 0:
        try:
            # xgboost < 2.0 supports early_stopping_rounds in fit
            model.fit(
                X.values, y.values,
                eval_set=[(X_val.values, y_val.values)],
                verbose=False,
                early_stopping_rounds=early_stopping_rounds,
            )
        except TypeError:
            # xgboost >= 2.0 uses callbacks API for early stopping
            cb = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)]
            model.fit(
                X.values, y.values,
                eval_set=[(X_val.values, y_val.values)],
                verbose=False,
                callbacks=cb,
            )
    else:
        model.fit(X.values, y.values)
    return model


def calibrate(model: xgb.XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series, method: str = "isotonic") -> CalibratedClassifierCV:
    calib = CalibratedClassifierCV(model, method=method, cv="prefit")
    calib.fit(X_val.values, y_val.values)
    return calib


def predict_proba(model_or_calib, X: pd.DataFrame) -> np.ndarray:
    p = model_or_calib.predict_proba(X.values)[:,1]
    return p


def save_artifacts(out_dir: Path, booster: xgb.XGBClassifier, calib: CalibratedClassifierCV, feat_cols: List[str], threshold: float, report: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.get_booster().save_model(str(out_dir/"model.json"))
    joblib.dump(calib, out_dir/"calibrator.joblib")
    (out_dir/"feature_list.json").write_text(json.dumps(feat_cols))
    (out_dir/"threshold.json").write_text(json.dumps({"prob_threshold": threshold}))
    (out_dir/"report.json").write_text(json.dumps(report, indent=2))
