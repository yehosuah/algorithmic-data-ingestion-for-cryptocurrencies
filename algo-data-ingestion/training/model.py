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

from .reporting import ensure_kpi_schema
from .model_registry import BaseModel, register_model

NON_FEAT = {"timestamp","dt","symbol","exchange","timeframe","feature_version","close","ret_next","y_dir"}
_NON_FEAT_PREFIXES = ("ret_next",)


def _is_feature_column(name: str) -> bool:
    if name in NON_FEAT:
        return False
    return not any(name.startswith(prefix) for prefix in _NON_FEAT_PREFIXES)


def extract_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    y = df["y_dir"].astype(int)
    feat_cols = [c for c in df.columns if _is_feature_column(c)]
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
    sample_weight: Optional[np.ndarray] = None,
) -> xgb.XGBClassifier:
    default_params = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "logloss",
        "min_child_weight": 1.0,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    }
    if params is None:
        params = default_params
    else:
        cfg = default_params.copy()
        cfg.update(params)
        params = cfg

    model = xgb.XGBClassifier(**params)
    fit_kwargs: Dict = {"verbose": False}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    if X_val is not None and y_val is not None and len(X_val) > 0:
        fit_kwargs["eval_set"] = [(X_val.values, y_val.values)]
        if early_stopping_rounds and early_stopping_rounds > 0:
            fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)

    model.fit(X.values, y.values, **fit_kwargs)
    return model


def calibrate(model: xgb.XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series, method: str = "isotonic") -> CalibratedClassifierCV:
    calib = CalibratedClassifierCV(model, method=method, cv="prefit")
    calib.fit(X_val.values, y_val.values)
    return calib


def predict_proba(model_or_calib, X: pd.DataFrame) -> np.ndarray:
    p = model_or_calib.predict_proba(X.values)[:,1]
    return p


def save_artifacts(out_dir: Path, booster: xgb.XGBClassifier, calib: CalibratedClassifierCV | None, feat_cols: List[str], threshold: float, report: Dict, gate_config: Optional[Dict] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.get_booster().save_model(str(out_dir/"model.json"))
    if calib is not None:
        joblib.dump(calib, out_dir/"calibrator.joblib")
    elif (out_dir/"calibrator.joblib").exists():
        (out_dir/"calibrator.joblib").unlink()
    (out_dir/"feature_list.json").write_text(json.dumps(feat_cols))
    (out_dir/"threshold.json").write_text(json.dumps({"prob_threshold": threshold}))
    report_payload = ensure_kpi_schema(report)
    (out_dir/"report.json").write_text(json.dumps(report_payload, indent=2))
    manifest = {
        "model_path": "model.json",
        "calibrator_path": "calibrator.joblib" if calib is not None else None,
        "feature_list_path": "feature_list.json",
        "threshold": {
            "value": float(threshold),
            "path": "threshold.json",
        },
        "report_path": "report.json",
        "gates": gate_config or {},
        "metadata": {
            "model_type": "xgboost_classifier",
            "calibrated": bool(calib is not None),
        },
    }
    (out_dir/"manifest.json").write_text(json.dumps(manifest, indent=2))


class XGBModel(BaseModel):
    """
    Thin wrapper around the existing XGBoost workflow, suitable for the model registry.
    """

    def __init__(self, config: Optional[Dict] = None):
        default_params = {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "logloss",
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "early_stopping_rounds": None,
        }
        self.config: Dict = default_params.copy()
        if config:
            self.config.update(config)
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_cols: Optional[List[str]] = None

    def fit(self, X_train, y_train, **kwargs):
        X_df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        self.feature_cols = list(X_df.columns)
        params = self.config.copy()

        model = xgb.XGBClassifier(**params)
        fit_kwargs: Dict = {"verbose": False}
        sample_weight = kwargs.get("sample_weight")
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        val_data = kwargs.get("val_data")
        if val_data is not None:
            X_val, y_val = val_data
            fit_kwargs["eval_set"] = [(np.asarray(X_val), np.asarray(y_val))]
            # early_stopping_rounds can be provided via model params; avoid duplicate kwarg

        model.fit(X_df.values, np.asarray(y_train), **fit_kwargs)
        self.model = model
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("XGBModel not trained; call fit first.")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_cols)
        if self.feature_cols:
            X_df = X_df[self.feature_cols]
        return self.model.predict_proba(X_df.values)[:, 1]

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.get_booster().save_model(str(out_dir / "model.json"))
        meta = {
            "config": self.config,
            "feature_cols": self.feature_cols,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str):
        p = Path(path)
        meta = {}
        if (p / "meta.json").exists():
            meta = json.loads((p / "meta.json").read_text())
        obj = cls(meta.get("config", {}))
        model = xgb.XGBClassifier(**obj.config)
        model.load_model(str(p / "model.json"))
        obj.model = model
        obj.feature_cols = meta.get("feature_cols")
        return obj


# Register the model for downstream scripts
register_model("xgb", XGBModel)
