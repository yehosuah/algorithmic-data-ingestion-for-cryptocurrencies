from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model_registry import BaseModel, register_model


class StackingMetaModel(BaseModel):
    def __init__(self, config: Optional[Dict] = None):
        cfg = {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
        }
        if config:
            cfg.update(config)
        self.config = cfg
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    penalty=self.config.get("penalty", "l2"),
                    C=float(self.config.get("C", 1.0)),
                    solver=self.config.get("solver", "lbfgs"),
                    max_iter=int(self.config.get("max_iter", 1000)),
                )),
            ]
        )
        self.feature_names: Optional[list[str]] = None

    def fit(self, X_meta, y_meta, **kwargs):
        if isinstance(X_meta, pd.DataFrame):
            self.feature_names = list(X_meta.columns)
            X_arr = X_meta.to_numpy()
        else:
            X_arr = np.asarray(X_meta)
        y_arr = np.asarray(y_meta).astype(int)
        mask = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
        self.model.fit(X_arr[mask], y_arr[mask])
        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_names] if self.feature_names else X.to_numpy()
        else:
            X_arr = np.asarray(X)
        return self.model.predict_proba(X_arr)[:, 1]

    def save(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out / "stacking_meta.joblib")
        meta = {"config": self.config, "feature_names": self.feature_names}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str):
        p = Path(path)
        meta = {}
        if (p / "meta.json").exists():
            meta = json.loads((p / "meta.json").read_text())
        obj = cls(meta.get("config", {}))
        obj.feature_names = meta.get("feature_names")
        obj.model = joblib.load(p / "stacking_meta.joblib")
        return obj


register_model("stacking_meta", StackingMetaModel)
