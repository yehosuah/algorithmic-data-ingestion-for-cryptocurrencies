from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model_registry import BaseModel, register_model


class RegimeBlender(BaseModel):
    """
    Regime-aware blender that learns separate weights per regime id.
    """

    def __init__(self, base_model_names: Optional[List[str]] = None, config: Optional[Dict] = None):
        self.base_model_names = base_model_names or []
        self.config = config or {}
        self.regime_models: Dict[str, Pipeline] = {}
        self.global_model: Optional[Pipeline] = None

    def _stack(self, base_preds: Dict[str, np.ndarray]) -> np.ndarray:
        used = self.base_model_names or list(base_preds.keys())
        missing = [m for m in used if m not in base_preds]
        if missing:
            raise KeyError(f"Missing base predictions for {missing}")
        cols = [np.asarray(base_preds[m]).reshape(-1) for m in used]
        return np.column_stack(cols)

    def _make_model(self) -> Pipeline:
        penalty = self.config.get("penalty", "l2")
        C = float(self.config.get("C", 1.0))
        solver = self.config.get("solver", "lbfgs")
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, penalty=penalty, solver=solver, C=C)),
            ]
        )

    def fit(self, base_preds: Dict[str, np.ndarray], y_true: np.ndarray, regime_ids: np.ndarray):
        X = self._stack(base_preds)
        y = np.asarray(y_true).astype(int)
        regimes = np.asarray(regime_ids).astype(str)
        mask_all = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X_all, y_all, regimes_all = X[mask_all], y[mask_all], regimes[mask_all]
        if len(y_all) == 0:
            raise ValueError("No valid samples to fit RegimeBlender")
        self.global_model = self._make_model()
        self.global_model.fit(X_all, y_all)
        for regime in np.unique(regimes_all):
            m = regimes_all == regime
            if m.sum() < 10:
                continue
            mdl = self._make_model()
            mdl.fit(X_all[m], y_all[m])
            self.regime_models[regime] = mdl
        return self

    def predict_proba(self, base_preds: Dict[str, np.ndarray], regime_ids: np.ndarray):
        X = self._stack(base_preds)
        regimes = np.asarray(regime_ids).astype(str)
        out = np.zeros(len(regimes), dtype=float)
        for i, r in enumerate(regimes):
            mdl = self.regime_models.get(r, self.global_model)
            if mdl is not None:
                out[i] = mdl.predict_proba(X[i : i + 1])[:, 1][0]
            else:
                out[i] = float(np.mean(X[i]))
        return out

    def save(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({"global": self.global_model, "per_regime": self.regime_models}, out / "regime_blender.joblib")
        meta = {"base_model_names": self.base_model_names, "config": self.config}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str):
        p = Path(path)
        meta = {}
        if (p / "meta.json").exists():
            meta = json.loads((p / "meta.json").read_text())
        obj = cls(meta.get("base_model_names"), meta.get("config"))
        payload = joblib.load(p / "regime_blender.joblib")
        obj.global_model = payload.get("global")
        obj.regime_models = payload.get("per_regime", {})
        return obj


register_model("regime_blender", RegimeBlender)
