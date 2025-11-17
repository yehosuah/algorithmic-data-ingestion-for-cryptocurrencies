from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model_registry import BaseModel, register_model
from .tcn_model import calibrate_logits
from .metrics import equity_curve, summary_stats
from .thresholds import select_prob_threshold


class DeepLOBNet(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 5), padding=(1, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        h = x.unsqueeze(1)  # (B,1,L,F)
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        h = self.dropout(self.act(self.conv3(h)))
        h = self.pool(h)
        logits = self.head(h).squeeze(-1)
        return logits.squeeze(-1)


class DeepLOBModel(BaseModel):
    def __init__(self, config: Optional[Dict] = None):
        cfg = {
            "epochs": 5,
            "batch_size": 128,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "dropout": 0.1,
            "calibration_method": "isotonic",
            "early_stopping_patience": 3,
            "cost_bps": 5.0,
            "long_only": False,
            "min_hold_bars": 1,
            "max_grad_norm": 1.0,
        }
        if config:
            cfg.update(config)
        self.config = cfg
        self.model: Optional[DeepLOBNet] = None
        self.calibrator = None
        self.n_features: Optional[int] = None

    def _iter_batches(self, X: np.ndarray, y: np.ndarray):
        bs = int(self.config.get("batch_size", 128))
        for start in range(0, len(y), bs):
            end = min(len(y), start + bs)
            yield X[start:end], y[start:end]

    def fit(self, X_train, y_train, **kwargs):
        X_arr = np.asarray(X_train, dtype=float)
        if X_arr.ndim != 3:
            raise ValueError(f"DeepLOBModel expects 3D input; got {X_arr.shape}")
        y_arr = np.asarray(y_train).astype(float)
        self.n_features = X_arr.shape[2]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_returns = kwargs.get("val_returns")
        val_data = kwargs.get("val_data")
        val_tuple = None
        y_val_arr = None
        if val_data is not None:
            Xv, yv = val_data
            Xv_arr = np.asarray(Xv, dtype=float)
            y_val_arr = np.asarray(yv).astype(float)
            val_tuple = (Xv_arr, y_val_arr)
        self.model = DeepLOBNet(self.n_features, dropout=float(self.config.get("dropout", 0.1))).to(device)
        opt = optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.get("lr", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )
        loss_fn = nn.BCEWithLogitsLoss()

        max_epochs = int(self.config.get("epochs", 5))
        patience = int(self.config.get("early_stopping_patience", 0) or 0)
        best_state = None
        best_metric = -np.inf
        patience_ctr = 0
        self.model.train()
        for epoch in range(max_epochs):
            perm = np.random.permutation(len(y_arr))
            Xp, yp = X_arr[perm], y_arr[perm]
            total_loss = 0.0
            for xb, yb in self._iter_batches(Xp, yp):
                xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
                yb_t = torch.tensor(yb, dtype=torch.float32, device=device)
                opt.zero_grad()
                logits = self.model(xb_t)
                loss = loss_fn(logits, yb_t)
                loss.backward()
                max_norm = float(self.config.get("max_grad_norm", 1.0) or 0.0)
                if max_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                opt.step()
                total_loss += float(loss.detach().cpu().item()) * len(yb)

            avg_loss = total_loss / max(1, len(y_arr))
            monitor_metric = -avg_loss
            if val_tuple is not None:
                Xv_arr, yv_arr = val_tuple
                with torch.no_grad():
                    logits_val_tmp = self.model(torch.tensor(Xv_arr, dtype=torch.float32, device=device)).cpu().numpy()
                prob_val = 1.0 / (1.0 + np.exp(-np.clip(logits_val_tmp, -20, 20)))
                returns = val_returns if val_returns is not None else np.where(yv_arr > 0.5, 1.0, -1.0)
                prob_series = pd.Series(prob_val, index=np.arange(len(prob_val)))
                ret_series = pd.Series(returns, index=np.arange(len(prob_val)))
                thr, _ = select_prob_threshold(
                    ret_series,
                    prob_series,
                    cost_bps=float(self.config.get("cost_bps", 5.0)),
                    long_only=bool(self.config.get("long_only", False)),
                    min_hold_bars=int(self.config.get("min_hold_bars", 1)),
                )
                eq = equity_curve(
                    ret_series,
                    prob_series,
                    threshold=thr,
                    cost_bps=float(self.config.get("cost_bps", 5.0)),
                    long_only=bool(self.config.get("long_only", False)),
                    min_hold_bars=int(self.config.get("min_hold_bars", 1)),
                )
                stats = summary_stats(eq)
                monitor_metric = stats.get("sharpe", monitor_metric)

            if monitor_metric > best_metric:
                best_metric = monitor_metric
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience and patience_ctr >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        with torch.no_grad():
            logits_train = self.model(torch.tensor(X_arr, dtype=torch.float32, device=device)).cpu().numpy()
            logits_val = None
            if val_tuple is not None:
                logits_val = self.model(torch.tensor(val_tuple[0], dtype=torch.float32, device=device)).cpu().numpy()
        if logits_val is not None and y_val_arr is not None:
            self.calibrator = calibrate_logits(logits_val, y_val_arr, method=self.config.get("calibration_method", "isotonic"))
        elif logits_train is not None:
            self.calibrator = calibrate_logits(logits_train, y_arr, method=self.config.get("calibration_method", "isotonic"))
        return self

    def _logits(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DeepLOBModel not trained; call fit first.")
        arr = np.asarray(X, dtype=float)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            logits = self.model(torch.tensor(arr, dtype=torch.float32, device=device)).cpu().numpy()
        return logits

    def predict_proba(self, X):
        logits = self._logits(np.asarray(X))
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        if self.calibrator is not None:
            prob = self.calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        return prob

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / "deeplob.pt")
        meta = {"config": self.config, "n_features": self.n_features}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if self.calibrator is not None:
            joblib.dump(self.calibrator, out_dir / "calibrator.joblib")

    @classmethod
    def load(cls, path: str):
        p = Path(path)
        meta = {}
        if (p / "meta.json").exists():
            meta = json.loads((p / "meta.json").read_text())
        obj = cls(meta.get("config", {}))
        obj.n_features = meta.get("n_features")
        if obj.n_features is None:
            raise ValueError("Missing n_features in DeepLOB metadata")
        model = DeepLOBNet(obj.n_features, dropout=float(obj.config.get("dropout", 0.1)))
        state = torch.load(p / "deeplob.pt", map_location="cpu")
        model.load_state_dict(state)
        obj.model = model
        calib_path = p / "calibrator.joblib"
        if calib_path.exists():
            obj.calibrator = joblib.load(calib_path)
        return obj


register_model("deeplob", DeepLOBModel)
