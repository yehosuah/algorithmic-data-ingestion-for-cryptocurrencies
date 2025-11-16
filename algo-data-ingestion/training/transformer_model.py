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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.pooling = pooling
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        h = self.input_proj(x)
        h = h.transpose(0, 1)  # (seq_len, batch, dim)
        h = self.pos_encoder(h)
        h = self.encoder(h)
        if self.pooling == "cls":
            pooled = h[0]
        else:
            pooled = h.mean(dim=0)
        logits = self.head(pooled).squeeze(-1)
        return logits


class TransformerModel(BaseModel):
    def __init__(self, config: Optional[Dict] = None):
        cfg = {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "epochs": 5,
            "batch_size": 128,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "pooling": "mean",
            "calibration_method": "isotonic",
        }
        if config:
            cfg.update(config)
        self.config = cfg
        self.model: Optional[TransformerClassifier] = None
        self.calibrator = None
        self.n_features: Optional[int] = None

    def _iter_batches(self, X: np.ndarray, y: np.ndarray):
        bs = int(self.config.get("batch_size", 128))
        for start in range(0, len(y), bs):
            end = min(len(y), start + bs)
            xb = torch.tensor(X[start:end], dtype=torch.float32)
            yb = torch.tensor(y[start:end], dtype=torch.float32)
            yield xb, yb

    def fit(self, X_train, y_train, **kwargs):
        X_arr = np.asarray(X_train, dtype=float)
        if X_arr.ndim != 3:
            raise ValueError(f"TransformerModel expects 3D input; got shape {X_arr.shape}")
        y_arr = np.asarray(y_train).astype(float)
        self.n_features = X_arr.shape[2]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerClassifier(
            n_features=self.n_features,
            d_model=int(self.config.get("d_model", 64)),
            nhead=int(self.config.get("nhead", 4)),
            num_layers=int(self.config.get("num_layers", 2)),
            dim_feedforward=int(self.config.get("dim_feedforward", 128)),
            dropout=float(self.config.get("dropout", 0.1)),
            pooling=self.config.get("pooling", "mean"),
        ).to(device)

        opt = optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.get("lr", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )
        loss_fn = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(int(self.config.get("epochs", 5))):
            perm = np.random.permutation(len(y_arr))
            Xp, yp = X_arr[perm], y_arr[perm]
            for xb, yb in self._iter_batches(Xp, yp):
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

        self.model.eval()
        with torch.no_grad():
            logits_train = self.model(torch.tensor(X_arr, dtype=torch.float32, device=device)).cpu().numpy()
            val_data = kwargs.get("val_data")
            logits_val = None
            y_val_arr = None
            if val_data is not None:
                Xv, yv = val_data
                Xv_arr = np.asarray(Xv, dtype=float)
                y_val_arr = np.asarray(yv).astype(float)
                logits_val = self.model(torch.tensor(Xv_arr, dtype=torch.float32, device=device)).cpu().numpy()
        if logits_val is not None and y_val_arr is not None:
            self.calibrator = calibrate_logits(logits_val, y_val_arr, method=self.config.get("calibration_method", "isotonic"))
        elif logits_train is not None:
            self.calibrator = calibrate_logits(logits_train, y_arr, method=self.config.get("calibration_method", "isotonic"))
        return self

    def _logits(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TransformerModel not trained; call fit first.")
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D input; got shape {arr.shape}")
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
        torch.save(self.model.state_dict(), out_dir / "transformer.pt")
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
            raise ValueError("Missing n_features in transformer metadata")
        model = TransformerClassifier(
            n_features=obj.n_features,
            d_model=int(obj.config.get("d_model", 64)),
            nhead=int(obj.config.get("nhead", 4)),
            num_layers=int(obj.config.get("num_layers", 2)),
            dim_feedforward=int(obj.config.get("dim_feedforward", 128)),
            dropout=float(obj.config.get("dropout", 0.1)),
            pooling=obj.config.get("pooling", "mean"),
        )
        state = torch.load(p / "transformer.pt", map_location="cpu")
        model.load_state_dict(state)
        obj.model = model
        calib_path = p / "calibrator.joblib"
        if calib_path.exists():
            obj.calibrator = joblib.load(calib_path)
        return obj


register_model("transformer", TransformerModel)
