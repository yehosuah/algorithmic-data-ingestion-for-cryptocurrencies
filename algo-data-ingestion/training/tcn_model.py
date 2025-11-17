from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from .model_registry import BaseModel, register_model
from .metrics import equity_curve, summary_stats
from .thresholds import select_prob_threshold


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TinyTCN(nn.Module):
    def __init__(self, n_inputs: int, channels: Tuple[int, int] = (32, 32), kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_ch = n_inputs
        dilation = 1
        for out_ch in channels:
            padding = (kernel_size - 1) * dilation
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation, padding=padding, dropout=dropout)]
            in_ch = out_ch
            dilation *= 2
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, max(16, in_ch // 2)),
            nn.ReLU(),
            nn.Linear(max(16, in_ch // 2), 1),
        )

    def forward(self, x):
        # x: (B, C, L)
        h = self.network(x)
        h = self.pool(h)
        logits = self.head(h).squeeze(-1)
        return logits


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    class_weight: Optional[float] = None  # weight for positive class
    early_stopping_patience: int = 5
    max_grad_norm: float = 1.0
    cost_bps: float = 5.0
    long_only: bool = False
    min_hold_bars: int = 1


def train_tcn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    val_returns: Optional[np.ndarray] = None,
    kernel_size: int = 3,
    channels: Tuple[int, int] = (32, 32),
    dropout: float = 0.05,
    config: TrainConfig = TrainConfig(),
    device: Optional[str] = None,
    progress_cb: Optional[Callable[[int, float], None]] = None,
) -> Tuple[TinyTCN, np.ndarray, Optional[np.ndarray]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTCN(X.shape[1], channels=channels, kernel_size=kernel_size, dropout=dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    pos_weight = None
    if config.class_weight is not None:
        # BCEWithLogitsLoss supports pos_weight tensor
        pos_weight = torch.tensor([float(config.class_weight)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _iter_batches(Xa, ya):
        n = len(ya)
        bs = config.batch_size
        idx = np.arange(n)
        for i in range(0, n, bs):
            j = min(n, i + bs)
            xb = torch.tensor(Xa[i:j], dtype=torch.float32, device=device)
            yb = torch.tensor(ya[i:j], dtype=torch.float32, device=device)
            yield xb, yb

    model.train()
    best_metric = -np.inf
    best_state = None
    patience = 0
    for epoch in range(config.epochs):
        perm = np.random.permutation(len(y))
        Xp, yp = X[perm], y[perm]
        total = 0.0
        for xb, yb in _iter_batches(Xp, yp):
            opt.zero_grad()
            logits = model(xb).view(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            if config.max_grad_norm and config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
            opt.step()
            total += float(loss.detach().cpu().item()) * len(yb)
        avg_loss = total / max(1, len(y))
        monitor_metric = -avg_loss
        if val is not None:
            Xv, yv = val
            with torch.no_grad():
                logits_val_tmp = model(torch.tensor(Xv, dtype=torch.float32, device=device)).view(-1).cpu().numpy()
            prob_val = 1.0 / (1.0 + np.exp(-np.clip(logits_val_tmp, -20, 20)))
            returns = val_returns if val_returns is not None else np.where(yv > 0.5, 1.0, -1.0)
            prob_series = pd.Series(prob_val, index=np.arange(len(prob_val)))
            ret_series = pd.Series(returns, index=np.arange(len(prob_val)))
            thr, _ = select_prob_threshold(
                ret_series,
                prob_series,
                cost_bps=config.cost_bps,
                long_only=config.long_only,
                min_hold_bars=config.min_hold_bars,
            )
            eq = equity_curve(
                ret_series,
                prob_series,
                threshold=thr,
                cost_bps=config.cost_bps,
                long_only=config.long_only,
                min_hold_bars=config.min_hold_bars,
            )
            stats = summary_stats(eq)
            monitor_metric = stats.get("sharpe", monitor_metric)

        if monitor_metric > best_metric:
            best_metric = monitor_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if progress_cb is not None:
            try:
                progress_cb(epoch + 1, float(avg_loss))
            except Exception:
                pass

        if config.early_stopping_patience and patience >= int(config.early_stopping_patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_train = model(torch.tensor(X, dtype=torch.float32, device=device)).view(-1).cpu().numpy()
        logits_val = None
        if val is not None:
            Xv, yv = val
            logits_val = model(torch.tensor(Xv, dtype=torch.float32, device=device)).view(-1).cpu().numpy()
    return model, logits_train, logits_val


def calibrate_logits(logits: np.ndarray, y: np.ndarray, method: str = "isotonic") -> CalibratedClassifierCV:
    # Use a simple LR with calibrator by wrapping logits as a single-feature input
    # Fit a dummy LR first then calibrate with preferred method
    base = LogisticRegression(max_iter=1000)
    z = logits.reshape(-1, 1)
    base.fit(z, y)
    calib = CalibratedClassifierCV(base, method=method, cv="prefit")
    calib.fit(z, y)
    return calib


def save_tcn(
    out_dir: Path,
    model: TinyTCN,
    scaler,
    calib: Optional[CalibratedClassifierCV],
    used_cols,
    meta: Optional[dict] = None,
    *,
    fold_logits: Optional[pd.DataFrame] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Torch model
    torch.save(model.state_dict(), out_dir / "tcn.pt")
    # Feature scaler + calibrator + series columns
    joblib.dump({"scaler": scaler, "series_cols": used_cols}, out_dir / "tcn_preproc.joblib")
    if calib is not None:
        joblib.dump(calib, out_dir / "tcn_calibrator.joblib")
    if meta is not None:
        (out_dir / "tcn_meta.json").write_text(__import__("json").dumps(meta, indent=2))
    if fold_logits is not None and len(fold_logits):
        df_logits = pd.DataFrame(fold_logits).copy()
        if "timestamp" in df_logits.columns:
            df_logits["timestamp"] = pd.to_datetime(df_logits["timestamp"], utc=True)
        df_logits.to_parquet(out_dir / "fold_logits.parquet", index=False)


class TCNModel(BaseModel):
    """
    Registry-friendly wrapper for the TinyTCN.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = {
            "kernel_size": 3,
            "channels": (32, 32),
            "dropout": 0.05,
            "epochs": 10,
            "batch_size": 256,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "class_weight": None,
            "calibration_method": "isotonic",
            "early_stopping_patience": 5,
            "max_grad_norm": 1.0,
            "cost_bps": 5.0,
            "long_only": False,
            "min_hold_bars": 1,
        }
        if config:
            cfg.update(config)
        # normalize channels
        ch = cfg.get("channels", (32, 32))
        if isinstance(ch, str):
            ch = tuple(int(x) for x in ch.split(",") if x.strip())
        cfg["channels"] = tuple(int(x) for x in ch)
        self.config = cfg
        self.model: Optional[TinyTCN] = None
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.n_inputs: Optional[int] = None

    def _ensure_ch_last_to_ch_first(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D input (N, L, F) or (N, C, L); got shape {arr.shape}")
        # Heuristic: if middle dim is small it is probably channels-first already
        return arr if arr.shape[1] <= arr.shape[2] else np.transpose(arr, (0, 2, 1))

    def fit(self, X_train, y_train, **kwargs):
        X_arr = self._ensure_ch_last_to_ch_first(np.asarray(X_train))
        y_arr = np.asarray(y_train).astype(np.float32)
        self.n_inputs = X_arr.shape[1]

        val_data = kwargs.get("val_data")
        val_tuple = None
        if val_data is not None:
            X_val, y_val = val_data
            X_val_arr = self._ensure_ch_last_to_ch_first(np.asarray(X_val))
            y_val_arr = np.asarray(y_val).astype(np.float32)
            val_tuple = (X_val_arr, y_val_arr)

        tcfg = TrainConfig(
            epochs=int(self.config.get("epochs", 10)),
            lr=float(self.config.get("lr", 1e-3)),
            batch_size=int(self.config.get("batch_size", 256)),
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
            class_weight=self.config.get("class_weight"),
            early_stopping_patience=int(self.config.get("early_stopping_patience", 5)),
            max_grad_norm=float(self.config.get("max_grad_norm", 1.0)),
            cost_bps=float(self.config.get("cost_bps", 5.0)),
            long_only=bool(self.config.get("long_only", False)),
            min_hold_bars=int(self.config.get("min_hold_bars", 1)),
        )
        model, logits_train, logits_val = train_tcn(
            X_arr,
            y_arr,
            val=val_tuple,
            val_returns=kwargs.get("val_returns"),
            kernel_size=int(self.config.get("kernel_size", 3)),
            channels=tuple(self.config.get("channels", (32, 32))),
            dropout=float(self.config.get("dropout", 0.05)),
            config=tcfg,
        )
        self.model = model
        self.calibrator = None
        calib_method = self.config.get("calibration_method", "isotonic")
        if val_tuple is not None and logits_val is not None:
            self.calibrator = calibrate_logits(logits_val, val_tuple[1], method=calib_method)
        elif logits_train is not None:
            # Fallback: calibrate on train logits if no validation provided
            self.calibrator = calibrate_logits(logits_train, y_arr, method=calib_method)
        return self

    def _logits(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TCNModel not trained; call fit first.")
        X_arr = self._ensure_ch_last_to_ch_first(np.asarray(X))
        device = next(self.model.parameters()).device
        with torch.no_grad():
            logits = self.model(torch.tensor(X_arr, dtype=torch.float32, device=device)).view(-1).cpu().numpy()
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
        torch.save(self.model.state_dict(), out_dir / "model.pt")
        meta = {
            "config": self.config,
            "n_inputs": self.n_inputs,
        }
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
        obj.n_inputs = meta.get("n_inputs")
        if obj.n_inputs is None:
            raise ValueError("Missing n_inputs in TCN metadata; cannot load model.")
        model = TinyTCN(
            obj.n_inputs,
            channels=tuple(obj.config.get("channels", (32, 32))),
            kernel_size=int(obj.config.get("kernel_size", 3)),
            dropout=float(obj.config.get("dropout", 0.05)),
        )
        state_path = p / "model.pt"
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        obj.model = model
        calib_path = p / "calibrator.joblib"
        if calib_path.exists():
            obj.calibrator = joblib.load(calib_path)
        return obj


register_model("tcn", TCNModel)
