from __future__ import annotations
from dataclasses import dataclass
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


def train_tcn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val: Optional[Tuple[np.ndarray, np.ndarray]] = None,
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
    for epoch in range(config.epochs):
        perm = np.random.permutation(len(y))
        Xp, yp = X[perm], y[perm]
        total = 0.0
        for xb, yb in _iter_batches(Xp, yp):
            opt.zero_grad()
            logits = model(xb).view(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu().item()) * len(yb)
        avg_loss = total / max(1, len(y))
        if progress_cb is not None:
            try:
                progress_cb(epoch + 1, float(avg_loss))
            except Exception:
                pass

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
