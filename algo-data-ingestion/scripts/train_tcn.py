#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from training.data import load_parquet_dataset, ensure_labels, sliding_windows
from training.walkforward import time_folds
from training.tcn_model import train_tcn, calibrate_logits, save_tcn, TrainConfig
from training.thresholds import select_prob_threshold


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train tiny TCN on sliding windows from market features")
    ap.add_argument("--data", default="datasets/market_btcusdt_1m_2024_2025.parquet")
    ap.add_argument("--out", default="models/tcn")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument("--channels", default="32,32", help="Comma-separated channels per block, e.g., 32,32")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--n-folds", type=int, default=6)
    ap.add_argument("--embargo-minutes", type=int, default=60)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--spread-col", default="hl_spread")
    ap.add_argument("--spread-scale", type=float, default=0.0)
    ap.add_argument("--fold-scheme", choices=["even", "calendar_month"], default="even")
    args = ap.parse_args(argv)

    df = load_parquet_dataset(args.data)
    df = ensure_labels(df)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Build windows aligned to df
    X, y, ts, series_cols, scaler = sliding_windows(df, window=args.window)
    win_df = pd.DataFrame({
        "timestamp": ts,
        "y_dir": y,
        "ret_next": df.loc[df.index[-len(ts):], "ret_next"].reset_index(drop=True),
    })

    # OOF evaluation
    logits_oof = np.zeros(len(win_df))
    mask_oof = np.zeros(len(win_df), dtype=bool)
    calib_for_artifact = None
    model_for_artifact = None

    channels = tuple(int(x) for x in args.channels.split(",") if x.strip())
    tcfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    for k, (tr_idx, va_idx) in enumerate(time_folds(win_df, n_folds=args.n_folds, embargo_minutes=args.embargo_minutes, scheme=args.fold_scheme)):
        fold = k + 1
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        model, logits_tr, logits_va = train_tcn(
            X_tr, y_tr,
            val=(X_va, y_va),
            kernel_size=args.kernel_size,
            channels=channels,
            dropout=args.dropout,
            config=tcfg,
        )
        calib = calibrate_logits(logits_va, y_va, method="isotonic")
        p_va = calib.predict_proba(logits_va.reshape(-1, 1))[:, 1]
        logits_oof[va_idx] = p_va
        mask_oof[va_idx] = True

        model_for_artifact = model
        calib_for_artifact = calib

    # Enrich win_df with spread column aligned to window ends (if it exists in df)
    if args.spread_col and args.spread_col in df.columns:
        win_df[args.spread_col] = df.loc[df.index[-len(ts):], args.spread_col].reset_index(drop=True)

    valid_idx = np.where(mask_oof)[0]
    spread_series = None
    if args.spread_scale != 0.0 and args.spread_col in win_df.columns:
        spread_series = win_df.loc[valid_idx, args.spread_col]
    thr, rep = select_prob_threshold(
        win_df.loc[valid_idx, "ret_next"],
        pd.Series(logits_oof[valid_idx], index=valid_idx),
        cost_bps=args.cost_bps,
        spread_series=spread_series,
        spread_scale=args.spread_scale,
        slippage_bps=args.slippage_bps,
    )
    rep.update({"oof_count": int(mask_oof.sum())})

    out_dir = Path(args.out)
    save_tcn(out_dir, model_for_artifact, scaler, calib_for_artifact, series_cols, meta={
        "window": int(args.window),
        "kernel_size": int(args.kernel_size),
        "channels": [int(x) for x in channels],
        "dropout": float(args.dropout),
    })
    (out_dir / "threshold.json").write_text(json.dumps({"prob_threshold": float(thr)}))
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))

    print(json.dumps({"out_dir": str(out_dir), "threshold": float(thr), "report": rep}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
