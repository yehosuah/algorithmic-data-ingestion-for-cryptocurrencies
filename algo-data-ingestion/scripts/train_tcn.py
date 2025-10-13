#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, List
import os
import sys

import numpy as np
import pandas as pd

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from training.data import load_parquet_dataset, ensure_labels, sliding_windows
from training.feature_eng import augment_market_features
from training.walkforward import time_folds
from training.tcn_model import train_tcn, calibrate_logits, save_tcn, TrainConfig
from training.thresholds import select_prob_threshold
from training.infer import load_base_predictor, predict_base, compute_gate_mask
from training.metrics import equity_curve, summary_stats


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
    ap.add_argument("--stride", type=int, default=1, help="Stride for sliding windows (>=1)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--class-weight", type=float, default=None, help="Optional positive-class weight for BCE loss")
    ap.add_argument("--n-folds", type=int, default=6)
    ap.add_argument("--embargo-minutes", type=int, default=60)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--spread-col", default="hl_spread")
    ap.add_argument("--spread-scale", type=float, default=0.0)
    ap.add_argument("--fold-scheme", choices=["even", "calendar_month"], default="even")
    ap.add_argument("--series-cols", default=None, help="Comma-separated feature columns for windows; defaults to curated list")
    ap.add_argument("--max-spread", type=float, default=None, help="Optional absolute spread ceiling to allow trades")
    ap.add_argument("--max-spread-z", type=float, default=0.25, help="Optional hl_spread_z ceiling to allow trades (default relaxed training gate)")
    ap.add_argument("--max-rvol20", type=float, default=2e-4, help="Optional rvol_20 ceiling to allow trades (default relaxed training gate)")
    ap.add_argument("--long-only", action="store_true", help="Use long-only rule when computing equity")
    ap.add_argument("--diagnostic-thresholds", default=None, help="Comma-separated thresholds to log additional equity diagnostics")
    ap.add_argument("--min-hold-bars", type=int, default=1, help="Minimum bars to hold a position once opened (>=1) during evaluation")
    ap.add_argument("--threshold-criterion", choices=["final_equity", "sharpe"], default="final_equity", help="Objective used to pick the probability threshold")
    ap.add_argument("--min-total-turnover", type=float, default=2.0, help="Minimum total turnover required when selecting thresholds")
    ap.add_argument("--label-threshold-bps", type=float, default=None, help="Optional label barrier: set y=1 if ret_next > threshold (bps)")
    ap.add_argument("--base-dir", default=None, help="Optional path to trained base model; adds base_prob to TCN series inputs")
    ap.add_argument("--horizon", type=int, default=1, help="Prediction horizon in bars for labels and PnL evaluation")
    ap.add_argument("--max-total-turnover", type=float, default=None, help="Maximum total turnover allowed when selecting thresholds")
    ap.add_argument("--inference-max-spread", type=float, default=5e-4, help="Inference gate: absolute spread ceiling in live trading")
    ap.add_argument("--inference-max-spread-z", type=float, default=-0.6, help="Inference gate: z-score ceiling using hl_spread_z")
    ap.add_argument("--inference-max-rvol20", type=float, default=4e-5, help="Inference gate: rvol_20 ceiling")
    ap.add_argument("--inference-prob-gate", type=float, default=0.85, help="Inference gate: minimum calibrated probability before thresholding")
    ap.add_argument("--inference-min-hold-bars", type=int, default=10, help="Inference gate: minimum hold bars constraint")
    args = ap.parse_args(argv)

    print(f"[TCN] Loading dataset: {args.data}", flush=True)
    df = load_parquet_dataset(args.data)
    df = ensure_labels(df)
    df = augment_market_features(df)
    df = df.sort_values("timestamp").reset_index(drop=True)
    horizon = max(1, int(args.horizon))
    ret_col = "ret_next"
    if horizon > 1:
        if "close" not in df.columns:
            raise ValueError("Dataset missing close column required for multi-step horizon computation")
        df[f"ret_next_{horizon}"] = df["close"].pct_change(horizon).shift(-horizon)
        ret_col = f"ret_next_{horizon}"
        df["y_dir"] = (df[ret_col] > 0).astype(int)
    if args.label_threshold_bps is not None:
        thr_val = float(args.label_threshold_bps) / 1e4
        if ret_col not in df.columns:
            raise ValueError("Dataset missing return column for label thresholding")
        df["y_dir"] = (df[ret_col] > thr_val).astype(int)

    df = df.dropna(subset=[ret_col]).reset_index(drop=True)

    if args.base_dir:
        base_path = Path(args.base_dir)
        calib_base, feat_cols_base = load_base_predictor(base_path)
        base_prob = predict_base(df, calib_base, feat_cols_base)
        df["base_prob"] = base_prob.values
    print(f"[TCN] Dataset rows after labeling: {len(df)}", flush=True)

    # Build windows aligned to df
    series_cols_override = None
    if args.series_cols:
        series_cols_override = [c.strip() for c in args.series_cols.split(",") if c.strip()]
    if args.base_dir and series_cols_override is not None and "base_prob" not in series_cols_override:
        series_cols_override.append("base_prob")
    X, y, ts, series_cols, scaler = sliding_windows(
        df,
        window=args.window,
        series_cols=series_cols_override,
        stride=max(1, int(args.stride)),
    )
    win_df = pd.DataFrame({
        "timestamp": ts,
        "y_dir": y,
        "ret_next": df.loc[df.index[-len(ts):], ret_col].reset_index(drop=True),
    })
    tail_aligned = df.loc[df.index[-len(ts):]].reset_index(drop=True)
    if args.spread_col and args.spread_col in tail_aligned.columns:
        win_df[args.spread_col] = tail_aligned[args.spread_col]
    if "hl_spread_z" in tail_aligned.columns:
        win_df["hl_spread_z"] = tail_aligned["hl_spread_z"]
    if "rvol_20" in tail_aligned.columns:
        win_df["rvol_20"] = tail_aligned["rvol_20"]

    # OOF evaluation
    logits_oof = np.zeros(len(win_df))
    mask_oof = np.zeros(len(win_df), dtype=bool)
    calib_for_artifact = None
    model_for_artifact = None

    channels = tuple(int(x) for x in args.channels.split(",") if x.strip())
    tcfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        class_weight=args.class_weight,
    )

    print(f"[TCN] Starting {args.n_folds}-fold walk-forward training (embargo={args.embargo_minutes} min)", flush=True)
    fold_logits_records = []
    for k, (tr_idx, va_idx) in enumerate(time_folds(win_df, n_folds=args.n_folds, embargo_minutes=args.embargo_minutes, scheme=args.fold_scheme)):
        fold = k + 1
        print(
            f"[TCN] Fold {fold}/{args.n_folds}: train={len(tr_idx)} windows, val={len(va_idx)} windows",
            flush=True,
        )
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        def _log_epoch(epoch: int, loss: float) -> None:
            print(
                f"[TCN] Fold {fold}/{args.n_folds} epoch {epoch}/{args.epochs} loss={loss:.4f}",
                flush=True,
            )

        model, logits_tr, logits_va = train_tcn(
            X_tr, y_tr,
            val=(X_va, y_va),
            kernel_size=args.kernel_size,
            channels=channels,
            dropout=args.dropout,
            config=tcfg,
            progress_cb=_log_epoch,
        )
        if logits_va is None or not np.isfinite(logits_va).all():
            raise RuntimeError(
                "TCN validation logits contain non-finite values; check input preprocessing and training stability"
            )
        calib = calibrate_logits(logits_va, y_va, method="isotonic")
        p_va = calib.predict_proba(logits_va.reshape(-1, 1))[:, 1]
        logits_oof[va_idx] = p_va
        mask_oof[va_idx] = True

        win_va = win_df.iloc[va_idx].copy()
        raw_prob = 1.0 / (1.0 + np.exp(-np.clip(logits_va, -20, 20)))
        fold_logits_records.append(pd.DataFrame({
            "timestamp": win_va["timestamp"].values,
            "fold": fold,
            "logit": logits_va,
            "prob_uncalibrated": raw_prob,
            "prob_calibrated": p_va,
            "label": y_va,
        }))

        model_for_artifact = model
        calib_for_artifact = calib
        print(f"[TCN] Fold {fold}/{args.n_folds} complete", flush=True)

    valid_idx = np.where(mask_oof)[0]
    spread_series = None
    if args.spread_scale != 0.0 and args.spread_col in win_df.columns:
        spread_series = win_df.loc[valid_idx, args.spread_col]
    training_gate = {
        "hl_spread_max": float(args.max_spread) if args.max_spread is not None else None,
        "hl_spread_z_max": float(args.max_spread_z) if args.max_spread_z is not None else None,
        "rvol20_max": float(args.max_rvol20) if args.max_rvol20 is not None else None,
        "prob_gate_min": None,
        "min_hold_bars": int(max(1, args.min_hold_bars)),
        "long_only": bool(args.long_only),
    }
    inference_gate = {
        "hl_spread_max": float(args.inference_max_spread) if args.inference_max_spread is not None else None,
        "hl_spread_z_max": float(args.inference_max_spread_z) if args.inference_max_spread_z is not None else None,
        "rvol20_max": float(args.inference_max_rvol20) if args.inference_max_rvol20 is not None else None,
        "prob_gate_min": float(args.inference_prob_gate) if args.inference_prob_gate is not None else None,
        "min_hold_bars": int(max(1, args.inference_min_hold_bars)),
        "long_only": bool(args.long_only),
    }
    gate_config = {
        "spread_column": args.spread_col,
        "prob_column": "tcn_prob",
        "training": training_gate,
        "inference": inference_gate,
    }
    full_gate_mask = compute_gate_mask(win_df, gate_config, mode="training")
    gate_mask = full_gate_mask.loc[valid_idx] if full_gate_mask is not None else None
    if gate_mask is not None and gate_mask.mean() < 0.999:
        print(f"[TCN] Trade gating active: coverage={gate_mask.mean():.3f}", flush=True)
    prob_oof = pd.Series(logits_oof[valid_idx], index=valid_idx)
    thr, rep = select_prob_threshold(
        win_df.loc[valid_idx, "ret_next"],
        prob_oof,
        cost_bps=args.cost_bps,
        spread_series=spread_series,
        spread_scale=args.spread_scale,
        slippage_bps=args.slippage_bps,
        long_only=bool(args.long_only),
        gate_mask=gate_mask,
        criterion=args.threshold_criterion,
        min_hold_bars=int(max(1, args.min_hold_bars)),
        min_total_turnover=float(args.min_total_turnover),
        max_total_turnover=args.max_total_turnover,
    )
    rep.update({"oof_count": int(mask_oof.sum())})
    if gate_mask is not None:
        rep["gate_fraction"] = float(gate_mask.mean())
        if args.max_spread is not None:
            rep["max_spread"] = float(args.max_spread)
        if args.max_spread_z is not None:
            rep["max_spread_z"] = float(args.max_spread_z)
        if args.max_rvol20 is not None:
            rep["max_rvol20"] = float(args.max_rvol20)
    rep["gate_config"] = gate_config
    if args.threshold_criterion != "final_equity":
        rep["threshold_criterion"] = args.threshold_criterion
    if args.label_threshold_bps is not None:
        rep["label_threshold_bps"] = float(args.label_threshold_bps)
    if args.diagnostic_thresholds:
        diag = {}
        for val in args.diagnostic_thresholds.split(","):
            val = val.strip()
            if not val:
                continue
            try:
                thr_val = float(val)
            except ValueError:
                continue
            eq = equity_curve(
                win_df.loc[valid_idx, "ret_next"],
                pd.Series(logits_oof[valid_idx], index=valid_idx),
                threshold=thr_val,
                cost_bps=args.cost_bps,
                spread_series=spread_series,
                spread_scale=args.spread_scale,
                slippage_bps=args.slippage_bps,
                long_only=bool(args.long_only),
                gate_mask=gate_mask,
                min_hold_bars=int(max(1, args.min_hold_bars)),
            )
            stats = summary_stats(eq)
            diag[str(thr_val)] = {
                "final_equity": stats["final_equity"],
                "total_turnover": stats["total_turnover"],
            }
        if diag:
            rep["diagnostic_final_equity"] = diag

    fold_logits_df = pd.concat(fold_logits_records, ignore_index=True) if fold_logits_records else pd.DataFrame(columns=[
        "timestamp", "fold", "logit", "prob_uncalibrated", "prob_calibrated", "label",
    ])
    if not fold_logits_df.empty:
        fold_logits_df["timestamp"] = pd.to_datetime(fold_logits_df["timestamp"], utc=True)
        ts_series = fold_logits_df["timestamp"]
        if hasattr(ts_series.dt, "tz") and ts_series.dt.tz is not None:
            ts_naive = ts_series.dt.tz_convert(None)
        else:
            ts_naive = ts_series
        fold_logits_df["month"] = ts_naive.dt.to_period("M").astype(str)
        sigma_by_month = fold_logits_df.groupby("month")["prob_calibrated"].std(ddof=0).fillna(0.0)
        rep["validation_prob_std_by_month"] = {m: float(v) for m, v in sigma_by_month.items()}
        low_sigma = [m for m, v in rep["validation_prob_std_by_month"].items() if v < 0.03]
        rep["prob_sigma_guardrail"] = {"threshold": 0.03, "months_below_threshold": low_sigma}
        if low_sigma:
            rep.setdefault("alerts", {})
            rep["alerts"]["prob_sigma_below_0.03"] = low_sigma
        rep["fold_logits_path"] = "fold_logits.parquet"

    out_dir = Path(args.out)
    model_meta = {
        "window": int(args.window),
        "kernel_size": int(args.kernel_size),
        "channels": [int(x) for x in channels],
        "dropout": float(args.dropout),
    }
    if not fold_logits_df.empty:
        model_meta["fold_logits_path"] = "fold_logits.parquet"
        model_meta["prob_sigma_guardrail"] = rep.get("prob_sigma_guardrail")
        model_meta["validation_prob_std_by_month"] = rep.get("validation_prob_std_by_month")
    save_tcn(
        out_dir,
        model_for_artifact,
        scaler,
        calib_for_artifact,
        series_cols,
        meta=model_meta,
        fold_logits=fold_logits_df,
    )
    (out_dir / "threshold.json").write_text(json.dumps({"prob_threshold": float(thr)}))
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))
    manifest_metadata = {
        "model_type": "tiny_tcn",
        "calibrated": calib_for_artifact is not None,
        "window": model_meta["window"],
        "kernel_size": model_meta["kernel_size"],
        "channels": model_meta["channels"],
    }
    if "fold_logits_path" in model_meta:
        manifest_metadata["fold_logits_path"] = model_meta["fold_logits_path"]
    if "prob_sigma_guardrail" in model_meta and model_meta["prob_sigma_guardrail"] is not None:
        manifest_metadata["prob_sigma_guardrail"] = model_meta["prob_sigma_guardrail"]
    if "validation_prob_std_by_month" in model_meta and model_meta["validation_prob_std_by_month"] is not None:
        manifest_metadata["validation_prob_std_by_month"] = model_meta["validation_prob_std_by_month"]

    manifest = {
        "model_path": "tcn.pt",
        "calibrator_path": "tcn_calibrator.joblib" if calib_for_artifact is not None else None,
        "preprocess_path": "tcn_preproc.joblib",
        "threshold": {
            "value": float(thr),
            "path": "threshold.json",
        },
        "report_path": "report.json",
        "gates": gate_config,
        "metadata": manifest_metadata,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps({"out_dir": str(out_dir), "threshold": float(thr), "report": rep}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
