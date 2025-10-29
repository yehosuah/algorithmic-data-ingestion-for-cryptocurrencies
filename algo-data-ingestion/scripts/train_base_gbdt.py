#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from training.data import load_parquet_dataset, ensure_labels
from training.feature_eng import augment_market_features
from training.walkforward import time_folds
from training.model import extract_features_labels, train_xgb, calibrate, predict_proba, save_artifacts
from training.thresholds import select_prob_threshold
from training.metrics import summary_stats, equity_curve
from training.reporting import ensure_kpi_schema, social_signal_audit
from sklearn.metrics import roc_auc_score


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train base XGBoost classifier with walk-forward CV and probability calibration")
    ap.add_argument("--data", default="datasets/market_btcusdt_1m_2024_2025.parquet")
    ap.add_argument("--out", default="models/base_xgb")
    ap.add_argument("--n-folds", type=int, default=6)
    ap.add_argument("--embargo-minutes", type=int, default=60)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--spread-col", default="hl_spread", help="Feature column to use as spread proxy for costs")
    ap.add_argument("--spread-scale", type=float, default=0.0, help="Scale factor applied to spread_col for additional costs")
    ap.add_argument("--max-spread", type=float, default=None, help="Optional absolute spread ceiling (same units as spread_col) for opening/holding trades")
    ap.add_argument("--max-spread-z", type=float, default=0.25, help="Optional z-score ceiling using hl_spread_z to gate trades (default relaxed training gate)")
    ap.add_argument("--max-rvol20", type=float, default=2e-4, help="Optional ceiling on rvol_20 to gate trades (default relaxed training gate)")
    ap.add_argument("--prob-gate", type=float, default=None, help="Optional minimum probability gate applied before thresholding")
    ap.add_argument("--diagnostic-thresholds", default=None, help="Comma-separated thresholds to log additional equity diagnostics")
    ap.add_argument("--threshold-grid-min", type=float, default=0.55, help="Lower bound for automatic threshold grid")
    ap.add_argument("--threshold-grid-max", type=float, default=0.995, help="Upper bound for automatic threshold grid")
    ap.add_argument("--min-hold-bars", type=int, default=1, help="Minimum bars to hold a position once opened (>=1) during evaluation")
    ap.add_argument("--sample-weight-scheme", choices=["none", "abs_return", "cost_margin"], default="none", help="Optional sample-weight scheme for model fitting")
    ap.add_argument("--label-threshold-bps", type=float, default=None, help="If provided, treat ret_next > threshold (in bps) as positive class")
    ap.add_argument("--threshold-criterion", choices=["final_equity", "sharpe"], default="final_equity", help="Objective used to pick the probability threshold")
    ap.add_argument("--min-total-turnover", type=float, default=2.0, help="Minimum total turnover required when selecting thresholds; helps avoid degenerate no-trade thresholds")
    ap.add_argument("--max-total-turnover", type=float, default=None, help="Maximum total turnover allowed when selecting thresholds")
    ap.add_argument("--horizon", type=int, default=1, help="Prediction horizon in bars for return label/equity evaluation")
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--fold-scheme", choices=["even", "calendar_month"], default="even")
    ap.add_argument("--long-only", action="store_true", help="Use long-only trading (no shorts) when selecting thresholds and computing equity")
    # XGBoost overrides
    ap.add_argument("--xgb-max-depth", type=int, default=None)
    ap.add_argument("--xgb-n-estimators", type=int, default=None)
    ap.add_argument("--xgb-learning-rate", type=float, default=None)
    ap.add_argument("--xgb-subsample", type=float, default=None)
    ap.add_argument("--xgb-colsample-bytree", type=float, default=None)
    ap.add_argument("--xgb-min-child-weight", type=float, default=None)
    ap.add_argument("--xgb-gamma", type=float, default=None)
    ap.add_argument("--xgb-reg-lambda", type=float, default=None)
    ap.add_argument("--xgb-reg-alpha", type=float, default=None)
    ap.add_argument("--xgb-scale-pos-weight", type=float, default=None, help="Optional class weight (positive class)")
    ap.add_argument("--auto-scale-pos-weight", action="store_true", help="Infer scale_pos_weight from label imbalance")
    ap.add_argument("--calibration-method", choices=["isotonic", "sigmoid", "none"], default="isotonic", help="Probability calibration method; 'none' keeps raw booster scores")
    ap.add_argument("--inference-max-spread", type=float, default=7e-4, help="Inference gate: absolute spread ceiling in live trading")
    ap.add_argument("--inference-max-spread-z", type=float, default=-0.25, help="Inference gate: z-score ceiling using hl_spread_z")
    ap.add_argument("--inference-max-rvol20", type=float, default=8e-5, help="Inference gate: rvol_20 ceiling")
    ap.add_argument("--inference-prob-gate", type=float, default=0.72, help="Inference gate: minimum calibrated probability before thresholding")
    ap.add_argument("--inference-min-hold-bars", type=int, default=10, help="Inference gate: minimum hold bars constraint to enforce downstream")
    args = ap.parse_args(argv)

    df = load_parquet_dataset(args.data)
    df = ensure_labels(df)
    df = augment_market_features(df)
    df = df.sort_values("timestamp").reset_index(drop=True)
    horizon = max(1, int(args.horizon))
    if horizon > 1:
        if "close" not in df.columns:
            raise ValueError("Dataset requires close column to compute multi-bar returns")
        df[f"ret_next_{horizon}"] = df["close"].pct_change(horizon).shift(-horizon)
        df["ret_next"] = df[f"ret_next_{horizon}"]
        df["y_dir"] = (df["ret_next"] > 0).astype(int)
    if args.label_threshold_bps is not None:
        thr_val = float(args.label_threshold_bps) / 1e4
        if "ret_next" not in df.columns:
            raise ValueError("Dataset missing ret_next for label thresholding")
        df["y_dir"] = (df["ret_next"] > thr_val).astype(int)
    df = df.dropna(subset=["ret_next"]).reset_index(drop=True)
    pos_rate = df["y_dir"].mean()
    if args.auto_scale_pos_weight and args.xgb_scale_pos_weight is None and pos_rate not in (0.0, 1.0):
        args.xgb_scale_pos_weight = float((1.0 - pos_rate) / max(pos_rate, 1e-6))
        print(f"[XGB] Auto scale_pos_weight={args.xgb_scale_pos_weight:.4f} (positive rate={pos_rate:.4f})")

    # Walk-forward evaluation
    oof_prob = np.zeros(len(df))
    oof_mask = np.zeros(len(df), dtype=bool)
    feat_cols_used: List[str] | None = None
    calib_for_artifact = None
    model_for_artifact = None
    last_val_idx = None

    xgb_param_override: Dict[str, float] = {}
    mapping = {
        "xgb_max_depth": "max_depth",
        "xgb_n_estimators": "n_estimators",
        "xgb_learning_rate": "learning_rate",
        "xgb_subsample": "subsample",
        "xgb_colsample_bytree": "colsample_bytree",
        "xgb_min_child_weight": "min_child_weight",
        "xgb_gamma": "gamma",
        "xgb_reg_lambda": "reg_lambda",
        "xgb_reg_alpha": "reg_alpha",
        "xgb_scale_pos_weight": "scale_pos_weight",
    }
    for arg_name, param_name in mapping.items():
        val = getattr(args, arg_name)
        if val is not None:
            xgb_param_override[param_name] = val
    if xgb_param_override:
        print(f"[XGB] Using parameter overrides: {xgb_param_override}")

    for k, (tr_idx, va_idx) in enumerate(time_folds(df, n_folds=args.n_folds, embargo_minutes=args.embargo_minutes, scheme=args.fold_scheme)):
        fold = k + 1
        tr = df.iloc[tr_idx]
        va = df.iloc[va_idx]

        X_tr, y_tr, feat_cols = extract_features_labels(tr)
        X_va, y_va, _ = extract_features_labels(va)
        feat_cols_used = feat_cols

        sw = None
        if args.sample_weight_scheme != "none":
            ret_tr = tr["ret_next"].to_numpy()
            if args.sample_weight_scheme == "abs_return":
                sw = np.abs(ret_tr) * 1e4
            elif args.sample_weight_scheme == "cost_margin":
                sw = np.abs(ret_tr) * 1e4 - float(args.cost_bps)
                sw = np.clip(sw, 0.0, None)
            if sw is not None:
                total = float(sw.sum())
                if total > 0:
                    sw = sw * (len(sw) / total)
                else:
                    sw = None

        booster = train_xgb(
            X_tr, y_tr,
            X_val=X_va, y_val=y_va,
            early_stopping_rounds=int(args.early_stopping_rounds),
            params=xgb_param_override or None,
            sample_weight=sw,
        )
        if args.calibration_method == "none":
            calib_cv = booster
        else:
            calib_cv = calibrate(booster, X_va, y_va, method=args.calibration_method)
        p_va = predict_proba(calib_cv, X_va)
        oof_prob[va_idx] = p_va
        oof_mask[va_idx] = True

        # Keep last fold artifacts for saving
        model_for_artifact = booster
        calib_for_artifact = calib_cv
        last_val_idx = va_idx

    # Build threshold and report on OOF predictions
    valid_idx = np.where(oof_mask)[0]
    ret_next = df.loc[valid_idx, "ret_next"]
    p = pd.Series(oof_prob[valid_idx], index=ret_next.index)
    spread_series = None
    if args.spread_scale != 0.0 and args.spread_col and args.spread_col in df.columns:
        spread_series = df.loc[valid_idx, args.spread_col]
    gate_mask = None
    if args.max_spread is not None and args.spread_col and args.spread_col in df.columns:
        gate_mask = (df.loc[valid_idx, args.spread_col] <= float(args.max_spread)).astype(bool)
    if args.max_spread_z is not None and "hl_spread_z" in df.columns:
        mask_z = (df.loc[valid_idx, "hl_spread_z"] <= float(args.max_spread_z)).astype(bool)
        if gate_mask is None:
            gate_mask = mask_z
        else:
            gate_mask = gate_mask & mask_z
    if args.prob_gate is not None:
        mask_prob = (p >= float(args.prob_gate)).astype(bool)
        gate_mask = mask_prob if gate_mask is None else (gate_mask & mask_prob)
    if args.max_rvol20 is not None and "rvol_20" in df.columns:
        mask_rvol = (df.loc[valid_idx, "rvol_20"] <= float(args.max_rvol20)).astype(bool)
        gate_mask = mask_rvol if gate_mask is None else (gate_mask & mask_rvol)
    if gate_mask is not None:
        gate_mask = gate_mask.reindex(ret_next.index).fillna(False)
        print(f"[XGB] Trade gating active: coverage={gate_mask.mean():.3f}")
    thr, rep = select_prob_threshold(
        ret_next,
        p,
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
        grid=np.linspace(float(args.threshold_grid_min), float(args.threshold_grid_max), 25),
    )

    # Additional fold-level metrics
    rep_extra: Dict[str, float] = {"oof_count": int(oof_mask.sum()), "oof_auc": float(roc_auc_score(df.loc[valid_idx, "y_dir"], p.values))}
    rep.update(rep_extra)
    if gate_mask is not None:
        rep["gate_fraction"] = float(gate_mask.mean())
        if args.max_spread is not None:
            rep["max_spread"] = float(args.max_spread)
        if args.max_spread_z is not None:
            rep["max_spread_z"] = float(args.max_spread_z)
        if args.max_rvol20 is not None:
            rep["max_rvol20"] = float(args.max_rvol20)
    if args.sample_weight_scheme != "none":
        rep["sample_weight_scheme"] = args.sample_weight_scheme
    if args.label_threshold_bps is not None:
        rep["label_threshold_bps"] = float(args.label_threshold_bps)
    if args.threshold_criterion != "final_equity":
        rep["threshold_criterion"] = args.threshold_criterion
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
                ret_next,
                p,
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

    monthly_diag: Dict[str, Dict] = {}
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df.loc[valid_idx, "timestamp"], utc=True, errors="coerce")
        ts.index = ret_next.index
        month_periods = ts.dt.to_period("M")
        months = sorted({period for period in month_periods.dropna().unique()})
        for month_period in months:
            month = str(month_period)
            mask = month_periods == month_period
            idx = ret_next.index[mask]
            if len(idx) == 0:
                continue
            ret_month = ret_next.loc[idx]
            prob_month = p.loc[idx]
            gate_month = gate_mask.loc[idx] if gate_mask is not None else None
            spread_month = spread_series.loc[idx] if spread_series is not None else None
            eq = equity_curve(
                ret_month,
                prob_month,
                threshold=float(thr),
                cost_bps=args.cost_bps,
                spread_series=spread_month,
                spread_scale=args.spread_scale,
                slippage_bps=args.slippage_bps,
                long_only=bool(args.long_only),
                gate_mask=gate_month,
                min_hold_bars=int(max(1, args.min_hold_bars)),
            )
            stats = summary_stats(eq)
            stats["n_samples"] = int(len(idx))
            if gate_month is not None:
                stats["gate_fraction"] = float(gate_month.mean())
            monthly_diag[month] = {
                k: (
                    float(v) if isinstance(v, (np.floating, np.float32, np.float64))
                    else int(v) if isinstance(v, np.integer)
                    else v
                )
                for k, v in stats.items()
            }
    if monthly_diag:
        rep["monthly_diagnostics"] = monthly_diag

    training_gate = {
        "hl_spread_max": float(args.max_spread) if args.max_spread is not None else None,
        "hl_spread_z_max": float(args.max_spread_z) if args.max_spread_z is not None else None,
        "rvol20_max": float(args.max_rvol20) if args.max_rvol20 is not None else None,
        "prob_gate_min": float(args.prob_gate) if args.prob_gate is not None else None,
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
        "prob_column": "base_prob",
        "training": training_gate,
        "inference": inference_gate,
    }
    rep["gate_config"] = gate_config
    rep["rss_audit"] = social_signal_audit(df)
    rep = ensure_kpi_schema(rep)

    # Save artifacts
    out_dir = Path(args.out)
    save_artifacts(out_dir, model_for_artifact, calib_for_artifact, feat_cols_used or [], float(thr), rep, gate_config)
    print(json.dumps({"out_dir": str(out_dir), "threshold": float(thr), "report": rep}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
