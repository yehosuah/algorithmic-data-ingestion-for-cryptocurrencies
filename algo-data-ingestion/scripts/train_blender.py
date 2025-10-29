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

from training.data import load_parquet_dataset, ensure_labels
from training.model import extract_features_labels
from training.blender import train_blender, save_blender
from training.infer import load_base_predictor, predict_base, load_tcn_predictor, predict_tcn




def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a blender model combining base and TCN predictions with RSS features")
    ap.add_argument("--data", default="datasets/training_matrix_months_2025-08-09.parquet")
    ap.add_argument("--base-dir", default="models/base_xgb")
    ap.add_argument("--tcn-dir", default="models/tcn")
    ap.add_argument("--out", default="models/blender")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--spread-col", default="hl_spread")
    ap.add_argument("--spread-scale", type=float, default=0.0)
    ap.add_argument("--tcn-stride", type=int, default=30, help="Stride used when generating TCN predictions for blender dataset")
    ap.add_argument("--max-total-turnover", type=float, default=200.0, help="Upper bound for total turnover when selecting probability thresholds")
    ap.add_argument("--min-total-turnover", type=float, default=80.0, help="Lower bound for total turnover when selecting probability thresholds")
    ap.add_argument("--min-toggle-count", type=int, default=2, help="Reject configurations that flip <= this many times when sweeping thresholds")
    ap.add_argument("--calibration-cv", type=int, default=5, help="Cross-validation folds for CalibratedClassifierCV")
    ap.add_argument(
        "--l1-ratio-grid",
        default="0.15,0.35,0.55,0.75,0.9",
        help="Comma-separated elastic-net l1_ratio candidates",
    )
    ap.add_argument(
        "--threshold-grid",
        default=None,
        help="Optional comma-separated list of probability thresholds to evaluate (defaults to a blended 0.45-0.995 grid).",
    )
    ap.add_argument("--min-daily-rss-coverage", type=float, default=0.80, help="Minimum acceptable rss_has_signal coverage before falling back to the no-RSS feature set")
    ap.add_argument("--min-minute-rss-share", type=float, default=0.0005, help="Minimum share of minutes with RSS hits before falling back to the no-RSS feature set")
    ap.add_argument(
        "--turnover-bonus-weight",
        type=float,
        default=0.003,
        help="Multiplier applied to turnover during model selection to prefer thresholds that trade more while remaining within limits",
    )
    ap.add_argument(
        "--sharpe-bonus-weight",
        type=float,
        default=0.05,
        help="Multiplier applied to Sharpe in model selection to avoid collapse into low-variance but inactive fits",
    )
    ap.add_argument(
        "--threshold-penalty-weight",
        type=float,
        default=0.25,
        help="Penalty weight applied when probability thresholds exceed the penalty floor (encourages looser activation).",
    )
    ap.add_argument(
        "--threshold-penalty-floor",
        type=float,
        default=0.88,
        help="Probability floor beyond which the threshold penalty is applied.",
    )
    ap.add_argument("--allow-shorts", action="store_true", help="Allow symmetric long/short thresholding (default: long-only gating)")
    ap.add_argument("--rss-gate-column", default="rss_spike_decay_fast", help="Column used to build the RSS activity gate mask.")
    ap.add_argument("--rss-gate-threshold", type=float, default=0.08, help="Gating threshold applied to the RSS gate column (strictly greater activates).")
    ap.add_argument("--disable-rss-gate", action="store_true", help="Disable gating by RSS activity before selecting thresholds.")
    ap.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default="balanced",
        help="Class weight strategy for the elastic-net blender (default balanced).",
    )
    args = ap.parse_args(argv)

    df = load_parquet_dataset(args.data)
    df = ensure_labels(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Base predictions over all rows (reuse if already present)
    base_series = None
    if "base_prob" in df.columns:
        base_series = pd.to_numeric(df["base_prob"], errors="coerce")
        if base_series.notna().sum() == 0:
            base_series = None
    if base_series is None:
        calib_base, feat_cols = load_base_predictor(Path(args.base_dir))
        base_prob = predict_base(df, calib_base, feat_cols)
        df["base_prob"] = base_prob.values
    else:
        df["base_prob"] = base_series.fillna(0.0)

    merged = df.copy()

    # TCN predictions (reuse if already present)
    tcn_series = None
    if "tcn_prob" in merged.columns:
        tcn_series = pd.to_numeric(merged["tcn_prob"], errors="coerce")
        if tcn_series.notna().sum() == 0:
            tcn_series = None
    if tcn_series is None:
        model_tcn, calib_tcn, series_cols, scaler, window = load_tcn_predictor(Path(args.tcn_dir))
        tcn_df = predict_tcn(
            merged,
            model_tcn,
            calib_tcn,
            series_cols,
            scaler,
            window,
            stride=max(1, int(args.tcn_stride)),
        )
        merged = merged.merge(tcn_df, on="timestamp", how="left")
    else:
        merged["tcn_prob"] = tcn_series

    # Keep rows where both prob and labels exist; drop NA tcn_prob for training blender
    bset = merged.dropna(subset=["base_prob", "tcn_prob", "y_dir", "ret_next"]).reset_index(drop=True)

    spread_series = None
    if args.spread_scale != 0.0 and args.spread_col in bset.columns:
        spread_series = bset[args.spread_col]
    l1_grid = [float(x.strip()) for x in args.l1_ratio_grid.split(",") if x.strip()]
    threshold_grid = None
    if args.threshold_grid:
        threshold_grid = [float(x.strip()) for x in args.threshold_grid.split(",") if x.strip()]
    gate_series = None
    gate_metadata = {}
    if not args.disable_rss_gate:
        gate_col = args.rss_gate_column
        if gate_col in bset.columns:
            gate_values = pd.to_numeric(bset[gate_col], errors="coerce").fillna(0.0)
            gate_series = (gate_values > float(args.rss_gate_threshold)).astype(float)
            gate_share = float(gate_series.mean())
            if gate_share <= 0.0:
                gate_metadata = {
                    "enabled": False,
                    "column": gate_col,
                    "threshold": float(args.rss_gate_threshold),
                    "reason": "empty_gate_after_threshold",
                }
                gate_series = None
            else:
                gate_metadata = {
                    "enabled": True,
                    "column": gate_col,
                    "threshold": float(args.rss_gate_threshold),
                    "share": gate_share,
                }
        else:
            gate_metadata = {
                "enabled": False,
                "column": gate_col,
                "threshold": float(args.rss_gate_threshold),
                "reason": "column_missing",
            }
    model, thr, rep, cols = train_blender(
        bset,
        cost_bps=args.cost_bps,
        spread_series=spread_series,
        spread_scale=args.spread_scale,
        slippage_bps=args.slippage_bps,
        l1_ratio_grid=l1_grid if l1_grid else None,
        calibration_cv=args.calibration_cv,
        max_total_turnover=args.max_total_turnover,
        min_total_turnover=args.min_total_turnover,
        min_toggle_count=args.min_toggle_count,
        min_daily_rss_coverage=args.min_daily_rss_coverage,
        min_minute_rss_share=args.min_minute_rss_share,
        turnover_bonus_weight=args.turnover_bonus_weight,
        sharpe_bonus_weight=args.sharpe_bonus_weight,
        threshold_penalty_weight=args.threshold_penalty_weight,
        threshold_penalty_floor=args.threshold_penalty_floor,
        threshold_grid=threshold_grid,
        long_only=not args.allow_shorts,
        gate_series=gate_series,
        class_weight=args.class_weight,
    )
    if gate_metadata:
        rep["rss_gate"] = gate_metadata
    out_dir = Path(args.out)
    save_blender(out_dir, model, cols, thr, rep)

    print(json.dumps({
        "out_dir": str(out_dir),
        "threshold": float(thr),
        "report": rep,
        "features": cols,
        "l1_ratio_grid": l1_grid,
        "long_only": not args.allow_shorts,
        "min_total_turnover": float(args.min_total_turnover),
        "turnover_bonus_weight": float(args.turnover_bonus_weight),
        "sharpe_bonus_weight": float(args.sharpe_bonus_weight),
        "threshold_penalty_weight": float(args.threshold_penalty_weight),
        "threshold_penalty_floor": float(args.threshold_penalty_floor),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
