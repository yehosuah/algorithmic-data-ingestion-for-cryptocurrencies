#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from training.data import load_parquet_dataset, ensure_labels
from training.meta import triple_barrier_events, rolling_vol
from training.metrics import equity_curve, summary_stats

from training.infer import load_base_predictor, predict_base, load_tcn_predictor, predict_tcn
from training.reporting import ensure_kpi_schema, social_signal_audit


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train meta-labeling logistic model using triple-barrier events")
    ap.add_argument("--data", default="datasets/training_matrix_months_2025-08-09.parquet")
    ap.add_argument("--base-dir", default="models/base_xgb")
    ap.add_argument("--tcn-dir", default="models/tcn")
    ap.add_argument("--out", default="models/meta")
    ap.add_argument("--pt-mult", type=float, default=2.0)
    ap.add_argument("--sl-mult", type=float, default=2.0)
    ap.add_argument("--max-hold", type=int, default=60)
    ap.add_argument("--primary-prob-column", default="base_prob", help="Column to apply masking against (base_prob or tcn_prob or blender_prob if available)")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--tcn-stride", type=int, default=30, help="Stride used when generating TCN probabilities for meta-label dataset")
    args = ap.parse_args(argv)

    df = load_parquet_dataset(args.data)
    df = ensure_labels(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Base and TCN probabilities
    calib_base, feat_cols = load_base_predictor(Path(args.base_dir))
    df["base_prob"] = predict_base(df, calib_base, feat_cols).values

    model_tcn, calib_tcn, series_cols, scaler, window = load_tcn_predictor(Path(args.tcn_dir))
    tcn_df = predict_tcn(
        df,
        model_tcn,
        calib_tcn,
        series_cols,
        scaler,
        window,
        stride=max(1, int(args.tcn_stride)),
    )
    df = df.merge(tcn_df, on="timestamp", how="left")

    # Triple barrier labels
    df_ts = df.set_index("timestamp")
    close = df_ts["close"].astype(float)
    logret = np.log(close).diff()
    vol = rolling_vol(logret)
    events = triple_barrier_events(
        close,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
        max_hold=args.max_hold,
        vol=vol,
    )
    lab = events["label"].reindex(df_ts.index).ffill().fillna(0).astype(int)

    # Meta features
    candidate_cols = [
        "base_prob", "tcn_prob", "rvol_5", "rvol_20", "rss_count", "rss_sent_mean", "reddit_count", "reddit_sent_mean"
    ]
    feat_cols = [c for c in candidate_cols if c in df.columns]
    X = df[feat_cols].astype(float)
    y = lab.values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    # Drop rows with NA in features
    mask = X.notna().all(axis=1)
    Xn = X[mask]
    yn = y[mask]
    dn = df.loc[mask].reset_index(drop=True)
    pipe.fit(Xn.values, yn)

    # Pick meta threshold by maximizing equity when masking primary prob
    primary = args.primary_prob_column
    if primary not in dn.columns:
        raise SystemExit(f"Primary prob column '{primary}' not present after predictions")

    raw_prob = pipe.predict_proba(Xn.values)[:, 1]

    # Evaluate a grid of meta thresholds as mask on the primary prob decisions
    thr_grid = np.linspace(0.50, 0.90, 9)
    trading_threshold = 0.6
    best_report: Optional[Dict[str, float]] = None
    for thr_meta in thr_grid:
        mask_keep = raw_prob >= thr_meta
        prob_for_trading = dn[primary].values.copy()
        # mask out low-quality trades by moving prob to neutral 0.5
        prob_for_trading[~mask_keep] = 0.5
        eq = equity_curve(
            dn["ret_next"],
            pd.Series(prob_for_trading, index=dn.index),
            threshold=trading_threshold,
            cost_bps=args.cost_bps,
        )
        stats = summary_stats(eq)
        candidate = ensure_kpi_schema(
            stats,
            overrides={
                "selected_threshold": trading_threshold,
                "criterion": "final_equity",
                "cost_bps": float(args.cost_bps),
                "spread_scale": 0.0,
                "slippage_bps": 0.0,
                "long_only": False,
                "min_hold_bars": 1,
                "min_total_turnover": 0.0,
                "max_total_turnover": None,
            },
        )
        candidate["meta_threshold"] = float(thr_meta)
        candidate["mask_keep_fraction"] = float(mask_keep.mean())
        if best_report is None or candidate["final_equity"] > best_report["final_equity"]:
            best_report = candidate

    if best_report is None:
        raise RuntimeError("Meta-threshold sweep did not produce any candidate reports.")

    best_report.update({
        "pt_mult": float(args.pt_mult),
        "sl_mult": float(args.sl_mult),
        "max_hold": int(args.max_hold),
        "label_pos_frac": float(yn.mean()) if len(yn) else 0.0,
    })
    best_report["rss_audit"] = social_signal_audit(dn)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "meta_model.joblib")
    (out_dir / "features.txt").write_text("\n".join(feat_cols))
    (out_dir / "meta_threshold.txt").write_text(str(best_report["meta_threshold"]))
    (out_dir / "report.json").write_text(json.dumps(best_report, indent=2))
    print(json.dumps({"out_dir": str(out_dir), "report": best_report, "features": feat_cols}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
