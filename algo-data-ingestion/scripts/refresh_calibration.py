#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

import os
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from training.calibration_utils import (
    fit_best_calibrator,
)
from training.calibration_store import save_calibrator
from training.data import load_parquet_dataset
from training.blender import build_blender_features

import joblib


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Refresh post-hoc probability calibration for base, TCN, and blender models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", required=True, help="Path to the blender matrix / evaluation parquet.")
    ap.add_argument("--base-model", required=True, help="Directory containing base XGB manifest/artifacts.")
    ap.add_argument("--tcn-model", required=True, help="Directory containing TCN manifest/artifacts.")
    ap.add_argument("--blender-model", required=True, help="Directory containing blender artifacts.")
    ap.add_argument("--start-date", help="Optional ISO datetime to filter rows from (inclusive).")
    ap.add_argument("--end-date", help="Optional ISO datetime to filter rows to (inclusive).")
    ap.add_argument("--split-ratio", type=float, default=0.65, help="Fraction of earliest rows used for calibration fit.")
    ap.add_argument("--min-train-rows", type=int, default=1500, help="Minimum rows required to fit a calibrator.")
    ap.add_argument("--out-dir", default="release/calibration/latest", help="Directory to write metrics/report artifacts.")
    ap.add_argument("--no-plots", action="store_true", help="Skip PNG generation for reliability curves.")
    ap.add_argument("--n-bins", type=int, default=20, help="Number of bins for reliability metrics.")
    return ap.parse_args(argv)


def _parse_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Unable to parse timestamp '{value}'")
    return ts


def _filter_frame(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    out = df.copy()
    if start is not None:
        out = out[out["timestamp"] >= start]
    if end is not None:
        out = out[out["timestamp"] <= end]
    return out.reset_index(drop=True)


def _split_frame(df: pd.DataFrame, ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Calibration dataset is empty after filtering.")
    ratio = min(max(ratio, 0.1), 0.9)
    split_idx = int(len(df) * ratio)
    if split_idx <= 0 or split_idx >= len(df):
        split_idx = len(df) // 2
    left = df.iloc[:split_idx].copy()
    right = df.iloc[split_idx:].copy()
    if left.empty or right.empty:
        half = len(df) // 2
        left = df.iloc[:half].copy()
        right = df.iloc[half:].copy()
    return left.reset_index(drop=True), right.reset_index(drop=True)


def _ensure_blender_prob(df: pd.DataFrame, blender_dir: Path) -> pd.Series:
    feature_path = blender_dir / "blender_features.txt"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing blender feature list: {feature_path}")
    candidate_cols = [
        line.strip() for line in feature_path.read_text().splitlines() if line.strip()
    ]
    if not candidate_cols:
        raise ValueError(f"No feature columns declared in {feature_path}")
    model_path = blender_dir / "blender.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing blender artifact: {model_path}")
    model = joblib.load(model_path)
    X, cols = build_blender_features(df, candidate_cols=candidate_cols)
    if X.empty:
        raise ValueError("Blender feature matrix is empty; verify dataset columns.")
    probabilities = model.predict_proba(X.values)[:, 1]
    series = pd.Series(probabilities, index=X.index, name="blender_prob").astype(float)
    aligned = pd.Series(index=df.index, dtype=float)
    aligned.loc[series.index] = series
    return aligned


def _maybe_plot(
    out_dir: Path,
    *,
    label: str,
    metrics_before: Dict[str, object],
    metrics_after: Dict[str, object],
) -> None:
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for metrics, style in ((metrics_before, "C1"), (metrics_after, "C0")):
        xs = [row["mean_pred"] for row in metrics["reliability"] if np.isfinite(row.get("mean_pred", np.nan))]
        ys = [row["empirical_rate"] for row in metrics["reliability"] if np.isfinite(row.get("empirical_rate", np.nan))]
        if not xs or not ys:
            continue
        ax.plot(xs, ys, marker="o", linestyle="-", color=style, label=("after" if style == "C0" else "before"))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"Reliability: {label}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_reliability.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    bins = np.linspace(0, 1, len(metrics_before["histogram"]) + 1)
    before_counts = [row["count"] for row in metrics_before["histogram"]]
    after_counts = [row["count"] for row in metrics_after["histogram"]]
    ax.bar(bins[:-1], before_counts, width=bins[1] - bins[0], alpha=0.5, label="before")
    ax.bar(bins[:-1], after_counts, width=bins[1] - bins[0], alpha=0.5, label="after")
    ax.set_title(f"Histogram: {label}")
    ax.set_xlabel("Probability bin")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_hist.png", dpi=150)
    plt.close(fig)


def _summarise_dataset(df: pd.DataFrame, name: str) -> Dict[str, object]:
    return {
        "name": name,
        "rows": int(len(df)),
        "start": df["timestamp"].min().isoformat() if not df.empty else None,
        "end": df["timestamp"].max().isoformat() if not df.empty else None,
    }


def _fit_model_calibration(
    name: str,
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    column: str,
    labels_col: str,
    model_dir: Path,
    methods: Sequence[str],
    min_rows: int,
    n_bins: int,
    dataset_summary: Dict[str, object],
    out_dir: Path,
    plot: bool,
) -> Dict[str, object]:
    train = train_df.dropna(subset=[column, labels_col])
    val = val_df.dropna(subset=[column, labels_col])
    if len(train) < min_rows or len(val) < max(200, min_rows // 3):
        raise ValueError(f"Not enough rows to calibrate {name}: train={len(train)} val={len(val)}")

    result = fit_best_calibrator(
        train[column],
        train[labels_col],
        val[column],
        val[labels_col],
        methods=methods,
        min_train_rows=min_rows,
        n_bins=n_bins,
    )

    dataset_info = {
        **dataset_summary,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
    }
    save_calibrator(model_dir, column, result=result, dataset_info=dataset_info)

    metrics_before = result.metrics_before.to_dict()
    metrics_after = result.metrics_after.to_dict()
    metrics_before["n_bins"] = n_bins
    metrics_after["n_bins"] = n_bins

    if plot:
        _maybe_plot(out_dir, label=name, metrics_before=metrics_before, metrics_after=metrics_after)

    payload = {
        "model": name,
        "prob_column": column,
        "method": result.method,
        "train_rows": len(train),
        "val_rows": len(val),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }
    metric_path = out_dir / f"{name}_metrics.json"
    metric_path.write_text(json.dumps(payload, indent=2))
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    data_path = Path(args.data)
    base_dir = Path(args.base_model)
    tcn_dir = Path(args.tcn_model)
    blender_dir = Path(args.blender_model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_parquet_dataset(data_path)
    if "timestamp" not in df.columns or "y_dir" not in df.columns:
        raise ValueError("Dataset must include 'timestamp' and 'y_dir' columns.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    start = _parse_timestamp(args.start_date)
    end = _parse_timestamp(args.end_date)
    df = _filter_frame(df, start, end)

    if "blender_prob" not in df.columns:
        df["blender_prob"] = _ensure_blender_prob(df, blender_dir)

    train_df, val_df = _split_frame(df, args.split_ratio)
    dataset_summary = {
        "source": str(data_path),
        "start": df["timestamp"].min().isoformat() if not df.empty else None,
        "end": df["timestamp"].max().isoformat() if not df.empty else None,
        "split_ratio": float(args.split_ratio),
    }

    report: Dict[str, object] = {
        "dataset": dataset_summary,
        "splits": {
            "train": _summarise_dataset(train_df, "train"),
            "validation": _summarise_dataset(val_df, "validation"),
        },
        "models": [],
    }

    models = [
        ("base_xgb", "base_prob", base_dir),
        ("tcn", "tcn_prob", tcn_dir),
        ("blender", "blender_prob", blender_dir),
    ]

    for name, column, model_path in models:
        try:
            payload = _fit_model_calibration(
                name,
                train_df=train_df,
                val_df=val_df,
                column=column,
                labels_col="y_dir",
                model_dir=model_path,
                methods=("isotonic", "isotonic_blend", "platt", "power"),
                min_rows=args.min_train_rows,
                n_bins=args.n_bins,
                dataset_summary=dataset_summary,
                out_dir=out_dir,
                plot=not args.no_plots,
            )
            report["models"].append(payload)
        except Exception as exc:
            payload = {
                "model": name,
                "prob_column": column,
                "error": str(exc),
            }
            report["models"].append(payload)

    summary_path = out_dir / "calibration_summary.json"
    summary_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
