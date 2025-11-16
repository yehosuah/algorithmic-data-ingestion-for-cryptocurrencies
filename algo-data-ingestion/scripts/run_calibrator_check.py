#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from training.infer import (  # type: ignore
    _apply_posthoc_if_available,
    load_base_predictor,
    load_manifest_artifacts,
    load_tcn_predictor,
    predict_base,
)


def _summarise(series: pd.Series) -> Dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0}
    summary: Dict[str, float] = {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "p05": float(numeric.quantile(0.05)),
        "p50": float(numeric.quantile(0.5)),
        "p95": float(numeric.quantile(0.95)),
    }
    return summary


def _score_base(df: pd.DataFrame, model_dir: Path) -> Dict[str, object]:
    artifacts = load_manifest_artifacts(model_dir)
    prob_col = artifacts.prob_column or "base_prob"
    calib, feat_cols = load_base_predictor(model_dir, prob_column=prob_col)
    probs = predict_base(df, calib, feat_cols)
    stats = _summarise(probs)
    threshold = (artifacts.manifest.get("metadata") or {}).get("prob_sigma_guardrail", {}).get("threshold")
    if threshold is not None and stats.get("count", 0) > 0:
        stats["sigma_threshold"] = float(threshold)
        stats["below_threshold"] = bool(stats.get("std", 0.0) < float(threshold))
    stats["prob_column"] = prob_col
    stats["model_label"] = artifacts.model_label
    return stats


def _tcn_with_logits(
    df: pd.DataFrame,
    model,
    calibrator,
    series_cols,
    scaler,
    window: int,
    *,
    stride: int,
) -> pd.DataFrame:
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    if ordered.empty:
        return pd.DataFrame(columns=["timestamp", "logit", "prob_uncalibrated", "prob_calibrated"])
    series_df = ordered[series_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    series_df = series_df.ffill().bfill().fillna(0.0)
    values = series_df.values
    if scaler is not None:
        try:
            values = scaler.transform(values)
        except Exception:
            pass
    n, c = values.shape
    L = int(window)
    stride = max(1, int(stride))
    starts = list(range(0, max(0, n - L), stride))
    if not starts:
        return pd.DataFrame(columns=["timestamp", "logit", "prob_uncalibrated", "prob_calibrated"])
    batch_size = max(128, min(2048, 8192 // stride))
    logits_all = []
    ts_idx = []
    model.eval()
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        batch_len = len(batch_starts)
        if batch_len == 0:
            continue
        X_batch = np.empty((batch_len, c, L), dtype=np.float32)
        for i, start in enumerate(batch_starts):
            seg = values[start : start + L, :].T
            mean = seg.mean(axis=1, keepdims=True)
            std = seg.std(axis=1, keepdims=True) + 1e-6
            seg = (seg - mean) / std
            X_batch[i] = seg
            ts_idx.append(start + L)
        with torch.no_grad():
            logits = model(torch.from_numpy(X_batch)).view(-1).cpu().numpy()
        logits_all.append(logits)
    if not logits_all:
        return pd.DataFrame(columns=["timestamp", "logit", "prob_uncalibrated", "prob_calibrated"])
    logits_concat = np.concatenate(logits_all)
    ts = pd.to_datetime(ordered["timestamp"], utc=True, errors="coerce")
    timestamps = ts.iloc[ts_idx].reset_index(drop=True)
    out = pd.DataFrame({"timestamp": timestamps, "logit": logits_concat})
    out["prob_uncalibrated"] = 1.0 / (1.0 + np.exp(-np.clip(out["logit"], -20, 20)))
    calibrated = calibrator.predict_proba(logits_concat.reshape(-1, 1))[:, 1]
    calibrated = _apply_posthoc_if_available(calibrator, calibrated)
    out["prob_calibrated"] = calibrated
    return out.dropna()


def _score_tcn(df: pd.DataFrame, model_dir: Path, stride: int) -> Dict[str, object]:
    artifacts = load_manifest_artifacts(model_dir)
    prob_col = artifacts.prob_column or "tcn_prob"
    model, calibrator, series_cols, scaler, window = load_tcn_predictor(model_dir, prob_column=prob_col)
    result = _tcn_with_logits(df, model, calibrator, series_cols, scaler, window, stride=stride)
    if result.empty:
        return {"count": 0, "model_label": artifacts.model_label, "prob_column": prob_col}
    stats = _summarise(result["prob_calibrated"])
    stats["prob_column"] = prob_col
    stats["model_label"] = artifacts.model_label
    stats["uncalibrated_std"] = float(result["prob_uncalibrated"].std(ddof=0))
    stats["uncalibrated_mean"] = float(result["prob_uncalibrated"].mean())
    threshold = (artifacts.manifest.get("metadata") or {}).get("prob_sigma_guardrail", {}).get("threshold")
    if threshold is not None and stats.get("count", 0) > 0:
        stats["sigma_threshold"] = float(threshold)
        stats["below_threshold"] = bool(stats.get("std", 0.0) < float(threshold))
    return stats


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Re-run manifest calibrators on a live batch to confirm probability health.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--live", required=True, help="Path to live feature parquet (e.g., export_feature_slice output).")
    ap.add_argument("--base-model", help="Base model directory.")
    ap.add_argument("--tcn-model", help="TCN model directory.")
    ap.add_argument("--tcn-stride", type=int, default=1, help="Stride to use when scoring the TCN.")
    ap.add_argument("--max-rows", type=int, default=0, help="Optional cap on the latest rows to inspect.")
    ap.add_argument("--summary-out", help="Optional path to write JSON summary.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.base_model and not args.tcn_model:
        raise SystemExit("Specify at least one of --base-model or --tcn-model.")
    df = pd.read_parquet(args.live)
    if args.max_rows and args.max_rows > 0:
        df = df.sort_values("timestamp").tail(args.max_rows).copy()
    if df.empty:
        raise SystemExit("Live feature frame is empty.")

    summary: Dict[str, object] = {}
    if args.base_model:
        base_stats = _score_base(df, Path(args.base_model))
        summary["base_model"] = base_stats
        print("[base] ", json.dumps(base_stats))
    if args.tcn_model:
        tcn_stats = _score_tcn(df, Path(args.tcn_model), stride=args.tcn_stride)
        summary["tcn_model"] = tcn_stats
        print("[tcn] ", json.dumps(tcn_stats))

    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
