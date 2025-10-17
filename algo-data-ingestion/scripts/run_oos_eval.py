#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from training.data import load_parquet_dataset
from training.eval_pipeline import (
    EvaluationSummary,
    compare_oos_frames,
    evaluate_base_xgb_oos,
    evaluate_tcn_oos,
    load_model_gate_config,
)


def _parse_channels(text: str) -> tuple[int, ...]:
    vals = [v.strip() for v in text.split(",") if v.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("channels string must contain at least one integer (e.g. '32,32').")
    try:
        return tuple(int(v) for v in vals)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid channel width in '{text}': {exc}") from exc


def _parse_series_cols(text: Optional[str]) -> Optional[list[str]]:
    if text is None:
        return None
    cols = [c.strip() for c in text.split(",") if c.strip()]
    return cols or None


def _diagnostic_summary(diag: Dict) -> Dict:
    summary: Dict = {}
    for key, value in diag.items():
        if isinstance(value, pd.DataFrame):
            summary[key] = {
                "rows": int(len(value)),
                "columns": list(value.columns),
            }
        else:
            summary[key] = value
    return summary


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run leak-proof out-of-sample evaluation for base_xgb or TCN model families.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--family", choices=("base_xgb", "tcn"), required=True, help="Model family to evaluate.")
    ap.add_argument("--data", required=True, help="Path to the evaluation dataset parquet.")
    ap.add_argument("--model-dir", required=True, help="Directory containing the trained model artifacts.")
    ap.add_argument("--save-oos", help="Optional path to save the generated oos_eval parquet.")
    ap.add_argument(
        "--baseline-oos",
        help="Path to an existing oos_eval parquet snapshot to compare against.",
    )
    ap.add_argument(
        "--save-fold-logits",
        help="Optional path to persist fold logits (TCN only); ignored for base_xgb.",
    )
    ap.add_argument("--align-gates", action="store_true", help="Force training gate to match inference gate.")
    ap.add_argument("--n-folds", type=int, default=6, help="Number of walk-forward folds.")
    ap.add_argument("--embargo-minutes", type=int, default=60, help="Embargo window between folds.")
    ap.add_argument("--fold-scheme", choices=("even", "calendar_month"), default="even")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost in basis points.")
    ap.add_argument("--slippage-bps", type=float, default=0.0, help="Additional slippage cost in basis points.")
    ap.add_argument("--spread-col", default="hl_spread", help="Spread column used for cost scaling.")
    ap.add_argument("--spread-scale", type=float, default=0.0, help="Multiplier applied to spread column for costs.")
    ap.add_argument("--min-total-turnover", type=float, default=2.0, help="Minimum turnover when selecting thresholds.")
    ap.add_argument("--max-total-turnover", type=float, default=None, help="Maximum turnover allowed for thresholds.")

    # Base XGB specific
    ap.add_argument("--sample-weight-scheme", choices=("none", "abs_return", "cost_margin"), default="none")
    ap.add_argument("--calibration-method", choices=("isotonic", "sigmoid", "none"), default="isotonic")

    # TCN specific knobs
    ap.add_argument("--window", type=int, default=32, help="Window length for sliding windows (TCN only).")
    ap.add_argument("--kernel-size", type=int, default=3, help="Kernel size for the TCN (TCN only).")
    ap.add_argument("--channels", type=_parse_channels, default="32,32", help="Comma-separated channels for TCN.")
    ap.add_argument("--dropout", type=float, default=0.05, help="Dropout applied within the TCN blocks.")
    ap.add_argument("--epochs", type=int, default=10, help="Number of epochs for each fold (TCN only).")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size for TCN training.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate for TCN AdamW optimizer.")
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for TCN optimizer.")
    ap.add_argument("--class-weight", type=float, default=None, help="Optional positive-class weight for BCE loss.")
    ap.add_argument("--stride", type=int, default=1, help="Stride for sliding windows (TCN only).")
    ap.add_argument(
        "--series-cols",
        default=None,
        help="Optional comma-separated list of series columns for the TCN windows.",
    )
    ap.add_argument(
        "--base-model-dir",
        default=None,
        help="Optional path to a base XGB model; when provided the TCN evaluation injects `base_prob` features.",
    )

    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    df = load_parquet_dataset(data_path)
    gate_cfg = load_model_gate_config(model_dir, fallback_align=args.align_gates)

    summary: EvaluationSummary
    if args.family == "base_xgb":
        summary = evaluate_base_xgb_oos(
            df,
            gate_config=gate_cfg,
            align_gates=args.align_gates,
            n_folds=int(args.n_folds),
            embargo_minutes=int(args.embargo_minutes),
            fold_scheme=args.fold_scheme,
            cost_bps=float(args.cost_bps),
            slippage_bps=float(args.slippage_bps),
            spread_col=args.spread_col,
            spread_scale=float(args.spread_scale),
            sample_weight_scheme=args.sample_weight_scheme,
            calibration_method=args.calibration_method,
            min_total_turnover=float(args.min_total_turnover),
            max_total_turnover=args.max_total_turnover,
        )
    else:
        summary = evaluate_tcn_oos(
            df,
            gate_config=gate_cfg,
            align_gates=args.align_gates,
            base_model_dir=Path(args.base_model_dir) if args.base_model_dir else None,
            window=int(args.window),
            kernel_size=int(args.kernel_size),
            channels=args.channels,
            dropout=float(args.dropout),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            class_weight=float(args.class_weight) if args.class_weight is not None else None,
            stride=int(args.stride),
            n_folds=int(args.n_folds),
            embargo_minutes=int(args.embargo_minutes),
            fold_scheme=args.fold_scheme,
            cost_bps=float(args.cost_bps),
            slippage_bps=float(args.slippage_bps),
            spread_col=args.spread_col,
            spread_scale=float(args.spread_scale),
            min_total_turnover=float(args.min_total_turnover),
            max_total_turnover=args.max_total_turnover,
            series_cols_override=_parse_series_cols(args.series_cols),
        )

    saved_oos = None
    if args.save_oos:
        out_path = Path(args.save_oos)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.oos_frame.to_parquet(out_path, index=False)
        saved_oos = str(out_path)

    saved_fold_logits = None
    if args.family == "tcn" and args.save_fold_logits:
        fold_logits = summary.diagnostics.get("fold_logits")
        if isinstance(fold_logits, pd.DataFrame) and not fold_logits.empty:
            out_path = Path(args.save_fold_logits)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fold_logits.to_parquet(out_path, index=False)
            saved_fold_logits = str(out_path)

    comparison = None
    if args.baseline_oos:
        baseline_path = Path(args.baseline_oos)
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline OOS snapshot not found: {baseline_path}")
        baseline_df = pd.read_parquet(baseline_path)
        comparison = compare_oos_frames(summary.oos_frame, baseline_df)

    diagnostics_repr = _diagnostic_summary(summary.diagnostics)
    result_payload = {
        "family": args.family,
        "model_dir": str(model_dir),
        "dataset": str(data_path),
        "threshold": summary.threshold,
        "auc": summary.auc,
        "training_report": summary.training_report,
        "inference_report": summary.inference_report,
        "gate_config": summary.gate_config,
        "oos_rows": int(len(summary.oos_frame)),
        "saved_oos": saved_oos,
        "saved_fold_logits": saved_fold_logits,
        "diagnostics": diagnostics_repr,
        "comparison": comparison,
    }
    print(json.dumps(_sanitize(result_payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
