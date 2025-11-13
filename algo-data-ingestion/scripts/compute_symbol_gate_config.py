#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from training.data import load_parquet_dataset, sanitize_market_dataset
from training.feature_eng import augment_market_features


def _quantile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q) if len(series) else np.nan)


def _round(value: float, digits: int = 8) -> float:
    if not math.isfinite(value):
        return value
    return float(round(value, digits))


def _with_default(mapping: Mapping[str, float], *, digits: int = 8) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    finite_vals = [val for val in mapping.values() if math.isfinite(val)]
    default_val = max(finite_vals) if finite_vals else math.nan
    for key, val in mapping.items():
        clean[key] = _round(val, digits=digits)
    if math.isfinite(default_val):
        clean["default"] = _round(default_val, digits=digits)
    return clean


def _build_gate_sections(
    df: pd.DataFrame,
    *,
    train_quantile: float,
    infer_quantile: float,
    spread_margin: float,
    vol_margin: float,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, Dict[str, float]]]:
    stats: Dict[str, Dict[str, float]] = {}
    training_gate: Dict[str, object] = {}
    inference_gate: Dict[str, object] = {}

    grouped = df.groupby("symbol")
    spread_train: Dict[str, float] = {}
    spread_infer: Dict[str, float] = {}
    spread_ratio_train: Dict[str, float] = {}
    spread_ratio_infer: Dict[str, float] = {}
    rvol_train: Dict[str, float] = {}
    rvol_infer: Dict[str, float] = {}
    rvol_ratio_train: Dict[str, float] = {}
    rvol_ratio_infer: Dict[str, float] = {}
    liq_ranks: Dict[str, float] = {}

    for symbol, grp in grouped:
        grp = grp.sort_values("timestamp")
        stats[symbol] = {
            "rows": float(len(grp)),
            "start": grp["timestamp"].min().isoformat() if len(grp) else None,
            "end": grp["timestamp"].max().isoformat() if len(grp) else None,
        }
        spread_train[symbol] = (1.0 + spread_margin) * _quantile(grp["hl_spread"], train_quantile)
        spread_infer[symbol] = (1.0 + spread_margin) * _quantile(grp["hl_spread"], infer_quantile)
        spread_ratio_train[symbol] = (1.0 + spread_margin) * _quantile(grp["sym_spread_ratio"], train_quantile)
        spread_ratio_infer[symbol] = (1.0 + spread_margin) * _quantile(grp["sym_spread_ratio"], infer_quantile)
        rvol_train[symbol] = (1.0 + vol_margin) * _quantile(grp["rvol_20"], train_quantile)
        rvol_infer[symbol] = (1.0 + vol_margin) * _quantile(grp["rvol_20"], infer_quantile)
        rvol_ratio_train[symbol] = (1.0 + vol_margin) * _quantile(grp["sym_rvol_ratio"], train_quantile)
        rvol_ratio_infer[symbol] = (1.0 + vol_margin) * _quantile(grp["sym_rvol_ratio"], infer_quantile)
        liq_ranks[symbol] = _quantile(grp["sym_liquidity_rank"], 0.99)

    training_gate["hl_spread_max"] = _with_default(spread_train)
    inference_gate["hl_spread_max"] = _with_default(spread_infer)
    training_gate["rvol20_max"] = _with_default(rvol_train, digits=12)
    inference_gate["rvol20_max"] = _with_default(rvol_infer, digits=12)
    training_gate["sym_spread_ratio_max"] = _with_default(spread_ratio_train)
    inference_gate["sym_spread_ratio_max"] = _with_default(spread_ratio_infer)
    training_gate["sym_rvol_ratio_max"] = _with_default(rvol_ratio_train)
    inference_gate["sym_rvol_ratio_max"] = _with_default(rvol_ratio_infer)
    training_gate["liquidity_rank_max"] = _with_default(liq_ranks, digits=3)
    inference_gate["liquidity_rank_max"] = _with_default(liq_ranks, digits=3)
    return training_gate, inference_gate, stats


def _parse_symbols(raw: Optional[str]) -> Optional[Iterable[str]]:
    if not raw:
        return None
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute per-symbol gate thresholds from a multi-symbol parquet dataset."
    )
    ap.add_argument("--data", default="datasets/market_multi_3symbol_1m.parquet", help="Path to parquet dataset.")
    ap.add_argument("--out", default="release/symbol_gates/latest.json", help="Output JSON path.")
    ap.add_argument("--train-quantile", type=float, default=0.975, help="Quantile used for training caps.")
    ap.add_argument("--infer-quantile", type=float, default=0.95, help="Quantile for inference caps.")
    ap.add_argument("--spread-margin", type=float, default=0.05, help="Relative padding applied to spread caps.")
    ap.add_argument("--vol-margin", type=float, default=0.05, help="Relative padding applied to volatility caps.")
    ap.add_argument(
        "--symbols",
        help="Optional comma-separated list of symbols to include; defaults to all symbols in the dataset.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    path = Path(args.data)
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")
    df = load_parquet_dataset(path, drop_duplicates=False)
    df = sanitize_market_dataset(df, verbose=True)
    df = augment_market_features(df)
    if args.symbols:
        wanted = set(_parse_symbols(args.symbols) or [])
        df = df[df["symbol"].isin(wanted)].copy()
    if df.empty:
        raise SystemExit("Dataset is empty after filtering; cannot compute gate stats.")

    training_gate, inference_gate, stats = _build_gate_sections(
        df,
        train_quantile=float(args.train_quantile),
        infer_quantile=float(args.infer_quantile),
        spread_margin=float(args.spread_margin),
        vol_margin=float(args.vol_margin),
    )
    payload = {
        "source": str(path),
        "rows": int(len(df)),
        "symbols": sorted(df["symbol"].dropna().unique()),
        "training": training_gate,
        "inference": inference_gate,
        "stats": stats,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out": str(out_path), "symbols": payload["symbols"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
