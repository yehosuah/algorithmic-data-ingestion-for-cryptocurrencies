from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os
from typing import Dict, Any

import pandas as pd
import yaml

# Ensure repository root on path for CLI execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from training.data import load_parquet_dataset


PRIMITIVE_KEYS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "exchange",
    "timeframe",
}


def load_market_dataset(path: str) -> pd.DataFrame:
    """Load the market parquet with sane defaults (UTC timestamps, deduped)."""
    return load_parquet_dataset(path)


def _detect_timestamp_column(df: pd.DataFrame) -> str | None:
    for cand in ("timestamp", "ts", "time", "date"):
        if cand in df.columns:
            return cand
    # try to detect datetime dtype columns
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    return dt_cols[0] if dt_cols else None


def _detect_freq(ts: pd.Series) -> str | None:
    if ts.empty:
        return None
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    ts = ts.dropna().sort_values()
    if ts.size < 2:
        return None
    deltas = ts.diff().dropna()
    if deltas.empty:
        return None
    median_sec = deltas.dt.total_seconds().median()
    # simple rounding to the nearest minute/second bucket
    if abs(median_sec - 60) < 5:
        return "1min"
    if abs(median_sec - 300) < 5:
        return "5min"
    if abs(median_sec - 900) < 5:
        return "15min"
    if abs(median_sec - 3600) < 30:
        return "1h"
    return f"{int(median_sec)}s"


def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Infer basic schema and roles from the loaded dataframe."""
    info: Dict[str, Any] = {"columns": {}, "meta": {}}

    ts_col = _detect_timestamp_column(df)
    ts_dtype = None
    tz = None
    if ts_col:
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        ts_dtype = str(ts.dtype)
        tz = "UTC"
        info["meta"]["freq"] = _detect_freq(ts)

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        role = "other"
        lc = col.lower()
        if col == ts_col:
            role = "time"
        elif lc in ("symbol", "asset"):
            role = "key"
            dtype_str = "category" if not pd.api.types.is_categorical_dtype(series) else dtype_str
        elif lc in PRIMITIVE_KEYS:
            role = "primitive_ohlc" if lc in {"open", "high", "low", "close"} else "primitive_volume"
        elif lc in {"ret_1", "logret_1", "rvol_5", "rvol_20"}:
            role = "derived_feature"
        info["columns"][col] = {"dtype": dtype_str, "role": role}
        if col == ts_col and tz:
            info["columns"][col]["tz"] = tz

    info["meta"].update(
        {
            "source": df.attrs.get("_source", None),
            "timestamp_column": ts_col,
        }
    )
    return info


def _save_schema(schema: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(schema, f, sort_keys=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect market parquet schema and emit YAML.")
    ap.add_argument("--input", default="datasets/market_multi_3symbol_1m.parquet")
    ap.add_argument("--output", default="configs/schema_market_multi_3symbol_1m.yaml")
    args = ap.parse_args()

    df = load_market_dataset(args.input)
    df.attrs["_source"] = Path(args.input).name
    schema = infer_schema(df)
    _save_schema(schema, Path(args.output))
    print(f"Wrote schema to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
