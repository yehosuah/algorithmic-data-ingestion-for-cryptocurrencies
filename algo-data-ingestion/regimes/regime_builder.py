from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from labels.label_generator import (
    generate_cost_adjusted_label,
    generate_directional_label,
    generate_meta_label,
    generate_continuous_return_label,
)


def _bucket(series: pd.Series, n_buckets: int) -> pd.Categorical:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.dropna().nunique() < n_buckets:
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="Int64")
    try:
        return pd.qcut(clean, q=n_buckets, labels=False, duplicates="drop").astype("Int64")
    except Exception:
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="Int64")


def assign_vol_regime(df: pd.DataFrame, n_buckets: int = 4) -> pd.Series:
    col = "feat_realized_vol_1h" if "feat_realized_vol_1h" in df.columns else "rvol_20"
    ser = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
    bucketed = _bucket(ser, n_buckets)
    bucketed.name = "vol_regime"
    return bucketed


def assign_liquidity_regime(df: pd.DataFrame, n_buckets: int = 4) -> pd.Series:
    for cand in ("feat_turnover_proxy_1h", "feat_rolling_volume_15m"):
        if cand in df.columns:
            ser = df[cand]
            break
    else:
        ser = pd.Series(np.nan, index=df.index)
    bucketed = _bucket(ser, n_buckets)
    bucketed.name = "liquidity_regime"
    return bucketed


def assign_spread_regime(df: pd.DataFrame, n_buckets: int = 3) -> pd.Series:
    col = "feat_spread_bps" if "feat_spread_bps" in df.columns else None
    ser = df[col] if col else pd.Series(np.nan, index=df.index)
    bucketed = _bucket(ser, n_buckets)
    bucketed.name = "spread_regime"
    return bucketed


def assign_event_flag(df: pd.DataFrame, return_col: str = "feat_log_return_1m", event_quantile: float = 0.99) -> pd.Series:
    if return_col not in df.columns:
        return pd.Series(0, index=df.index, name="event_flag", dtype="Int64")
    ser = pd.to_numeric(df[return_col], errors="coerce").abs()
    threshold = ser.quantile(event_quantile)
    flag = (ser >= threshold).astype("Int64")
    flag.name = "event_flag"
    return flag


def build_composite_regime_id(df: pd.DataFrame) -> pd.Series:
    cols = ["vol_regime", "liquidity_regime", "spread_regime", "event_flag"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(pd.NA, index=df.index)
    comp = (
        "vol"
        + df["vol_regime"].astype(str)
        + "_liq"
        + df["liquidity_regime"].astype(str)
        + "_spr"
        + df["spread_regime"].astype(str)
        + "_ev"
        + df["event_flag"].astype(str)
    )
    return pd.Series(comp.values, index=df.index, name="regime_id")


def _attach_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    directional = generate_directional_label(df, horizon)
    cost_adj = generate_cost_adjusted_label(df, horizon)
    df[directional.name] = directional
    df[cost_adj.name] = cost_adj
    # optional meta label using base return signal
    base_col = "feat_log_return_1m" if "feat_log_return_1m" in df.columns else None
    if base_col:
        meta = generate_meta_label(df, horizon, base_col, edge_threshold=0.0)
        df[meta.name] = meta
    cont = generate_continuous_return_label(df, horizon)
    df[cont.name] = cont
    return df


def attach_regimes(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = _attach_labels(df, horizon)
    out["vol_regime"] = assign_vol_regime(out)
    out["liquidity_regime"] = assign_liquidity_regime(out)
    out["spread_regime"] = assign_spread_regime(out)
    out["event_flag"] = assign_event_flag(out)
    out["regime_id"] = build_composite_regime_id(out)
    return out


def _quantile_edges(series: pd.Series, buckets: int) -> List[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return []
    qs = np.linspace(0, 1, buckets + 1)
    return [float(v) for v in clean.quantile(qs).unique()]


def export_regime_spec(df: pd.DataFrame, path: Path, horizon: int) -> None:
    spec = {
        "dimensions": [],
        "composite_format": "vol{v}_liq{l}_spr{s}_ev{e}",
        "label_horizon_minutes": horizon,
    }
    vol_source = df["feat_realized_vol_1h"] if "feat_realized_vol_1h" in df.columns else df.get("rvol_20", pd.Series(dtype=float))
    spec["dimensions"].append(
        {
            "name": "vol_regime",
            "buckets": _quantile_edges(vol_source, 4),
            "column": "feat_realized_vol_1h" if "feat_realized_vol_1h" in df.columns else "rvol_20",
        }
    )
    liq_source = None
    for cand in ("feat_turnover_proxy_1h", "feat_rolling_volume_15m"):
        if cand in df.columns:
            liq_source = df[cand]
            break
    spec["dimensions"].append(
        {
            "name": "liquidity_regime",
            "buckets": _quantile_edges(liq_source if liq_source is not None else pd.Series(dtype=float), 4),
            "column": liq_source.name if liq_source is not None else "feat_rolling_volume_15m",
        }
    )
    spread_source = df["feat_spread_bps"] if "feat_spread_bps" in df.columns else pd.Series(dtype=float)
    spec["dimensions"].append(
        {"name": "spread_regime", "buckets": _quantile_edges(spread_source, 3), "column": "feat_spread_bps"}
    )
    spec["dimensions"].append({"name": "event_flag", "definition": "abs(return) >= 99th percentile", "column": "event_flag"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(spec, sort_keys=False)
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Attach regime and label columns to feature dataset")
    ap.add_argument("--input", default="data/features_market_multi_3symbol_1m.parquet")
    ap.add_argument("--output", default="data/features_labels_regimes_market_multi_3symbol_1m.parquet")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--regime-spec-output", default="configs/regime_spec_market_multi_3symbol_1m.yaml")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.input)
    extended = attach_regimes(df, args.horizon)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extended.to_parquet(out_path, index=False)
    export_regime_spec(df, Path(args.regime_spec_output), args.horizon)
    print(f"Wrote features+labels+regimes to {out_path} rows={len(extended)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
