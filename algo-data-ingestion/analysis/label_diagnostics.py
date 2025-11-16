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
from labels.label_generator import _net_return  # type: ignore


def compute_label_stats(df: pd.DataFrame, label_col: str) -> Dict[str, object]:
    s = pd.to_numeric(df[label_col], errors="coerce")
    s = s.dropna()
    unique_vals = s.dropna().unique()
    if set(unique_vals).issubset({0, 1}) and len(unique_vals) <= 2:
        pos = float((s == 1).mean())
        neg = 1.0 - pos
        entropy = -(pos * np.log2(pos + 1e-9) + neg * np.log2(neg + 1e-9))
        return {"type": "binary", "pos_rate": pos, "count": int(len(s)), "entropy": entropy}
    desc = s.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "type": "continuous",
        "count": int(len(s)),
        "mean": float(desc.get("mean", np.nan)),
        "std": float(desc.get("std", np.nan)),
        "p01": float(desc.get("1%", np.nan)),
        "p05": float(desc.get("5%", np.nan)),
        "p50": float(desc.get("50%", np.nan)),
        "p95": float(desc.get("95%", np.nan)),
        "p99": float(desc.get("99%", np.nan)),
    }


def _plot_hist(series: pd.Series, path: Path) -> None:
    import matplotlib.pyplot as plt

    vals = pd.to_numeric(series, errors="coerce").dropna()
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=50, alpha=0.7)
    plt.title(series.name or "label")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def evaluate_naive_strategy(df: pd.DataFrame, label_col: str, horizon_minutes: int) -> Dict[str, float]:
    net = _net_return(df, horizon_minutes)
    label = df[label_col]
    mask = pd.to_numeric(label, errors="coerce") == 1
    trades = net[mask].dropna()
    if trades.empty:
        return {"count": 0, "pnl_sum": 0.0, "pnl_mean": 0.0, "sharpe_like": 0.0, "hit_rate": 0.0}
    pnl_sum = float(trades.sum())
    pnl_mean = float(trades.mean())
    pnl_std = float(trades.std(ddof=0)) if float(trades.std(ddof=0)) else 0.0
    sharpe_like = pnl_mean / (pnl_std + 1e-6)
    hit_rate = float((trades > 0).mean())
    return {
        "count": int(len(trades)),
        "pnl_sum": pnl_sum,
        "pnl_mean": pnl_mean,
        "sharpe_like": sharpe_like,
        "hit_rate": hit_rate,
    }


def _generate_label(df: pd.DataFrame, label_type: str, horizon: int, base_signal: Optional[str], edge: float) -> pd.Series:
    if label_type == "directional":
        return generate_directional_label(df, horizon)
    if label_type == "cost_adjusted":
        return generate_cost_adjusted_label(df, horizon)
    if label_type == "meta":
        if not base_signal:
            raise ValueError("base_signal_col required for meta label")
        return generate_meta_label(df, horizon, base_signal, edge)
    if label_type == "continuous":
        return generate_continuous_return_label(df, horizon)
    raise ValueError(f"Unsupported label type {label_type}")


def _to_native(obj):
    import numpy as np  # localized import to avoid polluting globals

    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Label diagnostics export")
    ap.add_argument("--input", default="data/features_market_multi_3symbol_1m.parquet")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--label-type", choices=["directional", "cost_adjusted", "meta", "continuous", "all"], default="cost_adjusted")
    ap.add_argument("--base-signal-col", default="feat_log_return_1m")
    ap.add_argument("--edge-threshold", type=float, default=0.0)
    ap.add_argument("--output", default="configs/label_spec_market_multi_3symbol_1m.yaml")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.input)
    label_types = [args.label_type] if args.label_type != "all" else ["directional", "cost_adjusted", "continuous", "meta"]
    specs = []
    df = df.copy()
    for lt in label_types:
        lbl = _generate_label(df, lt, args.horizon, args.base_signal_col, args.edge_threshold)
        df[lbl.name] = lbl
        stats = compute_label_stats(df, lbl.name)
        naive = evaluate_naive_strategy(df, lbl.name, args.horizon) if stats.get("type") == "binary" else {}
        plot_path = Path("reports") / f"label_hist_{lbl.name}.png"
        _plot_hist(df[lbl.name], plot_path)
        specs.append(
            {
                "name": lbl.name,
                "type": lt,
                "horizon_minutes": args.horizon,
                "stats": stats,
                "naive_strategy": naive,
                "base_signal": args.base_signal_col if lt == "meta" else None,
                "edge_threshold": args.edge_threshold if lt == "meta" else None,
                "histogram": str(plot_path),
            }
        )

    primary = next((s["name"] for s in specs if s["type"] == "cost_adjusted"), specs[0]["name"])
    secondaries = [s["name"] for s in specs if s["name"] != primary]

    spec = {
        "labels": specs,
        "primary_label": primary,
        "secondary_labels": secondaries,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(_to_native(spec), f, sort_keys=False)
    print(f"Label spec written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
