from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.label_diagnostics import evaluate_naive_strategy
from features.feature_engineering import list_feature_families


def _core_features() -> List[str]:
    families = list_feature_families()
    core = families.get("trend", []) + families.get("vol", [])
    return core


def summarize_regimes(df: pd.DataFrame, primary_label: str, horizon: int) -> Dict[str, dict]:
    if "regime_id" not in df.columns:
        raise ValueError("regime_id column missing")
    total = len(df)
    summary: Dict[str, dict] = {}
    for regime, part in df.groupby("regime_id"):
        stats = {
            "count": int(len(part)),
            "share": float(len(part) / max(total, 1)),
        }
        if primary_label in part.columns:
            pos_rate = float(pd.to_numeric(part[primary_label], errors="coerce").mean())
            stats["label_stats"] = {
                "pos_rate": pos_rate,
                "naive": evaluate_naive_strategy(part, primary_label, horizon_minutes=horizon) if len(part) else {},
            }
        feature_stats = {}
        for feat in _core_features():
            if feat in part.columns:
                series = pd.to_numeric(part[feat], errors="coerce")
                feature_stats[feat] = {"mean": float(series.mean()), "std": float(series.std(ddof=0))}
        stats["features"] = feature_stats
        summary[str(regime)] = stats
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Regime diagnostics summary")
    ap.add_argument("--input", default="data/features_labels_regimes_market_multi_3symbol_1m.parquet")
    ap.add_argument("--primary-label", default="cost_adjusted_15m")
    ap.add_argument("--output", default="reports/regime_summary_market_multi_3symbol_1m.json")
    ap.add_argument("--horizon", type=int, default=15)
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.input)
    summary = summarize_regimes(df, args.primary_label, args.horizon)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote regime diagnostics to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
