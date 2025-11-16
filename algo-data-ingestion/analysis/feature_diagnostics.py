from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from features.feature_engineering import FEATURE_REGISTRY, list_feature_families


def compute_feature_stats(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        missing = float(s.isna().mean())
        desc = s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        rows.append(
            {
                "feature": col,
                "missing_rate": missing,
                "mean": float(desc.get("mean", np.nan)),
                "std": float(desc.get("std", np.nan)),
                "skew": float(s.skew(skipna=True)),
                "p01": float(desc.get("1%", np.nan)),
                "p05": float(desc.get("5%", np.nan)),
                "p25": float(desc.get("25%", np.nan)),
                "p50": float(desc.get("50%", np.nan)),
                "p75": float(desc.get("75%", np.nan)),
                "p95": float(desc.get("95%", np.nan)),
                "p99": float(desc.get("99%", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _psi(base: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    base = base[~np.isnan(base)]
    target = target[~np.isnan(target)]
    if len(base) == 0 or len(target) == 0:
        return float("nan")
    edges = np.histogram_bin_edges(base, bins=bins)
    base_hist, _ = np.histogram(base, bins=edges)
    target_hist, _ = np.histogram(target, bins=edges)
    base_pct = base_hist / max(base_hist.sum(), 1)
    target_pct = target_hist / max(target_hist.sum(), 1)
    eps = 1e-6
    psi = np.sum((target_pct - base_pct) * np.log((target_pct + eps) / (base_pct + eps)))
    return float(psi)


def compute_feature_drift(df: pd.DataFrame, feature_cols: Sequence[str], time_splits: List[pd.Timestamp]) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column required for drift computation")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    splits = [ts.min()] + sorted(time_splits) + [ts.max()]
    intervals = []
    for start, end in zip(splits[:-1], splits[1:]):
        mask = (ts >= start) & (ts <= end)
        intervals.append(df.loc[mask])

    rows = []
    for col in feature_cols:
        for i in range(len(intervals) - 1):
            base = pd.to_numeric(intervals[i][col], errors="coerce").values
            target = pd.to_numeric(intervals[i + 1][col], errors="coerce").values
            rows.append({"feature": col, "split": f"{i}->{i+1}", "psi": _psi(base, target)})
    return pd.DataFrame(rows)


def _default_splits(df: pd.DataFrame, n_splits: int = 4) -> List[pd.Timestamp]:
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    quantiles = ts.dropna().quantile(np.linspace(0, 1, n_splits + 1)[1:-1]).tolist()
    return [pd.Timestamp(v) for v in quantiles]


def _build_registry(stats: pd.DataFrame, drift: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    families = list_feature_families()
    fam_lookup = {name: fam for fam, names in families.items() for name in names}
    registry: Dict[str, Dict[str, object]] = {}
    for _, row in stats.iterrows():
        name = row["feature"]
        feature_drift = drift[drift["feature"] == name]
        psi_summary = float(feature_drift["psi"].mean()) if not feature_drift.empty else float("nan")
        entry = {
            "name": name,
            "formula": f"See features.feature_engineering.{name}",
            "family": fam_lookup.get(name, "unknown"),
            "type": "experimental" if row["missing_rate"] > 0.2 else "core",
            "stats": {
                "missing_rate": float(row["missing_rate"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "skew": float(row["skew"]),
                "p01": float(row["p01"]),
                "p05": float(row["p05"]),
                "p25": float(row["p25"]),
                "p50": float(row["p50"]),
                "p75": float(row["p75"]),
                "p95": float(row["p95"]),
                "p99": float(row["p99"]),
                "psi_mean": psi_summary,
            },
        }
        registry[name] = entry
    return registry


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Feature diagnostics and registry export")
    ap.add_argument("--input", default="data/features_market_multi_3symbol_1m.parquet")
    ap.add_argument("--output", default="configs/feature_registry_market_multi_3symbol_1m.yaml")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.input)
    feature_cols = [c for c in df.columns if c in FEATURE_REGISTRY.keys()]
    if not feature_cols:
        raise SystemExit("No registered feature columns found in dataset. Run build_features first.")

    splits = _default_splits(df)
    stats = compute_feature_stats(df, feature_cols)
    drift = compute_feature_drift(df, feature_cols, splits)
    registry = _build_registry(stats, drift)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump({"features": list(registry.values())}, f, sort_keys=False)
    print(f"Wrote feature registry for {len(feature_cols)} features to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
