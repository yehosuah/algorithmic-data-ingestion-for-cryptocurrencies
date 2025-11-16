#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.append(str(ROOT))

try:  # Optional redis dependency (used only when pulling samples from a stream)
    import redis  # type: ignore
except Exception:  # pragma: no cover - redis may be absent in local envs
    redis = None  # type: ignore

# Column constants
PROB_COL = "probability"
TIMESTAMP_COL = "timestamp"


def _load_jsonl(path: Path, max_rows: int) -> List[dict]:
    buffer: deque[dict] = deque(maxlen=max_rows if max_rows > 0 else None)
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            buffer.append(obj)
    return list(buffer)


def _load_redis_stream(url: str, stream: str, count: int) -> List[dict]:
    if redis is None:
        raise RuntimeError("redis is not installed; install it or supply file-based samples instead.")
    client = redis.Redis.from_url(url, decode_responses=True)
    entries = client.xrevrange(stream, count=count)
    out: List[dict] = []
    for _, payload in entries:
        out.append(payload)
    return out


def _load_samples(
    sources: Sequence[str],
    *,
    prob_column: str,
    max_rows: int,
    redis_stream: Optional[str],
    redis_url: Optional[str],
) -> pd.DataFrame:
    rows: List[dict] = []
    for raw in sources:
        path = Path(raw).expanduser()
        if path.is_dir():
            for child in sorted(path.glob("*.jsonl")):
                rows.extend(_load_jsonl(child, max_rows))
        elif path.suffix.lower() == ".jsonl":
            rows.extend(_load_jsonl(path, max_rows))
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
            if PROB_COL not in df.columns:
                raise KeyError(f"Parquet sample {path} missing '{PROB_COL}' column")
            rows.extend(df.to_dict(orient="records"))
        else:
            raise ValueError(f"Unsupported sample source: {raw}")
    if redis_stream and redis_url:
        rows.extend(_load_redis_stream(redis_url, redis_stream, count=max_rows))
    if not rows:
        raise SystemExit("No probability samples found.")
    frame = pd.DataFrame(rows)
    if PROB_COL not in frame.columns:
        # Some streams may emit 'prob' instead of 'probability'
        if "prob" in frame.columns:
            frame[PROB_COL] = frame["prob"]
        else:
            raise KeyError(f"Samples missing '{PROB_COL}' column.")
    if "prob_column" not in frame.columns:
        frame["prob_column"] = prob_column
    frame[PROB_COL] = pd.to_numeric(frame[PROB_COL], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[PROB_COL])
    if TIMESTAMP_COL in frame.columns:
        frame[TIMESTAMP_COL] = pd.to_datetime(frame[TIMESTAMP_COL], utc=True, errors="coerce")
    if "model" not in frame.columns and "model_label" in frame.columns:
        frame["model"] = frame["model_label"]
    if "timeframe" not in frame.columns:
        frame["timeframe"] = frame.get("tf")  # fallback if tf is used upstream
    return frame


def _load_features(path_list: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for raw in path_list:
        p = Path(raw)
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True, errors="coerce")
    return df


def _session_from_hour(hour: int) -> str:
    if 0 <= hour < 8:
        return "Asia"
    if 8 <= hour < 16:
        return "EU"
    return "US"


def _symbol_cluster(symbol: Optional[str]) -> str:
    if not symbol:
        return "unknown"
    root = str(symbol).split("/")[0].upper()
    majors = {"BTC", "ETH"}
    alt_beta = {"SOL", "ADA", "BNB", "XRP", "BCH", "LTC", "AVAX", "LINK"}
    if root in majors:
        return "majors"
    if root in alt_beta:
        return "alt-beta"
    return "illiquid"


def _bucket(series: pd.Series, quantiles: Sequence[float]) -> pd.Categorical:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.dropna().nunique() < 3:
        return pd.Series(["missing"] * len(series), index=series.index, dtype="object")
    bins = np.unique(clean.quantile(quantiles).values)
    bins = np.concatenate(([clean.min() - 1e-9], bins, [clean.max() + 1e-9]))
    try:
        return pd.cut(clean, bins=bins, labels=False, duplicates="drop")
    except Exception:
        return pd.Series(["missing"] * len(series), index=series.index, dtype="object")


def _add_regime_tags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if TIMESTAMP_COL in out.columns:
        out["session"] = out[TIMESTAMP_COL].dt.hour.apply(_session_from_hour)
        out["date"] = out[TIMESTAMP_COL].dt.date.astype(str)
        out["hour"] = out[TIMESTAMP_COL].dt.floor("H")
    if "symbol" in out.columns:
        out["symbol_cluster"] = out["symbol"].apply(_symbol_cluster)
    if "rvol_20" in out.columns:
        out["vol_bucket"] = _bucket(out["rvol_20"], [0.25, 0.5, 0.75])
    elif "rvol20" in out.columns:
        out["vol_bucket"] = _bucket(out["rvol20"], [0.25, 0.5, 0.75])
    if "hl_spread" in out.columns:
        out["spread_bucket"] = _bucket(out["hl_spread"], [0.25, 0.5, 0.75])
    vol = None
    if "quote_volume" in out.columns:
        vol = out["quote_volume"]
    elif "volume" in out.columns:
        vol = out["volume"]
    if vol is not None:
        out["volume_decile"] = _bucket(pd.to_numeric(vol, errors="coerce"), np.linspace(0.1, 0.9, 9))
    if "vol_bucket" in out.columns and "spread_bucket" in out.columns:
        out["regime_label"] = out["vol_bucket"].astype(str) + "/" + out["spread_bucket"].astype(str)
    return out


def _histogram(series: pd.Series, bins: int) -> Dict[str, List[float]]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    hist, edges = np.histogram(numeric, bins=bins, range=(0.0, 1.0), density=True)
    cdf = np.cumsum(hist) / max(float(hist.sum()), 1.0)
    return {
        "edges": [float(v) for v in edges],
        "density": [float(v) for v in hist],
        "cdf": [float(v) for v in cdf],
    }


def _ks_stat(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) == 0 or len(b) == 0:
        return None
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    grid = np.unique(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, grid, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, grid, side="right") / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _wasserstein_approx(a: np.ndarray, b: np.ndarray, *, grid: int = 200) -> Optional[float]:
    if len(a) == 0 or len(b) == 0:
        return None
    qs = np.linspace(0.0, 1.0, grid)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.mean(np.abs(qa - qb)))


def _psi_stat(a: np.ndarray, b: np.ndarray, bins: int) -> Optional[float]:
    if len(a) == 0 or len(b) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    a_hist, _ = np.histogram(a, bins=edges, density=False)
    b_hist, _ = np.histogram(b, bins=edges, density=False)
    a_frac = a_hist / max(a_hist.sum(), 1.0)
    b_frac = b_hist / max(b_hist.sum(), 1.0)
    eps = 1e-6
    psi = np.sum((a_frac - b_frac) * np.log((a_frac + eps) / (b_frac + eps)))
    return float(psi)


def _summary(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"count": 0}
    quantiles = s.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
        "q01": float(quantiles[0.01]),
        "q05": float(quantiles[0.05]),
        "q50": float(quantiles[0.5]),
        "q95": float(quantiles[0.95]),
        "q99": float(quantiles[0.99]),
    }


def _calibration_metrics(prob: pd.Series, labels: pd.Series, *, n_bins: int) -> Optional[Dict[str, float]]:
    numeric = pd.to_numeric(prob, errors="coerce").dropna()
    if labels is None:
        return None
    y = pd.to_numeric(labels.loc[numeric.index], errors="coerce").dropna()
    aligned = numeric.loc[y.index]
    if aligned.empty:
        return None
    from training.calibration_utils import compute_metrics

    metrics = compute_metrics(aligned.values, y.values.astype(int), n_bins=n_bins)
    return metrics.to_dict()


def _collapse_flags(series: pd.Series) -> Dict[str, float | bool]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mid_mass_share": 0.0, "extreme_share": 0.0, "collapsed": False, "saturated": False}
    mid_mask = (s >= 0.45) & (s <= 0.55)
    extreme_mask = (s <= 0.02) | (s >= 0.98)
    mid_share = float(mid_mask.mean())
    extreme_share = float(extreme_mask.mean())
    return {
        "mid_mass_share": mid_share,
        "extreme_share": extreme_share,
        "collapsed": bool(mid_share > 0.6),
        "saturated": bool(extreme_share > 0.6),
    }


def _write_hourly_partitions(df: pd.DataFrame, root: Path) -> None:
    if TIMESTAMP_COL not in df.columns:
        return
    time_floor = df[TIMESTAMP_COL].dt.floor("H")
    model_col = df["model"] if "model" in df.columns else pd.Series(["unknown"] * len(df))
    prob_col = df["prob_column"] if "prob_column" in df.columns else pd.Series(["probability"] * len(df))
    for (model, prob, hour), group in df.groupby([model_col, prob_col, time_floor]):
        if hour is pd.NaT:
            continue
        safe_model = str(model or "unknown").replace("/", "_")
        safe_prob = str(prob or "prob").replace("/", "_")
        subdir = root / f"model={safe_model}" / f"prob={safe_prob}"
        subdir.mkdir(parents=True, exist_ok=True)
        fname = hour.strftime("%Y%m%dT%H00.parquet")
        group.to_parquet(subdir / fname, index=False)


@dataclass
class AuditConfig:
    group_cols: List[str]
    bins: int
    baseline_days: int
    label_col: Optional[str]


def _build_groups(df: pd.DataFrame, cfg: AuditConfig) -> List[Dict]:
    now = datetime.now(timezone.utc)
    baseline_cutoff = None
    if cfg.baseline_days > 0 and TIMESTAMP_COL in df.columns:
        baseline_cutoff = now - timedelta(days=cfg.baseline_days)
    baseline = pd.Series(dtype=float)
    if baseline_cutoff is not None:
        baseline = df[df[TIMESTAMP_COL] >= baseline_cutoff][PROB_COL]
    fold_ref: Optional[np.ndarray] = df.attrs.get("fold_reference")  # type: ignore

    usable_groups = [col for col in cfg.group_cols if col in df.columns]
    payloads: List[Dict] = []
    for keys, subset in df.groupby(usable_groups if usable_groups else [lambda _: True]):
        if not usable_groups:
            group_map = {"all": True}
        else:
            if not isinstance(keys, tuple):
                keys = (keys,)
            group_map = {col: (val if val is not pd.NA else "missing") for col, val in zip(usable_groups, keys)}
        probs = pd.to_numeric(subset[PROB_COL], errors="coerce").dropna()
        stats = _summary(probs)
        hist = _histogram(probs, bins=cfg.bins)
        distances: Dict[str, Optional[float]] = {}
        arr = probs.to_numpy(dtype=float)
        if fold_ref is not None:
            distances["ks_fold"] = _ks_stat(arr, fold_ref)
            distances["psi_fold"] = _psi_stat(arr, fold_ref, cfg.bins)
            distances["wasserstein_fold"] = _wasserstein_approx(arr, fold_ref)
        if not baseline.empty:
            b_arr = pd.to_numeric(baseline, errors="coerce").dropna().to_numpy(dtype=float)
            distances["ks_baseline"] = _ks_stat(arr, b_arr)
            distances["psi_baseline"] = _psi_stat(arr, b_arr, cfg.bins)
            distances["wasserstein_baseline"] = _wasserstein_approx(arr, b_arr)

        cal_metrics = None
        if cfg.label_col and cfg.label_col in subset.columns:
            cal_metrics = _calibration_metrics(subset[PROB_COL], subset[cfg.label_col], n_bins=cfg.bins)

        payloads.append(
            {
                "group": group_map,
                "count": stats.get("count", 0),
                "summary": stats,
                "histogram": hist,
                "distances": {k: (None if v is None or np.isnan(v) else float(v)) for k, v in distances.items()},
                "calibration": cal_metrics,
                "collapse": _collapse_flags(probs),
            }
        )
    return payloads


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Audit live probability distributions vs training fold logits and rolling baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--samples", nargs="+", required=True, help="JSONL/Parquet files or directories of samples.")
    ap.add_argument("--prob-column", default="probability", help="Probability column name inside samples.")
    ap.add_argument("--fold-logits", required=True, help="Path to training fold logits parquet.")
    ap.add_argument("--fold-column", default="prob_calibrated", help="Column to use from fold logits.")
    ap.add_argument("--features", nargs="*", default=[], help="Optional feature parquet(s) to tag regimes/sessions.")
    ap.add_argument("--label-column", default=None, help="Optional label/target column to compute calibration metrics.")
    ap.add_argument("--bins", type=int, default=30, help="Histogram/CDF bins.")
    ap.add_argument("--baseline-days", type=int, default=3, help="Days of live samples to form rolling baseline.")
    ap.add_argument("--max-sample-rows", type=int, default=500_000, help="Cap rows loaded from sample files.")
    ap.add_argument("--out-parquet", help="Path to write enriched samples with tags.")
    ap.add_argument("--hourly-dir", help="Directory to materialize hourly parquet snapshots (partitioned by model/prob).")
    ap.add_argument("--summary-out", default="release/calibration/latest/distribution_audit.json", help="JSON summary path.")
    ap.add_argument(
        "--redis-stream",
        help="Optional Redis stream name to pull the newest samples from (appended to file-based samples).",
    )
    ap.add_argument("--redis-url", help="Redis connection URL when pulling from a stream.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    samples = _load_samples(
        args.samples,
        prob_column=args.prob_column,
        max_rows=args.max_sample_rows,
        redis_stream=args.redis_stream,
        redis_url=args.redis_url,
    ).head(args.max_sample_rows)

    fold_df = pd.read_parquet(args.fold_logits)
    if args.fold_column not in fold_df.columns:
        raise KeyError(f"Fold logits parquet missing column '{args.fold_column}'")
    fold_ref = pd.to_numeric(fold_df[args.fold_column], errors="coerce").dropna().to_numpy(dtype=float)

    # Tag regime/session/context
    if args.features:
        feat_df = _load_features(args.features)
        if not feat_df.empty and TIMESTAMP_COL in feat_df.columns:
            join_cols = [c for c in ["timestamp", "symbol", "timeframe"] if c in feat_df.columns and c in samples.columns]
            if join_cols:
                samples = samples.merge(feat_df, on=join_cols, how="left", suffixes=("", "_feat"))
            else:
                samples["feature_version"] = feat_df.get("feature_version")
    samples = _add_regime_tags(samples)
    samples.attrs["fold_reference"] = fold_ref  # type: ignore[attr-defined]

    group_cols = [
        "model",
        "prob_column",
        "timeframe",
        "session",
        "regime_label",
        "symbol_cluster",
        "vol_bucket",
        "spread_bucket",
        "volume_decile",
    ]
    cfg = AuditConfig(
        group_cols=group_cols,
        bins=max(10, int(args.bins)),
        baseline_days=max(0, int(args.baseline_days)),
        label_col=args.label_column,
    )

    groups = _build_groups(samples, cfg)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "samples": args.samples,
            "fold_logits": str(args.fold_logits),
            "fold_column": args.fold_column,
            "bins": cfg.bins,
            "baseline_days": cfg.baseline_days,
            "label_column": args.label_column,
        },
        "fold_reference": _summary(pd.Series(fold_ref)),
        "groups": groups,
    }

    if args.out_parquet:
        out_path = Path(args.out_parquet)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        samples.to_parquet(out_path, index=False)
    if args.hourly_dir:
        _write_hourly_partitions(samples, Path(args.hourly_dir))

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"groups": len(groups), "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
