#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def _resolve_paths(raw: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for chunk in raw:
        expanded = Path(chunk).expanduser()
        if expanded.is_dir():
            paths.extend(sorted(expanded.glob("*.jsonl")))
        else:
            paths.append(expanded)
    return paths


def _load_live_samples(paths: List[Path], prob_column: str, max_rows: int) -> pd.Series:
    buffer: deque[float] = deque(maxlen=max_rows)
    for path in paths:
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("prob_column") not in {prob_column, None}:
                    continue
                prob = record.get("probability")
                if prob is None:
                    continue
                try:
                    buffer.append(float(prob))
                except (TypeError, ValueError):
                    continue
    return pd.Series(buffer, name="live_prob", dtype=float)


def _load_fold_logits(model_dir: Optional[Path], fold_path: Optional[Path], column: str) -> pd.Series:
    target: Optional[Path] = None
    if fold_path:
        target = Path(fold_path)
    elif model_dir:
        candidate = Path(model_dir) / "fold_logits.parquet"
        if candidate.exists():
            target = candidate
    if target is None or not target.exists():
        print(f"[WARN] fold logits parquet not found for column '{column}'", file=sys.stderr)
        return pd.Series(dtype=float, name="fold_prob")
    df = pd.read_parquet(target)
    if column not in df.columns:
        raise KeyError(f"Fold logits file {target} missing column '{column}'")
    return pd.to_numeric(df[column], errors="coerce").dropna().astype(float).rename("fold_prob")


def _summarise(series: pd.Series) -> dict:
    if series is None or series.empty:
        return {"count": 0}
    quantiles = series.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
        "quantiles": {f"{int(q*100):02d}": float(v) for q, v in quantiles.items()},
    }


def _plot(
    live: pd.Series,
    fold: pd.Series,
    *,
    title: str,
    bins: int,
    out_path: Path,
) -> None:
    if live.empty and fold.empty:
        raise ValueError("No data to plot. Ensure samples or fold logits exist.")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if not fold.empty:
        ax.hist(
            fold,
            bins=bins,
            density=True,
            alpha=0.5,
            label=f"fold_logits (n={len(fold)})",
            color="#1f77b4",
        )
    if not live.empty:
        ax.hist(
            live,
            bins=bins,
            density=True,
            alpha=0.5,
            label=f"live (n={len(live)})",
            color="#ff7f0e",
        )
    ax.set_title(title)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare live probability samples with training fold logits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--samples", nargs="+", required=True, help="One or more JSONL sample files or directories.")
    ap.add_argument("--prob-column", default="tcn_prob", help="Probability column to filter.")
    ap.add_argument("--model-dir", help="Model directory containing fold_logits.parquet.")
    ap.add_argument("--fold-logits", help="Explicit path to fold_logits parquet.")
    ap.add_argument("--fold-column", default="prob_calibrated", help="Column to read from fold_logits parquet.")
    ap.add_argument("--max-live-rows", type=int, default=5000, help="Maximum live rows to retain from samples.")
    ap.add_argument("--bins", type=int, default=50, help="Histogram bin count.")
    ap.add_argument("--out", default="release/calibration/latest/live_vs_fold_prob.png", help="Destination PNG path.")
    ap.add_argument("--summary-out", help="Optional JSON path for numeric summaries.")
    ap.add_argument("--title", default="Probability distribution", help="Plot title suffix.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    sample_paths = _resolve_paths(args.samples)
    if not sample_paths:
        raise SystemExit("No probability sample files were found.")
    live_series = _load_live_samples(sample_paths, args.prob_column, max_rows=max(100, args.max_live_rows))
    fold_series = _load_fold_logits(
        Path(args.model_dir) if args.model_dir else None,
        Path(args.fold_logits) if args.fold_logits else None,
        args.fold_column,
    )

    live_summary = _summarise(live_series)
    fold_summary = _summarise(fold_series)
    summary = {"live": live_summary, "fold": fold_summary}

    out_path = Path(args.out)
    title = f"{args.title} ({args.prob_column})"
    _plot(live_series, fold_series, title=title, bins=max(10, args.bins), out_path=out_path)
    print(json.dumps(summary, indent=2))
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
