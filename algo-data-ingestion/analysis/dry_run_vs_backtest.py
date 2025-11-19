from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def _frame_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def _daily_sharpe(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    daily = pnl.groupby(pnl.index.date).sum()
    mu = daily.mean()
    sigma = daily.std() + 1e-12
    return float(mu / sigma * np.sqrt(252))


def _parse_backtest_summary(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if df.empty:
            return {}
        row = df.iloc[0]
        return {
            "pnl_net": row.get("primary_pnl_net"),
            "sharpe": row.get("primary_sharpe"),
            "trade_count": row.get("trade_count"),
            "fraction_time_in_position": row.get("fraction_time_in_position"),
        }
    # crude parser for perf_sweep_summary.md where the top scenario CSV is embedded
    text = path.read_text()
    if "## Top scenarios" in text:
        block = text.split("## Top scenarios", 1)[1].strip().split("\n", 1)[-1]
        try:
            df = pd.read_csv(io.StringIO(block))
            if not df.empty:
                row = df.iloc[0]
                return {
                    "pnl_net": row.get("primary_pnl_net"),
                    "sharpe": row.get("primary_sharpe"),
                    "trade_count": row.get("trade_count"),
                    "fraction_time_in_position": row.get("fraction_time_in_position"),
                }
        except Exception:
            pass
    return {}


def analyze_dry_run(
    dry_run_trades_path: str,
    backtest_summary_path: str,
    output_path_md: str,
) -> None:
    trades_path = Path(dry_run_trades_path)
    backtest_path = Path(backtest_summary_path)
    df = _read_frame(trades_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    pnl_col = "pnl_net" if "pnl_net" in df.columns else "pnl"
    pnl_series = df[pnl_col].astype(float) if pnl_col in df.columns else pd.Series([], dtype=float)

    dry_metrics = {
        "pnl_net": float(pnl_series.sum()) if len(pnl_series) else 0.0,
        "sharpe": _daily_sharpe(pnl_series) if len(pnl_series) else 0.0,
        "trade_count": int(len(df)),
    }
    if "position" in df.columns:
        dry_metrics["fraction_time_in_position"] = float((df["position"].abs() > 0).mean())

    backtest_metrics = _parse_backtest_summary(backtest_path)

    lines = ["# Dry-run vs Backtest", ""]
    lines.append("## Dry-run metrics")
    lines.append(_frame_to_markdown(pd.DataFrame([dry_metrics])))
    if backtest_metrics:
        lines.append("")
        lines.append("## Backtest reference")
        lines.append(_frame_to_markdown(pd.DataFrame([backtest_metrics])))
        lines.append("")
        comparison = []
        for key in {"pnl_net", "sharpe", "trade_count", "fraction_time_in_position"}:
            if key in dry_metrics and key in backtest_metrics:
                comparison.append(
                    {
                        "metric": key,
                        "dry_run": dry_metrics.get(key),
                        "backtest": backtest_metrics.get(key),
                        "delta": (
                            dry_metrics.get(key, 0) - backtest_metrics.get(key, 0)
                            if dry_metrics.get(key) is not None and backtest_metrics.get(key) is not None
                            else None
                        ),
                    }
                )
        if comparison:
            lines.append("## Comparison")
            lines.append(_frame_to_markdown(pd.DataFrame(comparison)))

    Path(output_path_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path_md).write_text("\n".join(lines))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare dry-run trade logs against backtest summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dry-run-trades", required=True, help="Path to dry-run trade log (parquet/csv/jsonl).")
    ap.add_argument("--backtest-summary", required=True, help="Path to backtest summary (CSV or Markdown).")
    ap.add_argument("--output-md", required=True, help="Destination Markdown report.")
    args = ap.parse_args(argv)

    analyze_dry_run(args.dry_run_trades, args.backtest_summary, args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
