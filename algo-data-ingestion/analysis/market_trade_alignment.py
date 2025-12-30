from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


def _sanitize_ohlc(df: pd.DataFrame, *, max_dev_frac: float = 0.30) -> pd.DataFrame:
    """
    Clamp obviously-bad OHLC spikes (e.g., low far below open/close) that corrupt MFE/MAE.

    This is intentionally conservative: it only clamps extremes relative to the candle open/close,
    which are the most reliable fields in the dataset.
    """
    if df is None or df.empty:
        return df
    if not 0 < max_dev_frac < 1:
        max_dev_frac = 0.30
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return df
    working = df.copy()
    for col in ("open", "high", "low", "close"):
        working[col] = pd.to_numeric(working[col], errors="coerce")

    oc_min = working[["open", "close"]].min(axis=1)
    oc_max = working[["open", "close"]].max(axis=1)
    low_bad = working["low"] < (oc_min * (1.0 - max_dev_frac))
    high_bad = working["high"] > (oc_max * (1.0 + max_dev_frac))
    if low_bad.any():
        working.loc[low_bad, "low"] = oc_min[low_bad]
    if high_bad.any():
        working.loc[high_bad, "high"] = oc_max[high_bad]
    # Ensure OHLC consistency after clamping.
    working["low"] = working[["low", "open", "close"]].min(axis=1)
    working["high"] = working[["high", "open", "close"]].max(axis=1)
    return working


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align trades with market moves (MFE/MAE/regret).")
    parser.add_argument("--trades-csv", required=True, help="per_symbol_trades.csv from forensics output")
    parser.add_argument("--market-data", required=True, help="Root path to market parquet (partitioned by symbol/dt)")
    parser.add_argument("--output-dir", default="reports/log_forensics/alignment", help="Output directory")
    parser.add_argument("--window-mins", type=int, default=60, help="Lookahead window in minutes for MFE/MAE")
    return parser.parse_args()


def _date_range(start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
    dates = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def _collect_market_paths(root: Path, symbol: str, dates: Sequence[str]) -> List[Path]:
    sym_key = symbol.replace("/", "-")
    files: List[Path] = []
    for d in dates:
        modern = root / f"symbol={sym_key}" / f"dt={d}"
        if modern.exists():
            files.extend(sorted(modern.glob("*.parquet")))
        # nested year/month/day partitioning
        for dt_dir in root.glob(f"symbol={sym_key}/**/dt={d}"):
            if dt_dir.is_dir():
                files.extend(sorted(dt_dir.glob("*.parquet")))
        legacy = root / sym_key / f"date={d}"
        if legacy.exists():
            files.extend(sorted(legacy.glob("*.parquet")))
    return files


def _load_market(root: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = _date_range(start, end)
    paths = _collect_market_paths(root, symbol, dates)
    frames = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True, errors="coerce")
    df_all = df_all[(df_all["timestamp"] >= start - pd.Timedelta(minutes=5)) & (df_all["timestamp"] <= end + pd.Timedelta(minutes=5))]
    df_all = df_all.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df_all = df_all[["timestamp", "open", "high", "low", "close"]]
    df_all = _sanitize_ohlc(df_all)
    return df_all


def _price_at(df: pd.DataFrame, ts: pd.Timestamp) -> Optional[float]:
    subset = df[df["timestamp"] <= ts]
    if subset.empty:
        return None
    return float(subset.tail(1)["close"].iloc[0])


def _mfe_mae(df: pd.DataFrame, entry_ts: pd.Timestamp, window_min: int, entry_price: float) -> (Optional[float], Optional[float]):
    window_end = entry_ts + pd.Timedelta(minutes=window_min)
    window_df = df[(df["timestamp"] >= entry_ts) & (df["timestamp"] <= window_end)]
    if window_df.empty or entry_price is None:
        return None, None
    mfe = (window_df["high"].max() / entry_price) - 1.0
    mae = (window_df["low"].min() / entry_price) - 1.0
    return float(mfe), float(mae)


def _post_exit_regret(df: pd.DataFrame, exit_ts: pd.Timestamp, window_min: int, exit_price: float) -> (Optional[float], Optional[float]):
    window_end = exit_ts + pd.Timedelta(minutes=window_min)
    window_df = df[(df["timestamp"] >= exit_ts) & (df["timestamp"] <= window_end)]
    if window_df.empty or exit_price is None or exit_price == 0:
        return None, None
    max_after = (window_df["high"].max() / exit_price) - 1.0
    min_after = (window_df["low"].min() / exit_price) - 1.0
    return float(max_after), float(min_after)


def _align_trades(trades: pd.DataFrame, market_root: Path, window_min: int) -> pd.DataFrame:
    columns = [
        "symbol",
        "entry_ts",
        "exit_ts",
        "exit_reason",
        "pnl",
        "hold_minutes",
        "entry_price",
        "exit_price",
        "mfe_pct",
        "mae_pct",
        "exit_return_pct",
        "post_exit_max_return_pct",
        "post_exit_min_return_pct",
    ]
    aligned_rows = []
    if trades.empty:
        return pd.DataFrame(columns=columns)
    symbols = trades["symbol"].dropna().unique().tolist()
    for symbol in symbols:
        sym_trades = trades[trades["symbol"] == symbol].copy()
        if "occurred_at" not in sym_trades.columns:
            sym_trades["occurred_at"] = pd.NaT
        sym_trades["entry_ts"] = pd.to_datetime(sym_trades["entry_ts"], utc=True, errors="coerce")
        sym_trades["occurred_at"] = pd.to_datetime(sym_trades["occurred_at"], utc=True, errors="coerce")
        sym_trades["exit_ts"] = pd.to_datetime(sym_trades["exit_ts"], utc=True, errors="coerce")
        start = sym_trades["entry_ts"].min()
        end = sym_trades["exit_ts"].max()
        if pd.isna(start):
            start = sym_trades["occurred_at"].min()
        if pd.isna(start) or pd.isna(end):
            continue
        market = _load_market(market_root, symbol, start, end + pd.Timedelta(minutes=window_min))
        if market.empty:
            continue
        for _, row in sym_trades.iterrows():
            entry_ts = row["entry_ts"]
            if pd.isna(entry_ts):
                entry_ts = row["occurred_at"]
            exit_ts = row["exit_ts"] if pd.notna(row["exit_ts"]) else entry_ts
            entry_price = _price_at(market, entry_ts)
            exit_price = _price_at(market, exit_ts)
            mfe, mae = _mfe_mae(market, entry_ts, window_min, entry_price)
            regret_price = exit_price if exit_price is not None else entry_price
            post_max, post_min = _post_exit_regret(market, exit_ts, window_min, regret_price)
            exit_ret = None
            if entry_price not in (None, 0) and exit_price not in (None, 0):
                exit_ret = (exit_price / entry_price) - 1.0
            aligned_rows.append(
                {
                    "symbol": symbol,
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "exit_reason": row.get("exit_reason")
                    or row.get("exit_reason_primary")
                    or row.get("exit_trigger"),
                    "pnl": row.get("pnl"),
                    "hold_minutes": row.get("hold_minutes"),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "mfe_pct": mfe,
                    "mae_pct": mae,
                    "exit_return_pct": exit_ret,
                    "post_exit_max_return_pct": post_max,
                    "post_exit_min_return_pct": post_min,
                }
            )
    return pd.DataFrame(aligned_rows, columns=columns)


def _safe_mean(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    val = series.mean(skipna=True)
    return float(val) if pd.notna(val) else None


def _summaries(aligned: pd.DataFrame) -> Dict[str, object]:
    if aligned is None or aligned.empty or "symbol" not in aligned.columns:
        return {"per_symbol": {}, "recommendations": []}
    summary: Dict[str, object] = {}
    recs: List[str] = []
    for symbol in aligned["symbol"].dropna().unique():
        df = aligned[aligned["symbol"] == symbol]
        regret_mask = None
        try:
            regret_mask = (df["mfe_pct"] - df["exit_return_pct"]) > 0.002
        except Exception:
            regret_mask = None
        regret_frac = float(regret_mask.mean()) if regret_mask is not None else None
        short_hold_frac = float((df["hold_minutes"] < 5).mean()) if "hold_minutes" in df.columns else None
        sym_stats = {
            "trades_aligned": int(len(df)),
            "mfe_mean": _safe_mean(df["mfe_pct"]),
            "mae_mean": _safe_mean(df["mae_pct"]),
            "exit_return_mean": _safe_mean(df["exit_return_pct"]),
            "post_exit_max_return_mean": _safe_mean(df["post_exit_max_return_pct"]),
            "post_exit_min_return_mean": _safe_mean(df["post_exit_min_return_pct"]),
            "regret_fraction": regret_frac,
            "short_hold_fraction": short_hold_frac,
        }
        exit_reason_stats = (
            df.groupby("exit_reason")[["exit_return_pct", "post_exit_max_return_pct", "post_exit_min_return_pct"]]
            .mean()
            .reset_index()
            .sort_values("exit_return_pct")
        )
        sym_stats["exit_reason_stats"] = exit_reason_stats.to_dict(orient="records")
        summary[symbol] = sym_stats
        for _, row in exit_reason_stats.iterrows():
            if pd.notna(row["exit_return_pct"]) and pd.notna(row["post_exit_max_return_pct"]):
                if row["exit_return_pct"] < 0 and row["post_exit_max_return_pct"] > 0:
                    recs.append(
                        f"{symbol} exit_reason={row['exit_reason']} shows negative exit return "
                        f"but positive post-exit drift ({row['post_exit_max_return_pct']:.4f}); consider relaxing/retiming exits."
                    )
            else:
                recs.append(
                    f"{symbol} exit_reason={row['exit_reason']} insufficient data for regret; review exit logic."
                )
    return {"per_symbol": summary, "recommendations": recs}


def _render_markdown(aligned: pd.DataFrame, summary: Dict[str, object], output_dir: Path) -> None:
    lines: List[str] = []
    lines.append("# Market alignment")
    lines.append(f"- Trades aligned: {len(aligned)}")
    lines.append(f"- Recommendations: {len(summary['recommendations'])}")
    lines.append("")
    for symbol, stats in summary["per_symbol"].items():
        lines.append(f"## {symbol}")
        lines.append(f"- Aligned trades: {stats['trades_aligned']}")
        fmt = lambda v: "n/a" if v is None or pd.isna(v) else f"{v:.4f}"
        lines.append(
            f"- Mean MFE: {fmt(stats['mfe_mean'])} | Mean MAE: {fmt(stats['mae_mean'])} | Mean exit return: {fmt(stats['exit_return_mean'])}"
        )
        lines.append(
            f"- Post-exit drift: max {fmt(stats['post_exit_max_return_mean'])} | min {fmt(stats['post_exit_min_return_mean'])}"
        )
        lines.append(
            f"- Regret share (MFE >> exit): {fmt(stats.get('regret_fraction'))} | Short holds (<5m): {fmt(stats.get('short_hold_fraction'))}"
        )
        lines.append("- Exit reason effects:")
        for rec in stats["exit_reason_stats"][:5]:
            lines.append(
                f"  - {rec['exit_reason']}: exit_ret={fmt(rec['exit_return_pct'])}, post_max={fmt(rec['post_exit_max_return_pct'])}, post_min={fmt(rec['post_exit_min_return_pct'])}"
            )
        lines.append("")
    if summary["recommendations"]:
        lines.append("## Recommendations")
        for rec in summary["recommendations"]:
            lines.append(f"- {rec}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alignment_summary.md").write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    trades_path = Path(args.trades_csv).expanduser().resolve()
    market_root = Path(args.market_data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(trades_path, parse_dates=["entry_ts", "exit_ts"])
    aligned = _align_trades(trades, market_root, args.window_mins)
    summary = _summaries(aligned)

    aligned.to_csv(output_dir / "market_alignment.csv", index=False)
    (output_dir / "alignment_summary.json").write_text(aligned.to_json(orient="records", date_format="iso"))
    (output_dir / "alignment_stats.json").write_text(json.dumps(summary, indent=2, default=str))
    _render_markdown(aligned, summary, output_dir)


if __name__ == "__main__":
    main()
