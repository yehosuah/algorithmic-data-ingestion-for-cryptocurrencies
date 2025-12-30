from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from analysis.market_trade_alignment import _align_trades, _render_markdown as _render_alignment_markdown, _summaries
from analysis.trading_log_forensics import _build_trade_rows, _equity_curve, _safe_json_lines, _tail_stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expectancy diagnostics for executed exits (baseline-style).")
    parser.add_argument("--audit-log", required=True, help="Path to audit.log (JSON lines)")
    parser.add_argument(
        "--evidence-bundle",
        default=None,
        help="Optional evidence bundle directory (for report metadata)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/expectancy_fix",
        help="Output directory for markdown/json/csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="post_change",
        help="Prefix for outputs (e.g., baseline, post_change)",
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols to include (optional)")
    parser.add_argument("--start-ts", default=None, help="ISO timestamp filter (inclusive, uses exit_ts)")
    parser.add_argument("--end-ts", default=None, help="ISO timestamp filter (inclusive, uses exit_ts)")
    parser.add_argument(
        "--market-data",
        default="data_lake/market/exchange=binance",
        help="Root path to market parquet for alignment (optional; omit to skip alignment)",
    )
    parser.add_argument("--alignment-window-mins", type=int, default=60, help="Alignment lookahead window in minutes")
    return parser.parse_args()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def _filter_window(
    df: pd.DataFrame,
    *,
    symbols: Optional[List[str]],
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
) -> pd.DataFrame:
    out = df
    if symbols:
        out = out[out["symbol"].isin(symbols)]
    if start_ts is not None:
        out = out[out["exit_ts"] >= start_ts]
    if end_ts is not None:
        out = out[out["exit_ts"] <= end_ts]
    return out


def _executed_exits(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()
    side = trades_df["side"].astype(str).str.lower()
    executed = trades_df["executed"].apply(_truthy)
    has_entry = trades_df["entry_ts"].notna()
    exits = trades_df[side == "sell"].copy()
    exits = exits[executed.loc[exits.index]]
    exits = exits[has_entry.loc[exits.index]]
    return exits


def _compute_metrics(exits: pd.DataFrame) -> Dict[str, Optional[float]]:
    if exits.empty:
        return {
            "trades": 0,
            "pnl": 0.0,
            "win_rate": 0.0,
            "avg_win": None,
            "avg_loss": None,
            "payoff_ratio": None,
            "profit_factor": None,
            "max_drawdown_pnl": 0.0,
            "p95_loss": None,
            "p99_loss": None,
            "cvar_95": None,
            "max_loss": None,
            "max_win": None,
        }

    pnl = pd.to_numeric(exits["pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = float((pnl > 0).mean()) if len(pnl) else 0.0
    avg_win = float(wins.mean()) if not wins.empty else None
    avg_loss = float(losses.mean()) if not losses.empty else None

    payoff_ratio = None
    if avg_win is not None and avg_loss is not None and avg_loss != 0:
        payoff_ratio = float(avg_win / abs(avg_loss))

    profit_factor = None
    if not losses.empty:
        loss_sum = float(abs(losses.sum()))
        win_sum = float(wins.sum())
        profit_factor = win_sum / loss_sum if loss_sum != 0 else None

    _, max_dd = _equity_curve(exits)
    tail = _tail_stats(pnl)

    return {
        "trades": int(len(exits)),
        "pnl": float(pnl.sum()),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "max_drawdown_pnl": float(max_dd),
        "p95_loss": tail.get("p95_loss"),
        "p99_loss": tail.get("p99_loss"),
        "cvar_95": tail.get("cvar_95"),
        "max_loss": tail.get("max_loss"),
        "max_win": tail.get("max_win"),
    }


def _loss_attribution(exits: pd.DataFrame) -> List[Dict[str, object]]:
    if exits.empty:
        return []
    df = exits.copy()
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    losses = df[df["pnl"] < 0].copy()
    if losses.empty:
        return []
    losses["exit_reason_primary"] = losses["exit_reason_primary"].fillna("<none>")
    losses["loss_abs"] = losses["pnl"].abs()
    total = float(losses["loss_abs"].sum()) or 0.0
    grouped = (
        losses.groupby("exit_reason_primary", dropna=False)
        .agg(count=("pnl", "size"), loss_sum_abs=("loss_abs", "sum"))
        .reset_index()
        .sort_values("loss_sum_abs", ascending=False)
    )
    grouped["loss_share"] = grouped["loss_sum_abs"].apply(lambda v: float(v) / total if total else None)
    return grouped.to_dict(orient="records")


def _upside_starvation(aligned: Optional[pd.DataFrame], regret_gap_threshold: float = 0.002) -> Dict[str, Optional[float]]:
    if aligned is None or aligned.empty:
        return {
            "regret_fraction": None,
            "mean_regret_gap": None,
            "short_hold_fraction_lt_5m": None,
            "take_profit_trades": 0,
            "take_profit_avg_exit_return": None,
            "take_profit_avg_mfe": None,
            "take_profit_regret_fraction": None,
        }

    gap = pd.to_numeric(aligned["mfe_pct"], errors="coerce") - pd.to_numeric(aligned["exit_return_pct"], errors="coerce")
    regret_fraction = float((gap > regret_gap_threshold).mean()) if len(gap) else None
    mean_gap = float(gap.mean()) if gap.notna().any() else None
    short_hold = float((aligned["hold_minutes"] < 5).mean()) if "hold_minutes" in aligned.columns else None

    tp_mask = aligned["exit_reason"].fillna("") == "take_profit"
    tp_count = int(tp_mask.sum())
    tp_exit = pd.to_numeric(aligned.loc[tp_mask, "exit_return_pct"], errors="coerce")
    tp_mfe = pd.to_numeric(aligned.loc[tp_mask, "mfe_pct"], errors="coerce")
    tp_gap = tp_mfe - tp_exit
    return {
        "regret_fraction": regret_fraction,
        "mean_regret_gap": mean_gap,
        "short_hold_fraction_lt_5m": short_hold,
        "take_profit_trades": tp_count,
        "take_profit_avg_exit_return": float(tp_exit.mean()) if tp_exit.notna().any() else None,
        "take_profit_avg_mfe": float(tp_mfe.mean()) if tp_mfe.notna().any() else None,
        "take_profit_regret_fraction": float((tp_gap > regret_gap_threshold).mean()) if tp_count else None,
    }


def _markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(none)_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _render_forensics_markdown(
    *,
    title: str,
    evidence_bundle: Optional[str],
    audit_log: str,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
    headline: pd.DataFrame,
    portfolio_loss: pd.DataFrame,
    portfolio_upside: pd.DataFrame,
    per_symbol_tables: Dict[str, Dict[str, pd.DataFrame]],
) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    if evidence_bundle:
        lines.append(f"- Evidence bundle: `{evidence_bundle}`")
    lines.append(f"- Audit log: `{audit_log}`")
    lines.append(f"- Window: {start_ts.isoformat() if start_ts is not None else '<none>'} → {end_ts.isoformat() if end_ts is not None else '<none>'}")
    lines.append("- Trade universe: executed exits (audit event_type=trade, side=sell, executed=true, entry_ts present)")
    lines.append("")
    lines.append("## Headline")
    lines.append(_markdown_table(headline))
    lines.append("")
    lines.append("## Portfolio loss drivers (by exit_reason_primary)")
    lines.append(_markdown_table(portfolio_loss))
    lines.append("")
    lines.append("## Upside starvation")
    lines.append(_markdown_table(portfolio_upside))
    lines.append("")
    for sym, tables in per_symbol_tables.items():
        lines.append(f"## {sym}")
        lines.append("")
        lines.append("**Metrics**")
        lines.append(_markdown_table(tables["metrics"]))
        lines.append("")
        lines.append("**Loss drivers**")
        lines.append(_markdown_table(tables["loss_drivers"]))
        lines.append("")
        lines.append("**Upside starvation**")
        lines.append(_markdown_table(tables["upside"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    audit_path = Path(args.audit_log).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    start_ts = pd.to_datetime(args.start_ts, utc=True) if args.start_ts else None
    end_ts = pd.to_datetime(args.end_ts, utc=True) if args.end_ts else None

    events = _safe_json_lines(audit_path)
    trades_df = _build_trade_rows(events)
    trades_df = _filter_window(trades_df, symbols=symbols, start_ts=start_ts, end_ts=end_ts)
    exits_df = _executed_exits(trades_df).copy()

    # Persist the executed-exits universe for downstream inspection/repro.
    subdir = output_dir / f"{args.output_prefix}_forensics"
    subdir.mkdir(parents=True, exist_ok=True)
    exits_df.sort_values("exit_ts").to_csv(subdir / "executed_exits.csv", index=False)

    aligned: Optional[pd.DataFrame] = None
    market_root = None
    if args.market_data:
        market_root = Path(args.market_data).expanduser().resolve()
        if market_root.exists():
            align_in = exits_df[
                [
                    "symbol",
                    "entry_ts",
                    "exit_ts",
                    "exit_reason_primary",
                    "pnl",
                    "hold_minutes",
                    "entry_price",
                    "exit_price",
                ]
            ].copy()
            align_in = align_in.rename(columns={"exit_reason_primary": "exit_reason"})
            aligned = _align_trades(align_in, market_root, args.alignment_window_mins)
            align_out_dir = output_dir / f"{args.output_prefix}_alignment"
            align_out_dir.mkdir(parents=True, exist_ok=True)
            aligned.to_csv(align_out_dir / "market_alignment.csv", index=False)
            summary = _summaries(aligned)
            (align_out_dir / "alignment_stats.json").write_text(json.dumps(summary, indent=2, default=str))
            (align_out_dir / "alignment_summary.json").write_text(aligned.to_json(orient="records", date_format="iso"))
            _render_alignment_markdown(aligned, summary, align_out_dir)

    # Merge alignment columns back into the executed-exits universe (for per-symbol CSVs).
    exits_with_alignment = exits_df.copy()
    if aligned is not None and not aligned.empty:
        merged = exits_df.merge(
            aligned[
                [
                    "symbol",
                    "entry_ts",
                    "exit_ts",
                    "exit_reason",
                    "mfe_pct",
                    "mae_pct",
                    "exit_return_pct",
                    "post_exit_max_return_pct",
                    "post_exit_min_return_pct",
                ]
            ],
            left_on=["symbol", "entry_ts", "exit_ts", "exit_reason_primary"],
            right_on=["symbol", "entry_ts", "exit_ts", "exit_reason"],
            how="left",
        )
        merged = merged.drop(columns=["exit_reason"])
        exits_with_alignment = merged

    # Write per-symbol trade lists (executed exits + alignment columns) to the root output_dir.
    if not exits_with_alignment.empty:
        for sym in sorted(exits_with_alignment["symbol"].dropna().unique().tolist()):
            sym_key = sym.replace("/", "_")
            sym_df = exits_with_alignment[exits_with_alignment["symbol"] == sym].sort_values("exit_ts")
            sym_df.to_csv(output_dir / f"{args.output_prefix}_trades_{sym_key}.csv", index=False)

    # Build tables for markdown + json.
    symbols_out = sorted(exits_df["symbol"].dropna().unique().tolist())

    headline_rows: List[Dict[str, object]] = []
    per_symbol_json: Dict[str, Dict[str, object]] = {}
    per_symbol_tables: Dict[str, Dict[str, pd.DataFrame]] = {}

    # Compute portfolio-level upside starvation from aligned (if present).
    portfolio_upside = _upside_starvation(aligned)
    portfolio_metrics = _compute_metrics(exits_df)
    portfolio_loss = _loss_attribution(exits_df)

    for sym in symbols_out:
        sym_exits = exits_df[exits_df["symbol"] == sym].copy()
        sym_metrics = _compute_metrics(sym_exits)
        sym_loss = _loss_attribution(sym_exits)
        sym_aligned = aligned[aligned["symbol"] == sym].copy() if aligned is not None and not aligned.empty else None
        sym_upside = _upside_starvation(sym_aligned)

        headline_rows.append(
            {
                "symbol": sym,
                "trades": sym_metrics["trades"],
                "win_rate": sym_metrics["win_rate"],
                "avg_win": sym_metrics["avg_win"],
                "avg_loss": sym_metrics["avg_loss"],
                "payoff_ratio": sym_metrics["payoff_ratio"],
                "profit_factor": sym_metrics["profit_factor"],
                "cvar_95": sym_metrics["cvar_95"],
                "max_loss": sym_metrics["max_loss"],
                "pnl": sym_metrics["pnl"],
            }
        )

        per_symbol_json[sym] = {
            "metrics": sym_metrics,
            "loss_attribution": sym_loss,
            "upside_starvation": sym_upside,
        }
        per_symbol_tables[sym] = {
            "metrics": pd.DataFrame([sym_metrics]),
            "loss_drivers": pd.DataFrame(sym_loss),
            "upside": pd.DataFrame([sym_upside]),
        }

    headline_df = pd.DataFrame(headline_rows)
    portfolio_loss_df = pd.DataFrame(portfolio_loss)
    portfolio_upside_df = pd.DataFrame([portfolio_upside])

    window = {
        "start_ts": start_ts.isoformat() if start_ts is not None else None,
        "end_ts": end_ts.isoformat() if end_ts is not None else None,
    }
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "evidence_bundle": args.evidence_bundle,
        "audit_log": str(audit_path),
        "window": window,
        "trade_universe": "executed exits (trade side=sell, executed=true, entry_ts present)",
        "portfolio": {
            "metrics": portfolio_metrics,
            "loss_attribution": portfolio_loss,
            "upside_starvation": portfolio_upside,
        },
        "per_symbol": per_symbol_json,
        "headline": headline_rows,
    }

    (output_dir / f"{args.output_prefix}_forensics.json").write_text(json.dumps(payload, indent=2))
    title = f"{args.output_prefix.replace('_', ' ').title()} Forensics (Executed Exits)"
    md = _render_forensics_markdown(
        title=title,
        evidence_bundle=args.evidence_bundle,
        audit_log=str(audit_path),
        start_ts=start_ts,
        end_ts=end_ts,
        headline=headline_df,
        portfolio_loss=portfolio_loss_df,
        portfolio_upside=portfolio_upside_df,
        per_symbol_tables=per_symbol_tables,
    )
    (output_dir / f"{args.output_prefix}_forensics.md").write_text(md)


if __name__ == "__main__":
    main()
