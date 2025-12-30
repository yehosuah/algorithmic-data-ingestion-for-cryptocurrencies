from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forensics over trading audit logs.")
    parser.add_argument("--audit-log", required=True, help="Path to audit.log (JSON lines)")
    parser.add_argument("--output-dir", default="reports/log_forensics/forensics", help="Output directory")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols to include (optional)")
    parser.add_argument("--timeframe", default=None, help="Expected timeframe (e.g., 1m)")
    parser.add_argument("--start-ts", default=None, help="ISO timestamp filter (inclusive)")
    parser.add_argument("--end-ts", default=None, help="ISO timestamp filter (inclusive)")
    return parser.parse_args()


def _safe_json_lines(path: Path) -> List[dict]:
    events: List[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _to_ts(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def _build_trade_rows(events: List[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for ev in events:
        if ev.get("event_type") != "trade":
            continue
        payload = ev.get("payload") or {}
        symbol = ev.get("symbol") or payload.get("symbol")
        occurred_at = payload.get("ts") or ev.get("occurred_at")
        entry_ts = payload.get("entry_ts")
        exit_ts = payload.get("ts") or ev.get("occurred_at")
        row = {
            "symbol": symbol,
            "model": ev.get("model"),
            "policy_id": payload.get("policy_id"),
            "timeframe": payload.get("timeframe"),
            "occurred_at": _to_ts(occurred_at),
            "entry_ts": _to_ts(entry_ts),
            "exit_ts": _to_ts(exit_ts),
            "side": payload.get("side"),
            "gate_pass": payload.get("gate_pass"),
            "decision": payload.get("decision"),
            "skip_reason": payload.get("skip_reason") or payload.get("blocked_reason"),
            "risk_block_reason": payload.get("risk_block_reason"),
            "risk_clip_reasons": ",".join(payload.get("risk_clip_reasons", [])),
            "executed": payload.get("executed"),
            "entry_prob": payload.get("prob_entry") or payload.get("probability") or payload.get("prob"),
            "exit_prob": payload.get("prob") or payload.get("prob_now"),
            "entry_threshold": payload.get("entry_threshold"),
            "exit_threshold": payload.get("exit_threshold") or payload.get("threshold"),
            "exit_reason_primary": payload.get("exit_reason_primary") or payload.get("decision") or payload.get("skip_reason"),
            "exit_reason_all": ",".join(payload.get("exit_reason_all", [])),
            "pnl_gross": payload.get("pnl_gross"),
            "pnl_net": payload.get("pnl_net_estimate"),
            "pnl_notional": payload.get("pnl_notional"),
            "probability": payload.get("probability") or payload.get("prob"),
            "spread_entry_bps": payload.get("spread_feature_bps") or payload.get("spread_bps"),
            "spread_exit_bps": payload.get("spread_bps_now") or payload.get("quote_spread_bps"),
            "intended_notional": payload.get("intended_notional") or payload.get("notional"),
            "entry_price": payload.get("entry_price"),
            "exit_price": payload.get("price_now") or payload.get("price_used"),
        }
        pnl = row["pnl_net"]
        if pnl is None:
            pnl = row["pnl_gross"]
        row["pnl"] = pnl
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    df["hold_minutes"] = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / 60
    return df


def _tail_stats(pnl_series: pd.Series) -> Dict[str, Optional[float]]:
    losses = pnl_series[pnl_series < 0]
    wins = pnl_series[pnl_series > 0]
    out: Dict[str, Optional[float]] = {
        "p95_loss": None,
        "p99_loss": None,
        "cvar_95": None,
        "max_loss": float(losses.min()) if not losses.empty else None,
        "max_win": float(wins.max()) if not wins.empty else None,
    }
    if losses.empty:
        return out
    try:
        p95 = losses.quantile(0.95)
        out["p95_loss"] = float(p95)
        p99 = losses.quantile(0.99)
        out["p99_loss"] = float(p99)
        cvar = losses[losses <= p95].mean()
        out["cvar_95"] = float(cvar) if pd.notna(cvar) else None
    except Exception:
        pass
    return out


def _summarize_symbol(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {
            "trades": 0,
            "executed": 0,
            "pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": None,
            "median_hold_min": None,
            "exit_reasons": {},
            "skip_reasons": {},
        "gate_failures": 0,
    }
    pnl_series = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    profit_factor = None
    if not losses.empty:
        profit_factor = wins.sum() / abs(losses.sum()) if wins.sum() != 0 else 0.0
    tail = _tail_stats(pnl_series)
    avg_win = float(wins.mean()) if not wins.empty else None
    avg_loss = float(losses.mean()) if not losses.empty else None
    payoff_ratio = None
    if avg_win is not None and avg_loss is not None and avg_loss != 0:
        payoff_ratio = avg_win / abs(avg_loss)
    summary = {
        "trades": int(len(df)),
        "executed": int(df["executed"].fillna(False).sum()),
        "pnl": float(pnl_series.sum()),
        "win_rate": float((pnl_series > 0).mean()) if len(pnl_series) else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "median_hold_min": float(df["hold_minutes"].median()) if df["hold_minutes"].notna().any() else None,
        "mean_hold_min": float(df["hold_minutes"].mean()) if df["hold_minutes"].notna().any() else None,
        "exit_reasons": Counter(df["exit_reason_primary"].fillna("<none>")).most_common(),
        "skip_reasons": Counter(df["skip_reason"].fillna("<none>")).most_common(),
        "gate_failures": int((df["gate_pass"] == False).sum()),  # noqa: E712
    }
    summary.update(tail)
    return summary


def _equity_curve(df: pd.DataFrame) -> Tuple[List[Dict[str, object]], float]:
    if df.empty:
        return [], 0.0
    df_sorted = df.sort_values(by="exit_ts")
    equity = 0.0
    peak = 0.0
    curve = []
    for _, row in df_sorted.iterrows():
        pnl_val = row.get("pnl")
        if pd.isna(pnl_val):
            pnl_val = 0.0
        pnl = float(pnl_val)
        equity += pnl
        peak = max(peak, equity)
        drawdown = peak - equity
        curve.append(
            {
                "ts": row["exit_ts"].isoformat() if pd.notna(row["exit_ts"]) else None,
                "equity": equity,
                "drawdown": drawdown,
            }
        )
    max_dd = max((point["drawdown"] for point in curve), default=0.0)
    return curve, max_dd


def _render_markdown(summary: Dict[str, object], output_dir: Path) -> None:
    lines: List[str] = []
    lines.append("# Trading log forensics")
    lines.append("")
    portfolio = summary["portfolio"]
    lines.append(f"- Symbols: {', '.join(summary['symbols'])}")
    lines.append(f"- Trades: {portfolio['trades']} (executed: {portfolio['executed']})")
    lines.append(
        f"- PnL: {portfolio['pnl']:.6f} | Win rate: {portfolio['win_rate']:.2%} | Profit factor: {portfolio.get('profit_factor')} | Payoff: {portfolio.get('payoff_ratio')}"
    )
    lines.append(
        f"- Avg win: {portfolio.get('avg_win')} | Avg loss: {portfolio.get('avg_loss')} | P95 loss: {portfolio.get('p95_loss')} | CVaR95: {portfolio.get('cvar_95')}"
    )
    lines.append(f"- Max drawdown (pnl units): {portfolio.get('max_drawdown')} | Max loss: {portfolio.get('max_loss')} | Max win: {portfolio.get('max_win')}")
    lines.append("")
    for sym, data in summary["per_symbol"].items():
        lines.append(f"## {sym}")
        lines.append(f"- Trades: {data['trades']} (executed: {data['executed']})")
        lines.append(
            f"- PnL: {data['pnl']:.6f} | Win rate: {data['win_rate']:.2%} | Profit factor: {data.get('profit_factor')} | Payoff: {data.get('payoff_ratio')}"
        )
        lines.append(
            f"- Avg win: {data.get('avg_win')} | Avg loss: {data.get('avg_loss')} | P95 loss: {data.get('p95_loss')} | CVaR95: {data.get('cvar_95')}"
        )
        lines.append(f"- Median hold (min): {data.get('median_hold_min')}")
        top_exit = ", ".join(f"{k} ({v})" for k, v in data["exit_reasons"][:5])
        top_skip = ", ".join(f"{k} ({v})" for k, v in data["skip_reasons"][:5])
        lines.append(f"- Top exit reasons: {top_exit}")
        lines.append(f"- Top skip reasons: {top_skip}")
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "forensics_summary.md").write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    audit_path = Path(args.audit_log).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    events = _safe_json_lines(audit_path)
    trades_df = _build_trade_rows(events)
    if args.symbols:
        symbols_filter = {s.strip() for s in args.symbols.split(",") if s.strip()}
        trades_df = trades_df[trades_df["symbol"].isin(symbols_filter)]
    if args.start_ts or args.end_ts:
        start_ts = pd.to_datetime(args.start_ts, utc=True) if args.start_ts else None
        end_ts = pd.to_datetime(args.end_ts, utc=True) if args.end_ts else None
        if start_ts is not None:
            trades_df = trades_df[trades_df["exit_ts"] >= start_ts]
        if end_ts is not None:
            trades_df = trades_df[trades_df["exit_ts"] <= end_ts]
    symbols = sorted(trades_df["symbol"].dropna().unique().tolist())

    per_symbol_summary: Dict[str, dict] = {}
    for sym in symbols:
        per_symbol_summary[sym] = _summarize_symbol(trades_df[trades_df["symbol"] == sym])

    portfolio_curve, portfolio_max_dd = _equity_curve(trades_df)
    portfolio_summary = _summarize_symbol(trades_df)
    portfolio_summary["max_drawdown"] = portfolio_max_dd

    headline = []
    for sym in symbols:
        data = per_symbol_summary.get(sym, {})
        headline.append(
            {
                "symbol": sym,
                "trades": data.get("trades"),
                "win_rate": data.get("win_rate"),
                "avg_win": data.get("avg_win"),
                "avg_loss": data.get("avg_loss"),
                "payoff_ratio": data.get("payoff_ratio"),
                "profit_factor": data.get("profit_factor"),
                "cvar_95": data.get("cvar_95"),
                "p95_loss": data.get("p95_loss"),
                "max_loss": data.get("max_loss"),
                "pnl": data.get("pnl"),
            }
        )

    summary = {
        "symbols": symbols,
        "per_symbol": per_symbol_summary,
        "portfolio": portfolio_summary,
        "equity_curve": portfolio_curve,
        "headline": headline,
    }

    trades_df.sort_values(by="exit_ts").to_csv(output_dir / "per_symbol_trades.csv", index=False)
    (output_dir / "forensics_summary.json").write_text(json.dumps(summary, indent=2))
    _render_markdown(summary, output_dir)


if __name__ == "__main__":
    main()
