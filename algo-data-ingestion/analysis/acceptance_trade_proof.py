from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_ts(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def build_trade_records(audit_path: Path) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    with audit_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event_type") != "trade":
                continue
            payload = entry.get("payload") or {}
            if payload.get("side") != "sell":
                continue
            if payload.get("executed") is False:
                continue
            trade = {
                "model": entry.get("model"),
                "symbol": entry.get("symbol"),
                "policy_id": payload.get("policy_id"),
                "exit_reason": payload.get("exit_reason_primary") or payload.get("reason"),
                "exit_reason_all": payload.get("exit_reason_all") or [],
                "entry_ts": _parse_ts(payload.get("entry_ts")),
                "exit_ts": _parse_ts(entry.get("occurred_at") or payload.get("ts")),
                "entry_price": payload.get("entry_price"),
                "exit_price": payload.get("price_used") or payload.get("decision_price_used"),
                "entry_amount": payload.get("entry_amount"),
                "exit_amount": payload.get("decision_amount") or payload.get("amount"),
                "pnl_gross": payload.get("pnl_gross"),
                "pnl_net_estimate": payload.get("pnl_net_estimate"),
                "pnl_net_realized": payload.get("pnl_net_realized"),
                "spread_bps": payload.get("spread_bps") or payload.get("spread_bps_now"),
            }
            trades.append(trade)
    return trades


def summarize(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"trade_count": len(trades), "by_symbol": {}}
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for trade in trades:
        sym = trade.get("symbol")
        by_symbol.setdefault(sym, []).append(trade)
    for sym, rows in by_symbol.items():
        pnl_vals = [t.get("pnl_net_realized") for t in rows if isinstance(t.get("pnl_net_realized"), (int, float))]
        gross_vals = [t.get("pnl_gross") for t in rows if isinstance(t.get("pnl_gross"), (int, float))]
        summary["by_symbol"][sym] = {
            "count": len(rows),
            "positive_fraction_net": float(sum(1 for v in pnl_vals if v is not None and v > 0) / len(rows)) if rows else 0.0,
            "total_pnl_net": float(sum(pnl_vals)) if pnl_vals else None,
            "total_pnl_gross": float(sum(gross_vals)) if gross_vals else None,
        }
    return summary


def write_md(trades: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    lines = ["# Acceptance Trade Proof", ""]
    lines.append(f"Total trades: {summary.get('trade_count')}")
    lines.append("")
    lines.append("## Summary by symbol")
    for sym, stats in (summary.get("by_symbol") or {}).items():
        lines.append(f"- {sym}: count={stats.get('count')} positive_fraction_net={stats.get('positive_fraction_net'):.3f} total_pnl_net={stats.get('total_pnl_net')}")
    lines.append("")
    lines.append("## Trades")
    for t in trades:
        lines.append(
            f"- {t.get('symbol')} exit={t.get('exit_ts')} entry={t.get('entry_ts')} "
            f"entry_price={t.get('entry_price')} exit_price={t.get('exit_price')} "
            f"pnl_net={t.get('pnl_net_realized')} reason={t.get('exit_reason')}"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build acceptance trade proof from audit log.")
    ap.add_argument("--audit-log", required=True, help="Path to audit log.")
    ap.add_argument("--output-prefix", default=None, help="Prefix for output files.")
    args = ap.parse_args(argv)
    trades = build_trade_records(Path(args.audit_log))
    summary = summarize(trades)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prefix = Path(args.output_prefix) if args.output_prefix else Path("reports") / f"acceptance_trade_proof_{stamp}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    payload = {"generated_at": datetime.utcnow().isoformat() + "Z", "trades": trades, "summary": summary}
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(write_md(trades, summary))
    print(f"Wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
