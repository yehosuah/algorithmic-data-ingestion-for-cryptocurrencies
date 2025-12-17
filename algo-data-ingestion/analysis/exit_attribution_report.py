from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _parse_ts(value: str) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _load_audit(path: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
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
            reason = payload.get("exit_reason_primary") or payload.get("reason") or ""
            reasons_all = payload.get("exit_reason_all") or payload.get("reasons") or []
            ts = _parse_ts(entry.get("occurred_at", "") or payload.get("ts", ""))
            entry_ts = _parse_ts(payload.get("entry_ts", "") or payload.get("entry_time", ""))
            hold_seconds = None
            if ts and entry_ts:
                hold_seconds = (ts - entry_ts).total_seconds()
            bar_seconds = payload.get("bar_seconds") or payload.get("bar_sec") or 60
            pnl_gross = payload.get("pnl_gross")
            pnl_net = payload.get("pnl_net_estimate", payload.get("pnl_net"))
            pnl_realized = payload.get("pnl_net_realized") or payload.get("pnl_realized") or payload.get("pnl")
            record = {
                "symbol": payload.get("symbol") or entry.get("symbol"),
                "timestamp": ts,
                "reason": reason or None,
                "reasons_all": reasons_all if isinstance(reasons_all, list) else [reasons_all],
                "executed": bool(payload.get("executed", True)),
                "pnl_gross": float(pnl_gross) if pnl_gross is not None else None,
                "pnl_net": float(pnl_net) if pnl_net is not None else None,
                "pnl_realized": float(pnl_realized) if pnl_realized is not None else None,
                "hold_seconds": hold_seconds,
                "bar_seconds": float(bar_seconds) if bar_seconds is not None else 60.0,
                "min_hold_bars": payload.get("min_hold_bars"),
                "exit_reason_all": reasons_all if isinstance(reasons_all, list) else [reasons_all],
                "exit_reason_primary": reason or None,
            }
            records.append(record)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _safe_mean(series: pd.Series) -> float:
    valid = series.dropna()
    return float(valid.mean()) if len(valid) else 0.0


def build_report(df: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {"trade_count": int(len(df))}
    if df.empty:
        return report

    df["reason_clean"] = df["exit_reason_primary"].fillna("unknown")
    df["hour"] = df["timestamp"].dt.floor("h")
    df["pnl_use"] = df["pnl_realized"].fillna(df["pnl_net"]).fillna(df["pnl_gross"])
    executed_df = df[df["executed"] == True].copy()  # noqa: E712

    by_reason = []
    churn_rows = []
    negative_rates = []
    for reason, group in df.groupby("reason_clean"):
        pnl_vals = group["pnl_use"].dropna()
        by_reason.append(
            {
                "reason": reason,
                "count": int(len(group)),
                "avg_pnl": _safe_mean(group["pnl_use"]),
                "avg_pnl_gross": _safe_mean(group["pnl_gross"]),
                "avg_pnl_net": _safe_mean(group["pnl_net"]),
                "avg_hold_seconds": _safe_mean(group["hold_seconds"]),
                "executed_fraction": float(group["executed"].mean()),
            }
        )
        churn_rows.append(
            {
                "reason": reason,
                "fraction_le_one_bar": float((group["hold_seconds"] <= group["bar_seconds"]).mean())
                if "hold_seconds" in group
                else 0.0,
            }
        )
        negative_rates.append(
            {
                "reason": reason,
                "negative_fraction": float((pnl_vals < 0).mean()) if not pnl_vals.empty else 0.0,
            }
        )

    by_symbol_reason: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (sym, reason), count in df.groupby(["symbol", "reason_clean"]).size().items():
        by_symbol_reason[str(sym)][str(reason)] = int(count)

    by_hour_reason: List[Dict[str, Any]] = []
    for (hour, reason), count in df.groupby(["hour", "reason_clean"]).size().items():
        by_hour_reason.append({"hour": hour.isoformat() if pd.notna(hour) else None, "reason": reason, "count": int(count)})

    churn_fraction = float((df["hold_seconds"] <= df["bar_seconds"]).mean()) if "hold_seconds" in df else 0.0
    pnl_vals = df["pnl_use"].dropna()
    negative_fraction = float((pnl_vals < 0).mean()) if not pnl_vals.empty else 0.0
    executed_pnl_vals = executed_df["pnl_use"].dropna()
    executed_negative_fraction = float((executed_pnl_vals < 0).mean()) if not executed_pnl_vals.empty else 0.0

    executed_by_reason = []
    for reason, group in executed_df.groupby("reason_clean"):
        pnl_vals = group["pnl_use"].dropna()
        executed_by_reason.append(
            {
                "reason": reason,
                "count": int(len(group)),
                "avg_pnl": _safe_mean(group["pnl_use"]),
                "avg_pnl_gross": _safe_mean(group["pnl_gross"]),
                "avg_pnl_net": _safe_mean(group["pnl_net"]),
                "avg_hold_seconds": _safe_mean(group["hold_seconds"]),
                "negative_fraction": float((pnl_vals < 0).mean()) if not pnl_vals.empty else 0.0,
            }
        )

    report.update(
        {
            "by_reason": by_reason,
            "by_reason_executed": executed_by_reason,
            "by_symbol_reason": by_symbol_reason,
            "by_hour_reason": by_hour_reason,
            "churn_fraction_le_one_bar": churn_fraction,
            "negative_fraction": negative_fraction,
            "executed_trade_count": int(len(executed_df)),
            "executed_negative_fraction": executed_negative_fraction,
            "negative_fraction_by_reason": negative_rates,
            "churn_fraction_by_reason": churn_rows,
            "avg_hold_seconds": _safe_mean(df["hold_seconds"]),
            "avg_pnl": _safe_mean(df["pnl_use"]),
            "avg_pnl_gross": _safe_mean(df["pnl_gross"]),
            "avg_pnl_net": _safe_mean(df["pnl_net"]),
        }
    )
    return report


def _report_to_markdown(report: Dict[str, Any]) -> str:
    lines = ["# Exit Attribution", ""]
    lines.append(f"Total exits: {report.get('trade_count', 0)}")
    lines.append(f"Average hold seconds: {report.get('avg_hold_seconds', 0):.2f}")
    lines.append(f"Overall negative fraction: {report.get('negative_fraction', 0):.4f}")
    lines.append(f"Executed exits: {report.get('executed_trade_count', 0)}")
    lines.append(f"Executed negative fraction: {report.get('executed_negative_fraction', 0):.4f}")
    lines.append(f"Churn fraction (<=1 bar): {report.get('churn_fraction_le_one_bar', 0):.4f}")
    lines.append("")
    if report.get("by_reason"):
        lines.append("## By reason")
        df = pd.DataFrame(report["by_reason"])
        lines.append(df.to_markdown(index=False))
        lines.append("")
    if report.get("by_reason_executed"):
        lines.append("## Executed exits by reason")
        df = pd.DataFrame(report["by_reason_executed"])
        lines.append(df.to_markdown(index=False))
        lines.append("")
    if report.get("by_symbol_reason"):
        lines.append("## Reason frequency by symbol")
        entries = []
        for sym, reasons in report["by_symbol_reason"].items():
            for reason, count in reasons.items():
                entries.append({"symbol": sym, "reason": reason, "count": count})
        if entries:
            df = pd.DataFrame(entries)
            lines.append(df.to_markdown(index=False))
            lines.append("")
    if report.get("negative_fraction_by_reason"):
        lines.append("## Negative fraction by reason")
        df = pd.DataFrame(report["negative_fraction_by_reason"])
        lines.append(df.to_markdown(index=False))
        lines.append("")
    if report.get("by_hour_reason"):
        lines.append("## Exits by hour")
        df = pd.DataFrame(report["by_hour_reason"])
        lines.append(df.to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build exit attribution report from audit log.")
    ap.add_argument("--audit-log", required=True, help="Path to audit log (json lines).")
    ap.add_argument("--output-prefix", default=None, help="Prefix for report outputs (defaults to reports/exit_attribution_<ts>).")
    args = ap.parse_args(argv)

    audit_path = Path(args.audit_log)
    df = _load_audit(audit_path)
    report = build_report(df)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        prefix = Path("reports") / f"exit_attribution_{stamp}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, default=str))
    md_path.write_text(_report_to_markdown(report))
    print(f"Wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
