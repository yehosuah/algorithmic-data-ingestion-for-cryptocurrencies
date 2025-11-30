from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import os
import yaml
import hashlib
import hmac

DEFAULT_SYMBOLS = ("BTC/USDT", "SOL/USDT")


def _parse_ts(raw: object) -> Optional[datetime]:
    if raw is None:
        return None
    try:
        ts = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").replace(" ", "").upper()


def _load_risk_capital(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return 0.0
    try:
        return float(cfg.get("capital", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _compute_record_hmac(record: dict, key: bytes) -> str:
    body = {
        "audit_run_id": record.get("audit_run_id"),
        "audit_seq": record.get("audit_seq"),
        "occurred_at": record.get("occurred_at"),
        "event_type": record.get("event_type"),
        "model": record.get("model"),
        "symbol": record.get("symbol"),
        "payload": record.get("payload"),
    }
    payload_str = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hmac.new(key, payload_str.encode("utf-8"), hashlib.sha256).hexdigest()


def _load_audit_events(
    path: Path,
    symbols: Iterable[str],
    window_start: datetime,
    *,
    time_min: Optional[datetime],
    time_max: Optional[datetime],
    audit_source: str,
    allow_multi_run: bool,
    require_hmac: bool,
    hmac_key: Optional[str],
) -> Tuple[List[Tuple[datetime, dict]], dict]:
    target = {_normalize_symbol(sym) for sym in symbols}
    events: List[Tuple[datetime, dict]] = []
    state: Dict[str, int] = {}
    provenance = {
        "audit_source": audit_source,
        "run_ids": set(),
        "hmac_validated": False,
    }
    key_bytes = hmac_key.encode("utf-8") if hmac_key else None
    if not path.exists():
        raise FileNotFoundError(f"Audit log not found at {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(record.get("audit_source")) != audit_source:
                raise ValueError(f"Unexpected audit_source {record.get('audit_source')} in {path} (expected {audit_source})")
            run_id = record.get("audit_run_id")
            seq = record.get("audit_seq")
            if run_id is None or seq is None:
                raise ValueError("Audit record missing audit_run_id or audit_seq.")
            try:
                seq_int = int(seq)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid audit_seq value {seq!r}")
            if run_id in state and seq_int <= state[run_id]:
                raise ValueError(f"Non-monotonic audit_seq for run {run_id}: {seq_int} <= {state[run_id]}")
            state[run_id] = seq_int
            provenance["run_ids"].add(str(run_id))
            if len(provenance["run_ids"]) > 1 and not allow_multi_run:
                raise ValueError("Mixed audit_run_id values detected; rerun with --allow-multi-run to proceed.")
            if require_hmac:
                if not key_bytes:
                    raise ValueError("HMAC verification requested but no key provided.")
                expected = _compute_record_hmac(record, key_bytes)
                if str(record.get("audit_hmac") or "") != expected:
                    raise ValueError("Audit HMAC verification failed; aborting.")
            ts = _parse_ts(record.get("occurred_at"))
            if ts is None or ts < window_start:
                continue
            if time_min and ts < time_min:
                continue
            if time_max and ts > time_max:
                continue
            symbol = _normalize_symbol(record.get("symbol"))
            if target and symbol not in target:
                continue
            events.append((ts, record))
    events.sort(key=lambda pair: pair[0])
    provenance["run_ids"] = sorted(provenance["run_ids"])
    provenance["last_seq"] = {k: int(v) for k, v in state.items()}
    provenance["hmac_validated"] = require_hmac
    return events, provenance


def _should_count_spread_block(reason: str, risk_reason: str) -> bool:
    reason_lower = (reason or "").lower()
    risk_lower = (risk_reason or "").lower()
    return "spread" in reason_lower or "spread" in risk_lower


def _extract_notional(payload: dict) -> Optional[float]:
    for key in ("notional", "intended_notional", "risk_final_notional"):
        if key in payload and payload.get(key) is not None:
            try:
                value = float(payload[key])
                return value
            except (TypeError, ValueError):
                continue
    return None


def _evaluate_symbol_events(
    events: List[Tuple[datetime, dict]],
    symbol: str,
    *,
    window_start: datetime,
    window_end: datetime,
    capital: float,
    min_would_enter: int,
    max_risk_block_rate: float,
    max_spread_block_rate: float,
) -> Dict[str, object]:
    stats = {
        "would_enter": 0,
        "would_exit": 0,
        "implied_trades": 0,
        "risk_blocks": defaultdict(int),
        "spread_blocks": 0,
        "spread_samples": [],
        "turnover": 0.0,
        "time_in_position": 0.0,
        "attempts": 0,
    }
    state = {"in_position": False, "entry_ts": None}

    for ts, record in events:
        payload = record.get("payload") or {}
        side = str(payload.get("side") or "").lower()
        gate_pass = bool(payload.get("gate_pass"))
        if not gate_pass:
            continue
        stats["attempts"] += 1
        if side == "buy":
            stats["would_enter"] += 1
        elif side == "sell":
            stats["would_exit"] += 1

        risk_allowed = payload.get("risk_allowed")
        risk_reason = str(payload.get("risk_block_reason") or payload.get("blocked_reason") or "")
        spread_block = _should_count_spread_block(
            risk_reason,
            str(payload.get("risk_block_reason") or ""),
        )
        if risk_allowed is False:
            stats["risk_blocks"][risk_reason or "unknown"] += 1
        if spread_block:
            stats["spread_blocks"] += 1
        if payload.get("spread_bps") is not None:
            try:
                stats["spread_samples"].append(float(payload["spread_bps"]))
            except (TypeError, ValueError):
                pass

        would_execute = gate_pass and risk_allowed is not False and not spread_block
        would_execute = would_execute and (payload.get("executed") or payload.get("shadow_mode"))
        if not would_execute or payload.get("dedup_blocked"):
            continue

        notional = _extract_notional(payload)
        if notional is not None:
            stats["turnover"] += abs(float(notional))
        stats["implied_trades"] += 1

        if side == "buy" and not state["in_position"]:
            state["in_position"] = True
            state["entry_ts"] = ts
        elif side == "sell" and state["in_position"]:
            entry_ts = state["entry_ts"] or window_start
            stats["time_in_position"] += max(0.0, (ts - entry_ts).total_seconds())
            state["in_position"] = False
            state["entry_ts"] = None

    if state["in_position"] and state["entry_ts"]:
        stats["time_in_position"] += max(0.0, (window_end - state["entry_ts"]).total_seconds())

    window_seconds = max(1.0, (window_end - window_start).total_seconds())
    attempts = max(1, stats["attempts"])
    risk_block_rate = sum(stats["risk_blocks"].values()) / attempts
    spread_block_rate = stats["spread_blocks"] / attempts
    reasons: List[str] = []
    if stats["would_enter"] < min_would_enter:
        reasons.append(f"would_enter<{min_would_enter}")
    if stats["implied_trades"] <= 0:
        reasons.append("no_implied_trades")
    if risk_block_rate > max_risk_block_rate:
        reasons.append(f"risk_block_rate>{max_risk_block_rate:.2f}")
    if spread_block_rate > max_spread_block_rate:
        reasons.append(f"spread_block_rate>{max_spread_block_rate:.2f}")

    promotable = not reasons
    avg_spread = None
    if stats["spread_samples"]:
        avg_spread = sum(stats["spread_samples"]) / len(stats["spread_samples"])

    return {
        "symbol": symbol,
        "would_enter": stats["would_enter"],
        "would_exit": stats["would_exit"],
        "implied_trades": stats["implied_trades"],
        "fraction_time_in_position": stats["time_in_position"] / window_seconds if window_seconds else 0.0,
        "avg_spread_bps": avg_spread,
        "risk_block_breakdown": dict(stats["risk_blocks"]),
        "risk_block_rate": risk_block_rate,
        "spread_block_rate": spread_block_rate,
        "turnover_notional": stats["turnover"],
        "turnover_fraction": (stats["turnover"] / capital) if capital else None,
        "promotion_ready": promotable,
        "promotion_reasons": reasons,
        "samples": stats["attempts"],
    }


def _render_markdown(report: dict) -> str:
    lines = [
        f"# Shadow Readiness ({report.get('window_hours')}h window)",
        "",
        f"Generated at: {report.get('generated_at')}",
        f"Audit log: {report.get('audit_log')}",
        "",
    ]
    for sym, metrics in report.get("symbols", {}).items():
        lines.append(f"## {sym}")
        lines.append(f"- Promotion ready: {'YES' if metrics.get('promotion_ready') else 'NO'}")
        if metrics.get("promotion_reasons"):
            lines.append(f"- Reasons: {', '.join(metrics.get('promotion_reasons', []))}")
        lines.append(f"- Would-enter count: {metrics.get('would_enter')}")
        lines.append(f"- Would-exit count: {metrics.get('would_exit')}")
        lines.append(f"- Implied trades: {metrics.get('implied_trades')}")
        lines.append(f"- Fraction time in position: {metrics.get('fraction_time_in_position'):.4f}")
        lines.append(f"- Avg spread (bps): {metrics.get('avg_spread_bps')}")
        lines.append(f"- Risk block rate: {metrics.get('risk_block_rate'):.4f}")
        lines.append(f"- Spread block rate: {metrics.get('spread_block_rate'):.4f}")
        lines.append(f"- Turnover notional: {metrics.get('turnover_notional')}")
        lines.append(f"- Turnover fraction of capital: {metrics.get('turnover_fraction')}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute shadow-mode readiness scores from trading audit logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--audit-log", default="data_lake/trading/audit.log", help="Path to audit log file (JSON lines).")
    ap.add_argument("--risk-limits", default="configs/portfolio_risk_limits.yaml", help="Risk limits file for capital.")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Comma-separated list of symbols to evaluate.")
    ap.add_argument("--hours", type=int, default=24, help="Lookback window in hours.")
    ap.add_argument("--output-dir", default="reports", help="Directory to write readiness reports.")
    ap.add_argument("--min-would-enter", type=int, default=3, help="Minimum entry attempts required for readiness.")
    ap.add_argument("--max-risk-block-rate", type=float, default=0.2, help="Max allowable risk block rate.")
    ap.add_argument("--max-spread-block-rate", type=float, default=0.2, help="Max allowable spread block rate.")
    ap.add_argument(
        "--min-would-enter-map",
        default=None,
        help="Optional JSON mapping of symbol -> min would-enter threshold (overrides global).",
    )
    ap.add_argument(
        "--max-risk-block-rate-map",
        default=None,
        help="Optional JSON mapping of symbol -> max risk block rate (overrides global).",
    )
    ap.add_argument(
        "--max-spread-block-rate-map",
        default=None,
        help="Optional JSON mapping of symbol -> max spread block rate (overrides global).",
    )
    ap.add_argument("--time-min", default=None, help="Minimum occurred_at timestamp (ISO).")
    ap.add_argument("--time-max", default=None, help="Maximum occurred_at timestamp (ISO).")
    ap.add_argument("--allow-multi-run", action="store_true", help="Allow mixing audit_run_ids in readiness input.")
    ap.add_argument("--audit-source", default="runtime", help="Expected audit_source value.")
    ap.add_argument(
        "--require-hmac",
        action="store_true",
        default=False,
        help="Require audit_hmac validation (defaults to true when TRADING_AUDIT_HMAC_KEY is present).",
    )
    ap.add_argument("--hmac-key", default=None, help="Optional override for TRADING_AUDIT_HMAC_KEY.")
    args = ap.parse_args(argv)

    symbol_min = json.loads(args.min_would_enter_map) if args.min_would_enter_map else {}
    symbol_risk = json.loads(args.max_risk_block_rate_map) if args.max_risk_block_rate_map else {}
    symbol_spread = json.loads(args.max_spread_block_rate_map) if args.max_spread_block_rate_map else {}

    symbols = [_normalize_symbol(sym) for sym in args.symbols.split(",") if sym.strip()]
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=max(1, int(args.hours)))
    time_min = _parse_ts(args.time_min)
    time_max = _parse_ts(args.time_max)
    if time_min and time_min > window_start:
        window_start = time_min
    hmac_key = args.hmac_key or os.getenv("TRADING_AUDIT_HMAC_KEY")
    require_hmac = bool(args.require_hmac or hmac_key)
    audit_path = Path(args.audit_log)
    risk_capital = _load_risk_capital(Path(args.risk_limits))
    events, provenance = _load_audit_events(
        audit_path,
        symbols,
        window_start,
        time_min=time_min,
        time_max=time_max,
        audit_source=args.audit_source,
        allow_multi_run=bool(args.allow_multi_run),
        require_hmac=require_hmac,
        hmac_key=hmac_key,
    )

    per_symbol: Dict[str, object] = {}
    for sym in symbols:
        min_we = int(symbol_min.get(sym, args.min_would_enter))
        max_risk_rate = float(symbol_risk.get(sym, args.max_risk_block_rate))
        max_spread_rate = float(symbol_spread.get(sym, args.max_spread_block_rate))
        sym_events = [(ts, rec) for ts, rec in events if _normalize_symbol(rec.get("symbol")) == sym]
        per_symbol[sym] = _evaluate_symbol_events(
            sym_events,
            sym,
            window_start=window_start,
            window_end=window_end,
            capital=risk_capital,
            min_would_enter=min_we,
            max_risk_block_rate=max_risk_rate,
            max_spread_block_rate=max_spread_rate,
        )
        per_symbol[sym]["thresholds_used"] = {
            "min_would_enter": min_we,
            "max_risk_block_rate": max_risk_rate,
            "max_spread_block_rate": max_spread_rate,
        }

    report = {
        "generated_at": window_end.isoformat(),
        "window_hours": args.hours,
        "audit_log": str(audit_path),
        "risk_limits": str(args.risk_limits),
        "symbols": per_symbol,
        "thresholds": {
            "min_would_enter": args.min_would_enter,
            "max_risk_block_rate": args.max_risk_block_rate,
            "max_spread_block_rate": args.max_spread_block_rate,
        },
        "provenance": {
            **provenance,
            "time_min": time_min.isoformat() if time_min else None,
            "time_max": time_max.isoformat() if time_max else None,
        },
        "thresholds_by_symbol": {sym: per_symbol[sym]["thresholds_used"] for sym in per_symbol},
        "capital": risk_capital,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = window_end.strftime("%Y%m%d_%H%M")
    json_path = output_dir / f"shadow_readiness_{stamp}.json"
    md_path = output_dir / f"shadow_readiness_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    md_path.write_text(_render_markdown(report))
    print(f"Wrote shadow readiness report to {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
