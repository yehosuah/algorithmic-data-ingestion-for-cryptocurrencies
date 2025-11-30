from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from analysis.validate_deployment_contract import validate_deployment_contract


def _latest_readiness_report(reports_dir: Path) -> Optional[Path]:
    candidates = sorted(reports_dir.glob("shadow_readiness_*.json"))
    return candidates[-1] if candidates else None


def _resolve_shadow_symbols(contract_summary: dict) -> List[str]:
    symbol_cfg = contract_summary.get("symbols", {}) or {}
    shadow_map = symbol_cfg.get("symbol_shadow_mode", {}) or {}
    return [sym for sym, enabled in shadow_map.items() if enabled]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Preflight checks before promoting shadow symbols to live.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--contract", default="configs/deployment_portfolio_contract.yaml")
    ap.add_argument("--shadow-report", default=None, help="Optional explicit shadow readiness report path (.json).")
    ap.add_argument("--reports-dir", default="reports", help="Directory to scan for the latest readiness report.")
    ap.add_argument("--min-would-enter", type=int, default=None, help="Override minimum entry attempts required.")
    ap.add_argument(
        "--max-risk-block-rate",
        type=float,
        default=None,
        help="Override max allowable risk block rate for promotion readiness.",
    )
    ap.add_argument(
        "--max-spread-block-rate",
        type=float,
        default=None,
        help="Override max allowable spread block rate for promotion readiness.",
    )
    ap.add_argument(
        "--require-provenance",
        action="store_true",
        default=True,
        help="Fail if readiness report is missing audit provenance metadata.",
    )
    ap.add_argument(
        "--allow-unverified-provenance",
        dest="require_provenance",
        action="store_false",
        help="Allow readiness report without provenance metadata (unsafe).",
    )
    ap.add_argument(
        "--require-hmac",
        action="store_true",
        default=True,
        help="Fail if readiness report provenance is not HMAC-verified.",
    )
    ap.add_argument(
        "--allow-unauthenticated",
        dest="require_hmac",
        action="store_false",
        help="Allow readiness report without HMAC verification (unsafe).",
    )
    ap.add_argument(
        "--allow-multi-run",
        action="store_true",
        help="Allow multiple audit_run_ids in readiness provenance when set.",
    )
    args = ap.parse_args(argv)

    summary = validate_deployment_contract(args.contract)
    shadow_override = os.getenv("TRADING_SHADOW_SYMBOLS", "")
    if shadow_override.strip():
        target_symbols = [s.strip().upper() for s in shadow_override.split(",") if s.strip()]
    else:
        target_symbols = _resolve_shadow_symbols(summary)

    if not target_symbols:
        print("No shadow symbols detected; nothing to promote.")
        return 0

    report_path = Path(args.shadow_report) if args.shadow_report else _latest_readiness_report(Path(args.reports_dir))
    if not report_path or not report_path.exists():
        print("DO NOT promote: readiness report missing. Generate a shadow readiness report first.")
        return 1

    report = json.loads(report_path.read_text())
    provenance = report.get("provenance") or {}
    if args.require_provenance and not provenance:
        print("DO NOT promote: readiness report missing provenance metadata.")
        return 1
    if provenance:
        audit_source = provenance.get("audit_source")
        if audit_source and audit_source != "runtime":
            print(f"DO NOT promote: unexpected audit_source {audit_source} in readiness provenance.")
            return 1
        run_ids = provenance.get("run_ids") or []
        if run_ids and len(run_ids) > 1 and not args.allow_multi_run:
            print("DO NOT promote: readiness aggregates multiple audit_run_ids; rerun with --allow-multi-run to override.")
            return 1
        if args.require_hmac and not provenance.get("hmac_validated"):
            print("DO NOT promote: readiness provenance missing verified HMAC.")
            return 1
    thresholds = report.get("thresholds", {}) or {}
    thresholds_by_symbol = report.get("thresholds_by_symbol", {}) or {}
    min_would_enter = args.min_would_enter if args.min_would_enter is not None else thresholds.get("min_would_enter", 1)
    max_risk_block_rate = (
        args.max_risk_block_rate if args.max_risk_block_rate is not None else thresholds.get("max_risk_block_rate", 1.0)
    )
    max_spread_block_rate = (
        args.max_spread_block_rate
        if args.max_spread_block_rate is not None
        else thresholds.get("max_spread_block_rate", 1.0)
    )

    failures: List[str] = []
    for sym in target_symbols:
        sym_thresholds = thresholds_by_symbol.get(sym, {})
        sym_min_we = int(sym_thresholds.get("min_would_enter", min_would_enter))
        sym_max_risk = float(sym_thresholds.get("max_risk_block_rate", max_risk_block_rate))
        sym_max_spread = float(sym_thresholds.get("max_spread_block_rate", max_spread_block_rate))
        metrics = (report.get("symbols") or {}).get(sym)
        if not metrics:
            failures.append(f"{sym}: missing readiness metrics")
            continue
        reasons: List[str] = []
        if metrics.get("would_enter", 0) < sym_min_we:
            reasons.append(f"would_enter<{sym_min_we}")
        if metrics.get("implied_trades", 0) <= 0:
            reasons.append("no_implied_trades")
        if metrics.get("risk_block_rate", 0.0) > sym_max_risk:
            reasons.append(f"risk_block_rate>{sym_max_risk}")
        if metrics.get("spread_block_rate", 0.0) > sym_max_spread:
            reasons.append(f"spread_block_rate>{sym_max_spread}")
        if metrics.get("promotion_ready") is False:
            reasons.extend(metrics.get("promotion_reasons", []))
        if reasons:
            failures.append(f"{sym}: {', '.join(reasons)}")

    if failures:
        print("DO NOT promote:")
        for reason in failures:
            print(f"- {reason}")
        return 1

    print("OK to promote the following symbols to live:")
    for sym in target_symbols:
        print(f"- {sym}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
