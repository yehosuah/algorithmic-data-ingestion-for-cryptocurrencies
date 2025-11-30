from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import yaml
from analysis.preflight_coverage import main as coverage_main
from analysis.preflight_symbol_promotion import main as promotion_main
from analysis.shadow_readiness import _load_audit_events, _should_count_spread_block
from analysis.validate_deployment_contract import _normalize_symbol, validate_deployment_contract


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _ordered_stages(launch_ladder: Mapping[str, object]) -> List[str]:
    def _stage_index(name: str) -> int:
        if name.startswith("stage_"):
            try:
                return int(name.split("_", 1)[1])
            except Exception:
                return 1_000_000
        return 1_000_000

    return sorted(list(launch_ladder.keys()), key=_stage_index)


def _parse_stage(ladder_path: Path, stage_name: str) -> Tuple[dict, dict]:
    ladder = _load_yaml(ladder_path)
    stages = ladder.get("launch_ladder") or {}
    if stage_name not in stages:
        raise KeyError(f"Stage {stage_name} missing from ladder {ladder_path}")
    return ladder, stages[stage_name]


def _baseline_metrics(live_symbols: Sequence[str]) -> Dict[str, MutableMapping[str, Any]]:
    metrics: Dict[str, MutableMapping[str, Any]] = {}
    for sym in live_symbols:
        metrics[sym] = {
            "trade_count": 0,
            "executed_count": 0,
            "gate_samples": 0,
            "gate_pass": 0,
            "risk_blocks": 0,
            "spread_blocks": 0,
            "drawdown_pct": 0.0,
        }
    return metrics


def _collect_audit_metrics(
    events: List[Tuple[datetime, dict]],
    *,
    live_symbols: Sequence[str],
) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, Any]]:
    per_symbol = _baseline_metrics(live_symbols)
    live_set = {_normalize_symbol(s) for s in live_symbols}
    safe_mode_events = 0
    reconcile_mismatches = 0
    deadlock_actions = 0
    drawdown_samples: List[float] = []
    earliest: datetime | None = None
    latest: datetime | None = None

    for ts, record in events:
        if earliest is None or ts < earliest:
            earliest = ts
        if latest is None or ts > latest:
            latest = ts

        event_type = str(record.get("event_type") or "").lower()
        symbol = _normalize_symbol(record.get("symbol")) if record.get("symbol") else None
        payload = record.get("payload") or {}

        if event_type == "gate_toggle" and symbol in live_set:
            per_symbol[symbol]["gate_samples"] += 1
            if payload.get("gate_pass"):
                per_symbol[symbol]["gate_pass"] += 1
            continue

        if event_type == "trade" and symbol in live_set:
            gate_pass = bool(payload.get("gate_pass"))
            if gate_pass:
                per_symbol[symbol]["trade_count"] += 1
            if payload.get("executed"):
                per_symbol[symbol]["executed_count"] += 1
            risk_allowed = payload.get("risk_allowed")
            risk_reason = str(payload.get("risk_block_reason") or payload.get("blocked_reason") or "")
            if gate_pass and risk_allowed is False:
                per_symbol[symbol]["risk_blocks"] += 1
            if gate_pass and _should_count_spread_block(risk_reason, risk_reason):
                per_symbol[symbol]["spread_blocks"] += 1
            risk_snapshot = payload.get("risk_snapshot") or {}
            if isinstance(risk_snapshot, Mapping):
                if risk_snapshot.get("drawdown_pct") is not None:
                    try:
                        drawdown_samples.append(float(risk_snapshot.get("drawdown_pct")))
                        per_symbol[symbol]["drawdown_pct"] = max(
                            per_symbol[symbol]["drawdown_pct"], float(risk_snapshot.get("drawdown_pct"))
                        )
                    except (TypeError, ValueError):
                        pass
            continue

        if event_type == "safe_mode" and bool(payload.get("active")):
            safe_mode_events += 1
            continue

        if event_type == "reconciliation":
            results = payload.get("results") or []
            if isinstance(results, list):
                for entry in results:
                    if not isinstance(entry, Mapping):
                        continue
                    status = str(entry.get("status") or "")
                    if status and status.lower() != "ok":
                        reconcile_mismatches += 1
            continue

        if event_type == "deadlock_action":
            deadlock_actions += 1
            continue

    window_minutes = 0.0
    if earliest and latest:
        window_minutes = max(0.0, (latest - earliest).total_seconds() / 60.0)

    summary = {
        "safe_mode_events": safe_mode_events,
        "reconcile_mismatches": reconcile_mismatches,
        "deadlock_actions": deadlock_actions,
        "drawdown_pct_max": max(drawdown_samples) if drawdown_samples else 0.0,
        "runtime_minutes": window_minutes,
    }
    return per_symbol, summary


def _evaluate_gates(
    *,
    stage_name: str,
    gates: Mapping[str, Any],
    live_symbols: Sequence[str],
    per_symbol_metrics: Mapping[str, Mapping[str, Any]],
    summary_metrics: Mapping[str, Any],
) -> List[str]:
    failures: List[str] = []
    min_trade_count = gates.get("min_trade_count")
    min_coverage_ratio = gates.get("min_coverage_ratio")
    max_spread_block_rate = gates.get("max_spread_block_rate")
    max_risk_block_rate = gates.get("max_risk_block_rate")
    max_safe_mode_events = gates.get("max_safe_mode_events")
    max_reconcile_mismatches = gates.get("max_reconcile_mismatches")
    max_deadlock_actions = gates.get("max_deadlock_actions")
    max_drawdown_pct = gates.get("max_drawdown_pct")
    min_runtime_minutes = gates.get("min_runtime_minutes")

    for sym in live_symbols:
        metrics = per_symbol_metrics.get(sym) or {}
        attempts = float(metrics.get("trade_count") or 0)
        gate_samples = float(metrics.get("gate_samples") or 0)
        gate_pass = float(metrics.get("gate_pass") or 0)
        coverage_ratio = (gate_pass / gate_samples) if gate_samples else 0.0
        risk_rate = (float(metrics.get("risk_blocks") or 0) / attempts) if attempts else 0.0
        spread_rate = (float(metrics.get("spread_blocks") or 0) / attempts) if attempts else 0.0

        if min_trade_count is not None and attempts < float(min_trade_count):
            failures.append(f"{sym}: trade_count<{min_trade_count}")
        if min_coverage_ratio is not None and coverage_ratio < float(min_coverage_ratio):
            failures.append(f"{sym}: coverage<{min_coverage_ratio}")
        if max_spread_block_rate is not None and spread_rate > float(max_spread_block_rate):
            failures.append(f"{sym}: spread_block_rate>{max_spread_block_rate}")
        if max_risk_block_rate is not None and risk_rate > float(max_risk_block_rate):
            failures.append(f"{sym}: risk_block_rate>{max_risk_block_rate}")

    if max_safe_mode_events is not None and summary_metrics.get("safe_mode_events", 0) > max_safe_mode_events:
        failures.append(f"safe_mode_events>{max_safe_mode_events}")
    if max_reconcile_mismatches is not None and summary_metrics.get("reconcile_mismatches", 0) > max_reconcile_mismatches:
        failures.append(f"reconcile_mismatches>{max_reconcile_mismatches}")
    if max_deadlock_actions is not None and summary_metrics.get("deadlock_actions", 0) > max_deadlock_actions:
        failures.append(f"deadlock_actions>{max_deadlock_actions}")
    if max_drawdown_pct is not None and summary_metrics.get("drawdown_pct_max", 0.0) > float(max_drawdown_pct):
        failures.append(f"drawdown_pct>{max_drawdown_pct}")
    if min_runtime_minutes is not None and summary_metrics.get("runtime_minutes", 0.0) < float(min_runtime_minutes):
        failures.append(f"runtime_minutes<{min_runtime_minutes}")
    return failures


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Launch Stage Evaluation: {report.get('stage')}",
        f"- Status: {report.get('status')}",
        f"- Mode: {report.get('mode')}",
        f"- Audit log: {report.get('provenance', {}).get('audit_log')}",
        f"- Runtime window (min): {report.get('metrics', {}).get('summary', {}).get('runtime_minutes', 0):.2f}",
        "",
        "## Per-symbol metrics",
    ]
    for sym, metrics in (report.get("metrics", {}).get("per_symbol") or {}).items():
        lines.append(f"### {sym}")
        lines.append(f"- trade_count: {metrics.get('trade_count')}")
        lines.append(f"- executed_count: {metrics.get('executed_count')}")
        lines.append(f"- coverage_ratio: {metrics.get('coverage_ratio'):.4f}")
        lines.append(f"- risk_block_rate: {metrics.get('risk_block_rate'):.4f}")
        lines.append(f"- spread_block_rate: {metrics.get('spread_block_rate'):.4f}")
        lines.append("")
    lines.append("## Summary gates")
    lines.append(f"- safe_mode_events: {report.get('metrics', {}).get('summary', {}).get('safe_mode_events', 0)}")
    lines.append(
        f"- reconcile_mismatches: {report.get('metrics', {}).get('summary', {}).get('reconcile_mismatches', 0)}"
    )
    lines.append(f"- deadlock_actions: {report.get('metrics', {}).get('summary', {}).get('deadlock_actions', 0)}")
    lines.append(f"- drawdown_pct_max: {report.get('metrics', {}).get('summary', {}).get('drawdown_pct_max', 0.0):.4f}")
    lines.append("")
    if report.get("failures"):
        lines.append("## NO-GO reasons")
        for reason in report["failures"]:
            lines.append(f"- {reason}")
    else:
        lines.append("## GO")
        lines.append("All promotion gates satisfied.")
    return "\n".join(lines)


def _promotion_targets(launch_ladder: Mapping[str, Any], stage_name: str) -> Sequence[str]:
    ordered = _ordered_stages(launch_ladder)
    if stage_name not in ordered:
        return []
    idx = ordered.index(stage_name)
    if idx <= 0:
        return []
    previous_stage = launch_ladder.get(ordered[idx - 1]) or {}
    current_stage = launch_ladder.get(stage_name) or {}
    prev_live = [_normalize_symbol(s) for s in previous_stage.get("live_symbols") or []]
    current_live = [_normalize_symbol(s) for s in current_stage.get("live_symbols") or []]
    return [sym for sym in current_live if sym not in prev_live]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate whether a launch ladder stage meets promotion gates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--stage", required=True, help="Stage key from launch ladder (e.g., stage_2)")
    ap.add_argument("--ladder", default="configs/live_launch_ladder.yaml", help="Launch ladder config path.")
    ap.add_argument("--contract", default="configs/deployment_portfolio_contract.yaml", help="Deployment contract path.")
    ap.add_argument("--audit-log", default="data_lake/trading/audit.log", help="Audit log path (JSON lines).")
    ap.add_argument("--hours", type=int, default=48, help="Audit lookback window in hours.")
    ap.add_argument("--reports-dir", default="reports", help="Directory to write evaluation reports.")
    ap.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="Allow audit logs without verified HMAC (unsafe; defaults to False for live stages).",
    )
    ap.add_argument("--allow-multi-run", action="store_true", help="Allow mixed audit_run_ids in evaluation window.")
    ap.add_argument("--audit-source", default="runtime", help="Expected audit_source value.")
    args = ap.parse_args(argv)

    ladder_path = Path(args.ladder).expanduser().resolve()
    contract_path = Path(args.contract).expanduser().resolve()
    audit_path = Path(args.audit_log).expanduser().resolve()

    ladder, stage_cfg = _parse_stage(ladder_path, args.stage)
    live_symbols = [_normalize_symbol(s) for s in stage_cfg.get("live_symbols") or []]
    shadow_symbols = [_normalize_symbol(s) for s in stage_cfg.get("shadow_symbols") or []]
    mode = stage_cfg.get("mode", "dry_run")

    failures: List[str] = []
    if mode == "live":
        ledger_backend = os.getenv("TRADING_INTENT_LEDGER_BACKEND", "").lower()
        if ledger_backend != "redis":
            failures.append("intent_ledger_backend_not_redis")
        if not os.getenv("TRADING_AUDIT_HMAC_KEY"):
            failures.append("audit_hmac_key_missing")

    hmac_key = os.getenv("TRADING_AUDIT_HMAC_KEY")
    require_hmac = bool((mode == "live" or hmac_key) and not args.allow_unauthenticated and hmac_key)

    # Gate 1: contract validation
    try:
        validate_deployment_contract(str(contract_path))
    except Exception as exc:
        failures.append(f"contract_validation:{exc}")

    # Gate 2: coverage readiness
    try:
        coverage_rc = coverage_main(["--contract", str(contract_path), "--output-dir", str(args.reports_dir)])
        if coverage_rc != 0:
            failures.append("coverage_preflight_failed")
    except Exception as exc:
        failures.append(f"coverage_preflight_error:{exc}")

    # Gate 3: audit readiness
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=max(1, int(args.hours)))
    try:
        events, provenance = _load_audit_events(
            audit_path,
            live_symbols + shadow_symbols,
            window_start,
            time_min=window_start,
            time_max=window_end,
            audit_source=args.audit_source,
            allow_multi_run=bool(args.allow_multi_run),
            require_hmac=require_hmac,
            hmac_key=hmac_key,
        )
    except Exception as exc:
        failures.append(f"audit_read:{exc}")
        events = []
        provenance = {"audit_log": str(audit_path), "error": str(exc), "run_ids": [], "hmac_validated": False}

    per_symbol_raw, summary_metrics = _collect_audit_metrics(events, live_symbols=live_symbols)
    per_symbol_metrics: Dict[str, Dict[str, Any]] = {}
    for sym, metrics in per_symbol_raw.items():
        attempts = float(metrics.get("trade_count") or 0)
        gate_samples = float(metrics.get("gate_samples") or 0)
        gate_pass = float(metrics.get("gate_pass") or 0)
        coverage_ratio = (gate_pass / gate_samples) if gate_samples else 0.0
        per_symbol_metrics[sym] = {
            **metrics,
            "coverage_ratio": coverage_ratio,
            "risk_block_rate": (float(metrics.get("risk_blocks") or 0) / attempts) if attempts else 0.0,
            "spread_block_rate": (float(metrics.get("spread_blocks") or 0) / attempts) if attempts else 0.0,
        }

    promotion_gates = stage_cfg.get("promotion", {}) or {}
    gates = promotion_gates.get("gates") or {}
    gates["min_runtime_minutes"] = promotion_gates.get("min_runtime_minutes", gates.get("min_runtime_minutes"))
    gate_failures = _evaluate_gates(
        stage_name=args.stage,
        gates=gates,
        live_symbols=live_symbols,
        per_symbol_metrics=per_symbol_metrics,
        summary_metrics=summary_metrics,
    )
    failures.extend(gate_failures)

    # Gate 4: shadow->live promotion preflight
    promotion_targets = _promotion_targets(ladder.get("launch_ladder") or {}, args.stage)
    if promotion_targets:
        try:
            promo_rc = promotion_main(
                [
                    "--contract",
                    str(contract_path),
                    "--reports-dir",
                    str(args.reports_dir),
                ]
            )
            if promo_rc != 0:
                failures.append("shadow_promotion_preflight_failed")
        except Exception as exc:
            failures.append(f"shadow_promotion_preflight_error:{exc}")

    status = "GO" if not failures else "NO_GO"
    report = {
        "stage": args.stage,
        "mode": mode,
        "status": status,
        "failures": failures,
        "provenance": {
            **provenance,
            "audit_log": str(audit_path),
        },
        "metrics": {
            "per_symbol": per_symbol_metrics,
            "summary": summary_metrics,
        },
        "gates": gates,
        "live_symbols": live_symbols,
        "shadow_symbols": shadow_symbols,
        "generated_at": window_end.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "reports_dir": str(args.reports_dir),
    }

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = window_end.strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"launch_stage_eval_{args.stage}_{stamp}.json"
    md_path = reports_dir / f"launch_stage_eval_{args.stage}_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    md_path.write_text(_render_markdown(report))
    print(f"Wrote evaluation report to {json_path} and {md_path}")
    if failures:
        print("NO-GO: " + "; ".join(failures))
        return 1
    print("GO: all gates satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
