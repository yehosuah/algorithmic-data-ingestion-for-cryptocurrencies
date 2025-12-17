"""
Task 8.8 — End-to-end live-readiness smoke test (single command, GO/NO-GO).

Repo discovery (orchestrated checks, in order):
  1) analysis/validate_deployment_contract.py
  2) analysis/preflight_coverage.py
  3) analysis/shadow_readiness.py + analysis/preflight_symbol_promotion.py
  4) analysis/evaluate_launch_stage.py (optional; stage-level gating)

Requirement confirmations (from existing tools):
  - Redis intent ledger (live mode)
      * validate_deployment_contract: requires intent ledger backend == redis when contract live_invariants.mode == "live".
      * evaluate_launch_stage: requires TRADING_INTENT_LEDGER_BACKEND=redis when stage mode == "live".
  - Audit HMAC signing key (audit validation)
      * validate_deployment_contract: requires TRADING_AUDIT_HMAC_KEY when contract live_invariants.mode == "live".
      * shadow_readiness: enforces HMAC validation when --require-hmac is set OR TRADING_AUDIT_HMAC_KEY is present.
      * preflight_symbol_promotion: requires HMAC-verified provenance unless --allow-unauthenticated is set (unsafe).
      * evaluate_launch_stage: enforces HMAC requirements for live stages; can be overridden with --allow-unauthenticated (unsafe).
  - Audit log location
      * shadow_readiness requires a real audit log path (--audit-log).
      * evaluate_launch_stage requires a real audit log path (--audit-log).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from analysis.evaluate_launch_stage import main as launch_stage_main
from analysis.preflight_coverage import main as coverage_main
from analysis.preflight_symbol_promotion import main as promotion_main
from analysis.shadow_readiness import main as shadow_readiness_main
from analysis.validate_deployment_contract import _normalize_symbol, validate_deployment_contract


Status = str  # "PASS" | "FAIL" | "SKIP"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ")


def _parse_symbols_csv(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    symbols = [_normalize_symbol(sym) for sym in value.split(",") if sym.strip()]
    return symbols or None


def _snapshot(path: Path, pattern: str) -> set[Path]:
    return {p.resolve() for p in path.glob(pattern) if p.is_file()}


def _latest_new(path: Path, pattern: str, *, before: set[Path]) -> Optional[Path]:
    candidates = [p for p in path.glob(pattern) if p.is_file() and p.resolve() not in before]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _render_markdown(report: dict) -> str:
    lines: List[str] = [
        "# Live Readiness Check",
        "",
        f"Status: **{report.get('overall_status')}**",
        "",
        f"Timestamp (UTC): {report.get('timestamp_utc')}",
        f"Mode: {report.get('mode')}",
        f"Deployment contract: {report.get('deployment_contract_path')}",
    ]
    audit_log = report.get("audit_log")
    if audit_log:
        lines.append(f"Audit log: {audit_log}")
    lookback = report.get("lookback_hours")
    if lookback is not None:
        lines.append(f"Lookback hours: {lookback}")
    lines.extend(
        [
            "",
            "## Orchestrated checks (existing tools)",
            "- `analysis/validate_deployment_contract.py` (contract + live invariants; enforces Redis/HMAC in live mode)",
            "- `analysis/preflight_coverage.py` (coverage fraction + implied-trades proxy; deadlock-preflight)",
            "- `analysis/shadow_readiness.py` + `analysis/preflight_symbol_promotion.py` (audit provenance/HMAC + promotion gates)",
            "- `analysis/evaluate_launch_stage.py` (optional stage ladder GO/NO-GO; enforces Redis/HMAC for live stages)",
            "",
            "## Checks",
        ]
    )
    for check in report.get("checks") or []:
        name = check.get("name")
        status = check.get("status")
        duration = check.get("duration_ms")
        summary = check.get("summary")
        required = check.get("required")
        req_tag = "required" if required else "optional"
        lines.append(f"- **{name}**: `{status}` ({req_tag}, {duration} ms) — {summary}")
        artifacts = check.get("artifacts") or []
        for art in artifacts:
            lines.append(f"  - artifact: `{art}`")
    artifacts = report.get("artifacts") or {}
    if artifacts:
        lines.extend(["", "## Artifacts"])
        for key, value in artifacts.items():
            if not value:
                continue
            if isinstance(value, list):
                for item in value:
                    lines.append(f"- {key}: `{item}`")
            else:
                lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def run_readiness_check(args: argparse.Namespace) -> dict:
    started = _utc_now()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(started)

    contract_path = Path(args.deployment_contract).expanduser().resolve()
    mode = str(args.mode)
    lookback_hours = int(args.lookback_hours)
    audit_log = str(args.audit_log) if args.audit_log else None
    requested_symbols = _parse_symbols_csv(args.symbols)
    allow_zero_coverage = bool(args.allow_zero_coverage)
    fail_fast = bool(args.fail_fast)

    report: Dict[str, Any] = {
        "overall_status": "NO_GO",
        "timestamp_utc": started.isoformat(),
        "deployment_contract_path": str(contract_path),
        "mode": mode,
        "audit_log": audit_log,
        "lookback_hours": lookback_hours,
        "symbols_live": [],
        "symbols_shadow": [],
        "input_errors": [],
        "checks": [],
        "artifacts": {},
    }

    stop_after_failure = False

    # ---- Check 2.1: Deployment contract validation (Task 8.1 baseline) ----
    contract_required = bool(args.require_contract_validation)
    contract_check: Dict[str, Any] = {
        "name": "deployment_contract_validation",
        "status": "SKIP" if not contract_required else "FAIL",
        "required": contract_required,
        "summary": "Skipped by flag." if not contract_required else "Contract validation did not run.",
        "artifacts": [],
        "duration_ms": 0,
    }
    contract_summary: Optional[dict] = None
    t0 = time.monotonic()
    try:
        contract_summary = validate_deployment_contract(str(contract_path))
        contract_check.update(
            {
                "status": "PASS",
                "summary": "Deployment contract validated successfully.",
                "details": contract_summary,
            }
        )
    except Exception as exc:
        contract_check.update(
            {
                "status": "FAIL",
                "summary": f"Deployment contract validation failed: {exc}",
                "error": str(exc),
            }
        )
    contract_check["duration_ms"] = int((time.monotonic() - t0) * 1000)
    report["checks"].append(contract_check)

    if contract_summary:
        symbols_cfg = contract_summary.get("symbols", {}) or {}
        live_symbols = list(symbols_cfg.get("live_symbols") or [])
        shadow_map = symbols_cfg.get("symbol_shadow_mode") or {}
        symbols_shadow = [sym for sym in live_symbols if bool(shadow_map.get(sym))]
        symbols_live = [sym for sym in live_symbols if sym not in set(symbols_shadow)]
        report["symbols_live"] = symbols_live
        report["symbols_shadow"] = symbols_shadow
        report["contract_summary"] = contract_summary

    if requested_symbols and contract_summary:
        allowed = set((contract_summary.get("symbols") or {}).get("live_symbols") or [])
        extra = [sym for sym in requested_symbols if sym not in allowed]
        if extra:
            msg = f"--symbols includes symbols not present in deployment contract live_symbols: {extra}"
            report["input_errors"].append(msg)
            contract_check.update({"status": "FAIL", "summary": msg})
            contract_summary = None

    if requested_symbols is None and contract_summary:
        requested_symbols = list((contract_summary.get("symbols") or {}).get("live_symbols") or [])

    shadow_symbols_exist = bool(report.get("symbols_shadow"))
    shadow_required: bool
    if args.require_shadow_preflight is True:
        shadow_required = True
    elif args.require_shadow_preflight is False:
        shadow_required = False
    else:
        shadow_required = bool(shadow_symbols_exist)

    stage_required: bool = bool(args.stage)
    if args.require_launch_stage_eval is True:
        stage_required = True
    if args.require_launch_stage_eval is False and args.stage:
        stage_required = True  # --stage explicitly requests evaluation

    require_hmac_for_audit = bool(mode == "live" or (contract_summary and contract_summary.get("live_invariants", {}).get("mode") == "live"))

    # Helper: decide whether downstream checks should execute.
    contract_ok = bool(contract_summary is not None and contract_check.get("status") == "PASS")
    if fail_fast and contract_required and contract_check.get("status") == "FAIL":
        stop_after_failure = True

    # ---- Check 2.2: Coverage preflight (Task 8.6) ----
    coverage_required = bool(args.require_coverage_preflight)
    coverage_check: Dict[str, Any] = {
        "name": "coverage_preflight",
        "status": "SKIP",
        "required": coverage_required,
        "summary": "Skipped by flag." if not coverage_required else "Skipped.",
        "artifacts": [],
        "duration_ms": 0,
    }
    if stop_after_failure:
        coverage_check.update(
            {
                "status": "FAIL" if coverage_required else "SKIP",
                "summary": "Not executed (fail-fast after a prior required failure)." if coverage_required else "Skipped (fail-fast).",
            }
        )
    elif coverage_required and contract_ok:
        t0 = time.monotonic()
        before_json = _snapshot(output_dir, "preflight_coverage_*.json")
        before_md = _snapshot(output_dir, "preflight_coverage_*.md")
        argv: List[str] = ["--contract", str(contract_path), "--output-dir", str(output_dir), "--epsilon", "0"]
        if allow_zero_coverage:
            argv.append("--allow-no-go")
        rc: Optional[int] = None
        coverage_report: Optional[dict] = None
        try:
            rc = int(coverage_main(argv))
        except Exception as exc:
            rc = None
            coverage_check.update(
                {
                    "status": "FAIL",
                    "summary": f"Coverage preflight raised: {exc}",
                    "error": str(exc),
                }
            )
        json_art = _latest_new(output_dir, "preflight_coverage_*.json", before=before_json)
        md_art = _latest_new(output_dir, "preflight_coverage_*.md", before=before_md)
        artifacts: List[str] = [str(p) for p in (json_art, md_art) if p]
        coverage_check["artifacts"] = artifacts
        if json_art and json_art.exists():
            try:
                coverage_report = json.loads(json_art.read_text())
            except Exception:
                coverage_report = None

        if coverage_report is not None:
            per_symbol = coverage_report.get("symbols") or {}
            missing = [sym for sym in (requested_symbols or []) if sym not in per_symbol]
            zero_fraction = [
                sym
                for sym in (requested_symbols or [])
                if sym in per_symbol and float(per_symbol[sym].get("fraction_above_prob_gate_min") or 0.0) <= 0.0
            ]
            zero_trades = [
                sym
                for sym in (requested_symbols or [])
                if sym in per_symbol and int(per_symbol[sym].get("implied_trade_proxy") or 0) <= 0
            ]
            hard_fail_reasons: List[str] = []
            if missing:
                hard_fail_reasons.append(f"missing_symbols={missing}")
            if zero_fraction:
                hard_fail_reasons.append(f"zero_fraction_above_gate={zero_fraction}")
            if zero_trades:
                hard_fail_reasons.append(f"zero_implied_trades={zero_trades}")
            if missing:
                coverage_check.update(
                    {
                        "status": "FAIL",
                        "summary": "Coverage preflight blocked: missing per-symbol metrics.",
                        "details": {"missing_symbols": missing},
                    }
                )
            elif hard_fail_reasons and not allow_zero_coverage:
                coverage_check.update(
                    {
                        "status": "FAIL",
                        "summary": "Coverage preflight NO-GO: " + "; ".join(hard_fail_reasons),
                        "details": {
                            "hard_fail_reasons": hard_fail_reasons,
                            "no_go_reasons": coverage_report.get("no_go_reasons") or [],
                        },
                    }
                )
            else:
                status = "PASS" if rc == 0 else "FAIL"
                summary = "Coverage preflight passed." if status == "PASS" else "Coverage preflight failed."
                coverage_check.update(
                    {
                        "status": status,
                        "summary": summary,
                        "details": {
                            "no_go_reasons": coverage_report.get("no_go_reasons") or [],
                            "allow_zero_coverage": allow_zero_coverage,
                        },
                    }
                )
        else:
            if rc == 0:
                coverage_check.update({"status": "PASS", "summary": "Coverage preflight passed."})
            elif rc is None:
                # Exception already recorded above.
                pass
            else:
                coverage_check.update({"status": "FAIL", "summary": f"Coverage preflight failed (rc={rc})."})

        coverage_check["duration_ms"] = int((time.monotonic() - t0) * 1000)
    elif coverage_required and not contract_ok:
        coverage_check.update(
            {
                "status": "FAIL",
                "summary": "Coverage preflight blocked: deployment contract validation failed.",
            }
        )
    report["checks"].append(coverage_check)

    if fail_fast and coverage_required and coverage_check["status"] == "FAIL":
        stop_after_failure = True

    # ---- Check 2.3: Shadow readiness + promotion preflight (Task 8.5) ----
    shadow_check: Dict[str, Any] = {
        "name": "shadow_readiness_and_promotion",
        "status": "SKIP",
        "required": shadow_required,
        "summary": "Skipped (no shadow symbols and not required).",
        "artifacts": [],
        "duration_ms": 0,
    }
    if shadow_required:
        if stop_after_failure:
            shadow_check.update(
                {
                    "status": "FAIL",
                    "summary": "Not executed (fail-fast after a prior required failure).",
                }
            )
        elif not contract_ok:
            shadow_check.update(
                {
                    "status": "FAIL",
                    "summary": "Shadow readiness blocked: deployment contract validation failed.",
                }
            )
        elif not audit_log:
            shadow_check.update(
                {
                    "status": "FAIL",
                    "summary": "Shadow readiness requires --audit-log when enabled.",
                }
            )
        else:
            t0 = time.monotonic()
            before_json = _snapshot(output_dir, "shadow_readiness_*.json")
            before_md = _snapshot(output_dir, "shadow_readiness_*.md")
            shadow_symbols = list(report.get("symbols_shadow") or [])
            # If user forced shadow preflight but no shadow symbols exist, evaluate the live set for visibility.
            shadow_eval_symbols = shadow_symbols or list(report.get("symbols_live") or []) or (requested_symbols or [])
            argv_sr: List[str] = [
                "--audit-log",
                str(audit_log),
                "--risk-limits",
                str((contract_summary or {}).get("risk_limits") or "configs/portfolio_risk_limits.yaml"),
                "--symbols",
                ",".join(shadow_eval_symbols),
                "--hours",
                str(lookback_hours),
                "--output-dir",
                str(output_dir),
            ]
            if require_hmac_for_audit:
                argv_sr.append("--require-hmac")
            sr_exc: Optional[str] = None
            try:
                _ = shadow_readiness_main(argv_sr)
            except Exception as exc:
                sr_exc = str(exc)
            sr_json = _latest_new(output_dir, "shadow_readiness_*.json", before=before_json)
            sr_md = _latest_new(output_dir, "shadow_readiness_*.md", before=before_md)
            artifacts = [str(p) for p in (sr_json, sr_md) if p]
            shadow_check["artifacts"] = artifacts

            if sr_exc:
                shadow_check.update(
                    {
                        "status": "FAIL",
                        "summary": f"Shadow readiness failed: {sr_exc}",
                        "error": sr_exc,
                    }
                )
            else:
                promo_rc: Optional[int] = None
                promo_exc: Optional[str] = None
                promo_args: List[str] = [
                    "--contract",
                    str(contract_path),
                ]
                if sr_json:
                    promo_args += ["--shadow-report", str(sr_json)]
                else:
                    promo_args += ["--reports-dir", str(output_dir)]
                if not require_hmac_for_audit:
                    promo_args.append("--allow-unauthenticated")
                try:
                    promo_rc = int(promotion_main(promo_args))
                except Exception as exc:
                    promo_exc = str(exc)
                if promo_exc:
                    shadow_check.update(
                        {
                            "status": "FAIL",
                            "summary": f"Shadow promotion preflight raised: {promo_exc}",
                            "error": promo_exc,
                        }
                    )
                elif promo_rc != 0:
                    shadow_check.update(
                        {
                            "status": "FAIL",
                            "summary": "Shadow promotion preflight returned DO NOT promote.",
                            "details": {"rc": promo_rc},
                        }
                    )
                else:
                    shadow_check.update(
                        {
                            "status": "PASS",
                            "summary": "Shadow readiness + promotion preflight passed.",
                            "details": {"hmac_required": require_hmac_for_audit},
                        }
                    )

            shadow_check["duration_ms"] = int((time.monotonic() - t0) * 1000)
    report["checks"].append(shadow_check)

    if fail_fast and shadow_required and shadow_check["status"] == "FAIL":
        stop_after_failure = True

    # ---- Check 2.4: Launch stage evaluation (Task 8.7) ----
    stage_check: Dict[str, Any] = {
        "name": "launch_stage_evaluation",
        "status": "SKIP",
        "required": stage_required,
        "summary": "Skipped (no --stage provided).",
        "artifacts": [],
        "duration_ms": 0,
    }
    if stage_required:
        if stop_after_failure:
            stage_check.update(
                {
                    "status": "FAIL",
                    "summary": "Not executed (fail-fast after a prior required failure).",
                }
            )
        elif not args.stage:
            stage_check.update({"status": "FAIL", "summary": "--stage is required for launch stage evaluation."})
        elif not contract_ok:
            stage_check.update({"status": "FAIL", "summary": "Stage evaluation blocked: contract validation failed."})
        elif not audit_log:
            stage_check.update({"status": "FAIL", "summary": "Stage evaluation requires --audit-log when enabled."})
        else:
            t0 = time.monotonic()
            before_json = _snapshot(output_dir, f"launch_stage_eval_{args.stage}_*.json")
            before_md = _snapshot(output_dir, f"launch_stage_eval_{args.stage}_*.md")
            rc: Optional[int] = None
            exc_s: Optional[str] = None
            argv_stage = [
                "--stage",
                str(args.stage),
                "--contract",
                str(contract_path),
                "--audit-log",
                str(audit_log),
                "--hours",
                str(lookback_hours),
                "--reports-dir",
                str(output_dir),
            ]
            if args.ladder:
                argv_stage += ["--ladder", str(args.ladder)]
            try:
                rc = int(launch_stage_main(argv_stage))
            except Exception as exc:
                exc_s = str(exc)
            st_json = _latest_new(output_dir, f"launch_stage_eval_{args.stage}_*.json", before=before_json)
            st_md = _latest_new(output_dir, f"launch_stage_eval_{args.stage}_*.md", before=before_md)
            stage_check["artifacts"] = [str(p) for p in (st_json, st_md) if p]
            if exc_s:
                stage_check.update({"status": "FAIL", "summary": f"Stage evaluation raised: {exc_s}", "error": exc_s})
            elif rc != 0:
                failures: List[str] = []
                if st_json and st_json.exists():
                    try:
                        payload = json.loads(st_json.read_text())
                        failures = list(payload.get("failures") or [])
                    except Exception:
                        failures = []
                stage_check.update(
                    {
                        "status": "FAIL",
                        "summary": "Stage evaluation returned NO-GO.",
                        "details": {"failures": failures},
                    }
                )
            else:
                stage_check.update({"status": "PASS", "summary": "Stage evaluation returned GO."})
            stage_check["duration_ms"] = int((time.monotonic() - t0) * 1000)
    report["checks"].append(stage_check)

    # ---- Deterministic GO/NO-GO decision ----
    required_failures = [
        check
        for check in report["checks"]
        if bool(check.get("required")) and str(check.get("status")) != "PASS"
    ]
    report["overall_status"] = "GO" if (not required_failures and not report.get("input_errors")) else "NO_GO"

    # ---- Write readiness artifacts ----
    json_path = output_dir / f"{stamp}_live_readiness.json"
    md_path = output_dir / f"{stamp}_live_readiness.md"
    report["artifacts"]["readiness_json"] = str(json_path)
    report["artifacts"]["readiness_md"] = str(md_path)
    report["artifacts"]["coverage_reports"] = list(coverage_check.get("artifacts") or [])
    report["artifacts"]["shadow_reports"] = list(shadow_check.get("artifacts") or [])
    report["artifacts"]["stage_reports"] = list(stage_check.get("artifacts") or [])

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    md_path.write_text(_render_markdown(report))
    return report


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run an end-to-end live-readiness smoke test and emit a single GO/NO-GO decision.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--deployment-contract", required=True, help="Deployment contract path.")
    ap.add_argument("--mode", choices=("dry_run", "live_like", "live"), default="live_like")
    ap.add_argument("--audit-log", default=None, help="Audit log path (required for audit-based checks).")
    ap.add_argument("--lookback-hours", type=int, default=48)
    ap.add_argument("--symbols", default=None, help="Optional comma-separated override for contract live symbols.")
    ap.add_argument("--output-dir", required=True, help="Directory to write readiness artifacts.")
    ap.add_argument("--stage", default=None, help="Optional launch ladder stage id (enables stage evaluation).")
    ap.add_argument("--ladder", default=None, help="Optional launch ladder path for stage evaluation.")
    ap.add_argument("--fail-fast", action="store_true", default=False)
    ap.add_argument(
        "--allow-zero-coverage",
        action="store_true",
        help="Allow zero coverage/implied-trades (unsafe override).",
    )

    ap.add_argument("--require-contract-validation", dest="require_contract_validation", action="store_true", default=True)
    ap.add_argument("--no-require-contract-validation", dest="require_contract_validation", action="store_false")

    ap.add_argument("--require-coverage-preflight", dest="require_coverage_preflight", action="store_true", default=True)
    ap.add_argument("--no-require-coverage-preflight", dest="require_coverage_preflight", action="store_false")

    ap.add_argument("--require-shadow-preflight", dest="require_shadow_preflight", action="store_true", default=None)
    ap.add_argument("--no-require-shadow-preflight", dest="require_shadow_preflight", action="store_false")

    ap.add_argument("--require-launch-stage-eval", dest="require_launch_stage_eval", action="store_true", default=None)
    ap.add_argument("--no-require-launch-stage-eval", dest="require_launch_stage_eval", action="store_false")

    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_readiness_check(args)
    overall = report.get("overall_status")
    print(f"Live readiness: {overall}")
    print(f"- JSON: {report.get('artifacts', {}).get('readiness_json')}")
    print(f"- MD:  {report.get('artifacts', {}).get('readiness_md')}")
    return 0 if overall == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
