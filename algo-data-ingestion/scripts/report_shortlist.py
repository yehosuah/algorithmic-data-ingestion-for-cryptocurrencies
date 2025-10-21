#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from training.reporting import ensure_kpi_schema  # noqa: E402


def _load_report(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - diagnostics for ops usage
        raise RuntimeError(f"Failed to parse report at {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Report at {path} is not a JSON object.")
    return ensure_kpi_schema(payload)


def _default_score(report: Dict[str, Any]) -> float:
    final_equity = float(report.get("final_equity", 0.0))
    sharpe = float(report.get("sharpe", 0.0))
    turnover_bonus = 0.0
    if "total_turnover" in report:
        turnover_bonus = min(float(report.get("total_turnover") or 0.0), 500.0) * 1e-3
    return final_equity + 0.01 * sharpe + turnover_bonus


def _is_good(report: Dict[str, Any], *, criteria: Dict[str, Any]) -> bool:
    if report.get("rejected", False):
        return False
    if float(report.get("final_equity", 0.0)) < float(criteria["min_equity"]):
        return False
    if float(report.get("total_turnover", 0.0)) < float(criteria["min_turnover"]):
        return False
    if float(report.get("sharpe", 0.0)) < float(criteria["min_sharpe"]):
        return False
    if criteria.get("require_rss", True):
        audit = report.get("rss_audit")
        if not audit or not audit.get("passed", False):
            return False
    return True


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile shortlist of deployable model reports.")
    parser.add_argument("--models-root", default="models", help="Root directory containing model subdirectories.")
    parser.add_argument("--out", default="models/report_shortlist.json", help="Path to write shortlist JSON.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of candidates to keep in shortlist.")
    parser.add_argument("--min-equity", type=float, default=1.05, help="Minimum final equity required.")
    parser.add_argument("--min-turnover", type=float, default=10.0, help="Minimum total turnover required.")
    parser.add_argument("--min-sharpe", type=float, default=0.0, help="Minimum Sharpe required.")
    parser.add_argument("--require-rss", action="store_true", default=True, help="Require rss_audit pass flag when present.")
    parser.add_argument("--no-require-rss", action="store_false", dest="require_rss", help="Skip rss_audit requirement regardless of presence.")
    parser.add_argument("--allow-missing-rss", action="store_true", help="Do not drop reports missing rss_audit block.")
    args = parser.parse_args(argv)

    models_root = Path(args.models_root).resolve()
    if not models_root.exists():
        raise SystemExit(f"Models root not found: {models_root}")

    criteria = {
        "min_equity": float(args.min_equity),
        "min_turnover": float(args.min_turnover),
        "min_sharpe": float(args.min_sharpe),
        "require_rss": bool(args.require_rss and not args.allow_missing_rss),
    }

    reports: List[Dict[str, Any]] = []
    for report_path in sorted(models_root.glob("*/report.json")):
        report = _load_report(report_path)
        if args.allow_missing_rss and "rss_audit" not in report:
            report["rss_audit"] = {"passed": None, "reasons": ["rss_audit_missing"]}
        if not _is_good(report, criteria=criteria):
            continue
        model_dir = report_path.parent
        rel_model = model_dir.relative_to(models_root)
        rss_audit = report.get("rss_audit") or {}
        candidate = {
            "model": str(rel_model),
            "report_path": str(report_path.relative_to(models_root.parent)),
            "score": _default_score(report),
            "final_equity": float(report.get("final_equity", math.nan)),
            "sharpe": float(report.get("sharpe", math.nan)),
            "total_turnover": float(report.get("total_turnover", math.nan)),
            "avg_turnover": float(report.get("avg_turnover", math.nan)),
            "selected_threshold": report.get("selected_threshold"),
            "criterion": report.get("criterion"),
            "long_only": bool(report.get("long_only", False)),
            "cost_bps": float(report.get("cost_bps", 0.0)),
            "spread_scale": float(report.get("spread_scale", 0.0)),
            "slippage_bps": float(report.get("slippage_bps", 0.0)),
            "kpi_schema_version": int(report.get("kpi_schema_version", 0)),
            "rss_audit": rss_audit,
            "extra": {},
        }
        for key in ("model_family", "l1_ratio", "meta_threshold", "mask_keep_fraction"):
            if key in report:
                candidate["extra"][key] = report[key]
        reports.append(candidate)

    reports.sort(key=lambda row: row["score"], reverse=True)
    shortlisted = reports[: max(0, int(args.top_n))]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criteria": criteria,
        "total_candidates": len(shortlisted),
        "total_available": len(reports),
        "candidates": shortlisted,
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"shortlist_path": str(out_path), "candidate_count": len(shortlisted)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
