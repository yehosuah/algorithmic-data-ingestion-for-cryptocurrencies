from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def _git_cmd(args: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(["git"] + args, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _collect_commits(limit: int = 5) -> List[str]:
    log = _git_cmd(["log", f"-n{limit}", "--oneline"])
    if not log:
        return []
    return [line.strip() for line in log.splitlines() if line.strip()]


def _collect_diffstat() -> Optional[str]:
    return _git_cmd(["diff", "--stat"])


def _manifest_mtime(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    ts = datetime.utcfromtimestamp(path.stat().st_mtime)
    return ts.isoformat() + "Z"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _resolve_symbol_value(value: Any, symbol: Optional[str]) -> Any:
    if isinstance(value, dict) and symbol:
        if symbol in value:
            return value.get(symbol)
        if symbol.replace("/", "_") in value:
            return value.get(symbol.replace("/", "_"))
        if "default" in value:
            return value.get("default")
        if "*" in value:
            return value.get("*")
    return value


def _parse_compose_env(raw: Any) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key is None:
                continue
            env[str(key)] = "" if value is None else str(value)
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, str) or "=" not in item:
                continue
            key, value = item.split("=", 1)
            env[key] = value
    return env


def _load_compose_service_env(compose_path: Path, service: str) -> Dict[str, str]:
    if not compose_path.exists():
        return {}
    try:
        data = yaml.safe_load(compose_path.read_text()) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    services = data.get("services") or {}
    if not isinstance(services, dict):
        return {}
    svc = services.get(service) or {}
    if not isinstance(svc, dict):
        return {}
    return _parse_compose_env(svc.get("environment"))


def _parse_trading_models(raw: Optional[str]) -> List[Dict[str, Any]]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def build_snapshot(
    *,
    contract_path: Path,
    trigger_policy_path: Path,
    risk_limits_path: Optional[Path],
    compose_path: Optional[Path],
) -> Dict[str, Any]:
    contract = _load_yaml(contract_path)

    compose_env: Dict[str, str] = {}
    if compose_path is not None:
        compose_env = _load_compose_service_env(compose_path, "trading")
    trading_models = _parse_trading_models(compose_env.get("TRADING_MODELS"))

    models_cfg = contract.get("models") or {}
    symbol_model_key = contract.get("symbol_model_key") or {}

    resolved_risk_path = risk_limits_path
    if resolved_risk_path is None:
        raw = compose_env.get("TRADING_RISK_LIMITS_PATH") or contract.get("risk_limits")
        if isinstance(raw, str) and raw:
            resolved_risk_path = Path(raw)
    if resolved_risk_path is None:
        resolved_risk_path = Path("configs/portfolio_risk_limits.yaml")
    risk_limits = _load_yaml(resolved_risk_path)

    deadlock_policy = {}
    deadlock_path = compose_env.get("TRADING_DEADLOCK_POLICY_PATH") or contract.get("deadlock_policy")
    if isinstance(deadlock_path, str) and deadlock_path and Path(deadlock_path).suffix in {".yaml", ".yml"}:
        deadlock_policy = _load_yaml(Path(deadlock_path))

    models: List[Dict[str, Any]] = []
    symbols: List[Dict[str, Any]] = []
    for cfg in trading_models:
        symbol = cfg.get("symbol")
        model_label = cfg.get("model")
        if not isinstance(symbol, str) or not symbol:
            continue
        model_key = symbol_model_key.get(symbol)
        artifact_path = None
        if model_key and isinstance(models_cfg, dict):
            artifact_path = models_cfg.get(model_key)
        artifact = Path(artifact_path) if isinstance(artifact_path, str) else None
        manifest_payload = _load_json(artifact / "manifest.json") if artifact else {}
        gate_cfg = (manifest_payload.get("gates") or {}).get("inference") if isinstance(manifest_payload.get("gates"), dict) else {}
        threshold_meta = manifest_payload.get("threshold") if isinstance(manifest_payload.get("threshold"), dict) else {}
        metadata = manifest_payload.get("metadata") if isinstance(manifest_payload.get("metadata"), dict) else {}

        entry_thr = _resolve_symbol_value(gate_cfg.get("prob_gate_min") if isinstance(gate_cfg, dict) else None, symbol)
        exit_thr = threshold_meta.get("value") if isinstance(threshold_meta, dict) else None
        min_hold_manifest = _resolve_symbol_value(gate_cfg.get("min_hold_bars") if isinstance(gate_cfg, dict) else None, symbol)
        long_only = _resolve_symbol_value(gate_cfg.get("long_only") if isinstance(gate_cfg, dict) else None, symbol)
        exit_prob_drop = metadata.get("exit_prob_drop")
        rvol20_max = _resolve_symbol_value(gate_cfg.get("rvol20_max") if isinstance(gate_cfg, dict) else None, symbol)

        if entry_thr is None and exit_thr is not None:
            entry_thr = exit_thr
        if entry_thr is None:
            entry_thr = 0.5
        if exit_thr is None:
            exit_thr = entry_thr

        min_hold_override = cfg.get("min_hold_bars_override")
        effective_min_hold = min_hold_override or min_hold_manifest or 1

        symbols.append(
            {
                "symbol": symbol,
                "exchange": cfg.get("exchange"),
                "timeframe": cfg.get("timeframe"),
                "policy_id": cfg.get("policy_id"),
                "shadow_mode": bool(cfg.get("shadow_mode", False)),
                "order_notional": cfg.get("order_notional"),
                "max_spread_bps": cfg.get("max_spread_bps"),
                "stops": {
                    "stop_loss_pct": cfg.get("stop_loss_pct"),
                    "take_profit_pct": cfg.get("take_profit_pct"),
                    "max_hold_minutes": cfg.get("max_hold_minutes"),
                },
                "thresholds_manifest": {
                    "entry_threshold": entry_thr,
                    "exit_threshold": exit_thr,
                    "exit_prob_drop": exit_prob_drop,
                    "min_hold_bars": min_hold_manifest,
                    "long_only": long_only,
                    "rvol20_max": rvol20_max,
                },
                "thresholds_effective": {
                    "entry_threshold": entry_thr,
                    "exit_threshold": exit_thr,
                    "exit_prob_drop": exit_prob_drop,
                    "min_hold_bars": effective_min_hold,
                },
                "model": {
                    "label": model_label,
                    "model_key": model_key,
                    "artifact_path": str(artifact) if artifact else None,
                    "manifest_mtime": _manifest_mtime(artifact / "manifest.json") if artifact else None,
                },
            }
        )

        if model_key and all(m.get("model_key") != model_key for m in models):
            models.append(
                {
                    "model_key": model_key,
                    "model_label": model_label,
                    "artifact_path": str(artifact) if artifact else None,
                    "manifest_mtime": _manifest_mtime(artifact / "manifest.json") if artifact else None,
                }
            )

    env_snapshot = {
        k: v
        for k, v in (compose_env or os.environ).items()
        if k.startswith("TRADING_") or k in {"DECISION_QUEUE_URL", "DECISION_QUEUE_KEY"}
    }

    snapshot: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "contract_path": str(contract_path),
        "trigger_policy_path": str(trigger_policy_path),
        "risk_limits_path": str(resolved_risk_path),
        "models": models,
        "symbols": symbols,
        "risk_limits": risk_limits,
        "deadlock_policy": deadlock_policy or contract.get("deadlock_policy"),
        "reconciliation": contract.get("live_invariants", {}).get("reconciliation"),
        "safe_mode": contract.get("live_invariants", {}).get("safe_mode"),
        "kill_switch": contract.get("live_invariants", {}).get("kill_switch"),
        "env": env_snapshot,
        "git_commits": _collect_commits(),
        "git_diffstat": _collect_diffstat(),
    }
    return snapshot


def snapshot_to_markdown(snapshot: Dict[str, Any]) -> str:
    lines = ["# Project Stance Snapshot", ""]
    lines.append(f"Generated at: {snapshot.get('generated_at')}")
    lines.append("")
    lines.append("## Models")
    for model in snapshot.get("models", []):
        lines.append(
            f"- {model.get('model_key')}: label={model.get('model_label')} path={model.get('artifact_path')} "
            f"(manifest_mtime={model.get('manifest_mtime')})"
        )
    lines.append("")
    lines.append("## Symbols")
    for sym in snapshot.get("symbols", []):
        eff = sym.get("thresholds_effective") or {}
        stops = sym.get("stops") or {}
        lines.append(
            f"- {sym.get('symbol')}: policy={sym.get('policy_id')} shadow={sym.get('shadow_mode')} "
            f"notional={sym.get('order_notional')} max_spread_bps={sym.get('max_spread_bps')} "
            f"entry_thr={eff.get('entry_threshold')} exit_thr={eff.get('exit_threshold')} "
            f"min_hold_bars={eff.get('min_hold_bars')} take_profit_pct={stops.get('take_profit_pct')} "
            f"stop_loss_pct={stops.get('stop_loss_pct')}"
        )
    lines.append("")
    lines.append("## Risk limits")
    lines.append(f"Source: {snapshot.get('risk_limits_path')}")
    if snapshot.get("risk_limits"):
        lines.append(json.dumps(snapshot["risk_limits"], indent=2))
    lines.append("")
    lines.append("## Deadlock policy")
    if snapshot.get("deadlock_policy"):
        lines.append(json.dumps(snapshot["deadlock_policy"], indent=2))
    lines.append("")
    lines.append("## Env (trading)")
    for key, val in sorted((snapshot.get("env") or {}).items()):
        lines.append(f"- {key}={val}")
    lines.append("")
    lines.append("## Git")
    lines.append("Recent commits:")
    for commit in snapshot.get("git_commits", []):
        lines.append(f"- {commit}")
    if snapshot.get("git_diffstat"):
        lines.append("")
        lines.append("Diffstat:")
        lines.append(snapshot["git_diffstat"])
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate current stance snapshot report.")
    ap.add_argument("--contract", default="configs/deployment_portfolio_contract.yaml", help="Deployment contract path.")
    ap.add_argument("--trigger-policy", default="configs/final_trigger_policy.yaml", help="Trigger policy path.")
    ap.add_argument("--risk-limits", default=None, help="Risk limits path (defaults to contract.risk_limits).")
    ap.add_argument("--compose", default="docker-compose.yml", help="Optional docker-compose file (used to read runtime env).")
    ap.add_argument("--output-prefix", default=None, help="Prefix for output files.")
    args = ap.parse_args(argv)

    contract_path = Path(args.contract)
    trigger_path = Path(args.trigger_policy)
    risk_path = Path(args.risk_limits) if args.risk_limits else None
    compose_path = Path(args.compose) if args.compose else None
    snapshot = build_snapshot(
        contract_path=contract_path,
        trigger_policy_path=trigger_path,
        risk_limits_path=risk_path,
        compose_path=compose_path,
    )
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prefix = Path(args.output_prefix) if args.output_prefix else Path("reports") / f"project_stance_snapshot_{stamp}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(snapshot, indent=2))
    md_path.write_text(snapshot_to_markdown(snapshot))
    print(f"Wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
