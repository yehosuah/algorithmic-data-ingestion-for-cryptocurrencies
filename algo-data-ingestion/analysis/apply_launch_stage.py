from __future__ import annotations

import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

from analysis.validate_deployment_contract import _normalize_symbol


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _merge_risk(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = deepcopy(base)
    for key, value in overrides.items():
        if key == "symbols":
            continue
        merged[key] = value
    if "symbols" in overrides:
        merged.setdefault("symbols", {})
        for sym, sym_overrides in (overrides.get("symbols") or {}).items():
            norm = _normalize_symbol(sym)
            base_sym = merged["symbols"].get(norm, {}) if isinstance(merged.get("symbols"), dict) else {}
            next_cfg = dict(base_sym) if isinstance(base_sym, dict) else {}
            if isinstance(sym_overrides, Mapping):
                next_cfg.update(sym_overrides)
            merged["symbols"][norm] = next_cfg
    return merged


def _resolve_path(base: Path, raw: str | Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _stage_symbols(stage: Mapping[str, Any]) -> List[str]:
    live = [_normalize_symbol(s) for s in stage.get("live_symbols") or []]
    shadow = [_normalize_symbol(s) for s in stage.get("shadow_symbols") or []]
    seen = set()
    ordered: List[str] = []
    for sym in list(live) + list(shadow):
        if sym in seen:
            continue
        ordered.append(sym)
        seen.add(sym)
    if not ordered:
        raise ValueError("Stage must include at least one symbol in live_symbols/shadow_symbols.")
    return ordered


def _resolve_policy_map(stage: Mapping[str, Any], symbols: Sequence[str], contract: Mapping[str, Any]) -> Dict[str, str]:
    per_symbol = stage.get("per_symbol") or {}
    policy_map: Dict[str, str] = {}
    default_map = {k: v for k, v in (contract.get("symbol_policy_map") or {}).items()}
    default_policy = (contract.get("dry_run_integration") or {}).get("policy_default")
    for sym in symbols:
        stage_sym = per_symbol.get(sym) if isinstance(per_symbol, Mapping) else None
        policy_id = None
        if isinstance(stage_sym, Mapping):
            policy_id = stage_sym.get("policy_id")
        if policy_id is None:
            policy_id = default_map.get(sym) or default_policy
        if not policy_id:
            raise ValueError(f"No policy_id available for symbol {sym}; add it to ladder per_symbol or contract.")
        policy_map[_normalize_symbol(sym)] = str(policy_id)
    return policy_map


def _resolve_notional_map(stage: Mapping[str, Any], symbols: Sequence[str]) -> Dict[str, float]:
    per_symbol = stage.get("per_symbol") or {}
    notional_map: Dict[str, float] = {}
    for sym in symbols:
        entry = per_symbol.get(sym) if isinstance(per_symbol, Mapping) else None
        if not isinstance(entry, Mapping) or entry.get("order_notional") is None:
            raise ValueError(f"Stage missing order_notional for symbol {sym}")
        notional_map[_normalize_symbol(sym)] = float(entry["order_notional"])
    return notional_map


def _resolve_models(
    *,
    symbols: Sequence[str],
    symbol_model_key: Mapping[str, str],
    policy_map: Mapping[str, str],
    notional_map: Mapping[str, float],
    shadow_symbols: Sequence[str],
    risk_cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    shadow_set = {_normalize_symbol(s) for s in shadow_symbols}
    for sym in symbols:
        model_key = symbol_model_key.get(sym)
        if not model_key:
            raise ValueError(f"Missing symbol_model_key for {sym} in deployment contract.")
        sym_risk = ((risk_cfg.get("symbols") or {}) if isinstance(risk_cfg.get("symbols"), dict) else {}).get(sym, {})
        max_spread = sym_risk.get("max_spread_bps") or risk_cfg.get("halt_if_spread_bps_gt") or 10.0
        models.append(
            {
                "model": model_key,
                "symbol": sym,
                "exchange": "binance",
                "timeframe": "1m",
                "order_notional": notional_map.get(sym),
                "max_spread_bps": max_spread,
                "shadow_mode": sym in shadow_set,
                "policy_id": policy_map.get(sym),
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.0075,
                "max_hold_minutes": None,
            }
        )
    return models


def _write_stage_patch(
    *,
    stage_name: str,
    runtime_dir: Path,
    stage_env: Mapping[str, Any],
    risk_path: Path,
    deadlock_path: Path,
) -> Path:
    patch = {
        "stage": stage_name,
        "env": dict(stage_env),
        "risk_limits_path": str(risk_path),
        "deadlock_policy_path": str(deadlock_path),
    }
    patch_path = runtime_dir / f"{stage_name}.yaml"
    _atomic_write(patch_path, yaml.safe_dump(patch, sort_keys=False))
    return patch_path


def _apply_stage(
    *,
    stage_name: str,
    ladder_path: Path,
    contract_path: Path,
    runtime_overrides: Path,
) -> Dict[str, Any]:
    ladder_path = ladder_path.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    runtime_overrides = runtime_overrides.expanduser().resolve()

    ladder = _load_yaml(ladder_path)
    launch_ladder = ladder.get("launch_ladder") or {}
    if stage_name not in launch_ladder:
        raise KeyError(f"Stage {stage_name} missing from {ladder_path}")
    stage = launch_ladder[stage_name]

    contract = _load_yaml(contract_path)
    base_risk_path = ladder.get("base_risk_limits_path") or contract.get("risk_limits")
    risk_base_path_resolved = _resolve_path(contract_path.parent, base_risk_path)
    base_risk_cfg = _load_yaml(risk_base_path_resolved)

    symbols = _stage_symbols(stage)
    shadow_symbols = [_normalize_symbol(s) for s in (stage.get("shadow_symbols") or [])]
    policy_map = _resolve_policy_map(stage, symbols, contract)
    notional_map = _resolve_notional_map(stage, symbols)

    symbol_model_key = {_normalize_symbol(k): v for k, v in (contract.get("symbol_model_key") or {}).items()}
    missing_models = [sym for sym in symbols if sym not in symbol_model_key]
    if missing_models:
        raise ValueError(f"Contract missing symbol_model_key entries for: {missing_models}")

    risk_override_cfg = _merge_risk(base_risk_cfg, stage.get("risk_overrides") or {})
    deadlock_policy = stage.get("deadlock_policy") or {}

    contract["live_symbols"] = symbols
    contract["symbol_policy_map"] = policy_map
    contract["symbol_shadow_mode"] = {sym: sym in shadow_symbols for sym in symbols}
    contract["symbol_order_notional"] = notional_map
    contract["symbol_model_key"] = {sym: symbol_model_key[sym] for sym in symbols}

    runtime_overrides.mkdir(parents=True, exist_ok=True)
    risk_path = runtime_overrides / f"risk_limits_{stage_name}.yaml"
    deadlock_path = runtime_overrides / f"deadlock_policy_{stage_name}.yaml"
    _atomic_write(risk_path, yaml.safe_dump(risk_override_cfg, sort_keys=False))
    _atomic_write(deadlock_path, yaml.safe_dump(deadlock_policy, sort_keys=False))

    live_invariants = contract.get("live_invariants") or {}
    live_invariants["mode"] = stage.get("mode", live_invariants.get("mode", "dry_run"))
    risk_limits_cfg = live_invariants.get("risk_limits") or {}
    risk_limits_cfg["path"] = str(risk_path)
    live_invariants["risk_limits"] = risk_limits_cfg
    contract["live_invariants"] = live_invariants
    contract["intent_ledger_backend"] = "redis"
    contract["deadlock_policy"] = deadlock_policy

    contract["risk_limits"] = str(risk_path)

    models = _resolve_models(
        symbols=symbols,
        symbol_model_key=contract["symbol_model_key"],
        policy_map=policy_map,
        notional_map=notional_map,
        shadow_symbols=shadow_symbols,
        risk_cfg=risk_override_cfg,
    )
    stage_env = {
        "TRADING_DRY_RUN": "true" if live_invariants["mode"] != "live" else "false",
        "TRADING_SHADOW_SYMBOLS": ",".join(shadow_symbols),
        "TRADING_MODELS": json.dumps(models, separators=(",", ":"), sort_keys=True),
        "TRADING_RISK_LIMITS_PATH": str(risk_path),
        "TRADING_DEADLOCK_POLICY_PATH": str(deadlock_path),
        "TRADING_INTENT_LEDGER_BACKEND": "redis",
    }
    patch_path = _write_stage_patch(
        stage_name=stage_name,
        runtime_dir=runtime_overrides,
        stage_env=stage_env,
        risk_path=risk_path,
        deadlock_path=deadlock_path,
    )

    _atomic_write(contract_path, yaml.safe_dump(contract, sort_keys=False))

    return {
        "stage": stage_name,
        "contract_path": str(contract_path),
        "risk_limits_path": str(risk_path),
        "deadlock_policy_path": str(deadlock_path),
        "patch_path": str(patch_path),
        "live_symbols": symbols,
        "shadow_symbols": shadow_symbols,
        "policy_map": policy_map,
        "mode": live_invariants["mode"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Apply a launch stage to the deployment contract and runtime overrides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--stage", required=True, help="Stage key from launch ladder (e.g., stage_2)")
    ap.add_argument("--ladder", default="configs/live_launch_ladder.yaml", help="Path to launch ladder config.")
    ap.add_argument("--contract", default="configs/deployment_portfolio_contract.yaml", help="Deployment contract path.")
    ap.add_argument(
        "--runtime-overrides-dir",
        default="configs/runtime_overrides",
        help="Output directory for stage-specific runtime overrides.",
    )
    args = ap.parse_args(argv)

    summary = _apply_stage(
        stage_name=args.stage,
        ladder_path=Path(args.ladder),
        contract_path=Path(args.contract),
        runtime_overrides=Path(args.runtime_overrides_dir),
    )

    print(f"Applied stage {summary['stage']} to {summary['contract_path']}")
    print(f"- mode: {summary['mode']}")
    print(f"- live_symbols: {summary['live_symbols']}")
    print(f"- shadow_symbols: {summary['shadow_symbols']}")
    print(f"- policy map: {summary['policy_map']}")
    print(f"- risk limits override: {summary['risk_limits_path']}")
    print(f"- deadlock policy override: {summary['deadlock_policy_path']}")
    print(f"- stage patch: {summary['patch_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
