from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import yaml

ALLOWED_MODES = {"dry_run", "live"}
ALLOWED_KILL_BEHAVIOR = {"no_new_entries", "flatten_then_stop"}
REQUIRED_RISK_LIMIT_KEYS = (
    "capital",
    "max_gross_leverage",
    "max_net_exposure",
    "max_turnover_per_day",
    "max_orders_per_hour",
    "max_concurrent_positions",
    "daily_loss_limit_pct",
    "max_drawdown_pct",
    "cooldown_minutes_after_exit",
    "cooldown_minutes_after_loss",
    "halt_on_safe_mode",
    "halt_if_spread_bps_gt",
    "halt_if_vol_zscore_gt",
    "halt_if_missing_price_bars",
    "halt_if_data_stale_seconds",
    "symbols",
)
REQUIRED_SYMBOL_RISK_KEYS = (
    "max_symbol_notional",
    "max_symbol_weight",
    "max_spread_bps",
    "min_trade_notional",
)
PERCENTAGE_KEY_HINTS = ("pct", "percent", "ratio", "fraction")
DEFAULT_CODE_PATHS = (
    Path("app/trading/decision.py"),
    Path("app/trading/service.py"),
    Path("app/trading/config.py"),
    Path("app/trading/risk.py"),
)
DEFAULT_AUDIT_PATHS = (
    Path("app/trading/audit.py"),
    Path("app/trading/service.py"),
)
DEFAULT_METRICS_PATHS = (
    Path("app/monitoring/trading_metrics.py"),
    Path("app/trading/service.py"),
)

REQUIRED_AUDIT_PROVENANCE_FIELDS = ("audit_source", "audit_run_id", "audit_seq")


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _resolve_path(base: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _ensure_required_keys(section: dict, keys: Iterable[str], label: str) -> None:
    missing = [k for k in keys if k not in section]
    if missing:
        raise ValueError(f"Missing required keys in {label}: {missing}")


def _ensure_required_list(name: str, values: Iterable[str], required: Iterable[str]) -> None:
    missing = [k for k in required if k not in values]
    if missing:
        raise ValueError(f"Missing required entries in {name}: {missing}")


def _normalize_symbol(symbol: str) -> str:
    if symbol is None:
        raise ValueError("Symbol entries must be non-empty strings.")
    raw = str(symbol).strip()
    if not raw:
        raise ValueError("Symbol entries must be non-empty strings.")
    canonical = raw.replace(" ", "").upper()
    if "/" not in canonical and "-" in canonical:
        canonical = canonical.replace("-", "/")
    if "/" not in canonical:
        raise ValueError(f"Symbol '{symbol}' must include a quote leg (expected format like ETH/USDT).")
    return canonical


def _parse_trading_models_env() -> Set[str]:
    raw = os.getenv("TRADING_MODELS", "").strip()
    if not raw:
        return set()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid TRADING_MODELS payload: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("TRADING_MODELS must be a JSON list of model configs.")
    symbols: Set[str] = set()
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("TRADING_MODELS entries must be mappings with a 'symbol' field.")
        sym = entry.get("symbol")
        if not sym:
            continue
        symbols.add(_normalize_symbol(sym))
    return symbols


def _load_policies(contract: dict, base_dir: Path) -> Tuple[Dict[str, dict], str]:
    policy_path = contract.get("portfolio_policies")
    policies: Dict[str, dict] = {}
    if policy_path:
        p = _resolve_path(base_dir, policy_path)
        _require_path(p, "portfolio_policies")
        policies = _load_yaml(p)
    if not policies:
        policies = contract.get("portfolio_policies_payload") or contract.get("policies") or {}
    default_policy = contract.get("dry_run_integration", {}).get("policy_default", "primary")
    return policies, default_policy


def _assert_strings_present(
    needles: Iterable[str],
    paths: Sequence[Path],
    label: str,
    remediation: str,
) -> None:
    normalized: Dict[Path, str] = {}
    for path in paths:
        _require_path(path, label)
        normalized[path] = path.read_text(encoding="utf-8")
    missing = []
    for needle in needles:
        if not needle:
            continue
        found = any(needle in content for content in normalized.values())
        if not found:
            missing.append(needle)
    if missing:
        files = ", ".join(str(p) for p in paths)
        raise ValueError(f"{label} missing required strings {missing} in [{files}]. {remediation}")


def _validate_risk_limits(risk_limits_path: Path, required_keys: Iterable[str]) -> dict:
    _require_path(risk_limits_path, "risk_limits")
    cfg = _load_yaml(risk_limits_path)
    _ensure_required_keys(cfg, required_keys, f"risk_limits ({risk_limits_path})")

    def _require_positive(name: str, allow_zero: bool = False) -> float:
        raw = cfg.get(name)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Risk limit '{name}' must be numeric (got {raw!r})") from exc
        if allow_zero:
            if value < 0:
                raise ValueError(f"Risk limit '{name}' must be >= 0 (got {value})")
        else:
            if value <= 0:
                raise ValueError(f"Risk limit '{name}' must be > 0 (got {value})")
        return value

    _require_positive("capital")
    _require_positive("max_gross_leverage")
    _require_positive("max_net_exposure")
    _require_positive("max_turnover_per_day", allow_zero=True)
    _require_positive("max_orders_per_hour")
    concurrent_positions = _require_positive("max_concurrent_positions")
    if int(concurrent_positions) < 1:
        raise ValueError("Risk limit 'max_concurrent_positions' must be >= 1")
    _require_positive("daily_loss_limit_pct")
    if float(cfg.get("daily_loss_limit_pct")) >= 1.0:
        raise ValueError("daily_loss_limit_pct should be expressed as a fraction (e.g. 0.03 for 3%)")
    _require_positive("max_drawdown_pct")
    if float(cfg.get("max_drawdown_pct")) >= 1.0:
        raise ValueError("max_drawdown_pct should be expressed as a fraction (e.g. 0.1 for 10%)")
    _require_positive("cooldown_minutes_after_exit", allow_zero=True)
    _require_positive("cooldown_minutes_after_loss", allow_zero=True)
    _require_positive("halt_if_spread_bps_gt")
    _require_positive("halt_if_vol_zscore_gt", allow_zero=True)
    _require_positive("halt_if_data_stale_seconds", allow_zero=True)
    halt_on_safe = cfg.get("halt_on_safe_mode")
    if halt_on_safe not in (True, False):
        raise ValueError("Risk limit 'halt_on_safe_mode' must be a boolean.")
    if cfg.get("halt_if_missing_price_bars") not in (True, False):
        raise ValueError("Risk limit 'halt_if_missing_price_bars' must be a boolean.")
    if "max_symbol_weight" in cfg:
        global_weight = _require_positive("max_symbol_weight")
        if float(global_weight) > 1.0:
            raise ValueError(f"Risk limit 'max_symbol_weight' must be in (0,1] (got {global_weight})")
    if "max_total_notional" in cfg:
        _require_positive("max_total_notional")
    if "max_notional_per_symbol" in cfg:
        _require_positive("max_notional_per_symbol")
    symbols_cfg = cfg.get("symbols")
    if not isinstance(symbols_cfg, dict) or not symbols_cfg:
        raise ValueError("risk_limits.symbols must define per-symbol constraints.")
    for sym, sym_cfg in symbols_cfg.items():
        if not isinstance(sym_cfg, dict):
            raise ValueError(f"risk_limits.symbols.{sym} must be a mapping of limits.")
        _ensure_required_keys(sym_cfg, REQUIRED_SYMBOL_RISK_KEYS, f"risk_limits.symbols[{sym}]")
        try:
            max_sym_notional = float(sym_cfg.get("max_symbol_notional"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"risk_limits.symbols.{sym}.max_symbol_notional must be numeric.") from exc
        if max_sym_notional <= 0:
            raise ValueError(f"risk_limits.symbols.{sym}.max_symbol_notional must be > 0 (got {max_sym_notional})")
        try:
            weight = float(sym_cfg.get("max_symbol_weight"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"risk_limits.symbols.{sym}.max_symbol_weight must be numeric.") from exc
        if weight <= 0 or weight > 1.0:
            raise ValueError(f"risk_limits.symbols.{sym}.max_symbol_weight must be in (0,1]")
        try:
            spread_bps = float(sym_cfg.get("max_spread_bps"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"risk_limits.symbols.{sym}.max_spread_bps must be numeric.") from exc
        if spread_bps <= 0:
            raise ValueError(f"risk_limits.symbols.{sym}.max_spread_bps must be > 0")
        try:
            min_trade_notional = float(sym_cfg.get("min_trade_notional"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"risk_limits.symbols.{sym}.min_trade_notional must be numeric.") from exc
        if min_trade_notional <= 0:
            raise ValueError(f"risk_limits.symbols.{sym}.min_trade_notional must be > 0")
        if sym_cfg.get("max_position_age_minutes") is not None:
            try:
                max_age = float(sym_cfg.get("max_position_age_minutes"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"risk_limits.symbols.{sym}.max_position_age_minutes must be numeric.") from exc
            if max_age < 0:
                raise ValueError(f"risk_limits.symbols.{sym}.max_position_age_minutes must be >= 0 (got {max_age})")

    for key, value in cfg.items():
        if any(hint in key.lower() for hint in PERCENTAGE_KEY_HINTS):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric < 0 or numeric > 1:
                raise ValueError(f"Percentage-like risk limit '{key}' should be between 0 and 1 (got {value})")
    return cfg


def _guess_project_root(contract_file: Path) -> Path:
    candidates = [contract_file.parent] + list(contract_file.parents) + [Path.cwd()]
    for candidate in candidates:
        if (candidate / "app").exists() and (candidate / "configs").exists():
            return candidate
    return Path.cwd()


def _normalize_paths(base: Path, paths: Sequence[Path]) -> Tuple[Path, ...]:
    normalized = []
    for path in paths:
        p = path
        if not p.is_absolute():
            candidate = (base / p).resolve()
            p = candidate if candidate.exists() else p.resolve()
        normalized.append(p)
    return tuple(normalized)


def _validate_deadlock_policy(policy: object, mode: str) -> dict:
    if policy is None:
        if mode == "live":
            raise ValueError("deadlock_policy section required for live deployments.")
        return {}
    if not isinstance(policy, dict):
        raise ValueError("deadlock_policy must be a mapping.")
    enabled = bool(policy.get("enabled", False))
    window_minutes = int(policy.get("window_minutes") or 60)
    min_trades = int(policy.get("min_trades_window") or 0)
    min_coverage = float(policy.get("min_coverage_ratio_window") or 0.0)
    cooldown = int(policy.get("cooldown_minutes") or 0)
    max_actions = policy.get("max_actions_per_day")
    adjust_cfg = policy.get("adjust_prob_gate_min") or {}
    floor_raw = adjust_cfg.get("floor")
    if mode == "live" and not enabled:
        raise ValueError("deadlock_policy.enabled must be true for live deployments.")
    if window_minutes <= 0:
        raise ValueError("deadlock_policy.window_minutes must be > 0.")
    if min_trades < 0:
        raise ValueError("deadlock_policy.min_trades_window must be >= 0.")
    if min_coverage < 0:
        raise ValueError("deadlock_policy.min_coverage_ratio_window must be >= 0.")
    if cooldown < 0:
        raise ValueError("deadlock_policy.cooldown_minutes must be >= 0.")
    if max_actions is None:
        raise ValueError("deadlock_policy.max_actions_per_day must be provided.")
    try:
        max_actions_val = int(max_actions)
    except (TypeError, ValueError) as exc:
        raise ValueError("deadlock_policy.max_actions_per_day must be numeric.") from exc
    if max_actions_val <= 0:
        raise ValueError("deadlock_policy.max_actions_per_day must be > 0.")
    if floor_raw is None:
        raise ValueError("deadlock_policy.adjust_prob_gate_min.floor is required.")
    try:
        floor_val = float(floor_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("deadlock_policy.adjust_prob_gate_min.floor must be numeric.") from exc
    if floor_val < 0 or floor_val > 1:
        raise ValueError("deadlock_policy.adjust_prob_gate_min.floor must be within [0,1].")
    step_val = adjust_cfg.get("step") or adjust_cfg.get("delta") or adjust_cfg.get("adjust")
    if step_val is not None:
        try:
            float(step_val)
        except (TypeError, ValueError) as exc:
            raise ValueError("deadlock_policy.adjust_prob_gate_min.step must be numeric when provided.") from exc
    return {
        "enabled": enabled,
        "window_minutes": window_minutes,
        "min_trades_window": min_trades,
        "min_coverage_ratio_window": min_coverage,
        "cooldown_minutes": cooldown,
        "max_actions_per_day": max_actions_val,
        "adjust_floor": floor_val,
        "adjust_step": step_val,
        "actions": policy.get("actions") or [],
    }


def _validate_live_invariants(
    contract: dict,
    contract_file: Path,
    *,
    project_root: Path,
    code_paths: Sequence[Path],
    audit_paths: Sequence[Path],
    metrics_paths: Sequence[Path],
) -> dict:
    live = contract.get("live_invariants")
    if not isinstance(live, dict):
        raise ValueError("live_invariants section is required in the deployment contract.")

    _ensure_required_keys(
        live,
        [
            "mode",
            "kill_switch",
            "safe_mode",
            "time_integrity",
            "risk_limits",
            "idempotency",
            "reconciliation",
            "observability",
        ],
        "live_invariants",
    )

    mode = str(live.get("mode") or "").strip()
    if mode not in ALLOWED_MODES:
        raise ValueError(f"live_invariants.mode must be one of {sorted(ALLOWED_MODES)} (got {mode!r})")

    kill_switch = live.get("kill_switch") or {}
    if not isinstance(kill_switch, dict):
        raise ValueError("live_invariants.kill_switch must be a mapping.")
    kill_env = str(kill_switch.get("env_var") or "").strip()
    if not kill_env:
        raise ValueError("Kill switch env var is required at live_invariants.kill_switch.env_var.")
    kill_behavior = str(kill_switch.get("behavior") or "no_new_entries").strip() or "no_new_entries"
    if kill_behavior not in ALLOWED_KILL_BEHAVIOR:
        raise ValueError(
            f"live_invariants.kill_switch.behavior must be one of {sorted(ALLOWED_KILL_BEHAVIOR)} "
            f"(got {kill_behavior!r})"
        )

    safe_mode = live.get("safe_mode") or {}
    if not isinstance(safe_mode, dict):
        raise ValueError("live_invariants.safe_mode must be a mapping.")
    safe_env = str(safe_mode.get("env_var") or "").strip()
    if not safe_env:
        raise ValueError("Safe mode env var is required at live_invariants.safe_mode.env_var.")

    time_integrity = live.get("time_integrity") or {}
    if not isinstance(time_integrity, dict):
        raise ValueError("live_invariants.time_integrity must be a mapping.")
    _ensure_required_keys(time_integrity, ["require_monotonic_timestamps", "max_clock_skew_seconds"], "time_integrity")
    if not isinstance(time_integrity.get("require_monotonic_timestamps"), bool):
        raise ValueError("live_invariants.time_integrity.require_monotonic_timestamps must be a boolean.")
    try:
        skew = float(time_integrity.get("max_clock_skew_seconds"))
    except (TypeError, ValueError) as exc:
        raise ValueError("live_invariants.time_integrity.max_clock_skew_seconds must be numeric.") from exc
    if skew <= 0:
        raise ValueError("live_invariants.time_integrity.max_clock_skew_seconds must be > 0.")

    risk_section = live.get("risk_limits") or {}
    if not isinstance(risk_section, dict):
        raise ValueError("live_invariants.risk_limits must be a mapping.")
    _ensure_required_keys(risk_section, ["path", "require"], "live_invariants.risk_limits")
    risk_limits_path = _resolve_path(project_root, str(risk_section.get("path") or ""))
    required_risk_keys = risk_section.get("require") or []
    if not isinstance(required_risk_keys, (list, tuple)):
        raise ValueError("live_invariants.risk_limits.require must be a list of required risk limit keys.")
    _ensure_required_list("live_invariants.risk_limits.require", required_risk_keys, REQUIRED_RISK_LIMIT_KEYS)
    risk_cfg = _validate_risk_limits(risk_limits_path, required_risk_keys)

    idempotency = live.get("idempotency") or {}
    if not isinstance(idempotency, dict):
        raise ValueError("live_invariants.idempotency must be a mapping.")
    require_intent_id = idempotency.get("require_order_intent_id")
    if require_intent_id not in (True, False):
        raise ValueError("live_invariants.idempotency.require_order_intent_id must be a boolean.")

    reconciliation = live.get("reconciliation") or {}
    if not isinstance(reconciliation, dict):
        raise ValueError("live_invariants.reconciliation must be a mapping.")
    require_reconcile = reconciliation.get("require_live_reconcile_on_startup")
    if require_reconcile not in (True, False):
        raise ValueError("live_invariants.reconciliation.require_live_reconcile_on_startup must be a boolean.")

    observability = live.get("observability") or {}
    if not isinstance(observability, dict):
        raise ValueError("live_invariants.observability must be a mapping.")
    _ensure_required_keys(
        observability, ["required_counters", "required_fields_in_audit_log"], "live_invariants.observability"
    )
    required_counters = observability.get("required_counters") or []
    required_audit_fields = observability.get("required_fields_in_audit_log") or []
    if not isinstance(required_counters, (list, tuple)) or not required_counters:
        raise ValueError("live_invariants.observability.required_counters must list expected counters.")
    if not isinstance(required_audit_fields, (list, tuple)) or not required_audit_fields:
        raise ValueError("live_invariants.observability.required_fields_in_audit_log must list audit fields.")

    ledger_backend = os.getenv("TRADING_INTENT_LEDGER_BACKEND", "").strip().lower()
    if not ledger_backend:
        ledger_backend = str(contract.get("intent_ledger_backend") or "").strip().lower()

    if mode == "live":
        dry_run_cfg = contract.get("dry_run_integration") or {}
        risky_flags = []
        for flag in ("state_reset", "reset_state", "reset_on_start", "reset_state_on_start"):
            if dry_run_cfg.get(flag) or contract.get(flag):
                risky_flags.append(flag)
        if risky_flags:
            raise ValueError(
                "Live mode forbids dry-run reset settings. Disable the following keys before going live: "
                f"{sorted(set(risky_flags))}"
            )
        if not kill_env or not safe_env:
            raise ValueError("Live mode requires kill_switch.env_var and safe_mode.env_var to be provided.")
        if ledger_backend != "redis":
            raise ValueError("Live mode requires TRADING_INTENT_LEDGER_BACKEND=redis (no memory fallback).")

    _assert_strings_present(
        [kill_env],
        code_paths,
        "kill-switch wiring",
        f"Add an env check for {kill_env} near the entry logic in app/trading/decision.py or service.py.",
    )
    _assert_strings_present(
        [safe_env],
        code_paths,
        "safe-mode wiring",
        f"Add an env check for {safe_env} near the entry logic in app/trading/decision.py or service.py.",
    )
    if require_intent_id:
        _assert_strings_present(
            ["order_intent_id", "intent_id", "order_intent"],
            code_paths,
            "idempotency hooks",
            "Add or expose an order intent id in the trading executor/service.",
        )
    _assert_strings_present(
        required_audit_fields,
        audit_paths,
        "audit logging fields",
        "Ensure audit logging emits the required fields in app/trading/audit.py.",
    )
    for required in REQUIRED_AUDIT_PROVENANCE_FIELDS:
        if required not in required_audit_fields:
            raise ValueError(
                f"live_invariants.observability.required_fields_in_audit_log must include {required} for provenance."
            )
    _assert_strings_present(
        [str(counter) for counter in required_counters],
        metrics_paths,
        "observability counters",
        "Expose counters or gauges matching the contract in monitoring.",
    )
    _assert_strings_present(
        ["assess_and_adjust_order"],
        code_paths,
        "runtime risk engine wiring",
        "Call assess_and_adjust_order before submitting orders in the trading service.",
    )

    deadlock_policy = _validate_deadlock_policy(contract.get("deadlock_policy"), mode)

    if mode == "live":
        if not os.getenv("TRADING_AUDIT_HMAC_KEY"):
            raise ValueError("Live mode requires TRADING_AUDIT_HMAC_KEY for audit provenance integrity.")
        if "deadlock_action_taken_total" not in required_counters:
            raise ValueError(
                "deadlock_action_taken_total must be present in observability.required_counters for live deployments."
            )

    return {
        "mode": mode,
        "kill_switch_env": kill_env,
        "kill_switch_behavior": kill_behavior,
        "safe_mode_env": safe_env,
        "risk_limits_path": str(risk_limits_path),
        "required_risk_limit_keys": list(required_risk_keys),
        "require_order_intent_id": bool(require_intent_id),
        "require_live_reconcile_on_startup": bool(require_reconcile),
        "intent_ledger_backend": ledger_backend or "",
        "observability_counters": list(required_counters),
        "observability_audit_fields": list(required_audit_fields),
        "deadlock_policy": deadlock_policy,
    }


def _validate_symbol_contract(
    contract: dict,
    *,
    risk_cfg: dict,
    policies: Dict[str, dict],
    default_policy: str,
    contract_models: Dict[str, object],
) -> dict:
    live_symbols_raw = contract.get("live_symbols")
    if not isinstance(live_symbols_raw, (list, tuple)) or not live_symbols_raw:
        raise ValueError("live_symbols must list at least one enabled symbol.")
    live_symbols: List[str] = []
    seen: Set[str] = set()
    for raw_sym in live_symbols_raw:
        normalized = _normalize_symbol(raw_sym)
        canonical_input = str(raw_sym).strip().replace(" ", "").upper()
        if normalized != canonical_input:
            raise ValueError(
                f"live_symbols must use normalized CCXT formatting (got {raw_sym!r}, expected {normalized!r})."
            )
        if normalized in seen:
            raise ValueError(f"Duplicate symbol detected in live_symbols: {normalized}")
        seen.add(normalized)
        live_symbols.append(normalized)

    risk_symbols = risk_cfg.get("symbols") or {}
    missing_risk = [sym for sym in live_symbols if sym not in risk_symbols]
    if missing_risk:
        raise ValueError(f"Each live_symbol must have a risk_limits.symbols entry (missing: {missing_risk})")
    for sym in live_symbols:
        _ensure_required_keys(risk_symbols.get(sym, {}), REQUIRED_SYMBOL_RISK_KEYS, f"risk_limits.symbols[{sym}]")

    policy_map_raw = contract.get("symbol_policy_map") or {}
    if not isinstance(policy_map_raw, dict):
        raise ValueError("symbol_policy_map must be a mapping of symbol -> policy_id.")
    policy_map: Dict[str, str] = {_normalize_symbol(k): str(v) for k, v in policy_map_raw.items()}
    extra_policy_symbols = [sym for sym in policy_map if sym not in live_symbols]
    if extra_policy_symbols:
        raise ValueError(f"symbol_policy_map contains symbols not marked live: {extra_policy_symbols}")
    policy_ids = set(policies.keys())
    resolved_policy_map: Dict[str, str] = {}
    for sym in live_symbols:
        policy_id = policy_map.get(sym, default_policy)
        if policy_id not in policy_ids:
            raise ValueError(f"Policy '{policy_id}' for symbol {sym} missing from portfolio policies.")
        resolved_policy_map[sym] = policy_id

    shadow_map_raw = contract.get("symbol_shadow_mode") or {}
    if shadow_map_raw and not isinstance(shadow_map_raw, dict):
        raise ValueError("symbol_shadow_mode must be a mapping of symbol -> boolean.")
    shadow_map: Dict[str, bool] = {_normalize_symbol(k): bool(v) for k, v in shadow_map_raw.items()}
    shadow_outliers = [sym for sym in shadow_map if sym not in live_symbols]
    if shadow_outliers:
        raise ValueError(f"Shadow-mode symbols must be a subset of live_symbols (got extras {shadow_outliers})")

    model_map_raw = contract.get("symbol_model_key") or {}
    if model_map_raw and not isinstance(model_map_raw, dict):
        raise ValueError("symbol_model_key must be a mapping of symbol -> model key.")
    available_models = set(contract_models.keys())
    if not available_models:
        raise ValueError("Deployment contract must declare models before mapping symbols.")
    model_map: Dict[str, str] = {_normalize_symbol(k): str(v) for k, v in (model_map_raw or {}).items()}
    resolved_model_map: Dict[str, str] = {}
    missing_model_map: List[str] = []
    for sym in live_symbols:
        model_key = model_map.get(sym)
        if model_key is None and len(available_models) == 1:
            model_key = next(iter(available_models))
        if model_key is None:
            missing_model_map.append(sym)
            continue
        if model_key not in available_models:
            raise ValueError(
                f"symbol_model_key for {sym} references unknown model '{model_key}'. "
                f"Valid options: {sorted(available_models)}"
            )
        resolved_model_map[sym] = model_key
    if missing_model_map:
        raise ValueError(f"symbol_model_key must specify a model for each live_symbol (missing: {missing_model_map})")

    env_trading_symbols = _parse_trading_models_env()
    if env_trading_symbols:
        missing_env = [sym for sym in live_symbols if sym not in env_trading_symbols]
        if missing_env:
            raise ValueError(f"live_symbols missing from TRADING_MODELS env: {missing_env}")
    else:
        missing_from_map = [sym for sym in live_symbols if sym not in resolved_model_map]
        if missing_from_map:
            raise ValueError(
                "TRADING_MODELS env is empty; symbol_model_key must map every live symbol "
                f"(missing: {missing_from_map})"
            )

    return {
        "live_symbols": live_symbols,
        "symbol_policy_map": resolved_policy_map,
        "symbol_shadow_mode": shadow_map,
        "symbol_model_key": resolved_model_map,
    }


def validate_deployment_contract(
    contract_path: str,
    *,
    code_paths: Sequence[Path] | None = None,
    audit_paths: Sequence[Path] | None = None,
    metrics_paths: Sequence[Path] | None = None,
) -> dict:
    """
    Validate that the deployment contract references existing artifacts and policies.
    """
    contract_file = Path(contract_path).expanduser().resolve()
    _require_path(contract_file, "deployment_contract")
    contract = _load_yaml(contract_file)
    project_root = _guess_project_root(contract_file)

    code_paths = _normalize_paths(project_root, tuple(code_paths or DEFAULT_CODE_PATHS))
    audit_paths = _normalize_paths(project_root, tuple(audit_paths or DEFAULT_AUDIT_PATHS))
    metrics_paths = _normalize_paths(project_root, tuple(metrics_paths or DEFAULT_METRICS_PATHS))

    dataset_contract = _resolve_path(project_root, contract.get("dataset_contract", ""))
    best_model_cfg = _resolve_path(project_root, contract.get("best_model_configs", ""))
    risk_limits = _resolve_path(project_root, contract.get("risk_limits", ""))
    portfolio_policies_path = _resolve_path(project_root, contract.get("portfolio_policies", ""))

    for path, label in [
        (dataset_contract, "dataset_contract"),
        (best_model_cfg, "best_model_configs"),
        (risk_limits, "risk_limits"),
        (portfolio_policies_path, "portfolio_policies"),
    ]:
        _require_path(path, label)

    risk_cfg = _load_yaml(risk_limits)
    live_summary = _validate_live_invariants(
        contract,
        contract_file,
        project_root=project_root,
        code_paths=code_paths,
        audit_paths=audit_paths,
        metrics_paths=metrics_paths,
    )
    live_risk_path = Path(live_summary["risk_limits_path"])
    if live_risk_path.resolve() != risk_limits.resolve():
        raise ValueError(
            f"risk_limits path ({risk_limits}) must match live_invariants.risk_limits.path ({live_risk_path})"
        )
    risk_cfg = _validate_risk_limits(live_risk_path, REQUIRED_RISK_LIMIT_KEYS)

    policies, default_policy = _load_policies(contract, project_root)
    if not policies:
        raise ValueError("No policies available in deployment contract.")
    if default_policy not in policies:
        raise ValueError(f"Default policy '{default_policy}' missing from portfolio_policies.")

    models = contract.get("models") or {}
    if not models:
        raise ValueError("No models section found in deployment contract.")
    missing_models = [
        name for name, path in models.items() if not _resolve_path(project_root, str(path)).expanduser().exists()
    ]
    if missing_models:
        raise FileNotFoundError(f"Missing model artifact paths: {missing_models}")

    symbol_summary = _validate_symbol_contract(
        contract,
        risk_cfg=risk_cfg,
        policies=policies,
        default_policy=default_policy,
        contract_models=models,
    )

    summary = {
        "contract_path": str(contract_file),
        "dataset_contract": str(dataset_contract),
        "best_model_configs": str(best_model_cfg),
        "risk_limits": str(risk_limits),
        "portfolio_policies": str(portfolio_policies_path),
        "default_policy": default_policy,
        "policy_ids": sorted(list(policies.keys())),
        "models": {k: str(v) for k, v in models.items()},
        "risk_limits_keys": list(risk_cfg.keys()),
        "live_invariants": live_summary,
        "symbols": symbol_summary,
    }
    print("Deployment contract validation passed.")
    print(f"- dataset_contract: {dataset_contract}")
    print(f"- best_model_configs: {best_model_cfg}")
    print(f"- risk_limits: {risk_limits}")
    print(f"- portfolio_policies: {portfolio_policies_path} (default policy: {default_policy})")
    print(f"- models: {summary['models']}")
    print(
        f"- live_invariants: mode={live_summary['mode']} "
        f"kill_switch={live_summary['kill_switch_env']} behavior={live_summary['kill_switch_behavior']} "
        f"safe_mode={live_summary['safe_mode_env']}"
    )
    print(
        f"- symbols: {symbol_summary['live_symbols']} "
        f"shadow={symbol_summary['symbol_shadow_mode']} "
        f"policies={symbol_summary['symbol_policy_map']} "
        f"models={symbol_summary['symbol_model_key']}"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate deployment_portfolio_contract.yaml for completeness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--contract", required=True, help="Path to deployment_portfolio_contract.yaml")
    args = ap.parse_args(argv)

    validate_deployment_contract(args.contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
