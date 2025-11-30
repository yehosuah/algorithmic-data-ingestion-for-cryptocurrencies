import json
from pathlib import Path

import pytest
import yaml

from analysis.validate_deployment_contract import (
    REQUIRED_RISK_LIMIT_KEYS,
    validate_deployment_contract,
)


@pytest.fixture(autouse=True)
def _trading_models_env(monkeypatch):
    monkeypatch.setenv(
        "TRADING_MODELS",
        json.dumps(
            [
                {
                    "model": "xgb_primary",
                    "exchange": "binance",
                    "symbol": "BTC/USDT",
                    "timeframe": "1m",
                    "order_notional": 50.0,
                }
            ]
        ),
    )
    monkeypatch.setenv("TRADING_AUDIT_HMAC_KEY", "test-key")
    yield


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload))


def _build_risk_limits(tmp_path: Path, *, include_missing: bool = False) -> Path:
    payload = {
        "capital": 1_000_000,
        "max_gross_leverage": 3.0,
        "max_net_exposure": 1.5,
        "max_turnover_per_day": 1.0,
        "max_orders_per_hour": 30,
        "max_concurrent_positions": 3,
        "daily_loss_limit_pct": 0.03,
        "max_drawdown_pct": 0.1,
        "cooldown_minutes_after_exit": 5,
        "cooldown_minutes_after_loss": 15,
        "halt_on_safe_mode": True,
        "halt_if_spread_bps_gt": 25.0,
        "halt_if_vol_zscore_gt": 4.0,
        "halt_if_missing_price_bars": False,
        "halt_if_data_stale_seconds": 90,
        "symbols": {
            "BTC/USDT": {
                "max_symbol_notional": 1_000,
                "max_symbol_weight": 0.2,
                "max_spread_bps": 25,
                "min_trade_notional": 10.0,
            }
        },
    }
    if include_missing:
        payload.pop("max_drawdown_pct")
    path = tmp_path / "risk_limits.yaml"
    _write_yaml(path, payload)
    return path


def _minimal_contract(tmp_path: Path, *, live_mode: str = "dry_run", kill_env: str = "TRADING_KILL_SWITCH"):
    dataset = tmp_path / "dataset.yaml"
    best_model = tmp_path / "best_model.yaml"
    policies = tmp_path / "policies.yaml"
    model_artifact = tmp_path / "model.bin"
    for p in (dataset, best_model, policies, model_artifact):
        p.write_text("ok\n")
    _write_yaml(policies, {"primary": {"id": "policy_001"}})
    risk_path = _build_risk_limits(tmp_path)
    contract = {
        "dataset_contract": str(dataset),
        "best_model_configs": str(best_model),
        "risk_limits": str(risk_path),
        "portfolio_policies": str(policies),
        "models": {"xgb_primary": str(model_artifact)},
        "live_symbols": ["BTC/USDT"],
        "symbol_policy_map": {"BTC/USDT": "primary"},
        "symbol_shadow_mode": {"BTC/USDT": False},
        "symbol_model_key": {"BTC/USDT": "xgb_primary"},
        "live_invariants": {
            "mode": live_mode,
            "kill_switch": {"env_var": kill_env, "behavior": "no_new_entries"},
            "safe_mode": {"env_var": "TRADING_SAFE_MODE"},
            "time_integrity": {"require_monotonic_timestamps": True, "max_clock_skew_seconds": 5},
            "risk_limits": {"path": str(risk_path), "require": list(REQUIRED_RISK_LIMIT_KEYS)},
            "idempotency": {"require_order_intent_id": True},
            "reconciliation": {"require_live_reconcile_on_startup": True},
            "observability": {
                "required_counters": ["trade_count", "coverage", "skips_by_reason", "deadlock_action_taken_total"],
                "required_fields_in_audit_log": [
                    "symbol",
                    "timeframe",
                    "ts",
                    "policy_id",
                    "gate_pass",
                    "prob",
                    "decision",
                    "skip_reason",
                    "audit_source",
                    "audit_run_id",
                    "audit_seq",
                ],
            },
        },
        "deadlock_policy": {
            "enabled": True,
            "window_minutes": 60,
            "min_trades_window": 1,
            "min_coverage_ratio_window": 0.01,
            "cooldown_minutes": 30,
            "max_actions_per_day": 3,
            "adjust_prob_gate_min": {"step": 0.02, "floor": 0.48},
            "actions": [
                {"adjust_prob_gate_min": {"step": 0.02, "floor": 0.48}},
                {"enter_safe_mode": True},
            ],
        },
    }
    contract_path = tmp_path / "contract.yaml"
    _write_yaml(contract_path, contract)
    return contract_path


def test_valid_contract_passes(tmp_path: Path):
    contract_path = _minimal_contract(tmp_path)
    summary = validate_deployment_contract(str(contract_path))
    assert summary["live_invariants"]["mode"] == "dry_run"
    assert summary["policy_ids"]


def test_missing_live_invariants_fails(tmp_path: Path):
    contract_path = _minimal_contract(tmp_path)
    contract = yaml.safe_load(contract_path.read_text())
    contract.pop("live_invariants", None)
    _write_yaml(contract_path, contract)
    with pytest.raises(ValueError):
        validate_deployment_contract(str(contract_path))


def test_missing_risk_for_live_symbol_fails(tmp_path: Path, monkeypatch):
    contract_path = _minimal_contract(tmp_path)
    contract = yaml.safe_load(contract_path.read_text())
    contract["live_symbols"] = ["BTC/USDT", "ETH/USDT"]
    contract["symbol_policy_map"]["ETH/USDT"] = "primary"
    contract["symbol_model_key"]["ETH/USDT"] = "xgb_primary"
    _write_yaml(contract_path, contract)
    monkeypatch.setenv(
        "TRADING_MODELS",
        json.dumps(
            [
                {
                    "model": "xgb_primary",
                    "exchange": "binance",
                    "symbol": "BTC/USDT",
                    "timeframe": "1m",
                    "order_notional": 50.0,
                },
                {
                    "model": "xgb_primary",
                    "exchange": "binance",
                    "symbol": "ETH/USDT",
                    "timeframe": "1m",
                    "order_notional": 50.0,
                },
            ]
        ),
    )
    with pytest.raises(ValueError):
        validate_deployment_contract(str(contract_path))


def test_duplicate_live_symbols_fail(tmp_path: Path):
    contract_path = _minimal_contract(tmp_path)
    contract = yaml.safe_load(contract_path.read_text())
    contract["live_symbols"] = ["BTC/USDT", "btc/usdt"]
    _write_yaml(contract_path, contract)
    with pytest.raises(ValueError):
        validate_deployment_contract(str(contract_path))


def test_missing_risk_limit_key_fails(tmp_path: Path):
    contract_path = _minimal_contract(tmp_path)
    missing_risk = tmp_path / "risk_limits_missing.yaml"
    _write_yaml(
        missing_risk,
        {
            "capital": 1_000_000,
            "max_gross_leverage": 3.0,
            "max_net_exposure": 1.5,
            "max_turnover_per_day": 1.0,
            "max_concurrent_positions": 3,
            "daily_loss_limit_pct": 0.02,
            "max_drawdown_pct": 0.1,
            "cooldown_minutes_after_exit": 5,
            "cooldown_minutes_after_loss": 10,
            "halt_on_safe_mode": True,
            "halt_if_spread_bps_gt": 20.0,
            "halt_if_vol_zscore_gt": 4.0,
            "halt_if_missing_price_bars": False,
            "halt_if_data_stale_seconds": 60,
            "symbols": {
                "BTC/USDT": {
                    "max_symbol_notional": 1_000,
                    "max_symbol_weight": 0.2,
                    "max_spread_bps": 25,
                    "min_trade_notional": 10.0,
                }
            },
        },
    )
    contract = yaml.safe_load(contract_path.read_text())
    contract["risk_limits"] = str(missing_risk)
    contract["live_invariants"]["risk_limits"]["path"] = str(missing_risk)
    _write_yaml(contract_path, contract)
    with pytest.raises(ValueError):
        validate_deployment_contract(str(contract_path))


def test_live_mode_without_kill_switch_env_var_fails(tmp_path: Path):
    contract_path = _minimal_contract(tmp_path, live_mode="live", kill_env="")
    with pytest.raises(ValueError):
        validate_deployment_contract(str(contract_path))


def test_validator_detects_missing_kill_switch_string(tmp_path: Path, monkeypatch):
    contract_path = _minimal_contract(tmp_path, live_mode="live")
    monkeypatch.setenv("TRADING_INTENT_LEDGER_BACKEND", "redis")
    dummy_code = tmp_path / "code.py"
    dummy_code.write_text(
        "\n".join(
            [
                "TRADING_SAFE_MODE",
                "order_intent_id",
                "trade_count",
                "coverage",
                "skips_by_reason",
                "symbol",
                "ts",
                "policy_id",
                "gate_pass",
                "prob",
                "decision",
                "skip_reason",
            ]
        )
    )
    with pytest.raises(ValueError):
        validate_deployment_contract(
            str(contract_path),
            code_paths=[dummy_code],
            audit_paths=[dummy_code],
            metrics_paths=[dummy_code],
        )


def test_trading_models_env_must_cover_symbols(tmp_path: Path, monkeypatch):
    contract_path = _minimal_contract(tmp_path)
    contract = yaml.safe_load(contract_path.read_text())
    risk = yaml.safe_load(Path(contract["risk_limits"]).read_text())
    risk["symbols"]["ETH/USDT"] = dict(risk["symbols"]["BTC/USDT"])
    risk_path = tmp_path / "risk_limits_two.yaml"
    _write_yaml(risk_path, risk)
    contract["risk_limits"] = str(risk_path)
    contract["live_invariants"]["risk_limits"]["path"] = str(risk_path)
    contract["live_symbols"] = ["BTC/USDT", "ETH/USDT"]
    contract["symbol_policy_map"]["ETH/USDT"] = "primary"
    contract["symbol_model_key"]["ETH/USDT"] = "xgb_primary"
    _write_yaml(contract_path, contract)
    monkeypatch.setenv(
        "TRADING_MODELS",
        json.dumps(
            [
                {
                    "model": "xgb_primary",
                    "exchange": "binance",
                    "symbol": "BTC/USDT",
                    "timeframe": "1m",
                    "order_notional": 50.0,
                }
            ]
        ),
    )
    with pytest.raises(ValueError):
        validate_deployment_contract(str(contract_path))


def test_live_mode_requires_redis_intent_ledger_backend(tmp_path: Path, monkeypatch):
    contract_path = _minimal_contract(tmp_path, live_mode="live")
    monkeypatch.delenv("TRADING_INTENT_LEDGER_BACKEND", raising=False)
    with pytest.raises(ValueError):
        validate_deployment_contract(str(contract_path))
    monkeypatch.setenv("TRADING_INTENT_LEDGER_BACKEND", "redis")
    summary = validate_deployment_contract(str(contract_path))
    assert summary["live_invariants"]["intent_ledger_backend"] == "redis"


def test_live_mode_requires_deadlock_policy(tmp_path: Path, monkeypatch):
    contract_path = _minimal_contract(tmp_path, live_mode="live")
    contract = yaml.safe_load(contract_path.read_text())
    contract.pop("deadlock_policy", None)
    _write_yaml(contract_path, contract)
    monkeypatch.setenv("TRADING_INTENT_LEDGER_BACKEND", "redis")
    with pytest.raises(ValueError):
        validate_deployment_contract(str(contract_path))
