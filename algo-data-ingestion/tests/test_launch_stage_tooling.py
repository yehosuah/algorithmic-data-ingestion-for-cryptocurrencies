import json
from pathlib import Path

import pytest
import yaml

from analysis.apply_launch_stage import _apply_stage
from analysis.evaluate_launch_stage import main as eval_main
from analysis.shadow_readiness import _compute_record_hmac
from tests.test_live_invariants_validation import _minimal_contract


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload))


def test_apply_stage_updates_contract_and_overrides(tmp_path: Path):
    base_risk = {
        "capital": 1000000,
        "max_gross_leverage": 3.0,
        "max_net_exposure": 1.5,
        "max_turnover_per_day": 1.0,
        "max_orders_per_hour": 50,
        "max_concurrent_positions": 3,
        "daily_loss_limit_pct": 0.05,
        "max_drawdown_pct": 0.1,
        "cooldown_minutes_after_exit": 5,
        "cooldown_minutes_after_loss": 5,
        "halt_on_safe_mode": True,
        "halt_if_spread_bps_gt": 35.0,
        "halt_if_vol_zscore_gt": 4.0,
        "halt_if_missing_price_bars": False,
        "halt_if_data_stale_seconds": 120,
        "symbols": {
            "ETH/USDT": {"max_symbol_notional": 50000, "max_symbol_weight": 0.2, "max_spread_bps": 30, "min_trade_notional": 10.0},
            "BTC/USDT": {"max_symbol_notional": 60000, "max_symbol_weight": 0.2, "max_spread_bps": 30, "min_trade_notional": 10.0},
        },
    }
    base_risk_path = tmp_path / "risk.yaml"
    _write_yaml(base_risk_path, base_risk)

    contract_path = tmp_path / "contract.yaml"
    contract = {
        "dataset_contract": str(tmp_path / "dataset.yaml"),
        "best_model_configs": str(tmp_path / "best.yaml"),
        "risk_limits": str(base_risk_path),
        "portfolio_policies": str(tmp_path / "policies.yaml"),
        "models_root": str(tmp_path),
        "models": {"xgb_primary": str(tmp_path / "model.bin")},
        "live_symbols": ["ETH/USDT"],
        "symbol_model_key": {"ETH/USDT": "xgb_primary", "BTC/USDT": "xgb_primary"},
        "symbol_policy_map": {"ETH/USDT": "primary", "BTC/USDT": "primary"},
        "symbol_shadow_mode": {"ETH/USDT": False, "BTC/USDT": True},
        "live_invariants": {
            "mode": "dry_run",
            "kill_switch": {"env_var": "TRADING_KILL_SWITCH", "behavior": "no_new_entries"},
            "safe_mode": {"env_var": "TRADING_SAFE_MODE"},
            "time_integrity": {"require_monotonic_timestamps": True, "max_clock_skew_seconds": 5},
            "risk_limits": {"path": str(base_risk_path), "require": []},
            "idempotency": {"require_order_intent_id": True},
            "reconciliation": {"require_live_reconcile_on_startup": True},
            "observability": {"required_counters": ["trade_count"], "required_fields_in_audit_log": ["symbol", "ts"]},
        },
    }
    _write_yaml(contract_path, contract)

    ladder_path = tmp_path / "ladder.yaml"
    _write_yaml(
        ladder_path,
        {
            "base_risk_limits_path": str(base_risk_path),
            "launch_ladder": {
                "stage_0": {
                    "mode": "dry_run",
                    "live_symbols": ["ETH/USDT"],
                    "shadow_symbols": ["BTC/USDT"],
                    "per_symbol": {
                        "ETH/USDT": {"order_notional": 50, "policy_id": "primary"},
                        "BTC/USDT": {"order_notional": 40, "policy_id": "conservative"},
                    },
                    "risk_overrides": {"max_orders_per_hour": 10, "symbols": {"ETH/USDT": {"max_spread_bps": 25}}},
                    "deadlock_policy": {
                        "enabled": True,
                        "window_minutes": 30,
                        "min_trades_window": 1,
                        "min_coverage_ratio_window": 0.01,
                        "cooldown_minutes": 30,
                        "max_actions_per_day": 2,
                        "adjust_prob_gate_min": {"step": 0.02, "floor": 0.48},
                        "actions": [{"enter_safe_mode": True}],
                    },
                }
            },
        },
    )

    overrides_dir = tmp_path / "runtime_overrides"
    summary = _apply_stage(
        stage_name="stage_0",
        ladder_path=ladder_path,
        contract_path=contract_path,
        runtime_overrides=overrides_dir,
    )

    updated_contract = yaml.safe_load(contract_path.read_text())
    assert updated_contract["live_symbols"] == ["ETH/USDT", "BTC/USDT"]
    assert updated_contract["symbol_shadow_mode"]["BTC/USDT"] is True
    assert updated_contract["symbol_policy_map"]["BTC/USDT"] == "conservative"
    assert Path(summary["risk_limits_path"]).exists()
    risk_override = yaml.safe_load(Path(summary["risk_limits_path"]).read_text())
    assert risk_override["max_orders_per_hour"] == 10
    assert Path(summary["patch_path"]).exists()
    patch = yaml.safe_load(Path(summary["patch_path"]).read_text())
    assert patch["env"]["TRADING_SHADOW_SYMBOLS"] == "BTC/USDT"


def test_evaluate_stage_fails_on_gate_violation(tmp_path: Path, monkeypatch):
    key = "secret-key"
    monkeypatch.setenv("TRADING_INTENT_LEDGER_BACKEND", "redis")
    monkeypatch.setenv("TRADING_AUDIT_HMAC_KEY", key)
    monkeypatch.setenv(
        "TRADING_MODELS",
        json.dumps([{"model": "xgb_primary", "exchange": "binance", "symbol": "ETH/USDT", "timeframe": "1m", "order_notional": 10.0}]),
    )

    contract_path = _minimal_contract(tmp_path, live_mode="live")
    contract = yaml.safe_load(contract_path.read_text())
    contract["live_symbols"] = ["ETH/USDT"]
    contract["symbol_policy_map"] = {"ETH/USDT": "primary"}
    contract["symbol_shadow_mode"] = {"ETH/USDT": False}
    contract["symbol_model_key"] = {"ETH/USDT": "xgb_primary"}
    _write_yaml(contract_path, contract)

    risk_path = Path(contract["risk_limits"])
    risk_cfg = yaml.safe_load(risk_path.read_text())
    risk_cfg["symbols"]["ETH/USDT"] = {
        "max_symbol_notional": 1000,
        "max_symbol_weight": 0.2,
        "max_spread_bps": 30,
        "min_trade_notional": 1.0,
    }
    _write_yaml(risk_path, risk_cfg)

    ladder_path = tmp_path / "ladder.yaml"
    _write_yaml(
        ladder_path,
        {
            "launch_ladder": {
                "stage_0": {
                    "mode": "live",
                    "live_symbols": ["ETH/USDT"],
                    "shadow_symbols": [],
                    "per_symbol": {"ETH/USDT": {"order_notional": 25, "policy_id": "primary"}},
                    "promotion": {
                        "min_runtime_minutes": 1,
                        "gates": {
                            "min_trade_count": 2,
                            "min_coverage_ratio": 0.5,
                            "max_spread_block_rate": 0.5,
                            "max_risk_block_rate": 0.5,
                            "max_safe_mode_events": 0,
                            "max_reconcile_mismatches": 0,
                            "max_deadlock_actions": 0,
                            "max_drawdown_pct": 0.5,
                        },
                    },
                }
            }
        },
    )

    audit_path = tmp_path / "audit.log"
    now = "2024-01-01T00:00:00+00:00"
    record = {
        "occurred_at": now,
        "event_type": "trade",
        "model": "xgb",
        "symbol": "ETH/USDT",
        "payload": {"gate_pass": False, "risk_allowed": True},
        "audit_source": "runtime",
        "audit_run_id": "run-1",
        "audit_seq": 1,
    }
    record["audit_hmac"] = _compute_record_hmac(record, key.encode())
    audit_path.write_text(json.dumps(record) + "\n")

    monkeypatch.setattr("analysis.evaluate_launch_stage.coverage_main", lambda argv=None: 0)
    reports_dir = tmp_path / "reports"
    rc = eval_main(
        [
            "--stage",
            "stage_0",
            "--ladder",
            str(ladder_path),
            "--contract",
            str(contract_path),
            "--audit-log",
            str(audit_path),
            "--reports-dir",
            str(reports_dir),
            "--hours",
            "2",
        ]
    )
    assert rc == 1
    reports = list(reports_dir.glob("launch_stage_eval_stage_0_*.json"))
    assert reports
    report = json.loads(reports[0].read_text())
    assert report["status"] == "NO_GO"
    assert any("trade_count" in reason for reason in report["failures"])
