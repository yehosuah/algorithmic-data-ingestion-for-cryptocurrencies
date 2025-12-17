import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from analysis.live_readiness_check import _build_parser, run_readiness_check
from analysis.validate_deployment_contract import REQUIRED_RISK_LIMIT_KEYS


class FixedProbaModel:
    """
    Pickle-friendly test helper for preflight_coverage.
    """

    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        n = len(X)
        probs = np.resize(self._probs, n).astype(float)
        probs = np.clip(probs, 0.0, 1.0)
        return np.column_stack([1.0 - probs, probs])


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload))


def _build_contract_fixture(tmp_path: Path, *, prob_gate_min: float) -> Path:
    # 1) Minimal model directory (base model branch: model.json + feature_list.json + calibrator.joblib)
    model_dir = tmp_path / "model_xgb"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.json").write_text("{}\n")
    (model_dir / "feature_list.json").write_text(json.dumps(["f1"]))
    joblib.dump(FixedProbaModel([0.4, 0.6, 0.4, 0.6]), model_dir / "calibrator.joblib")

    # 2) Features file at the default preflight_coverage fallback location for a tmp_path contract.
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "symbol": ["BTC/USDT"] * 4,
            "f1": [0.0, 1.0, 2.0, 3.0],
        }
    )
    df.to_parquet(exp_dir / "dummy_live_features.parquet", index=False)

    # 3) Policies + risk limits referenced by the contract.
    policies_path = tmp_path / "policies.yaml"
    _write_yaml(
        policies_path,
        {
            "primary": {
                "id": "policy_001",
                "thresholds": {"global": {"entry_long": 0.5, "exit_long": 0.49, "min_hold_bars": 1}},
            }
        },
    )

    risk_path = tmp_path / "risk_limits.yaml"
    risk_cfg = {
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
        "gate_mode": "inference",
        "gate_config": {
            "spread_column": "hl_spread",
            "prob_column": "base_prob",
            "training": {
                "hl_spread_max": None,
                "hl_spread_z_max": None,
                "rvol20_max": None,
                "prob_gate_min": prob_gate_min,
            },
            "inference": {
                "hl_spread_max": None,
                "hl_spread_z_max": None,
                "rvol20_max": None,
                "prob_gate_min": prob_gate_min,
            },
        },
        "symbols": {
            "BTC/USDT": {
                "max_symbol_notional": 1_000,
                "max_symbol_weight": 0.2,
                "max_spread_bps": 25,
                "min_trade_notional": 10.0,
            }
        },
    }
    _write_yaml(risk_path, risk_cfg)

    dataset = tmp_path / "dataset.yaml"
    best_model = tmp_path / "best_model.yaml"
    dataset.write_text("ok\n")
    best_model.write_text("ok\n")

    contract = {
        "dataset_contract": str(dataset),
        "best_model_configs": str(best_model),
        "risk_limits": str(risk_path),
        "portfolio_policies": str(policies_path),
        "models_root": str(tmp_path),
        "models": {"xgb_primary": str(model_dir)},
        "live_symbols": ["BTC/USDT"],
        "symbol_policy_map": {"BTC/USDT": "primary"},
        "symbol_shadow_mode": {"BTC/USDT": False},
        "symbol_model_key": {"BTC/USDT": "xgb_primary"},
        "live_invariants": {
            "mode": "dry_run",
            "kill_switch": {"env_var": "TRADING_KILL_SWITCH", "behavior": "no_new_entries"},
            "safe_mode": {"env_var": "TRADING_SAFE_MODE"},
            "time_integrity": {"require_monotonic_timestamps": True, "max_clock_skew_seconds": 5},
            "risk_limits": {"path": str(risk_path), "require": list(REQUIRED_RISK_LIMIT_KEYS)},
            "idempotency": {"require_order_intent_id": False},
            "reconciliation": {"require_live_reconcile_on_startup": False},
            "observability": {
                "required_counters": ["trade_count"],
                "required_fields_in_audit_log": [
                    "symbol",
                    "timeframe",
                    "ts",
                    "audit_source",
                    "audit_run_id",
                    "audit_seq",
                ],
            },
        },
    }
    contract_path = tmp_path / "contract.yaml"
    _write_yaml(contract_path, contract)
    return contract_path


def _check_status(report: dict, name: str) -> str:
    for check in report.get("checks") or []:
        if check.get("name") == name:
            return str(check.get("status"))
    raise AssertionError(f"missing check {name}")


def test_live_readiness_happy_path_go(tmp_path: Path, monkeypatch):
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
    contract_path = _build_contract_fixture(tmp_path, prob_gate_min=0.5)
    out_dir = tmp_path / "reports"
    args = _build_parser().parse_args(
        [
            "--deployment-contract",
            str(contract_path),
            "--mode",
            "live_like",
            "--no-require-shadow-preflight",
            "--output-dir",
            str(out_dir),
        ]
    )
    report = run_readiness_check(args)
    assert report["overall_status"] == "GO"
    assert _check_status(report, "deployment_contract_validation") == "PASS"
    assert _check_status(report, "coverage_preflight") == "PASS"
    assert Path(report["artifacts"]["readiness_json"]).exists()
    assert Path(report["artifacts"]["readiness_md"]).exists()


def test_live_readiness_contract_fail_no_go(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TRADING_MODELS", json.dumps([]))
    contract_path = _build_contract_fixture(tmp_path, prob_gate_min=0.5)
    payload = yaml.safe_load(contract_path.read_text())
    payload.pop("live_invariants", None)
    _write_yaml(contract_path, payload)

    out_dir = tmp_path / "reports"
    args = _build_parser().parse_args(
        [
            "--deployment-contract",
            str(contract_path),
            "--mode",
            "dry_run",
            "--no-require-shadow-preflight",
            "--output-dir",
            str(out_dir),
        ]
    )
    report = run_readiness_check(args)
    assert report["overall_status"] == "NO_GO"
    assert _check_status(report, "deployment_contract_validation") == "FAIL"


def test_live_readiness_coverage_fail_no_go(tmp_path: Path, monkeypatch):
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
    # prob_gate_min high enough to yield zero fraction above gate (no synthetic audit injection).
    contract_path = _build_contract_fixture(tmp_path, prob_gate_min=1.0)
    out_dir = tmp_path / "reports"
    args = _build_parser().parse_args(
        [
            "--deployment-contract",
            str(contract_path),
            "--mode",
            "dry_run",
            "--no-require-shadow-preflight",
            "--output-dir",
            str(out_dir),
        ]
    )
    report = run_readiness_check(args)
    assert report["overall_status"] == "NO_GO"
    assert _check_status(report, "deployment_contract_validation") == "PASS"
    assert _check_status(report, "coverage_preflight") == "FAIL"

