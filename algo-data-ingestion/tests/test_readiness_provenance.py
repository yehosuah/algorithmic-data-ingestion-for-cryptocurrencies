import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from analysis.preflight_symbol_promotion import main as promotion_main
from analysis.shadow_readiness import _compute_record_hmac, _load_audit_events
from tests.test_live_invariants_validation import _minimal_contract


def test_shadow_readiness_rejects_missing_provenance(tmp_path: Path):
    now = datetime.now(timezone.utc)
    audit_path = tmp_path / "audit.log"
    record = {
        "occurred_at": now.isoformat(),
        "event_type": "trade",
        "model": "xgb",
        "symbol": "BTC/USDT",
        "payload": {},
    }
    audit_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        _load_audit_events(
            audit_path,
            symbols=["BTC/USDT"],
            window_start=now - timedelta(minutes=5),
            time_min=None,
            time_max=None,
            audit_source="runtime",
            allow_multi_run=False,
            require_hmac=False,
            hmac_key=None,
        )


def test_shadow_readiness_validates_hmac(tmp_path: Path):
    now = datetime.now(timezone.utc)
    audit_path = tmp_path / "audit.log"
    key = "secret"
    record = {
        "occurred_at": now.isoformat(),
        "event_type": "trade",
        "model": "xgb",
        "symbol": "BTC/USDT",
        "payload": {},
        "audit_source": "runtime",
        "audit_run_id": "run-1",
        "audit_seq": 1,
    }
    record["audit_hmac"] = _compute_record_hmac(record, key.encode())
    audit_path.write_text(json.dumps(record) + "\n")
    events, provenance = _load_audit_events(
        audit_path,
        symbols=["BTC/USDT"],
        window_start=now - timedelta(minutes=5),
        time_min=None,
        time_max=None,
        audit_source="runtime",
        allow_multi_run=False,
        require_hmac=True,
        hmac_key=key,
    )
    assert events
    assert provenance["run_ids"] == ["run-1"]
    assert provenance["hmac_validated"] is True


def test_preflight_promotion_requires_provenance(tmp_path: Path, monkeypatch):
    contract_path = _minimal_contract(tmp_path)
    contract = yaml.safe_load(contract_path.read_text())
    contract["symbol_shadow_mode"]["BTC/USDT"] = True
    contract["live_symbols"] = ["BTC/USDT"]
    contract_path.write_text(yaml.safe_dump(contract))
    readiness_report = {
        "symbols": {
            "BTC/USDT": {
                "would_enter": 5,
                "would_exit": 5,
                "implied_trades": 2,
                "promotion_ready": True,
                "promotion_reasons": [],
                "risk_block_rate": 0.0,
                "spread_block_rate": 0.0,
            }
        },
        "thresholds": {"min_would_enter": 1, "max_risk_block_rate": 1.0, "max_spread_block_rate": 1.0},
        "thresholds_by_symbol": {"BTC/USDT": {"min_would_enter": 1, "max_risk_block_rate": 1.0, "max_spread_block_rate": 1.0}},
    }
    report_path = tmp_path / "readiness.json"
    report_path.write_text(json.dumps(readiness_report))
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
    result = promotion_main(
        [
            "--contract",
            str(contract_path),
            "--shadow-report",
            str(report_path),
        ]
    )
    assert result == 1
