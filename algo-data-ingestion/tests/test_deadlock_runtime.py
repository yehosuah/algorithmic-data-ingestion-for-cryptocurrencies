import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.monitoring.trading_metrics import TRADING_DEADLOCK_ACTIONS
from app.trading.config import TradingConfig
from app.trading.deadlock import DeadlockMonitor, DeadlockPolicy
from app.trading.service import TradingService


class _StubAuditLogger:
    def __init__(self):
        self.deadlock_status = []
        self.deadlock_actions = []
        self.safe_modes = []

    async def log_deadlock_status(self, **kwargs):
        self.deadlock_status.append(kwargs)

    async def log_deadlock_action(self, **kwargs):
        self.deadlock_actions.append(kwargs)

    async def log_safe_mode(self, **kwargs):
        self.safe_modes.append(kwargs)


def _run(coro):
    return asyncio.run(coro)


def test_deadlock_action_emits_audit_and_metrics(tmp_path, monkeypatch):
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
                    "max_spread_bps": 10.0,
                }
            ]
        ),
    )
    monkeypatch.setenv("TRADING_STATE_BACKEND", "file")
    monkeypatch.setenv("TRADING_AUDIT_BACKEND", "file")
    cfg = TradingConfig(
        decision_queue_url="redis://example/0",
        decision_queue_key="queue",
        state_path=tmp_path / "state.json",
        models_root=tmp_path,
        dry_run=True,
        audit_log_path=tmp_path / "audit.log",
    )
    service = TradingService(cfg)
    service.audit_logger = _StubAuditLogger()
    service.deadlock_policy = DeadlockPolicy(
        enabled=True,
        window_minutes=1,
        min_trades_window=1,
        min_coverage_ratio_window=0.5,
        cooldown_minutes=0,
        max_actions_per_day=1,
        actions=[{"enter_safe_mode": True}],
        adjust_floor=0.1,
        adjust_step=0.05,
        audit_every_action=True,
    )
    service.deadlock_monitor = DeadlockMonitor(window_minutes=1)
    service._deadlock_action_state = {"last_action_ts": None, "actions_taken_today": 0, "day": None, "next_index": 0}
    now = datetime.now(timezone.utc)
    service.deadlock_monitor.record_gate(
        model="xgb_primary",
        symbol="BTC/USDT",
        ts=now,
        gate_pass=False,
        probability=0.1,
        prob_gate_min=0.6,
        timeframe="1m",
    )

    _run(service._evaluate_deadlock(now))

    assert service.audit_logger.deadlock_status
    assert service.audit_logger.deadlock_actions
    assert service.audit_logger.safe_modes
    metric_value = TRADING_DEADLOCK_ACTIONS.labels(action="enter_safe_mode")._value.get()
    assert metric_value >= 1.0
    assert service._safe_mode_reason == "deadlock_policy"
