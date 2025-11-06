import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.trading.config import TradingConfig, TradingModelConfig
from app.trading.executor import OrderDecision
from app.trading.service import TradingService


class FakeExecutor:
    def __init__(self):
        self.calls = []

    async def submit(
        self,
        *,
        exchange,
        symbol,
        side,
        order_amount,
        order_notional,
        max_spread_bps,
    ):
        record = {
            "exchange": exchange,
            "symbol": symbol,
            "side": side,
            "order_amount": order_amount,
            "order_notional": order_notional,
            "max_spread_bps": max_spread_bps,
        }
        self.calls.append(record)
        return OrderDecision(executed=True, price_used=1.0, amount=1.0, spread_bps=1.0)

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_trading_service_respects_min_hold(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TRADING_MODELS", "[]")
    monkeypatch.setenv("TRADING_STATE_BACKEND", "file")
    models_root = Path("models").resolve()
    manifest_dir = models_root / "tcn_h120_calmon_relaxed"
    cfg = TradingConfig(
        decision_queue_url="redis://example:6379/0",
        decision_queue_key="queue",
        state_path=tmp_path / "state.json",
        models_root=models_root,
        dry_run=True,
        state_backend="file",
        trading_models=[
            TradingModelConfig(
                model="tcn_h120_calmon_relaxed",
                exchange="binance",
                symbol="ETH/USDT",
                timeframe="1m",
                order_amount=0.01,
                max_spread_bps=15.0,
            )
        ],
    )
    service = TradingService(cfg)
    fake_executor = FakeExecutor()
    service.executor = fake_executor
    state = service.state_store.get("binance:ETH/USDT")
    assert state.ready_for_entry(datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc))

    manifest = service._resolve_manifest({"artifact_dir": str(manifest_dir)}, "tcn_h120_calmon_relaxed")
    assert manifest is not None
    assert manifest.threshold == pytest.approx(0.25, abs=1e-6)
    assert manifest.min_hold_bars == 10
    assert manifest.long_only is True

    payload = json.dumps(
        {
            "model": "tcn_h120_calmon_relaxed",
            "artifact_dir": str(manifest_dir),
            "items": [
                {"timestamp": "2025-10-01T00:00:00+00:00", "probability": 0.60, "gate_pass": True},
                {"timestamp": "2025-10-01T00:01:00+00:00", "probability": 0.40, "gate_pass": False},
                {"timestamp": "2025-10-01T00:09:00+00:00", "probability": 0.40, "gate_pass": False},
                {"timestamp": "2025-10-01T00:11:00+00:00", "probability": 0.40, "gate_pass": False},
            ],
        }
    )

    await service._handle_payload(payload)

    assert [call["side"] for call in fake_executor.calls] == ["buy", "sell"]

    state = service.state_store.get("binance:ETH/USDT")
    assert not state.in_position
    assert state.last_timestamp.isoformat() == "2025-10-01T00:11:00+00:00"
    assert cfg.state_path.exists()


@pytest.mark.asyncio
async def test_trading_service_handles_duplicate_model_labels(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TRADING_MODELS", "[]")
    monkeypatch.setenv("TRADING_STATE_BACKEND", "file")
    models_root = Path("models").resolve()
    manifest_dir = models_root / "tcn_h120_calmon_relaxed"
    cfg = TradingConfig(
        decision_queue_url="redis://example:6379/0",
        decision_queue_key="queue",
        state_path=tmp_path / "state.json",
        models_root=models_root,
        dry_run=True,
        state_backend="file",
        trading_models=[
            TradingModelConfig(
                model="tcn_h120_calmon_relaxed",
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                order_amount=0.01,
                max_spread_bps=15.0,
            ),
            TradingModelConfig(
                model="tcn_h120_calmon_relaxed",
                exchange="binance",
                symbol="ETH/USDT",
                timeframe="1m",
                order_amount=0.01,
                max_spread_bps=15.0,
            ),
        ],
    )
    service = TradingService(cfg)
    fake_executor = FakeExecutor()
    service.executor = fake_executor

    payload = json.dumps(
        {
            "model": "tcn_h120_calmon_relaxed",
            "symbol": "ETH/USDT",
            "artifact_dir": str(manifest_dir),
            "items": [
                {"timestamp": "2025-10-01T00:00:00+00:00", "probability": 0.90, "gate_pass": True},
            ],
        }
    )

    await service._handle_payload(payload)

    assert fake_executor.calls
    assert fake_executor.calls[0]["symbol"] == "ETH/USDT"
