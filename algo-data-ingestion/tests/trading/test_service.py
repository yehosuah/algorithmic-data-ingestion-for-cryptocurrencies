import json
from datetime import datetime, timezone
from pathlib import Path
import pytest

from app.trading.config import TradingConfig
from app.trading.executor import OrderDecision
from app.trading.service import TradingService
from app.trading.state import PositionState


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


class DummyRedis:
    def __init__(self):
        self._hashes = {}

    async def hget(self, name, field):
        return self._hashes.get(name, {}).get(field)

    async def hset(self, name, field, value):
        bucket = self._hashes.setdefault(name, {})
        bucket[field] = value
        return 1


def _reset_state(service: TradingService, state_key: str) -> PositionState:
    state = service.state_store.get(state_key)
    state.metadata.clear()
    state.in_position = False
    state.entry_ts = None
    state.hold_until = None
    state.last_exit_ts = None
    state.last_gate = None
    state.last_probability = None
    state.last_timestamp = None
    state.bars_in_position = 0
    service.state_store.update(state_key, state)
    return state


@pytest.mark.asyncio
async def test_trading_service_respects_min_hold(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "TRADING_MODELS",
        json.dumps(
            [
                {
                    "model": "tcn_h120_calmon_relaxed",
                    "exchange": "binance",
                    "symbol": "BTC/USDT",
                    "timeframe": "1m",
                    "order_amount": 0.01,
                    "max_spread_bps": 15.0,
                    "stop_loss_pct": 0.005,
                    "take_profit_pct": 0.007,
                },
                {
                    "model": "tcn_h120_calmon_relaxed",
                    "exchange": "binance",
                    "symbol": "ETH/USDT",
                    "timeframe": "1m",
                    "order_amount": 0.01,
                    "max_spread_bps": 15.0,
                    "stop_loss_pct": 0.005,
                    "take_profit_pct": 0.007,
                },
            ]
        ),
    )
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
    )
    service = TradingService(cfg)
    fake_executor = FakeExecutor()
    service.executor = fake_executor
    state_key = service.config.trading_models[0].state_key
    state = _reset_state(service, state_key)
    assert state.ready_for_entry(datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc))

    manifest = service._resolve_manifest(
        {"artifact_dir": str(manifest_dir), "symbol": "BTC/USDT"},
        "tcn_h120_calmon_relaxed",
    )
    assert manifest is not None
    assert manifest.entry_threshold == pytest.approx(0.55, abs=1e-6)
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

    state = service.state_store.get(state_key)
    assert not state.in_position
    assert state.last_timestamp.isoformat() == "2025-10-01T00:11:00+00:00"
    assert cfg.state_path.exists()


@pytest.mark.asyncio
async def test_trading_service_handles_duplicate_model_labels(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "TRADING_MODELS",
        json.dumps(
            [
                {
                    "model": "tcn_h120_calmon_relaxed",
                    "exchange": "binance",
                    "symbol": "BTC/USDT",
                    "timeframe": "1m",
                    "order_amount": 0.01,
                    "max_spread_bps": 15.0,
                    "stop_loss_pct": 0.005,
                    "take_profit_pct": 0.007,
                },
                {
                    "model": "tcn_h120_calmon_relaxed",
                    "exchange": "binance",
                    "symbol": "ETH/USDT",
                    "timeframe": "1m",
                    "order_amount": 0.01,
                    "max_spread_bps": 15.0,
                    "stop_loss_pct": 0.005,
                    "take_profit_pct": 0.007,
                }
            ]
        ),
    )
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
    )
    service = TradingService(cfg)
    fake_executor = FakeExecutor()
    service.executor = fake_executor
    _reset_state(service, service.config.trading_models[0].state_key)
    _reset_state(service, service.config.trading_models[1].state_key)

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


def test_manifest_resolution_supports_symbol_specific_thresholds(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "TRADING_MODELS",
        json.dumps(
            [
                {
                    "model": "base_xgb_cost_spread",
                    "exchange": "binance",
                    "symbol": "BTC/USDT",
                    "timeframe": "1m",
                    "order_amount": 0.01,
                    "max_spread_bps": 15.0,
                },
                {
                    "model": "base_xgb_cost_spread",
                    "exchange": "binance",
                    "symbol": "ETH/USDT",
                    "timeframe": "1m",
                    "order_amount": 0.01,
                    "max_spread_bps": 15.0,
                },
            ]
        ),
    )
    monkeypatch.setenv("TRADING_STATE_BACKEND", "file")
    models_root = Path("models").resolve()
    manifest_dir = models_root / "base_xgb_cost_spread"
    cfg = TradingConfig(
        decision_queue_url="redis://example:6379/0",
        decision_queue_key="queue",
        state_path=tmp_path / "state.json",
        models_root=models_root,
        dry_run=True,
        state_backend="file",
    )
    service = TradingService(cfg)

    btc_snapshot = service._resolve_manifest(
        {"artifact_dir": str(manifest_dir), "symbol": "BTC/USDT"},
        "base_xgb_cost_spread",
    )
    eth_snapshot = service._resolve_manifest(
        {"artifact_dir": str(manifest_dir), "symbol": "ETH/USDT"},
        "base_xgb_cost_spread",
    )

    assert btc_snapshot is not None
    assert eth_snapshot is not None
    assert btc_snapshot.entry_threshold == pytest.approx(0.6, abs=1e-6)
    assert eth_snapshot.entry_threshold == pytest.approx(0.55, abs=1e-6)


@pytest.mark.asyncio
async def test_trading_service_stop_loss_exit(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "TRADING_MODELS",
        json.dumps(
            [
                {
                    "model": "tcn_h120_calmon_relaxed",
                    "exchange": "binance",
                    "symbol": "ETH/USDT",
                    "timeframe": "1m",
                    "order_amount": 0.01,
                    "max_spread_bps": 15.0,
                    "stop_loss_pct": 0.005,
                    "take_profit_pct": None,
                }
            ]
        ),
    )
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
    )
    service = TradingService(cfg)
    fake_executor = FakeExecutor()
    service.executor = fake_executor
    state_key = service.config.trading_models[0].state_key
    assert service.config.trading_models[0].take_profit_pct is None
    _reset_state(service, state_key)

    payload = json.dumps(
        {
            "model": "tcn_h120_calmon_relaxed",
            "artifact_dir": str(manifest_dir),
            "items": [
                {"timestamp": "2025-10-01T00:00:00+00:00", "probability": 0.90, "gate_pass": True, "close": 1.0},
                {"timestamp": "2025-10-01T00:01:00+00:00", "probability": 0.92, "gate_pass": True, "close": 0.994},
            ],
        }
    )

    await service._handle_payload(payload)

    assert [call["side"] for call in fake_executor.calls] == ["buy", "sell"]
    state_key = service.config.trading_models[0].state_key
    state = service.state_store.get(state_key)
    assert state.metadata.get("last_exit_trigger") == "stop_loss"


@pytest.mark.asyncio
async def test_trading_service_take_profit_exit(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "TRADING_MODELS",
        json.dumps(
            [
                {
                    "model": "tcn_h120_calmon_relaxed",
                    "exchange": "binance",
                    "symbol": "ETH/USDT",
                    "timeframe": "1m",
                    "order_amount": 0.01,
                    "max_spread_bps": 15.0,
                    "stop_loss_pct": 0.005,
                    "take_profit_pct": 0.007,
                }
            ]
        ),
    )
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
    )
    service = TradingService(cfg)
    fake_executor = FakeExecutor()
    service.executor = fake_executor
    state_key = service.config.trading_models[0].state_key
    take_profit_value = service.config.trading_models[0].take_profit_pct
    assert take_profit_value == pytest.approx(0.007)
    _reset_state(service, state_key)

    payload = json.dumps(
        {
            "model": "tcn_h120_calmon_relaxed",
            "artifact_dir": str(manifest_dir),
            "items": [
                {"timestamp": "2025-10-01T00:00:00+00:00", "probability": 0.90, "gate_pass": True, "close": 1.0},
                {"timestamp": "2025-10-01T00:01:00+00:00", "probability": 0.92, "gate_pass": True, "close": 1.010},
            ],
        }
    )

    await service._handle_payload(payload)

    state = service.state_store.get(state_key)
    assert state.metadata.get("last_exit_trigger") == "take_profit"
    assert [call["side"] for call in fake_executor.calls] == ["buy", "sell"], fake_executor.calls


@pytest.mark.asyncio
async def test_trading_service_drops_stale_payloads_from_redis_cutoff(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "TRADING_MODELS",
        json.dumps(
            [
                {
                    "model": "tcn_h120_calmon_relaxed",
                    "exchange": "binance",
                    "symbol": "BTC/USDT",
                    "timeframe": "1m",
                    "order_amount": 0.01,
                    "max_spread_bps": 15.0,
                    "min_hold_bars_override": 1,
                }
            ]
        ),
    )
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
    )
    service = TradingService(cfg)
    fake_executor = FakeExecutor()
    service.executor = fake_executor
    redis_stub = DummyRedis()
    service._redis = redis_stub
    state_key = service.config.trading_models[0].state_key
    await redis_stub.hset(cfg.last_timestamp_hash, state_key, "2025-10-01T00:10:00+00:00")

    payload = json.dumps(
        {
            "model": "tcn_h120_calmon_relaxed",
            "artifact_dir": str(manifest_dir),
            "items": [
                {"timestamp": "2025-10-01T00:09:00+00:00", "probability": 0.90, "gate_pass": True},
            ],
        }
    )

    await service._handle_payload(payload)

    assert fake_executor.calls == []
    state = service.state_store.get(state_key)
    assert state.last_timestamp is None


@pytest.mark.asyncio
async def test_trading_service_persists_last_timestamp_to_redis(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "TRADING_MODELS",
        json.dumps(
            [
                {
                    "model": "tcn_h120_calmon_relaxed",
                    "exchange": "binance",
                    "symbol": "BTC/USDT",
                    "timeframe": "1m",
                    "order_amount": 0.01,
                    "max_spread_bps": 15.0,
                    "min_hold_bars_override": 1,
                }
            ]
        ),
    )
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
    )
    service = TradingService(cfg)
    fake_executor = FakeExecutor()
    service.executor = fake_executor
    redis_stub = DummyRedis()
    service._redis = redis_stub
    state_key = service.config.trading_models[0].state_key

    payload = json.dumps(
        {
            "model": "tcn_h120_calmon_relaxed",
            "artifact_dir": str(manifest_dir),
            "items": [
                {"timestamp": "2025-10-02T00:00:00+00:00", "probability": 0.90, "gate_pass": True},
                {"timestamp": "2025-10-02T00:02:00+00:00", "probability": 0.30, "gate_pass": False},
            ],
        }
    )

    await service._handle_payload(payload)

    assert [call["side"] for call in fake_executor.calls] == ["buy", "sell"]
    state = service.state_store.get(state_key)
    assert state.last_timestamp is not None
    assert state.last_timestamp.isoformat() == "2025-10-02T00:02:00+00:00"
    stored_ts = await redis_stub.hget(cfg.last_timestamp_hash, state_key)
    assert stored_ts == "2025-10-02T00:02:00+00:00"
