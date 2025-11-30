import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
import pytest

from app.trading.config import TradingConfig
from app.trading.executor import IntentLedger, OrderDecision, OrderExecutor
from app.trading.service import ManifestSnapshot, TradingService, _extract_price_from_item
from app.trading.state import PositionState

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_RISK_LIMITS = REPO_ROOT / "configs" / "portfolio_risk_limits.yaml"


def _run(coro):
    return asyncio.run(coro)


def _set_file_backends(monkeypatch):
    monkeypatch.setenv("TRADING_STATE_BACKEND", "file")
    monkeypatch.setenv("TRADING_AUDIT_BACKEND", "file")
    monkeypatch.setenv("TRADING_AUDIT_BACKEND", "file")
    monkeypatch.setenv("TRADING_RISK_LIMITS_PATH", str(TEST_RISK_LIMITS))


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
        shadow_mode=False,
        order_intent_id=None,
    ):
        record = {
            "exchange": exchange,
            "symbol": symbol,
            "side": side,
            "order_amount": order_amount,
            "order_notional": order_notional,
            "max_spread_bps": max_spread_bps,
            "shadow_mode": shadow_mode,
            "order_intent_id": order_intent_id,
        }
        self.calls.append(record)
        return OrderDecision(
            executed=True,
            price_used=1.0,
            amount=1.0,
            spread_bps=1.0,
            order_intent_id=order_intent_id,
            shadow_mode=bool(shadow_mode),
        )

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


class ShadowBlockingExecutor:
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
        shadow_mode=False,
        order_intent_id=None,
    ):
        payload = {
            "exchange": exchange,
            "symbol": symbol,
            "side": side,
            "order_amount": order_amount,
            "order_notional": order_notional,
            "max_spread_bps": max_spread_bps,
            "shadow_mode": shadow_mode,
            "order_intent_id": order_intent_id,
        }
        self.calls.append(payload)
        return OrderDecision(
            executed=False,
            price_used=20000.0,
            amount=0.1,
            spread_bps=2.0,
            reason="shadow_mode",
            blocked_reason="shadow_mode",
            shadow_mode=True,
            order_payload={"status": "shadow_blocked"},
            order_intent_id=order_intent_id,
            notional=2000.0,
        )

    async def close(self):
        return None


class StubAuditLogger:
    def __init__(self):
        self.trades = []
        self.toggles = []

    async def log_gate_toggle(self, **kwargs):
        self.toggles.append(kwargs)

    async def log_trade(self, **kwargs):
        self.trades.append(kwargs)

    async def close(self):
        return None


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


def test_portfolio_state_counts_pending_intents(tmp_path: Path, monkeypatch):
    async def _test():
        monkeypatch.setenv(
            "TRADING_MODELS",
            json.dumps(
                [
                    {
                        "model": "xgb_primary",
                        "exchange": "binance",
                        "symbol": "ETH/USDT",
                        "timeframe": "1m",
                        "order_notional": 100.0,
                        "max_spread_bps": 10.0,
                    },
                    {
                        "model": "xgb_primary",
                        "exchange": "binance",
                        "symbol": "BTC/USDT",
                        "timeframe": "1m",
                        "order_notional": 100.0,
                        "max_spread_bps": 10.0,
                    },
                ]
            ),
        )
        _set_file_backends(monkeypatch)
        cfg = TradingConfig(
            decision_queue_url="redis://example:6379/0",
            decision_queue_key="queue",
            state_path=tmp_path / "state.json",
            models_root=tmp_path,
            dry_run=True,
            state_backend="file",
        )
        service = TradingService(cfg)
        eth_key = service.config.trading_models[0].state_key
        btc_key = service.config.trading_models[1].state_key
        _reset_state(service, eth_key)
        _reset_state(service, btc_key)
        eth_state = service.state_store.get(eth_key)
        eth_state.in_position = True
        eth_state.metadata["open_price"] = "1000"
        eth_state.metadata["open_amount"] = "0.5"
        service.state_store.update(eth_key, eth_state)

        btc_state = service.state_store.get(btc_key)
        btc_state.metadata["pending_entry_intent_id"] = "intent-btc"
        btc_state.metadata["pending_entry_amount"] = "0.1"
        btc_state.metadata["pending_entry_price"] = "30000"
        service.state_store.update(btc_key, btc_state)

        ts = datetime.now(timezone.utc)
        portfolio = service._build_portfolio_state(
            ts=ts,
            price_hints={"ETH/USDT": 1000.0, "BTC/USDT": 30_000.0},
        )

        assert portfolio["gross_exposure"] == pytest.approx(3_500.0)
        assert portfolio["net_exposure"] == pytest.approx(3_500.0)
        assert portfolio["open_positions"] == 2
        assert "BTC/USDT" in portfolio.get("pending_symbols", [])
        assert portfolio.get("pending_notional") == pytest.approx(3_000.0)

    _run(_test())


def test_extract_price_from_nested_features():
    item = {"features": {"close": 123.45}}
    assert _extract_price_from_item(item) == pytest.approx(123.45)


def test_trading_service_respects_min_hold(tmp_path: Path, monkeypatch):
    async def _test():
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
        _set_file_backends(monkeypatch)
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
        assert manifest.entry_threshold >= 0.55
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

    _run(_test())


def test_trading_service_handles_duplicate_model_labels(tmp_path: Path, monkeypatch):
    async def _test():
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
                        "min_hold_bars_override": 1,
                    }
                ]
            ),
        )
        _set_file_backends(monkeypatch)
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

    _run(_test())


def test_manifest_resolution_supports_symbol_specific_thresholds(tmp_path: Path, monkeypatch):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
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
    _set_file_backends(monkeypatch)
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
    service._prob_gate_overrides["ETH/USDT"] = 0.5
    loop.close()

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
    assert eth_snapshot.entry_threshold == pytest.approx(0.5, abs=1e-6)


def test_trading_service_stop_loss_exit(tmp_path: Path, monkeypatch):
    async def _test():
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
                        "min_hold_bars_override": 1,
                    }
                ]
            ),
        )
        _set_file_backends(monkeypatch)
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

    _run(_test())


def test_trading_service_take_profit_exit(tmp_path: Path, monkeypatch):
    async def _test():
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
                        "min_hold_bars_override": 1,
                    }
                ]
            ),
        )
        _set_file_backends(monkeypatch)
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

    _run(_test())


def test_trading_service_drops_stale_payloads_from_redis_cutoff(tmp_path: Path, monkeypatch):
    async def _test():
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
        _set_file_backends(monkeypatch)
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

    _run(_test())


def test_trading_service_persists_last_timestamp_to_redis(tmp_path: Path, monkeypatch):
    async def _test():
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
        _set_file_backends(monkeypatch)
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

    _run(_test())


def test_order_executor_shadow_mode_blocks_submission():
    async def _test():
        executor = OrderExecutor(dry_run=False)

        class StubAdapter:
            def __init__(self):
                self.create_calls = 0
                self.ensure_calls = 0

            async def fetch_ticker(self, symbol):
                return {"bid": 100.0, "ask": 101.0}

            async def ensure_markets(self):
                self.ensure_calls += 1

            def amount_to_precision(self, symbol, amount):
                return amount

            async def create_market_order(self, symbol, side, amount):
                self.create_calls += 1
                return {"id": "abc", "amount": amount}

        adapter = StubAdapter()

        async def fake_get_adapter(exchange: str):
            return adapter

        executor._get_adapter = fake_get_adapter  # type: ignore[assignment]

        decision = await executor.submit(
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            order_amount=1.0,
            order_notional=None,
            max_spread_bps=5000.0,
            shadow_mode=True,
            order_intent_id="intent-123",
        )

        assert decision.executed is False
        assert decision.reason == "shadow_mode"
        assert decision.shadow_mode is True
        assert decision.order_intent_id == "intent-123"
        assert adapter.create_calls == 0
        assert adapter.ensure_calls == 0
        assert decision.notional == pytest.approx(101.0)

    _run(_test())


def test_shadow_mode_blocks_position_and_logs(tmp_path: Path, monkeypatch):
    async def _test():
        monkeypatch.setenv(
            "TRADING_MODELS",
            json.dumps(
                [
                    {
                        "model": "xgb_primary",
                        "exchange": "binance",
                        "symbol": "BTC/USDT",
                        "timeframe": "1m",
                        "order_notional": 100.0,
                        "max_spread_bps": 10.0,
                    }
                ]
            ),
        )
        _set_file_backends(monkeypatch)
        cfg = TradingConfig(
            decision_queue_url="redis://example:6379/0",
            decision_queue_key="queue",
            state_path=tmp_path / "state.json",
            models_root=tmp_path,
            dry_run=False,
            state_backend="file",
            shadow_symbols=["BTC/USDT"],
        )
        service = TradingService(cfg)
        shadow_executor = ShadowBlockingExecutor()
        audit_logger = StubAuditLogger()
        service.executor = shadow_executor
        service.audit_logger = audit_logger
        state_key = service.config.trading_models[0].state_key
        _reset_state(service, state_key)
        service._resolve_manifest = lambda payload, model_label, symbol_hint=None: ManifestSnapshot(
            entry_threshold=0.5,
            exit_threshold=0.5,
            exit_prob_drop=0.1,
            min_hold_bars=1,
            long_only=True,
        )

        payload = json.dumps(
            {
                "model": "xgb_primary",
                "symbol": "BTC/USDT",
                "items": [
                    {"timestamp": "2025-10-01T00:00:00+00:00", "probability": 0.9, "gate_pass": True, "close": 20000.0}
                ],
            }
        )

        await service._handle_payload(payload)

        assert shadow_executor.calls, "Expected executor to receive shadow-routed order"
        assert shadow_executor.calls[0]["shadow_mode"] is True

        state = service.state_store.get(state_key)
        assert state.in_position is False
        assert state.metadata.get("open_price") is None
        assert state.metadata.get("open_amount") is None
        assert state.metadata.get("last_entry_reason") == "shadow_mode"

        assert audit_logger.trades, "Audit logger should record shadow-blocked attempt"
        logged_trade = audit_logger.trades[0]
        decision_payload = audit_logger.trades[0]["decision"]
        assert isinstance(decision_payload, OrderDecision)
        assert decision_payload.shadow_mode is True
        assert decision_payload.reason == "shadow_mode"
        assert decision_payload.order_intent_id
        assert logged_trade["timeframe"] == "1m"
        assert logged_trade["policy_id"] == "xgb_primary"
        assert "decision_namespace" in logged_trade

    _run(_test())


def test_executor_dedup_blocks_duplicate_intents(monkeypatch):
    async def _test():
        ledger = IntentLedger(backend="memory", lock_ttl_seconds=1000)
        executor = OrderExecutor(dry_run=False, intent_ledger=ledger)

        class StubAdapter:
            def __init__(self):
                self.create_calls = 0
                self.ensure_calls = 0

            async def fetch_ticker(self, symbol):
                return {"bid": 100.0, "ask": 101.0}

            async def ensure_markets(self):
                self.ensure_calls += 1

            def amount_to_precision(self, symbol, amount):
                return amount

            async def create_market_order(self, symbol, side, amount):
                self.create_calls += 1
                return {"id": f"order-{self.create_calls}", "status": "closed", "amount": amount}

        adapter = StubAdapter()

        async def fake_get_adapter(exchange: str):
            return adapter

        executor._get_adapter = fake_get_adapter  # type: ignore[assignment]

        decision1 = await executor.submit(
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            order_amount=1.0,
            order_notional=None,
            max_spread_bps=5000.0,
            shadow_mode=False,
            order_intent_id="dup-intent",
        )
        decision2 = await executor.submit(
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            order_amount=1.0,
            order_notional=None,
            max_spread_bps=5000.0,
            shadow_mode=False,
            order_intent_id="dup-intent",
        )

        assert decision1.executed is True
        assert decision2.dedup_blocked is True
        assert decision2.reason == "duplicate_intent"
        assert adapter.create_calls == 1

    _run(_test())


def test_executor_respects_existing_intent_lock(monkeypatch):
    async def _test():
        ledger = IntentLedger(backend="memory", lock_ttl_seconds=1000)
        await ledger.acquire("locked-intent")
        executor = OrderExecutor(dry_run=False, intent_ledger=ledger)

        class StubAdapter:
            async def fetch_ticker(self, symbol):
                return {"bid": 100.0, "ask": 101.0}

            async def ensure_markets(self):
                return None

            def amount_to_precision(self, symbol, amount):
                return amount

            async def create_market_order(self, symbol, side, amount):
                raise AssertionError("create_market_order should not be called when intent is locked")

        async def fake_get_adapter(exchange: str):
            return StubAdapter()

        executor._get_adapter = fake_get_adapter  # type: ignore[assignment]

        decision = await executor.submit(
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            order_amount=1.0,
            order_notional=None,
            max_spread_bps=5000.0,
            shadow_mode=False,
            order_intent_id="locked-intent",
        )

        assert decision.dedup_blocked is True
        assert decision.executed is False
        assert decision.reason == "duplicate_intent"

    _run(_test())


def test_reconciliation_mismatch_latches_safe_mode(tmp_path: Path, monkeypatch):
    async def _test():
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
        _set_file_backends(monkeypatch)
        cfg = TradingConfig(
            decision_queue_url="redis://example:6379/0",
            decision_queue_key="queue",
            state_path=tmp_path / "state.json",
            models_root=tmp_path,
            dry_run=False,
            state_backend="file",
            reconcile_healthy_streak=2,
        )
        service = TradingService(cfg)

        class StubAdapter:
            def __init__(self):
                self.balance_payload = {"free": {"BTC": 1.0}, "used": {"BTC": 0.0}}
                self.ticker_payload = {"bid": 10000.0, "ask": 10010.0}
                self.open_orders_payload = []

            async def fetch_balance(self):
                return self.balance_payload

            async def fetch_ticker(self, symbol):
                return self.ticker_payload

            async def fetch_open_orders(self, symbol):
                return list(self.open_orders_payload)

        stub = StubAdapter()

        async def fake_get_adapter(exchange: str):
            return stub

        service.executor.get_adapter = fake_get_adapter  # type: ignore[assignment]

        healthy = await service._reconcile_once()
        assert healthy is False
        assert service._safe_mode_reason == "reconciliation_failed"

    _run(_test())


def test_reconciliation_clears_safe_mode_after_healthy_streak(tmp_path: Path, monkeypatch):
    async def _test():
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
        _set_file_backends(monkeypatch)
        cfg = TradingConfig(
            decision_queue_url="redis://example:6379/0",
            decision_queue_key="queue",
            state_path=tmp_path / "state.json",
            models_root=tmp_path,
            dry_run=False,
            state_backend="file",
            reconcile_healthy_streak=2,
        )
        service = TradingService(cfg)

        class StubAdapter:
            def __init__(self):
                self.balance_payload = {"free": {"BTC": 1.0}, "used": {"BTC": 0.0}}
                self.ticker_payload = {"bid": 10000.0, "ask": 10010.0}
                self.open_orders_payload = []

            async def fetch_balance(self):
                return self.balance_payload

            async def fetch_ticker(self, symbol):
                return self.ticker_payload

            async def fetch_open_orders(self, symbol):
                return list(self.open_orders_payload)

        stub = StubAdapter()

        async def fake_get_adapter(exchange: str):
            return stub

        service.executor.get_adapter = fake_get_adapter  # type: ignore[assignment]

        await service._reconcile_once()
        assert service._safe_mode_reason == "reconciliation_failed"

        stub.balance_payload = {"free": {"BTC": 0.0}, "used": {"BTC": 0.0}}
        stub.open_orders_payload = []

        await service._reconcile_once()
        await service._reconcile_once()

        assert service._safe_mode_reason is None

    _run(_test())
