from datetime import datetime, timezone

import pandas as pd
import pytest

fakeredis = pytest.importorskip("fakeredis.aioredis")
from fakeredis.aioredis import FakeRedis  # type: ignore

from app.features.store import redis_store as rs
from app.features.store.redis_store import RedisFeatureStore, get_store
from app.ingestion_service.main import app


pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
async def fake_store(monkeypatch):
    """
    Replace DI singleton with a RedisFeatureStore that uses fakeredis.
    """
    r = FakeRedis(decode_responses=True)
    store = RedisFeatureStore(url="redis://fake", namespace="features", default_ttl=None, redis_client=r)

    original_singleton = getattr(rs, "_store_singleton", None)
    rs._store_singleton = store

    monkeypatch.setattr("app.ingestion_service.routes.get_store", lambda: store, raising=True)
    app.dependency_overrides[get_store] = lambda: store

    try:
        yield store
    finally:
        app.dependency_overrides.clear()
        rs._store_singleton = original_singleton
        await store.aclose()


def _mk_df(n: int = 30):
    # Generate n 1-min bars, tz-naive (route will tz_localize to UTC)
    base = pd.Timestamp("2025-08-01 00:00:00")
    ts = pd.date_range(base, periods=n, freq="1min", tz=None)
    opens = pd.Series(range(100, 100 + n), dtype="float")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": opens + 0.5,
            "low": opens - 0.5,
            "close": opens + 0.01,  # tiny drift so ret1 != 0
            "volume": pd.Series([10.0 + i for i in range(n)], dtype="float"),
            "symbol": pd.Series(["BTCUSDT"] * n, dtype="string"),
            "exchange": pd.Series(["binance"] * n, dtype="string"),
            "timeframe": pd.Series(["1m"] * n, dtype="string"),
        }
    )
    return df


async def test_ingest_then_retrieve_features(async_client, fake_store, monkeypatch):
    """
    /ingest/market -> features written to Redis by route -> /ingest/features/market returns them.
    """

    async def fake_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
        return _mk_df(30)

    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv),
    )

    async with async_client() as client:
        response = await client.post(
            "/ingest/market/binance",
            json={"symbol": "BTC/USDT", "granularity": "1m", "limit": 2},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "ok"
        assert body["features_written"] >= 1

        t0 = int(datetime(2025, 8, 1, 0, 14, tzinfo=timezone.utc).timestamp())
        t1 = t0 + 60

        response2 = await client.get(
            "/ingest/features/market",
            params={"symbol": "BTC/USDT", "timeframe": "1m", "ts": [t0, t1]},
        )

    assert response2.status_code == 200, response2.text
    out = response2.json()
    assert out["rows"] in (1, 2)
    assert isinstance(out["data"], list)
    expected_keys = {"ret_1", "hl_spread", "hl_spread_z", "rvol_20"}
    for row in out["data"]:
        assert "timestamp" in row
        assert expected_keys.issubset(row.keys())


async def test_cache_miss_returns_empty(async_client, fake_store):
    """
    If keys are not present in Redis, the GET should return rows=0 data=[].
    """
    now = int(datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp())

    async with async_client() as client:
        response = await client.get(
            "/ingest/features/market",
            params={"symbol": "ETH/USDT", "timeframe": "1m", "ts": [now, now + 60]},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"] == 0
    assert payload["data"] == []


async def test_metrics_increment_after_ops(async_client, fake_store, monkeypatch):
    """
    After a write + read flow, /metrics should include our custom counters/histogram buckets.
    """

    async def fake_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
        return _mk_df(30)

    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv),
    )

    async with async_client() as client:
        response = await client.post(
            "/ingest/market/binance",
            json={"symbol": "BTC/USDT", "granularity": "1m", "limit": 2},
        )
        assert response.status_code == 200

        t0 = int(datetime(2025, 8, 1, 0, 14, tzinfo=timezone.utc).timestamp())
        await client.get(
            "/ingest/features/market",
            params={"symbol": "BTC/USDT", "timeframe": "1m", "ts": [t0]},
        )

        metrics_response = await client.get("/metrics/")

    assert metrics_response.status_code == 200
    text = metrics_response.text
    assert "feature_writes_total" in text
    assert "feature_reads_total" in text
    assert ("feature_hits_total" in text) or ("feature_misses_total" in text)
    assert "feature_op_latency_seconds_bucket" in text
