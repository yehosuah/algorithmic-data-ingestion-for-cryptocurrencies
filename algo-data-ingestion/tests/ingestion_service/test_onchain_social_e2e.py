# tests/ingestion_service/test_onchain_social_e2e.py
import pandas as pd
import pytest

from app.ingestion_service import routes
from app.ingestion_service.main import app


pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture(autouse=True)
def _stub_parquet(monkeypatch, tmp_path):
    out = tmp_path / "ok.parquet"
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        lambda df, base, parts: str(out),
    )


class _MemStore:
    def __init__(self):
        self._kv = {}

    @staticmethod
    def _epoch_seconds(ts):
        if isinstance(ts, (int, float)):
            return int(ts // 1000) if ts > 10_000_000_000 else int(ts)
        s = pd.Series([ts])
        dt = pd.to_datetime(s, utc=True)
        ns = dt.astype("int64").iloc[0]
        return int(ns // 1_000_000_000)

    async def batch_write(self, items):
        for it in items:
            epoch_s = self._epoch_seconds(it["ts"])
            payload = dict(it["payload"])
            key = (
                it["domain"],
                str(it["symbol"]),
                str(it["timeframe"]),
                epoch_s,
            )
            self._kv[key] = payload

            metric = payload.get("metric")
            if str(it["domain"]).lower() == "onchain" and metric:
                alias_key = (
                    it["domain"],
                    str(it["symbol"]),
                    str(metric),
                    epoch_s,
                )
                self._kv[alias_key] = payload
        return list(self._kv.keys())

    async def batch_read(self, queries):
        out = []
        for (domain, symbol, timeframe, ts) in queries:
            k = (domain, str(symbol), str(timeframe), self._epoch_seconds(ts))
            out.append(self._kv.get(k))
        return out


@pytest.fixture
def mem_store(monkeypatch):
    store = _MemStore()
    monkeypatch.setattr("app.ingestion_service.routes.get_store", lambda: store)
    app.dependency_overrides[routes.provide_store] = lambda: store
    try:
        yield store
    finally:
        app.dependency_overrides.pop(routes.provide_store, None)


def _mk_onchain_df():
    now = pd.Timestamp.utcnow().floor("D")
    ts = pd.date_range(end=now, periods=2, freq="D")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["BTC", "BTC"],
            "metric": ["active_addresses", "active_addresses"],
            "value": [123.0, 125.5],
            "timeframe": ["1d", "1d"],
        }
    )


def _mk_social_df(query="bitcoin", n=3):
    now = pd.Timestamp.utcnow().floor("T")
    ts = pd.date_range(end=now, periods=n, freq="T")
    return pd.DataFrame(
        {
            "ts": ts,
            "user": [f"user{i}" for i in range(n)],
            "text": [f"{query} sample {i}" for i in range(n)],
            "sentiment_score": [0.1, 0.2, -0.1][:n],
            "timeframe": ["1m"] * n,
        }
    )


async def test_onchain_ingest_then_retrieve(async_client, mem_store, monkeypatch):
    """
    POST /ingest/onchain/glassnode writes rows -> GET /ingest/features/onchain returns them.
    """
    monkeypatch.setattr("app.ingestion_service.routes.fetch_glassnode", lambda *a, **k: _mk_onchain_df())

    async with async_client() as client:
        response = await client.post(
            "/ingest/onchain/glassnode",
            json={"symbol": "BTC", "metric": "active_addresses", "days": 1},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "ok"
        assert body["features_written"] >= 1

        df = _mk_onchain_df()
        ts_epoch = int(df["timestamp"].iloc[-1].value // 1_000_000_000)

        response2 = await client.get(
            "/ingest/features/onchain",
            params={"symbol": "BTC", "metric": "active_addresses", "ts": [ts_epoch]},
        )

    assert response2.status_code == 200, response2.text
    out = response2.json()
    assert out["rows"] >= 1
    assert isinstance(out["data"][0]["timestamp"], int)
    assert "value" in out["data"][0]


async def test_social_ingest_then_retrieve(async_client, mem_store, monkeypatch):
    """
    POST /ingest/social/twitter writes rows -> GET /ingest/features/social returns them.
    """
    monkeypatch.setattr(
        "app.ingestion_service.routes.fetch_twitter_sentiment",
        lambda query, since, until, max_results: _mk_social_df(query=query, n=3),
    )

    async with async_client() as client:
        response = await client.post(
            "/ingest/social/twitter",
            json={"query": "bitcoin", "max_results": 5},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "ok"
        assert body["features_written"] >= 1

        df = _mk_social_df("bitcoin", 3)
        ts_epoch = int(df["ts"].iloc[-1].value // 1_000_000_000)

        response2 = await client.get(
            "/ingest/features/social",
            params={"topic": "twitter", "timeframe": "1m", "ts": [ts_epoch]},
        )

    assert response2.status_code == 200, response2.text
    out = response2.json()
    assert out["rows"] >= 1
    row = out["data"][0]
    assert isinstance(row["timestamp"], int)
    assert "user" in row and "text" in row and "sentiment" in row
