import pytest
import pandas as pd

from app.ingestion_service import routes as ingestion_routes
from app.ingestion_service.main import app, get_ccxt, get_onchain, get_social


pytestmark = pytest.mark.anyio("asyncio")


async def test_get_news_search_envelope(async_client):
    class FakeNews:
        async def get_crypto_news(self, since=None, until=None, source="api", limit: int = 2):
            ts = pd.to_datetime(["2025-08-01T00:00:00Z"], utc=True)
            return pd.DataFrame(
                {
                    "published_at": ts,
                    "id": pd.Series(["n1"], dtype="string"),
                    "title": pd.Series(["Hello"], dtype="string"),
                    "url": pd.Series(["http://x"], dtype="string"),
                    "source": pd.Series([source], dtype="string"),
                    "author": pd.Series(["me"], dtype="string"),
                    "description": pd.Series(["desc"], dtype="string"),
                }
            )

        async def aclose(self):  # pragma: no cover - just satisfies interface
            pass

    app.dependency_overrides[ingestion_routes.provide_news] = lambda: FakeNews()

    async with async_client() as client:
        response = await client.get(
            "/ingest/news/search",
            params={"source": "api", "limit": 1, "since": 0, "until": 1},
        )

    assert response.status_code == 200


async def test_get_onchain_glassnode_envelope(async_client):
    class FakeOnchain:
        async def get_glassnode_metric(self, symbol: str, metric: str, days: int = 1):
            ts = pd.to_datetime(["2025-08-01T00:00:00Z"], utc=True)
            return pd.DataFrame(
                {
                    "timestamp": ts,
                    "source": ["glassnode"],
                    "symbol": [symbol],
                    "metric": [metric],
                    "value": [1.23],
                }
            )

        async def aclose(self):  # pragma: no cover - interface stub
            pass

    app.dependency_overrides[get_onchain] = lambda *_: FakeOnchain()

    async with async_client() as client:
        response = await client.get(
            "/ingest/onchain/glassnode",
            params={"symbol": "BTC", "metric": "active_addresses", "days": 1},
        )

    assert response.status_code == 200
    payload = response.json()
    assert "rows" in payload
    assert isinstance(payload["data"], list)


async def test_get_social_reddit_envelope(async_client):
    class FakeSocial:
        async def fetch_reddit_api(self, subreddit: str, since=None, until=None, limit: int = 2):
            ts = pd.to_datetime(["2025-08-01T00:00:00Z"], utc=True)
            return pd.DataFrame(
                {
                    "ts": ts,
                    "author": pd.Series(["r1"], dtype="string"),
                    "title": pd.Series(["t"], dtype="string"),
                    "selftext": pd.Series(["s"], dtype="string"),
                    "score": pd.Series([10], dtype="Int64"),
                    "num_comments": pd.Series([2], dtype="Int64"),
                    "id": pd.Series(["abc"], dtype="string"),
                    "subreddit": pd.Series([subreddit], dtype="string"),
                    "source": pd.Series(["reddit"], dtype="string"),
                }
            )

        async def aclose(self):  # pragma: no cover
            pass

    app.dependency_overrides[get_social] = lambda *_: FakeSocial()

    async with async_client() as client:
        response = await client.get(
            "/ingest/social/reddit",
            params={"subreddit": "CryptoCurrency", "limit": 1},
        )

    assert response.status_code == 200
    payload = response.json()
    assert "rows" in payload
    assert isinstance(payload["data"], list)


async def test_get_ccxt_historical_envelope(async_client):
    class FakeCCXT:
        async def fetch_historical(self, symbol: str, timeframe: str = "1m", since=None, limit: int = 2):
            ts = pd.to_datetime(["2025-08-01T00:00:00Z", "2025-08-01T00:01:00Z"], utc=True)
            return pd.DataFrame(
                {
                    "timestamp": ts,
                    "open": [1.0, 2.0],
                    "high": [1.1, 2.1],
                    "low": [0.9, 1.9],
                    "close": [1.05, 2.05],
                    "volume": [10.0, 20.0],
                    "symbol": pd.Series([symbol, symbol], dtype="string"),
                    "exchange": pd.Series(["binance", "binance"], dtype="string"),
                    "timeframe": pd.Series([timeframe, timeframe], dtype="string"),
                }
            )

        async def aclose(self):  # pragma: no cover
            pass

    app.dependency_overrides[get_ccxt] = lambda *_: FakeCCXT()

    async with async_client() as client:
        response = await client.get(
            "/ingest/ccxt/binance/historical",
            params={"symbol": "BTC/USDT", "timeframe": "1m", "limit": 2},
        )

    assert response.status_code == 200
    payload = response.json()
    if isinstance(payload, dict) and "rows" in payload and "data" in payload:
        assert isinstance(payload["data"], list)
    else:
        for key in ("timestamp", "open", "high", "low", "close", "volume", "symbol", "exchange", "timeframe", "dt"):
            assert key in payload


async def test_post_market_success(async_client, monkeypatch):
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-08-01 00:00:00"]),
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [100.0],
            "symbol": pd.Series(["BTCUSDT"], dtype="string"),
            "exchange": pd.Series(["binance"], dtype="string"),
            "timeframe": pd.Series(["1m"], dtype="string"),
        }
    )

    async def fake_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
        return df

    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv),
    )

    def capture_write(df_in, base, partitions, filename=None):
        assert "UTC" in str(df_in["timestamp"].dtype)
        return "/fake/path.parquet"

    monkeypatch.setattr("app.ingestion_service.routes.write_to_parquet", capture_write)

    async with async_client() as client:
        response = await client.post(
            "/ingest/market/binance",
            json={"symbol": "BTC-USDT", "granularity": "1m"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["path"] == "/fake/path.parquet"
    if "features_written" in body:
        assert body["features_written"] >= 0


async def test_post_market_no_data(async_client, monkeypatch):
    async def fake_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
        return pd.DataFrame()

    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv),
    )
    monkeypatch.setattr("app.ingestion_service.routes.write_to_parquet", lambda *a, **k: None)

    async with async_client() as client:
        response = await client.post(
            "/ingest/market/binance",
            json={"symbol": "BTC-USDT", "granularity": "1m"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_data"
    assert body["path"] is None


async def test_post_news_rss_success(async_client, monkeypatch, tmp_path):
    ts = pd.to_datetime(["2025-10-01T00:00:00Z", "2025-10-01T00:05:00Z"], utc=True)
    df = pd.DataFrame(
        {
            "published_at": ts,
            "id": pd.Series(["n1", "n2"], dtype="string"),
            "title": pd.Series(["t1", "t2"], dtype="string"),
            "url": pd.Series(["https://example.com/a", "https://example.com/b"], dtype="string"),
            "source": pd.Series(["example.com", "example.com"], dtype="string"),
            "author": pd.Series(["alice", "bob"], dtype="string"),
            "description": pd.Series(["d1", "d2"], dtype="string"),
            "dt": pd.Series(["2025-10-01", "2025-10-01"], dtype="string"),
        }
    )

    async def fake_fetch(feed_url: str, limit: int = 500):
        assert feed_url == "https://example.com/rss"
        return df

    writes = []

    def fake_write(df_in: pd.DataFrame, base: str, partitions: dict, filename: str | None = None):
        writes.append((df_in.copy(), base, partitions))
        return f"{base}/dt={df_in['dt'].iloc[0]}/part.parquet"

    async def fake_store(df_in: pd.DataFrame) -> int:
        return len(df_in)

    monkeypatch.setattr(ingestion_routes, "fetch_news_rss_once", fake_fetch)
    monkeypatch.setattr(ingestion_routes, "write_to_parquet", fake_write)
    monkeypatch.setattr(ingestion_routes, "_write_news_features_to_store", fake_store)
    monkeypatch.setattr(ingestion_routes.settings, "NEWS_PATH", str(tmp_path / "news"), raising=False)

    async with async_client() as client:
        response = await client.post(
            "/ingest/news",
            json={"source_type": "rss", "feed_url": "https://example.com/rss"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["features_written"] == len(df)
    assert payload["path"].endswith("part.parquet")
    assert writes, "expected write_to_parquet to be called"
    _, base, parts = writes[0]
    assert base.endswith("/rss")
    assert parts.get("source") == "example.com"

async def test_post_market_write_error(async_client, monkeypatch):
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-08-01 00:00:00"]),
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [100.0],
            "symbol": pd.Series(["BTCUSDT"], dtype="string"),
            "exchange": pd.Series(["binance"], dtype="string"),
            "timeframe": pd.Series(["1m"], dtype="string"),
        }
    )

    async def fake_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
        return df

    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv),
    )

    def bad_write(*_a, **_k):
        raise IOError("disk full")

    monkeypatch.setattr("app.ingestion_service.routes.write_to_parquet", bad_write)

    async with async_client() as client:
        response = await client.post(
            "/ingest/market/binance",
            json={"symbol": "BTC-USDT", "granularity": "1m"},
        )

    assert response.status_code == 500
    assert "Write failed" in response.json()["detail"]


async def test_post_market_schema_error(async_client, monkeypatch):
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-08-01 00:00:00"]),
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [100.0],
            "symbol": pd.Series(["BTCUSDT"], dtype="string"),
            "exchange": pd.Series(["binance"], dtype="string"),
            "timeframe": pd.Series(["1m"], dtype="string"),
        }
    )

    async def fake_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
        return df

    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv),
    )
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("Missing columns: ['x']")),
    )

    async with async_client() as client:
        response = await client.post(
            "/ingest/market/binance",
            json={"symbol": "BTC-USDT", "granularity": "1m"},
        )

    assert response.status_code == 422
    assert "Missing columns" in response.json()["detail"]
