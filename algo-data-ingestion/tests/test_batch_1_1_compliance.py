# tests/test_batch_1_1_compliance.py
import asyncio
import types
import time

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from app.ingestion_service.main import app as fastapi_app, get_http_dep
from app.features.ingestion.news_client import NewsClient
from app.features.ingestion.social_client import SocialClient
from app.features.ingestion.onchain_client import OnchainClient
from app.common.async_infra import get_http, close_http

# ---------- Lifespan & singleton ----------
def test_lifespan_singleton_http():
    with TestClient(fastapi_app) as client:
        # Two requests should see the same shared instance
        http1 = fastapi_app.state.http
        assert http1 is get_http()
        client.get("/docs")
        http2 = fastapi_app.state.http
        assert http1 is http2

@pytest.mark.asyncio
async def test_close_and_recreate_http():
    http1 = get_http()
    await close_http()
    http2 = get_http()
    assert http1 is not http2  # new instance created

# ---------- Retry policy (simulated transient) ----------
@pytest.mark.asyncio
async def test_news_retry(monkeypatch):
    http = get_http()
    client = NewsClient(http=http)

    calls = {"n": 0}

    async def fake_fetch_news_api(since, until, source, limit, http=None):
        calls["n"] += 1
        # Fail first two times, succeed on third
        if calls["n"] < 3:
            raise RuntimeError("simulated transient")
        import pandas as pd
        return pd.DataFrame([{"ts": 0, "title": "ok", "url": "x", "source": source}])

    from app.adapters import news_adapter
    monkeypatch.setattr(news_adapter, "fetch_news_api", fake_fetch_news_api)

    df = await client.get_crypto_news(since=None, until=None, source="test", limit=1)
    assert not df.empty
    assert calls["n"] == 3  # retried

@pytest.mark.asyncio
async def test_social_retry_and_di(monkeypatch):
    http = get_http()
    client = SocialClient(http=http)

    calls = {"n": 0}
    async def fake_fetch_twitter_sentiment(query, since, until, limit, http=None):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        import pandas as pd
        return pd.DataFrame([{"ts": 0, "author": "a", "text": "t", "likes": 0, "retweets": 0, "sentiment_score": 0.1}])

    from app.adapters import sentiment_adapter
    monkeypatch.setattr(sentiment_adapter, "fetch_twitter_sentiment", fake_fetch_twitter_sentiment)

    df = await client.fetch_tweets("btc", None, None, 1)
    assert not df.empty
    assert calls["n"] == 2

@pytest.mark.asyncio
async def test_onchain_retry_and_di(monkeypatch):
    http = get_http()
    client = OnchainClient(http=http)

    calls = {"g": 0, "c": 0}

    async def fake_glassnode(symbol, metric, days, http=None):
        calls["g"] += 1
        if calls["g"] < 2:
            raise RuntimeError("transient")
        import pandas as pd
        return pd.DataFrame([{"timestamp": 0, "value": 1.23}])

    async def fake_covalent(chain_id, address, http=None):
        calls["c"] += 1
        if calls["c"] < 2:
            raise RuntimeError("transient")
        import pandas as pd
        return pd.DataFrame([{"timestamp": 0, "value": 0.0, "contract_address": "0x00", "contract_name": "X"}])

    from app.adapters import onchain_adapter
    monkeypatch.setattr(onchain_adapter, "fetch_glassnode", fake_glassnode)
    monkeypatch.setattr(onchain_adapter, "fetch_covalent", fake_covalent)

    df1 = await client.get_glassnode_metric("BTC", "active_addresses", 1)
    df2 = await client.get_covalent_balances(1, "0xabc")
    assert not df1.empty and not df2.empty
    assert calls["g"] == 2 and calls["c"] == 2