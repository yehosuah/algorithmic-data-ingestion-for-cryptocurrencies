import asyncio
from datetime import datetime
import pandas as pd
import pytest
from types import SimpleNamespace
from httpx import Response
import json

import feedparser

# FakeClient for patching httpx.AsyncClient
class FakeClient:
    def __init__(self, get_fn):
        self.get = get_fn

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

# Adjust this import to your module path
from app.adapters.news_adapter import (
    fetch_news_api,
    fetch_news_rss,
    NEWS_API_KEY,
)


class DummyResponse:
    def __init__(self, status_code: int, json_data=None, text_data=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


@pytest.fixture(autouse=True)
def fix_env(monkeypatch):
    # ensure NEWS_API_KEY is set
    monkeypatch.setenv("NEWS_API_KEY", "test-token")
    import importlib
    import app.adapters.news_adapter as news_mod
    importlib.reload(news_mod)
    yield


@pytest.mark.asyncio
async def test_fetch_news_api_success(monkeypatch):
    # Prepare fake articles
    data = {
        "data": [
            {
                "news_url": "https://example.com/abcd1234",
                "title": "Test Title",
                "source_name": "ExampleSource",
                "author": "Alice",
                "text": "Description here",
                "date": "2025-08-01T12:00:00"
            }
        ]
    }

    async def fake_get(url, params):
        # check parameters passed correctly
        assert params["section"] == "general"
        assert params["items"] == 1
        assert params["token"] == "test-token"
        return DummyResponse(200, json_data=data)

    # patch httpx.AsyncClient.get
    monkeypatch.setattr(
        "app.adapters.news_adapter.httpx.AsyncClient",
        lambda *args, **kwargs: FakeClient(fake_get)
    )

    results = await fetch_news_api("general", limit=1)
    # We now return a normalized pandas DataFrame
    assert isinstance(results, pd.DataFrame)
    assert results.shape[0] == 1
    # dtype for published_at should be tz-aware UTC
    assert str(results["published_at"].dtype).endswith("UTC]")
    row = results.iloc[0]
    assert row["id"] == "abcd1234"
    assert row["title"] == "Test Title"
    assert row["url"] == "https://example.com/abcd1234"
    assert row["source"] == "ExampleSource"
    assert row["author"] == "Alice"
    assert row["description"] == "Description here"
    assert row["published_at"] == pd.Timestamp("2025-08-01T12:00:00Z")


@pytest.mark.asyncio
async def test_fetch_news_api_http_error(monkeypatch):
    async def fake_get(url, params):
        return DummyResponse(500)

    monkeypatch.setattr(
        "app.adapters.news_adapter.httpx.AsyncClient",
        lambda *args, **kwargs: FakeClient(fake_get)
    )

    with pytest.raises(Exception):
        await fetch_news_api("exchange", limit=5)


@pytest.mark.asyncio
async def test_fetch_news_rss(monkeypatch):
    monkeypatch.setattr(
        "app.adapters.news_adapter.socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 80))],
    )
    # Fake RSS content
    xml = "<rss><channel><item><guid>1</guid><title>News 1</title><link>url1</link><description>Sum1</description><pubDate>Wed, 01 Aug 2025 12:00:00 GMT</pubDate></item></channel></rss>"

    async def fake_get(url):
        return DummyResponse(200, text_data=xml)

    # patch httpx.AsyncClient.get
    monkeypatch.setattr(
        "app.adapters.news_adapter.httpx.AsyncClient",
        lambda *args, **kwargs: FakeClient(lambda url: fake_get(url))
    )

    # patch feedparser.parse
    monkeypatch.setattr(feedparser, "parse", lambda content: SimpleNamespace(entries=[
        {
            "id": "1",
            "title": "News 1",
            "link": "url1",
            "summary": "Sum1",
            "published": "Wed, 01 Aug 2025 12:00:00 GMT"
        }
    ]))

    updates = []
    # replace asyncio.sleep to break after first loop
    monkeypatch.setattr("app.adapters.news_adapter.asyncio.sleep", lambda s: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        await fetch_news_rss("http://fake.feed/", lambda item: updates.append(item), poll_interval=1)

    assert len(updates) == 1
    item = updates[0]
    assert item["id"] == "1"
    assert item["title"] == "News 1"
    assert item["url"] == "url1"
    assert item["summary"] == "Sum1"
    assert item["published_at"] == "2025-08-01T12:00:00+00:00"


# 1. Test retry logic in fetch_news_api
@pytest.mark.asyncio
async def test_fetch_news_api_retry(monkeypatch):
    # Simulate HTTP failure on first attempt, then success
    data = {"data": [{"news_url": "https://example.com/xyz", "title": "T", "source_name": "S", "author": "A", "text": "D", "date": "2025-08-02T00:00:00"}]}
    call_count = {"n": 0}
    async def fake_get(url, params):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("HTTP error")
        return DummyResponse(200, json_data=data)
    monkeypatch.setenv("NEWS_API_KEY", "test-token")
    monkeypatch.setattr("app.adapters.news_adapter.httpx.AsyncClient", lambda *args, **kwargs: FakeClient(fake_get))
    results = await fetch_news_api("general", limit=1)
    assert len(results) == 1
    assert call_count["n"] == 2


# 2. Test JSON parse error handling in fetch_news_api
@pytest.mark.asyncio
async def test_fetch_news_api_parse_error(monkeypatch):
    # DummyResponse.json raises
    class BadResponse(DummyResponse):
        def json(self):
            raise ValueError("bad json")
    async def fake_get(url, params):
        return BadResponse(200)
    # Track parse error counter
    import app.adapters.news_adapter as mod
    called = {"n": 0}
    monkeypatch.setattr(mod, "NEWS_PARSE_ERRORS", SimpleNamespace(inc=lambda: called.__setitem__("n", called.get("n", 0)+1)))
    monkeypatch.setattr("app.adapters.news_adapter.httpx.AsyncClient", lambda *args, **kwargs: FakeClient(fake_get))
    results = await fetch_news_api("general", limit=1)
    assert isinstance(results, pd.DataFrame)
    assert results.empty
    assert called["n"] == 1


# 3. Test RSS parse error handling in fetch_news_rss
@pytest.mark.asyncio
async def test_fetch_news_rss_parse_error(monkeypatch):
    monkeypatch.setattr(
        "app.adapters.news_adapter.socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 80))],
    )
    # Simulate parse failure then KeyboardInterrupt
    xml = "<bad</xml>"
    async def fake_get(url):
        return DummyResponse(200, text_data=xml)
    # Monkeypatch parse to raise
    import app.adapters.news_adapter as mod
    called = {"n": 0}
    monkeypatch.setattr(mod, "NEWS_PARSE_ERRORS", SimpleNamespace(inc=lambda: called.__setitem__("n", called.get("n", 0)+1)))
    monkeypatch.setattr(feedparser, "parse", lambda content: (_ for _ in ()).throw(Exception("xml error")))
    monkeypatch.setenv("RSS_POLL_INTERVAL", "1")
    # Break out of loop after error
    monkeypatch.setattr("app.adapters.news_adapter.asyncio.sleep", lambda s: (_ for _ in ()).throw(KeyboardInterrupt()))
    # Patch client
    monkeypatch.setattr("app.adapters.news_adapter.httpx.AsyncClient", lambda *args, **kwargs: FakeClient(fake_get))
    updates = []
    with pytest.raises(KeyboardInterrupt):
        await fetch_news_rss("http://fake", lambda x: updates.append(x))
    assert called["n"] == 1
    assert updates == []


# 4. Test default poll interval
def test_default_poll_interval(monkeypatch):
    monkeypatch.delenv("RSS_POLL_INTERVAL", raising=False)
    import importlib, os
    import app.adapters.news_adapter as mod
    importlib.reload(mod)
    assert mod.DEFAULT_POLL_INTERVAL == 60
    monkeypatch.setenv("RSS_POLL_INTERVAL", "5")
    importlib.reload(mod)
    assert mod.DEFAULT_POLL_INTERVAL == 5

@pytest.mark.asyncio
async def test_news_api_calls_metric(monkeypatch):
    # Track NEWS_API_CALLS.inc calls
    import app.adapters.news_adapter as mod
    calls = {"n": 0}
    monkeypatch.setattr(mod, "NEWS_API_CALLS", SimpleNamespace(inc=lambda: calls.__setitem__("n", calls.get("n", 0) + 1)))
    # Fake HTTP client returning empty data
    async def fake_get(url, params):
        return DummyResponse(200, json_data={"data": []})
    monkeypatch.setattr(
        "app.adapters.news_adapter.httpx.AsyncClient",
        lambda *args, **kwargs: FakeClient(fake_get)
    )
    results = await fetch_news_api("general", limit=1)
    assert isinstance(results, pd.DataFrame)
    assert results.empty
    # Should have incremented once at function start
    assert calls["n"] == 1

@pytest.mark.asyncio
async def test_news_rss_polls_metric(monkeypatch):
    monkeypatch.setattr(
        "app.adapters.news_adapter.socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 80))],
    )
    # Track NEWS_RSS_POLLS.inc calls
    import app.adapters.news_adapter as mod
    calls = {"n": 0}
    monkeypatch.setenv("RSS_POLL_INTERVAL", "1")
    monkeypatch.setattr(mod, "NEWS_RSS_POLLS", SimpleNamespace(inc=lambda: calls.__setitem__("n", calls.get("n", 0) + 1)))
    # Fake RSS content and break after first iteration
    xml = "<rss></rss>"
    async def fake_get(url):
        return DummyResponse(200, text_data=xml)
    monkeypatch.setattr(
        "app.adapters.news_adapter.httpx.AsyncClient",
        lambda *args, **kwargs: FakeClient(fake_get)
    )
    # Break loop after first poll
    monkeypatch.setattr("app.adapters.news_adapter.asyncio.sleep", lambda s: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        await fetch_news_rss("http://fake", lambda x: None)
    # Should have incremented once before parsing
    assert calls["n"] == 1

def test_validate_rss_url_blocks_private_literal():
    from app.adapters.news_adapter import validate_rss_url

    with pytest.raises(ValueError, match="non-public IP"):
        validate_rss_url("http://127.0.0.1/latest.xml")


def test_validate_rss_url_blocks_private_dns(monkeypatch):
    from app.adapters.news_adapter import validate_rss_url

    monkeypatch.setattr(
        "app.adapters.news_adapter.socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("10.0.0.5", 80))],
    )
    with pytest.raises(ValueError, match="non-public IP"):
        validate_rss_url("https://feeds.example.test/rss")
