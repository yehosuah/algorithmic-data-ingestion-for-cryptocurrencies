from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable

import httpx
import pytest

try:
    from fakeredis.aioredis import FakeRedis as _FakeRedis  # type: ignore
except Exception:  # pragma: no cover
    _FakeRedis = None

from app.ingestion_service.main import app


@pytest.fixture
def async_client() -> Callable[[], AsyncIterator[httpx.AsyncClient]]:
    """Factory that yields an AsyncClient wired to the FastAPI app with lifespan."""

    @asynccontextmanager
    async def _client() -> AsyncIterator[httpx.AsyncClient]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            follow_redirects=True,
        ) as client:
            yield client

    return _client


@pytest.fixture(autouse=True)
def _reset_dependency_overrides():
    """Ensure dependency overrides never leak across tests."""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def anyio_backend():
    """Force anyio-powered tests to use asyncio only."""
    return "asyncio"


@pytest.fixture(autouse=True)
def _stub_async_redis(monkeypatch):
    """Avoid real Redis connections during app lifespan."""
    if _FakeRedis is not None:
        stub = _FakeRedis(decode_responses=True)
    else:
        class _StubRedis:
            async def close(self):
                pass

            async def pipeline(self, *args, **kwargs):
                raise RuntimeError("Redis pipeline requested but fakeredis unavailable")

        stub = _StubRedis()

    monkeypatch.setattr("redis.asyncio.from_url", lambda *a, **k: stub)
    yield
