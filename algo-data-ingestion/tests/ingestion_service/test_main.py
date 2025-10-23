import pytest

pytestmark = pytest.mark.anyio("asyncio")


async def test_root_endpoint(async_client):
    async with async_client() as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"service": "raw-data-ingestion", "version": "1.0.0"}


async def test_health_endpoint(async_client):
    async with async_client() as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_metrics_endpoint(async_client):
    async with async_client() as client:
        response = await client.get("/metrics")
    assert response.status_code == 200
    # Should include Prometheus metric for writes (even if zero)
    assert "parquet_writes_total" in response.text
    assert "parquet_write_errors_total" in response.text
    assert "parquet_write_latency_seconds" in response.text
