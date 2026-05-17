"""Async Redis-backed feature store (Batch 2.1)

Responsibilities
- Normalize timestamps to epoch seconds (UTC)
- Keys: {namespace}:{domain}:{symbol}:{timeframe}:{epoch_sec}
- Async write/read (+ batch variants) with optional TTL
- Prometheus metrics: writes/reads/hits/misses + op latency
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import time
import logging
import os

import pandas as pd
from prometheus_client import Counter, Histogram
from redis import asyncio as aioredis  # redis>=4.5

from app.common.time_norm import to_utc_dt
from app.ingestion_service.config import settings
from app.ingestion_service.utils import _METRICS_REGISTRY

logger = logging.getLogger(__name__)

# ------------------------------
# Metrics
# ------------------------------
FEATURE_WRITES_TOTAL = Counter(
    "feature_writes_total",
    "Number of feature write operations",
    labelnames=("domain",),
    registry=_METRICS_REGISTRY,
)
FEATURE_READS_TOTAL = Counter(
    "feature_reads_total",
    "Number of feature read operations",
    labelnames=("domain",),
    registry=_METRICS_REGISTRY,
)
FEATURE_HITS_TOTAL = Counter(
    "feature_hits_total",
    "Number of cache hits on read",
    labelnames=("domain",),
    registry=_METRICS_REGISTRY,
)
FEATURE_MISSES_TOTAL = Counter(
    "feature_misses_total",
    "Number of cache misses on read",
    labelnames=("domain",),
    registry=_METRICS_REGISTRY,
)
FEATURE_OP_LATENCY = Histogram(
    "feature_op_latency_seconds",
    "Latency of feature store operations",
    labelnames=("op",),
    registry=_METRICS_REGISTRY,
)


def _safe_symbol(symbol: str) -> str:
    """Canonicalize symbol for redis keys (avoid path-ish chars)."""
    s = (symbol or "").strip()
    return s.replace("/", "-").replace(":", "-").upper()


def _epoch_seconds(ts: Any) -> int:
    """Coerce timestamp into epoch seconds (UTC)."""
    if isinstance(ts, int):
        # Heuristic: ms vs s
        return int(ts // 1000) if ts > 10_000_000_000 else int(ts)
    if isinstance(ts, float):
        return int(ts)

    ser = pd.Series([ts])
    dt = to_utc_dt(ser)
    if dt.isna().iloc[0]:
        raise ValueError(f"Could not parse timestamp to UTC: {ts!r}")
    stamp = pd.Timestamp(dt.iloc[0])
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return int(stamp.timestamp())


class RedisFeatureStore:
    """Async feature store wrapper around Redis."""

    def __init__(
        self,
        url: str,
        *,
        namespace: str = "features",
        default_ttl: Optional[int] = None,
        redis_client: Optional[aioredis.Redis] = None,
        decode_responses: bool = True,
    ) -> None:
        if not url:
            raise RuntimeError("REDIS_URL is not configured")
        self._url = url
        self._namespace = (namespace or "features").strip()
        self._default_ttl = int(default_ttl) if (default_ttl not in (None, "", "None")) else None
        self._client: Optional[aioredis.Redis] = redis_client
        self._decode_responses = decode_responses

    def _key(self, domain: str, symbol: str, timeframe: str, epoch_s: int) -> str:
        dom = (domain or "").strip().lower()
        tf = (timeframe or "").strip().lower()
        sym = _safe_symbol(symbol)
        return f"{self._namespace}:{dom}:{sym}:{tf}:{int(epoch_s)}"

    def _index_key(self, domain: str, symbol: str, timeframe: str) -> str:
        """
        Per-(domain,symbol,timeframe) sorted index for range queries.
        Members are feature keys, score is epoch seconds.
        """
        dom = (domain or "").strip().lower()
        tf = (timeframe or "").strip().lower()
        sym = _safe_symbol(symbol)
        return f"{self._namespace}:{dom}:{sym}:{tf}:_idx"

    def _ensure_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=self._decode_responses,
            )
        return self._client

    async def write(
        self,
        *,
        domain: str,
        symbol: str,
        timeframe: str,
        ts: Any,
        payload: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> str:
        """Write a single feature blob; returns redis key."""
        start = time.perf_counter()
        epoch_s = _epoch_seconds(ts)
        key = self._key(domain, symbol, timeframe, epoch_s)
        client = self._ensure_client()
        try:
            await client.set(key, json.dumps(payload, default=str), ex=ttl or self._default_ttl)
            FEATURE_WRITES_TOTAL.labels(domain=domain).inc()
            return key
        finally:
            FEATURE_OP_LATENCY.labels(op="write").observe(time.perf_counter() - start)

    async def read(
        self, *, domain: str, symbol: str, timeframe: str, ts: Any
    ) -> Optional[Dict[str, Any]]:
        """Read a single feature blob by (domain, symbol, timeframe, ts)."""
        start = time.perf_counter()
        epoch_s = _epoch_seconds(ts)
        key = self._key(domain, symbol, timeframe, epoch_s)
        client = self._ensure_client()
        try:
            FEATURE_READS_TOTAL.labels(domain=domain).inc()
            raw = await client.get(key)
            if raw is None:
                FEATURE_MISSES_TOTAL.labels(domain=domain).inc()
                return None
            FEATURE_HITS_TOTAL.labels(domain=domain).inc()
            return json.loads(raw)
        finally:
            FEATURE_OP_LATENCY.labels(op="read").observe(time.perf_counter() - start)

    async def batch_write(self, items: Sequence[Dict[str, Any]]) -> List[str]:
        """Batch write using a pipeline.
        Each item: {domain, symbol, timeframe, ts, payload, [ttl]}.
        Returns the list of keys written.
        """
        start = time.perf_counter()
        client = self._ensure_client()
        keys: List[str] = []
        async with client.pipeline(transaction=False) as pipe:
            for it in items:
                domain = it["domain"]
                symbol = it["symbol"]
                timeframe = it["timeframe"]
                ttl = it.get("ttl")
                epoch_s = _epoch_seconds(it["ts"])
                key = self._key(domain, symbol, timeframe, epoch_s)
                keys.append(key)
                payload = json.dumps(it["payload"], default=str)
                # IMPORTANT: do NOT `await` individual pipeline commands
                pipe.set(key, payload, ex=ttl or self._default_ttl)
                # Maintain time index for range queries
                idx = self._index_key(domain, symbol, timeframe)
                pipe.zadd(idx, {key: float(epoch_s)})
                FEATURE_WRITES_TOTAL.labels(domain=domain).inc()
            await pipe.execute()
        FEATURE_OP_LATENCY.labels(op="batch_write").observe(time.perf_counter() - start)
        return keys

    async def batch_read(
        self, queries: Sequence[Tuple[str, str, str, Any]]
    ) -> List[Optional[Dict[str, Any]]]:
        """Batch read: queries = [(domain, symbol, timeframe, ts), ...]."""
        start = time.perf_counter()
        client = self._ensure_client()
        keys: List[str] = []
        for (domain, symbol, timeframe, ts) in queries:
            FEATURE_READS_TOTAL.labels(domain=domain).inc()
            keys.append(self._key(domain, symbol, timeframe, _epoch_seconds(ts)))
        raw_list = await client.mget(keys)
        out: List[Optional[Dict[str, Any]]] = []
        for i, raw in enumerate(raw_list):
            domain = queries[i][0]
            if raw is None:
                FEATURE_MISSES_TOTAL.labels(domain=domain).inc()
                out.append(None)
            else:
                FEATURE_HITS_TOTAL.labels(domain=domain).inc()
                out.append(json.loads(raw))
        FEATURE_OP_LATENCY.labels(op="batch_read").observe(time.perf_counter() - start)
        return out

    async def range_read(
        self,
        *,
        domain: str,
        symbol: str,
        timeframe: str,
        start: int,
        end: int,
        limit: int = 200,
        reverse: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return list of payload dicts for keys in [start,end] (epoch seconds).

        Uses a per-(domain,symbol,timeframe) ZSET index whose score is epoch seconds
        and whose member is the full feature key.
        """
        t0 = time.perf_counter()
        client = self._ensure_client()
        idx = self._index_key(domain, symbol, timeframe)
        try:
            # fetch matching keys from sorted index
            if reverse:
                key_list = await client.zrevrangebyscore(idx, end, start, start=0, num=limit)
            else:
                key_list = await client.zrangebyscore(idx, start, end, start=0, num=limit)
            if not key_list:
                return []
            raw_list = await client.mget(key_list)
            out: List[Dict[str, Any]] = []
            for raw in raw_list:
                if not raw:
                    continue
                try:
                    out.append(json.loads(raw))
                except Exception:
                    continue
            return out
        finally:
            FEATURE_OP_LATENCY.labels(op="range_read").observe(time.perf_counter() - t0)

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            finally:
                try:
                    await self._client.connection_pool.disconnect()
                except Exception:
                    pass


# ------------------------------
# Singleton accessor for DI
# ------------------------------
_store_singleton: Optional[RedisFeatureStore] = None


def get_store() -> RedisFeatureStore:
    """Return a process-wide singleton store using settings/env values.

    settings/env keys:
      - REDIS_URL (str)                e.g. redis://redis:6379/0
      - REDIS_HOST/PORT/DB             used to synthesize URL if REDIS_URL missing
      - FEATURE_TTL_SEC (int|None)     default TTL for writes
      - FEATURE_NAMESPACE (str)        key namespace, default 'features'
    """
    global _store_singleton
    if _store_singleton is None:
        # Prefer settings.REDIS_URL, then env, then synthesize from host/port/db
        url = getattr(settings, "REDIS_URL", None) or os.getenv("REDIS_URL")
        if not url:
            host = getattr(settings, "REDIS_HOST", None) or os.getenv("REDIS_HOST", "redis")
            port = getattr(settings, "REDIS_PORT", None) or os.getenv("REDIS_PORT", "6379")
            db = getattr(settings, "REDIS_DB", None) or os.getenv("REDIS_DB", "0")
            url = f"redis://{host}:{port}/{db}"

        ttl_raw = getattr(settings, "FEATURE_TTL_SEC", None) or os.getenv("FEATURE_TTL_SEC")
        ttl = int(ttl_raw) if ttl_raw not in (None, "", "None") else None

        ns = getattr(settings, "FEATURE_NAMESPACE", None) or os.getenv("FEATURE_NAMESPACE", "features")

        _store_singleton = RedisFeatureStore(url=url, namespace=ns, default_ttl=ttl)
        logger.info("RedisFeatureStore initialized: url=%s namespace=%s ttl=%s", url, ns, ttl)
    return _store_singleton

# Ensure our metric families exist in the registry as soon as this
# module is imported (before any reads/writes happen).
def _ensure_metric_families() -> None:
    try:
        FEATURE_WRITES_TOTAL.labels(domain="bootstrap").inc(0)
        FEATURE_READS_TOTAL.labels(domain="bootstrap").inc(0)
        FEATURE_HITS_TOTAL.labels(domain="bootstrap").inc(0)
        FEATURE_MISSES_TOTAL.labels(domain="bootstrap").inc(0)
        FEATURE_OP_LATENCY.labels(op="init").observe(0.0)
    except Exception:
        # Don't crash import if prometheus_client changes; it's just a warmup.
        pass

_ensure_metric_families()

__all__ = [
    "RedisFeatureStore",
    "get_store",
    "FEATURE_WRITES_TOTAL",
    "FEATURE_READS_TOTAL",
    "FEATURE_HITS_TOTAL",
    "FEATURE_MISSES_TOTAL",
    "FEATURE_OP_LATENCY",
]