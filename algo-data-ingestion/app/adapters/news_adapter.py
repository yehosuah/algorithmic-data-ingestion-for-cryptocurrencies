# Status: ✅ Exists & handles both REST news and RSS feeds
import os
import asyncio
import httpx
import feedparser
import logging
import socket
import ipaddress
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse, urljoin
import pandas as pd
from prometheus_client import Counter, CollectorRegistry
from app.common.time_norm import standardize_time_column, add_dt_partition, coerce_schema, to_utc_dt
from app.ingestion_service.parquet_schemas import NEWS_SCHEMA

_METRICS_REGISTRY = CollectorRegistry()

NEWS_API_CALLS = Counter('news_api_calls_total', 'Total number of news API calls', registry=_METRICS_REGISTRY)
NEWS_RSS_POLLS = Counter('news_rss_polls_total', 'Total number of RSS polls', registry=_METRICS_REGISTRY)
NEWS_PARSE_ERRORS = Counter('news_parse_errors_total', 'Total number of parse errors (JSON/XML)', registry=_METRICS_REGISTRY)

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = int(os.getenv("RSS_POLL_INTERVAL", "60"))
RSS_MAX_BYTES = max(1024, int(os.getenv("RSS_MAX_BYTES", str(2 * 1024 * 1024))))
RSS_MAX_REDIRECTS = max(0, int(os.getenv("RSS_MAX_REDIRECTS", "3")))
_ALLOWED_HOSTS_RAW = os.getenv("RSS_ALLOWED_HOSTS", "").strip()
RSS_ALLOWED_HOSTS = {host.strip().lower() for host in _ALLOWED_HOSTS_RAW.split(",") if host.strip()}


def _host_allowed_by_policy(host: str) -> bool:
    if not RSS_ALLOWED_HOSTS:
        return True
    host = host.rstrip(".").lower()
    for allowed in RSS_ALLOWED_HOSTS:
        allowed = allowed.rstrip(".").lower()
        if host == allowed or host.endswith(f".{allowed}"):
            return True
    return False


def _is_public_ip(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_rss_url(feed_url: str) -> str:
    """
    Validate an operator/user supplied RSS URL before issuing outbound requests.

    Only HTTP(S) URLs are accepted. Hostnames must pass the optional
    RSS_ALLOWED_HOSTS allow-list and must not resolve to loopback, link-local,
    private, reserved, multicast, or unspecified IP ranges.
    """
    parsed = urlparse(str(feed_url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("RSS feed URL must use http or https")
    if not parsed.hostname:
        raise ValueError("RSS feed URL must include a hostname")
    host = parsed.hostname.rstrip(".").lower()
    if not _host_allowed_by_policy(host):
        raise ValueError("RSS feed hostname is not allow-listed")

    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        if not _is_public_ip(str(literal_ip)):
            raise ValueError("RSS feed URL resolves to a non-public IP address")
        return feed_url

    try:
        infos = socket.getaddrinfo(
            host,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError("RSS feed hostname could not be resolved") from exc
    addresses = {info[4][0] for info in infos if info and info[4]}
    if not addresses:
        raise ValueError("RSS feed hostname did not resolve to any address")
    for address in addresses:
        if not _is_public_ip(address):
            raise ValueError("RSS feed hostname resolves to a non-public IP address")
    return feed_url


async def _read_limited_text(resp: httpx.Response, *, max_bytes: int = RSS_MAX_BYTES) -> str:
    content_length = getattr(resp, "headers", {}).get("content-length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise ValueError("RSS response exceeds maximum allowed size")
        except ValueError:
            raise
        except Exception:
            pass
    data = getattr(resp, "content", None)
    if data is None:
        data = str(getattr(resp, "text", "")).encode("utf-8")
    if len(data) > max_bytes:
        raise ValueError("RSS response exceeds maximum allowed size")
    return resp.text


async def _get_validated_rss(
    client: httpx.AsyncClient,
    feed_url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
) -> httpx.Response:
    url = validate_rss_url(feed_url)
    for _ in range(RSS_MAX_REDIRECTS + 1):
        request_kwargs = {"headers": headers} if headers is not None else {}
        resp = await _retry_http(client.get, url, **request_kwargs)
        if resp.status_code not in {301, 302, 303, 307, 308}:
            return resp
        location = getattr(resp, "headers", {}).get("location")
        if not location:
            return resp
        url = validate_rss_url(urljoin(url, location))
    raise ValueError("RSS feed exceeded maximum redirects")


async def _retry_http(method, *args, retries: int = 3, backoff_factor: float = 1.0, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return await method(*args, **kwargs)
        except Exception as e:
            NEWS_API_CALLS.inc()  # count each failed attempt
            try:
                logger.warning(f"HTTP call {method.__name__} failed attempt {attempt}/{retries}: {e}")
            except Exception:
                pass
            if attempt < retries:
                await asyncio.sleep(backoff_factor * 2 ** (attempt - 1))
    # Last attempt
    return await method(*args, **kwargs)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
async def fetch_news_api(*args, **kwargs) -> pd.DataFrame:
    """
    Fetch crypto news via REST and return a normalized DataFrame.

    Supports two call styles for backward compatibility:
      - fetch_news_api(query: str, limit: int)
      - fetch_news_api(since: datetime|None, until: datetime|None, source: str, limit: int)
    """
    NEWS_API_CALLS.inc()

    # Parse arguments
    query: Optional[str] = None
    since: Optional[datetime] = kwargs.get("since")
    until: Optional[datetime] = kwargs.get("until")
    source: Optional[str] = kwargs.get("source")
    limit: Optional[int] = kwargs.get("limit")

    # Legacy positional signature: (query: str, limit: int)
    if len(args) >= 1 and isinstance(args[0], str) and (source is None):
        query = args[0]
        if limit is None and len(args) >= 2 and isinstance(args[1], int):
            limit = args[1]

    section = (query or source or "general")
    if limit is None:
        limit = 100

    url = "https://cryptonews-api.com/api/v1/category"
    params = {
        "section": section,
        "items": limit,
        "token": NEWS_API_KEY,
    }

    async with httpx.AsyncClient() as client:
        resp = await _retry_http(client.get, url, params=params)
        resp.raise_for_status()
        try:
            raw = resp.json().get("data", [])
        except Exception as e:
            NEWS_PARSE_ERRORS.inc()
            logger.error(f"JSON parse error in fetch_news_api: {e}")
            # Return schema-stable empty frame
            empty_schema = {
                "id": "string",
                "title": "string",
                "url": "string",
                "source": "string",
                "author": "string",
                "description": "string",
                "published_at": "datetime64[ns, UTC]",
            }
            return coerce_schema(pd.DataFrame(), empty_schema)

    rows: List[Dict[str, Any]] = []
    for art in raw:
        news_url = art.get("news_url", "")
        art_id = news_url.rstrip("/").split("/")[-1] if news_url else None
        published_raw = art.get("date")
        # Convert published time to UTC
        try:
            published = pd.to_datetime(published_raw, utc=True)
        except Exception:
            published = pd.NaT
        rows.append({
            "id": art_id,
            "title": art.get("title", ""),
            "url": news_url,
            "source": art.get("source_name", ""),
            "author": art.get("author", None),
            "description": art.get("text", None),
            "published_at": published,
        })

    df = pd.DataFrame(rows)
    df = standardize_time_column(df, candidates=["published_at", "date"], dest="published_at")
    schema = {
        "id": "string",
        "title": "string",
        "url": "string",
        "source": "string",
        "author": "string",
        "description": "string",
        "published_at": "datetime64[ns, UTC]",
    }
    df = coerce_schema(df, schema)

    # Optional range filter if since/until were provided in kwargs
    if since is not None:
        since_utc = pd.Timestamp(since, tz="UTC") if not isinstance(since, pd.Timestamp) else (since if since.tzinfo else since.tz_localize("UTC"))
        df = df[df["published_at"] >= since_utc]
    if until is not None:
        until_utc = pd.Timestamp(until, tz="UTC") if not isinstance(until, pd.Timestamp) else (until if until.tzinfo else until.tz_localize("UTC"))
        df = df[df["published_at"] <= until_utc]

    add_dt_partition(df, ts_col="published_at")
    return df


async def fetch_news_rss_once(
    feed_url: str,
    *,
    limit: int = 200,
    http: Optional[httpx.AsyncClient] = None,
) -> pd.DataFrame:
    """Fetch a single RSS snapshot and return a normalized DataFrame."""
    headers = {"User-Agent": "AlgoDataIngestion/1.0 (+https://example.local)"}
    limit = max(limit, 1)

    if http is None:
        async with httpx.AsyncClient(follow_redirects=False, headers=headers, timeout=15) as client:
            resp = await _get_validated_rss(client, feed_url, headers=headers)
    else:
        resp = await _get_validated_rss(http, feed_url, headers=headers)

    resp.raise_for_status()
    content = await _read_limited_text(resp)

    try:
        feed = feedparser.parse(content)
    except Exception as e:
        NEWS_PARSE_ERRORS.inc()
        logger.error(f"RSS parse error in snapshot fetch: {e}")
        return coerce_schema(pd.DataFrame(), NEWS_SCHEMA)

    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(feed.entries[:limit]):
        entry_id = entry.get("id") or entry.get("link") or f"idx-{idx}"
        link = entry.get("link") or ""
        published_raw = entry.get("published") or entry.get("updated")
        rows.append({
            "id": entry_id,
            "title": entry.get("title"),
            "url": link,
            "source": urlparse(link).netloc,
            "author": entry.get("author"),
            "description": entry.get("summary"),
            "published_at": published_raw,
        })

    df = pd.DataFrame(rows)
    df = standardize_time_column(df, candidates=["published_at", "updated", "pubDate"], dest="published_at")
    df = coerce_schema(df, NEWS_SCHEMA)
    add_dt_partition(df, ts_col="published_at")
    return df

async def fetch_news_rss(feed_url: str, handle_update: callable, poll_interval: int = DEFAULT_POLL_INTERVAL):
    seen_ids = set()
    while True:
        NEWS_RSS_POLLS.inc()
        async with httpx.AsyncClient(follow_redirects=False, timeout=15) as client:
            resp = await _get_validated_rss(client, feed_url)
            resp.raise_for_status()
            content = await _read_limited_text(resp)
        try:
            feed = feedparser.parse(content)
        except Exception as e:
            NEWS_PARSE_ERRORS.inc()
            logger.error(f"RSS parse error: {e}")
            await asyncio.sleep(poll_interval)
            continue
        for entry in feed.entries:
            entry_id = entry.get("id") or entry.get("link")
            if entry_id not in seen_ids:
                seen_ids.add(entry_id)
                # Normalize published time to UTC ISO string
                published_raw = entry.get("published") or entry.get("updated")
                published_utc = pd.to_datetime(published_raw, utc=True, errors="coerce")
                published_iso = published_utc.isoformat() if pd.notnull(published_utc) else None
                item = {
                    "id": entry_id,
                    "title": entry.get("title"),
                    "url": entry.get("link"),
                    "summary": entry.get("summary"),
                    "published_at": published_iso,
                }
                result = handle_update(item)
                if asyncio.iscoroutine(result):
                    await result
        await asyncio.sleep(poll_interval)
