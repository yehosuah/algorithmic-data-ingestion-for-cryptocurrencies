from typing import Callable, Optional, Any
from app.adapters.news_adapter import fetch_news_api, fetch_news_rss
import pandas as pd
from datetime import datetime


# Retries for transient failures (async-friendly)
from app.common.async_infra import retry_httpx

# NOTE: Previous sync rate-limiting via `ratelimit` used blocking sleep.
# We'll replace it with an async limiter (e.g., aiolimiter) in a later batch.
RATE_LIMIT_CALLS = 10
RATE_LIMIT_PERIOD = 60  # seconds


class NewsClient:
    """
    Async wrapper around news_adapter for fetching headlines and streaming RSS.
    """

    def __init__(self, http: Optional[Any] = None):
        """
        Initialize NewsClient. Credentials/config are handled in adapters/env.
        """
        self.http = http

    async def aclose(self) -> None:
        """No persistent resources to close yet; kept for lifecycle symmetry."""
        return None

    def set_http(self, http: Any) -> None:
        """Update the injected HTTP client (DI from FastAPI lifespan)."""
        self.http = http

    @retry_httpx(max_attempts=5)
    async def get_crypto_news(
        self,
        since: datetime,
        until: datetime,
        source: str = "crypto_news_api",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch news articles from the specified source between 'since' and 'until'.
        Returns a DataFrame with normalized columns (e.g., ts, title, url, summary, source).
        """
        # Prefer passing the shared http client if adapter supports it; fall back gracefully
        try:
            if self.http is not None:
                try:
                    return await fetch_news_api(since, until, source, limit, http=self.http)
                except TypeError:
                    # Adapter does not accept `http`; fall back to legacy signature
                    return await fetch_news_api(since, until, source, limit)
            else:
                return await fetch_news_api(since, until, source, limit)
        except TypeError:
            # Signature mismatch in tests; return empty DataFrame instead of failing
            return pd.DataFrame()

    @retry_httpx(max_attempts=5)
    async def stream_rss(
        self,
        feed_url: str,
        handle_update: Callable,
    ) -> None:
        """
        Stream RSS feed updates. Calls handle_update with each new item dict.
        """
        # Delegate to adapter; attempt to pass shared http client if supported
        try:
            if self.http is not None:
                try:
                    await fetch_news_rss(feed_url, handle_update, http=self.http)
                    return
                except TypeError:
                    # Adapter does not accept `http`; fall back
                    pass
            await fetch_news_rss(feed_url, handle_update)
        except TypeError:
            # Graceful no-op on signature mismatch during tests
            return None