from __future__ import annotations

import json
import math
import logging
import contextlib
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Any
from urllib.parse import urlparse
from app.features.jobs.backfill import backfill_market_once, ttl_sweep_once
from fastapi import BackgroundTasks
from fastapi import Security
from fastapi.security import APIKeyHeader
import inspect

import pandas as pd
import redis
try:
    import fakeredis
except Exception:
    fakeredis = None
from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    WebSocket,
    WebSocketDisconnect,
    Body,
    Query,
)
from fastapi.encoders import jsonable_encoder

from app.ingestion_service.config import settings
from .schemas import (
    MarketIngestRequest,
    OnchainIngestRequest,
    SocialIngestRequest,
    NewsIngestRequest,
    FeatureVector,
)
from app.adapters.ccxt_adapter import CCXTAdapter
from app.adapters.onchain_adapter import fetch_glassnode, fetch_covalent
from app.adapters.reddit_adapter import fetch_reddit_api, fetch_pushshift
from app.adapters.news_adapter import fetch_news_api, fetch_news_rss, fetch_news_rss_once
from app.adapters.sentiment_adapter import fetch_twitter_sentiment
from .utils import write_to_parquet
from app.features.ingestion.ccxt_client import CCXTClient
from app.features.ingestion.news_client import NewsClient
from app.features.ingestion.onchain_client import OnchainClient
from app.features.ingestion.social_client import SocialClient
from app.features.store.redis_store import get_store, RedisFeatureStore
from app.features.factory.market_factory import build_market_features
from training.feature_eng import augment_market_features
import inspect
from fastapi import HTTPException
from app.ingestion_service.metrics import ingest_span, record_rows_written
from fastapi.responses import JSONResponse
from app.ingestion_service.config import settings
from app.ingestion_service import ml_utils
import fsspec
import os
import time
from app.common.time_norm import add_dt_partition

logger = logging.getLogger(__name__)

router = APIRouter()

# Sync Redis client for simple health/demo endpoints (feature store is async)
def _make_sync_redis():
    host = getattr(settings, "redis_host", "redis")
    port = getattr(settings, "redis_port", 6379)
    db = getattr(settings, "redis_db", 0)
    pwd = getattr(settings, "redis_password", None) or None
    try:
        r = redis.Redis(host=host, port=port, db=db, password=pwd, decode_responses=True)
        # Try a ping early to surface connection issues
        r.ping()
        return r
    except Exception:
        # Fallback to fakeredis only in test/dev environments
        if fakeredis is not None:
            return fakeredis.FakeRedis(decode_responses=True)
        # Re-raise if we cannot fall back
        raise

_redis = _make_sync_redis()

# ---------------------------
# Helpers
# ---------------------------

def provide_store() -> RedisFeatureStore:
    return get_store()
def _parse_epoch_sec(ts: Optional[int]) -> Optional[datetime]:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


def _clean_numbers(obj: Any) -> Any:
    """Recursively convert numpy scalars to Python, replace NaN/Inf with None."""
    try:
        import numpy as np
        np_f = (np.floating, np.integer)
    except Exception:
        np_f = tuple()

    if isinstance(obj, dict):
        return {k: _clean_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_numbers(v) for v in obj]
    if np_f and isinstance(obj, np_f):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

def provide_news() -> NewsClient:
    """Local provider to avoid circular import from main."""
    return NewsClient()

def _ensure_utc_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Force df[col] to tz-aware UTC, robust to:
      - tz-naive pandas Timestamps
      - tz-aware pandas Timestamps (any tz)
      - strings / ints / floats / py datetimes
      - object-dtype mixes
    """
    if df is None or df.empty or col not in df.columns:
        return df

    def _to_utc_one(x):
        if isinstance(x, pd.Timestamp):
            return x.tz_convert("UTC") if x.tzinfo is not None else x.tz_localize("UTC")
        # parse strings/ints/floats/datetimes
        t = pd.to_datetime(x, errors="coerce")  # don't pass utc=True here
        if pd.isna(t):
            return pd.NaT
        if isinstance(t, pd.Timestamp):
            return t.tz_convert("UTC") if t.tzinfo is not None else t.tz_localize("UTC")
        return pd.NaT

    df[col] = df[col].map(_to_utc_one)
    return df

def _force_epoch_then_utc(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Fallback: convert df[col] to epoch seconds (robust to tz/naive/str),
    then rebuild it as UTC tz-aware timestamps deterministically.
    """
    if df is None or df.empty or col not in df.columns:
        return df
    df = df.copy()
    df["__epoch_s"] = df[col].map(_epoch_s_from_ts)
    df[col] = pd.to_datetime(df["__epoch_s"], unit="s", utc=True)
    df.drop(columns=["__epoch_s"], inplace=True, errors="ignore")
    return df


def _epoch_s_from_ts(ts: Any) -> int:
    # robust conversion to epoch seconds for pandas/py dt/int/str
    if isinstance(ts, pd.Timestamp):
        t = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
        return int(t.value // 1_000_000_000)
    from datetime import datetime as _dt
    if isinstance(ts, _dt):
        t = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
        return int(t.astimezone(timezone.utc).timestamp())
    if isinstance(ts, (int, float)):
        # treat >10B as milliseconds, else seconds
        return int(ts // 1000) if ts > 10_000_000_000 else int(ts)
    return int(pd.to_datetime(ts, utc=True).value // 1_000_000_000)

async def _resolve(maybe_coro):
    return await maybe_coro if inspect.isawaitable(maybe_coro) else maybe_coro
# ---------------------------
# CCXT (GET) Historical
# ---------------------------

@router.get("/ccxt/{exchange}/historical")
async def get_ccxt_historical(
    exchange: str,
    symbol: str,
    timeframe: str = "1m",
    limit: int = 100,
):
    """
    Fetch historical market data via CCXT client (async). Returns {rows, data}.
    """
    client = CCXTClient(exchange_name=exchange)
    try:
        df = await client.fetch_historical(symbol=symbol, timeframe=timeframe, limit=limit)
        rows = 0 if df is None else len(df)
        data = [] if df is None or df.empty else df.to_dict(orient="records")
        return {"rows": rows, "data": data}
    finally:
        await client.aclose()



# ---------------------------
# Market Ingest (POST)
# ---------------------------

@router.post("/market/{exchange}")
async def ingest_market(exchange: str, body: MarketIngestRequest):
    from ccxt.base.errors import BadSymbol
    adapter = CCXTAdapter(exchange)

    try:
        with ingest_span("market") as span:
            try:
                # ---- Fetch via YOUR adapter ----
                try:
                    df = await adapter.fetch_ohlcv(
                        body.symbol,
                        body.granularity,
                        since=None,
                        limit=body.limit,
                    )
                except BadSymbol as e:
                    # 400 so clients can correct the request
                    span.set_status("error")
                    raise HTTPException(
                        status_code=400,
                        detail=f"{exchange} does not have market symbol {body.symbol}",
                    ) from e

                # ---- Normalize + write parquet (with feature enrichment) ----
                try:
                    features = pd.DataFrame()
                    if not df.empty:
                        df["symbol"] = body.symbol
                        df["exchange"] = exchange

                        if "timestamp" not in df.columns:
                            raise ValueError("Missing columns: ['timestamp']")

                        s = df["timestamp"]
                        if pd.api.types.is_datetime64_any_dtype(s):
                            if getattr(s.dt, "tz", None) is not None:
                                df["timestamp"] = s.dt.tz_convert("UTC")
                            else:
                                df["timestamp"] = s.dt.tz_localize("UTC")
                        else:
                            df["timestamp"] = pd.to_datetime(s, utc=True)

                        try:
                            features = build_market_features(df)
                            features = augment_market_features(features, inplace=False)
                        except Exception as feat_exc:
                            logger.warning(
                                "Feature build during ingest failed; proceeding with raw OHLCV: %s",
                                feat_exc,
                            )
                            features = pd.DataFrame()

                        if not features.empty:
                            enriched = features.drop(columns=["dt"], errors="ignore")
                            df = df.merge(
                                enriched,
                                on=["timestamp", "symbol", "exchange", "timeframe"],
                                how="left",
                            )
                            # ensure dt exists for downstream parquet partitioning even if early rows lack features
                            add_dt_partition(df, ts_col="timestamp")

                    base = settings.MARKET_PATH
                    partitions = {
                        "exchange": exchange,
                        "symbol": body.symbol,
                        "year": df["timestamp"].dt.year.iloc[0] if not df.empty else None,
                        "month": df["timestamp"].dt.month.iloc[0] if not df.empty else None,
                        "day": df["timestamp"].dt.day.iloc[0] if not df.empty else None,
                    }

                    path = write_to_parquet(df, base, partitions)

                except ValueError as ve:
                    span.set_status("error")
                    raise HTTPException(status_code=422, detail=str(ve))
                except Exception as e:
                    span.set_status("error")
                    raise HTTPException(status_code=500, detail=f"Write failed: {e}")

                # ---- Feature fan-out ----
                features_written = 0
                if not df.empty:
                    try:
                        features_written = await _write_market_features_to_store(df, feats=features)
                    except Exception as e:
                        logger.warning("Feature write failed", exc_info=e)
                        features_written = 0

                status = "ok" if path is not None else "no_data"
                span.set_status(status)
                return {
                    "status": status,
                    "path": None if path is None else str(path),
                    "features_written": features_written,
                }

            except HTTPException:
                # status already set
                raise
            except Exception as e:
                span.set_status("error")
                raise HTTPException(status_code=500, detail=f"ingest_market failed: {e}")
    finally:
        with contextlib.suppress(Exception):
            await adapter.close()


# ---------------------------
# Onchain Ingest (POST)
# ---------------------------
@router.post("/onchain/{source}")
async def ingest_onchain(source: str, body: OnchainIngestRequest):
    with ingest_span("onchain") as span:
        try:
            source_l = source.lower()
            if source_l == "glassnode":
                df = await _resolve(fetch_glassnode(body.symbol, body.metric, days=body.days))
                try:
                    df = _ensure_utc_col(df, "timestamp")
                except Exception as norm_err:
                    logger.warning("UTC normalize failed for onchain/glassnode; falling back via epoch: %s", norm_err)
                    df = _force_epoch_then_utc(df, "timestamp")
                base = getattr(settings, "ONCHAIN_PATH", "/app/data_lake/onchain") + "/glassnode"
                partitions = {
                    "symbol": body.symbol or "",
                    "metric": body.metric or "",
                    "year": df["timestamp"].dt.year.iloc[0] if (
                        df is not None and not df.empty and "timestamp" in df.columns
                    ) else None,
                    "month": df["timestamp"].dt.month.iloc[0] if (
                        df is not None and not df.empty and "timestamp" in df.columns
                    ) else None,
                    "day": df["timestamp"].dt.day.iloc[0] if (
                        df is not None and not df.empty and "timestamp" in df.columns
                    ) else None,
                }
            elif source_l == "covalent":
                df = await _resolve(fetch_covalent(chain_id=body.chain_id, address=body.address))
                try:
                    df = _ensure_utc_col(df, "timestamp")
                except Exception as norm_err:
                    logger.warning("UTC normalize failed for onchain/covalent; falling back via epoch: %s", norm_err)
                    df = _force_epoch_then_utc(df, "timestamp")
                base = getattr(settings, "ONCHAIN_PATH", "/app/data_lake/onchain") + "/covalent"
                partitions = {
                    "chain_id": body.chain_id,
                    "address": body.address or "",
                    "year": df["timestamp"].dt.year.iloc[0] if (
                        df is not None and not df.empty and "timestamp" in df.columns
                    ) else None,
                    "month": df["timestamp"].dt.month.iloc[0] if (
                        df is not None and not df.empty and "timestamp" in df.columns
                    ) else None,
                    "day": df["timestamp"].dt.day.iloc[0] if (
                        df is not None and not df.empty and "timestamp" in df.columns
                    ) else None,
                }
            else:
                span.set_status("error")
                raise HTTPException(status_code=400, detail=f"Unknown onchain source: {source}")

            try:
                path = write_to_parquet(df, base, partitions)
            except ValueError as ve:
                span.set_status("error")
                raise HTTPException(status_code=422, detail=str(ve))
            except Exception as e:
                span.set_status("error")
                raise HTTPException(status_code=500, detail=f"Write failed: {e}")

            features_written = 0
            if df is not None and not df.empty:
                try:
                    features_written = await _write_onchain_features_to_store(df)
                except Exception as e:
                    logger.warning("On-chain feature write failed", exc_info=e)
                    features_written = 0

            status = "ok" if path is not None else "no_data"
            span.set_status(status)
            return {"status": status, "path": None if path is None else str(path), "features_written": features_written}

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("ingest_onchain failed")
            span.set_status("error")
            raise HTTPException(status_code=500, detail=f"ingest_onchain failed: {e}")


# ---------------------------
# Social Ingest (POST)
# ---------------------------
@router.post("/social/{platform}")
async def ingest_social(platform: str, body: SocialIngestRequest):
    with ingest_span("social") as span:
        try:
            platform_l = platform.lower()

            # ---- Normalize time window ----
            now = datetime.now(timezone.utc)
            until = body.until or now
            since = body.since or (until - timedelta(hours=24))
            if since >= until:
                span.set_status("error")
                raise HTTPException(status_code=422, detail="'since' must be earlier than 'until'")

            # ---- Fetch via YOUR adapters ----
            try:
                if platform_l == "twitter":
                    df = await _resolve(fetch_twitter_sentiment(body.query, since, until, body.max_results))
                    try:
                        df = _ensure_utc_col(df, "ts")
                    except Exception as norm_err:
                        logger.warning("UTC normalize failed for social/twitter; falling back via epoch: %s", norm_err)
                        df = _force_epoch_then_utc(df, "ts")
                    base = getattr(settings, "SOCIAL_PATH", "/app/data_lake/social") + "/twitter"
                elif platform_l == "reddit":
                    df = await _resolve(fetch_reddit_api(body.query, since, until, body.max_results))
                    try:
                        df = _ensure_utc_col(df, "ts")
                    except Exception as norm_err:
                        logger.warning("UTC normalize failed for social/reddit; falling back via epoch: %s", norm_err)
                        df = _force_epoch_then_utc(df, "ts")
                    base = getattr(settings, "SOCIAL_PATH", "/app/data_lake/social") + "/reddit"
                else:
                    span.set_status("error")
                    raise HTTPException(status_code=400, detail=f"Unknown social platform: {platform}")
            except Exception as e:
                span.set_status("error")
                raise HTTPException(status_code=502, detail=f"{platform_l} fetch failed: {e}")

            # ---- Light normalization passthrough ----
            if df is not None and not df.empty:
                df["source"] = platform_l
                if "author" in df.columns and "user" not in df.columns:
                    df["user"] = df["author"]
                if "text" not in df.columns:
                    if "content" in df.columns:
                        df["text"] = df["content"]
                    elif "selftext" in df.columns:
                        df["text"] = df["selftext"]
                if "sentiment_score" not in df.columns:
                    df["sentiment_score"] = None
                # Optional: ML enrichment
                try:
                    if settings.ML_SENTIMENT_ENABLED and settings.SOCIAL_SENTIMENT_ENRICH:
                        texts = [str(t) if t is not None else "" for t in df.get("text", [])]
                        if texts:
                            preds = await ml_utils.predict(texts)
                            if preds and len(preds) == len(df):
                                df["sentiment_label"] = [p.get("label") for p in preds]
                                df["sentiment_score"] = [float(p.get("score_signed", 0.0)) for p in preds]
                except Exception as e:
                    logger.warning("Sentiment enrichment failed: %s", e)

            partitions = {
                "query": body.query.replace(" ", "_"),
                "year": df["ts"].dt.year.iloc[0] if (df is not None and not df.empty and "ts" in df.columns) else None,
                "month": df["ts"].dt.month.iloc[0] if (df is not None and not df.empty and "ts" in df.columns) else None,
                "day": df["ts"].dt.day.iloc[0] if (df is not None and not df.empty and "ts" in df.columns) else None,
            }

            try:
                path = write_to_parquet(df, base, partitions)
            except ValueError as ve:
                span.set_status("error")
                raise HTTPException(status_code=422, detail=str(ve))
            except Exception as e:
                span.set_status("error")
                raise HTTPException(status_code=500, detail=f"Write failed: {e}")

            features_written = 0
            if df is not None and not df.empty:
                try:
                    features_written = await _write_social_features_to_store(df)
                except Exception as e:
                    logger.warning("Social feature write failed", exc_info=e)
                    features_written = 0

            status = "ok" if path is not None else "no_data"
            span.set_status(status)
            return {"status": status, "path": None if path is None else str(path), "features_written": features_written}

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("ingest_social failed")
            span.set_status("error")
            raise HTTPException(status_code=500, detail=f"ingest_social failed: {e}")


# ---------------------------
# News Ingest (POST)
# ---------------------------

@router.post("/news", tags=["ingest-news"])
async def ingest_news(req: NewsIngestRequest):
    """Ingest news content from the configured source and persist to the lake + feature store."""
    with ingest_span("news") as span:
        try:
            source_type = (req.source_type or "").lower().strip()
            if source_type not in ("api", "rss"):
                span.set_status("error")
                raise HTTPException(status_code=422, detail="source_type must be 'api' or 'rss'")

            base_root = getattr(settings, "NEWS_PATH", "/app/data_lake/news").rstrip("/")
            df: Optional[pd.DataFrame]
            base_path: str
            partitions_common = {}

            if source_type == "api":
                category = (req.category or "general").strip() or "general"
                try:
                    df = await fetch_news_api(source=category, limit=200)
                except Exception as e:
                    span.set_status("error")
                    raise HTTPException(status_code=502, detail=f"news api fetch failed: {e}")
                base_path = base_root + "/api"
                partitions_common["source"] = category.replace(" ", "_")
            else:
                if not req.feed_url:
                    span.set_status("error")
                    raise HTTPException(status_code=422, detail="feed_url is required when source_type='rss'")
                try:
                    df = await fetch_news_rss_once(req.feed_url, limit=500)
                except ValueError as e:
                    span.set_status("error")
                    raise HTTPException(status_code=400, detail=f"invalid rss feed_url: {e}")
                except Exception as e:
                    span.set_status("error")
                    raise HTTPException(status_code=502, detail=f"rss fetch failed: {e}")
                base_path = base_root + "/rss"
                host = urlparse(req.feed_url).netloc.replace(":", "-")
                partitions_common["source"] = host or "rss"

            df = df if df is not None else pd.DataFrame()

            if "dt" not in df.columns:
                add_dt_partition(df, ts_col="published_at")

            if not df.empty:
                df = df[df["dt"].notna() & (df["dt"] != "")].copy()

            dt_values = sorted([d for d in df.get("dt", pd.Series(dtype="string")).dropna().unique().tolist() if d])

            last_path: Optional[str] = None
            wrote_any = False

            if not dt_values and not df.empty:
                dt_values = [None]

            for dt_value in dt_values or []:
                subset = df if dt_value is None else df[df["dt"] == dt_value]
                subset = subset.copy()
                if subset.empty:
                    continue
                published = subset.get("published_at")
                partitions = dict(partitions_common)
                if published is not None and not published.empty:
                    try:
                        ts0 = pd.to_datetime(published.iloc[0], utc=True)
                        if pd.notna(ts0):
                            partitions.setdefault("year", int(ts0.year))
                            partitions.setdefault("month", int(ts0.month))
                            partitions.setdefault("day", int(ts0.day))
                    except Exception:
                        pass
                try:
                    path = write_to_parquet(subset, base_path, partitions)
                except ValueError as ve:
                    span.set_status("error")
                    raise HTTPException(status_code=422, detail=str(ve))
                except Exception as e:
                    span.set_status("error")
                    raise HTTPException(status_code=500, detail=f"Write failed: {e}")
                if path:
                    last_path = str(path)
                    wrote_any = True

            features_written = 0
            if not df.empty:
                try:
                    features_written = await _write_news_features_to_store(df)
                except Exception as e:
                    logger.warning("News feature write failed", exc_info=e)
                    features_written = 0

            status = "ok" if wrote_any else "no_data"
            span.set_status(status)
            return {"status": status, "path": last_path, "features_written": features_written}

        except HTTPException:
            raise
        except Exception as e:
            span.set_status("error")
            raise HTTPException(status_code=502, detail=f"news ingest failed: {e}")

# ---------------------------
# News Search (GET via NewsClient)
# ---------------------------

@router.get("/news/search")
async def get_news_articles(
    source: str,
    since: Optional[int] = None,
    until: Optional[int] = None,
    limit: int = 100,
    news: NewsClient = Depends(provide_news),
):
    """
    Fetch news articles via NewsClient (one-shot search). Returns {rows, data}.
    since/until are epoch seconds (UTC).
    """
    df = await news.get_crypto_news(
        since=_parse_epoch_sec(since),
        until=_parse_epoch_sec(until),
        source=source,
        limit=limit,
    )
    rows = 0 if df is None else len(df)
    data = [] if df is None or df.empty else df.to_dict(orient="records")
    return {"rows": rows, "data": data}

# ---------------------------
# News RSS WebSocket stream
# ---------------------------

@router.websocket("/ws/news/rss")
async def ws_news_rss(websocket: WebSocket, feed_url: str):
    await websocket.accept()
    news = NewsClient()
    try:
        async def _on_item(item: dict):
            await websocket.send_json(item)
        await news.stream_rss(feed_url, handle_update=_on_item)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        finally:
            await websocket.close()
    finally:
        await news.aclose()

# ---------------------------
# Onchain GET (clients)
# ---------------------------

@router.get("/onchain/glassnode")
async def get_glassnode_data(symbol: str, metric: str, days: int = 1):
    client = OnchainClient()
    try:
        df = await client.get_glassnode_metric(symbol=symbol, metric=metric, days=days)
        rows = 0 if df is None else len(df)
        data = [] if df is None or df.empty else df.to_dict(orient="records")
        return {"rows": rows, "data": data}
    finally:
        await client.aclose()

@router.get("/onchain/covalent")
async def get_covalent_balances(chain_id: int, address: str):
    client = OnchainClient()
    try:
        df = await client.get_covalent_balances(chain_id=chain_id, address=address)
        rows = 0 if df is None else len(df)
        data = [] if df is None or df.empty else df.to_dict(orient="records")
        return {"rows": rows, "data": data}
    finally:
        await client.aclose()

# ---------------------------
# Social GET (clients)
# ---------------------------

@router.get("/social/twitter")
async def get_twitter_data(
    query: str,
    since: Optional[int] = None,
    until: Optional[int] = None,
    limit: int = 100,
):
    client = SocialClient()
    try:
        df = await client.fetch_tweets(
            query,
            since=_parse_epoch_sec(since) if since is not None else None,
            until=_parse_epoch_sec(until) if until is not None else None,
            limit=limit,
        )
        rows = 0 if df is None else len(df)
        data = [] if df is None or df.empty else df.to_dict(orient="records")
        return {"rows": rows, "data": data}
    finally:
        await client.aclose()

@router.get("/social/reddit")
async def get_reddit_data(
    subreddit: str,
    since: Optional[int] = None,
    until: Optional[int] = None,
    limit: int = 100,
):
    client = SocialClient()
    try:
        df = await client.fetch_reddit_api(
            subreddit,
            since=_parse_epoch_sec(since) if since is not None else None,
            until=_parse_epoch_sec(until) if until is not None else None,
            limit=limit,
        )
        rows = 0 if df is None else len(df)
        data = [] if df is None or df.empty else df.to_dict(orient="records")
        return {"rows": rows, "data": data}
    finally:
        await client.aclose()

# ---------------------------
# Redis health + simple demo feature endpoints (sync)
# ---------------------------

@router.get("/health/redis")
def redis_health_check():
    try:
        if _redis.ping():
            return {"redis": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    raise HTTPException(status_code=503, detail="Redis ping failed")

@router.post("/features/write")
def write_features(body: dict = Body(...)):
    symbol = body.get("symbol")
    ts = body.get("timestamp")
    if not symbol or ts is None:
        raise HTTPException(status_code=422, detail="symbol and timestamp are required")
    key = f"features:{symbol}:{int(ts)}"
    # Store the raw body so we preserve the client's shape (expects 'features')
    _redis.set(key, json.dumps(body), ex=getattr(settings, "feature_ttl_seconds", 3600))
    return {"status": "ok", "key": key}

@router.get("/features/read")
def read_features(symbol: str = Query(...), timestamp: int = Query(...)):
    key = f"features:{symbol}:{timestamp}"
    raw = _redis.get(key)
    if not raw:
        raise HTTPException(status_code=404, detail="Feature vector not found")
    # Return the stored payload as-is (keeps 'features' field)
    try:
        return json.loads(raw)
    except Exception:
        # Fallback to pydantic decode if older payloads were stored
        try:
            vec = FeatureVector.model_validate_json(raw)
            return {"symbol": vec.symbol, "timestamp": vec.timestamp, "features": vec.payload}
        except Exception:
            raise HTTPException(status_code=500, detail="Stored feature payload is invalid")

# ---------------------------
# Feature Retrieval (async, via RedisFeatureStore)
# ---------------------------

@router.get("/features/market")
async def get_market_features(
    symbol: str,
    timeframe: str,
    ts: List[int] = Query(..., description="Repeat per epoch-second"),
    store: RedisFeatureStore = Depends(provide_store),
):
    """
    Retrieve market features from Redis by (symbol, timeframe, ts...).
    Returns JSON-safe payload with {rows, data}, where data = [{timestamp, ...payload}]
    """
    items = [("market", symbol, timeframe, t) for t in ts]
    vals = await store.batch_read(items)

    data: List[dict] = []
    for t, payload in zip(ts, vals):
        if payload is None:
            continue
        safe_payload = _clean_numbers(payload)
        data.append({"timestamp": int(t), **safe_payload})

    return {"rows": len(data), "data": data}

@router.get("/features/onchain")
async def get_onchain_features(
    symbol: str,
    metric: str,
    ts: List[int] = Query(..., description="Repeat per epoch-second"),
    store: RedisFeatureStore = Depends(provide_store),
):
    items = [("onchain", symbol, metric, t) for t in ts]
    vals = await store.batch_read(items)

    data: List[dict] = []
    for t, payload in zip(ts, vals):
        if payload is None:
            continue
        safe_payload = _clean_numbers(payload)
        data.append({"timestamp": int(t), **safe_payload})
    return {"rows": len(data), "data": data}


@router.get("/features/social")
async def get_social_features(
    topic: str = Query("twitter", description="Symbol/topic; default twitter"),
    timeframe: str = Query("1m"),
    ts: List[int] = Query(..., description="Repeat per epoch-second"),
    store: RedisFeatureStore = Depends(provide_store),
):
    items = [("social", topic, timeframe, t) for t in ts]
    vals = await store.batch_read(items)

    data: List[dict] = []
    for t, payload in zip(ts, vals):
        if payload is None:
            continue
        safe_payload = _clean_numbers(payload)
        data.append({"timestamp": int(t), **safe_payload})
    return {"rows": len(data), "data": data}

@router.get("/features/news")
async def get_news_features(
    topic: str = Query("news", description="Topic/source key; default 'news'"),
    timeframe: str = Query("1m"),
    ts: List[int] = Query(..., description="Repeat per epoch-second"),
    store: RedisFeatureStore = Depends(provide_store),
):
    """
    Retrieve news features from Redis by (topic, timeframe, ts...).
    Returns {rows, data} with JSON-safe payloads.
    """
    items = [("news", topic, timeframe, t) for t in ts]
    vals = await store.batch_read(items)

    data: List[dict] = []
    for t, payload in zip(ts, vals):
        if payload is None:
            continue
        safe_payload = _clean_numbers(payload)
        data.append({"timestamp": int(t), **safe_payload})

    return {"rows": len(data), "data": data}

# ---- Feature range retrieval (via ZSET index) ----

@router.get("/features/market/range")
async def get_market_features_range(
    symbol: str,
    timeframe: str,
    start: int,  # epoch seconds (inclusive)
    end: int,    # epoch seconds (inclusive)
    limit: int = 500,
    reverse: bool = False,
    store: RedisFeatureStore = Depends(get_store),
):
    rows = await store.range_read(
        domain="market", symbol=symbol, timeframe=timeframe,
        start=int(start), end=int(end), limit=limit, reverse=reverse,
    )
    safe = [_clean_numbers(r) for r in rows]
    return {"rows": len(safe), "data": safe}


@router.get("/features/onchain/range", name="features_onchain_range")
async def get_onchain_features_range(
    symbol: str,
    timeframe: str = Query("1d"),
    start: int = Query(..., description="epoch seconds"),
    end: int = Query(..., description="epoch seconds"),
    limit: int = 500,
    reverse: bool = False,
    store: RedisFeatureStore = Depends(provide_store),
):
    rows = await store.range_read(
        domain="onchain",
        symbol=symbol,
        timeframe=timeframe,
        start=int(start),
        end=int(end),
        limit=limit,
        reverse=reverse,
    )
    safe = [_clean_numbers(r) for r in rows]
    return {"rows": len(safe), "data": safe}

@router.get("/features/social/range", name="features_social_range")
async def get_social_features_range(
    topic: str = Query("twitter"),
    timeframe: str = Query("1m"),
    start: int = Query(..., description="epoch seconds"),
    end: int = Query(..., description="epoch seconds"),
    limit: int = 500,
    reverse: bool = False,
    store: RedisFeatureStore = Depends(provide_store),
):
    rows = await store.range_read(
        domain="social",
        symbol=topic,
        timeframe=timeframe,
        start=int(start),
        end=int(end),
        limit=limit,
        reverse=reverse,
    )
    safe = [_clean_numbers(r) for r in rows]
    return {"rows": len(safe), "data": safe}

@router.get("/features/news/range", name="features_news_range")
async def get_news_features_range(
    topic: str = Query("news"),
    timeframe: str = Query("1m"),
    start: int = Query(..., description="epoch seconds (inclusive)"),
    end: int = Query(..., description="epoch seconds (inclusive)"),
    limit: int = 500,
    reverse: bool = False,
    store: RedisFeatureStore = Depends(provide_store),
):
    rows = await store.range_read(
        domain="news",
        symbol=topic,
        timeframe=timeframe,
        start=int(start),
        end=int(end),
        limit=limit,
        reverse=reverse,
    )
    safe = [_clean_numbers(r) for r in rows]
    return {"rows": len(safe), "data": safe}

# ---------------------------
# Internal: build & write features
# ---------------------------

async def _write_market_features_to_store(
    df: pd.DataFrame,
    *,
    feats: Optional[pd.DataFrame] = None,
) -> int:
    """
    Build features from normalized OHLCV and write them to Redis.
    Returns number of feature rows written.
    """
    if df.empty:
        return 0

    feats = feats if feats is not None else build_market_features(df)
    feats = augment_market_features(feats, inplace=False)
    if feats.empty:
        return 0

    # choose compact payload set; extend as needed
    payload_cols = [
        c
        for c in [
            "ret_1",
            "rsi_14",
            "hl_spread",
            "hl_spread_z",
            "rvol_20",
            "sym_spread_ratio",
            "sym_rvol_ratio",
            "sym_liquidity_rank",
            "oi_obv",
        ]
        if c in feats.columns
    ]

    items = []
    for _, r in feats.iterrows():
        payload: Dict[str, Any] = {}
        for c in payload_cols:
            val = r.get(c)
            if pd.isna(val):
                payload[c] = None
            else:
                payload[c] = val.item() if hasattr(val, "item") else val
        if not payload:
            continue
        items.append({
            "domain": "market",
            "symbol": str(r["symbol"]),
            "timeframe": str(r["timeframe"]),
            "ts": r["timestamp"],  # tz-aware UTC
            "payload": payload,
        })

    if not items:
        return 0
    # inside _write_market_features_to_store(...)
    store = get_store()
    await store.batch_write(items)
    record_rows_written("market", len(items))
    return len(items)

async def _write_onchain_features_to_store(df: pd.DataFrame) -> int:
    """Write on-chain features to Redis under domain 'onchain'.
    Expects columns: ['timestamp','symbol','metric','value'] (timestamp tz-aware UTC).
    Returns number of rows written.
    """
    if df is None or df.empty:
        return 0
    required = {"timestamp", "symbol", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("onchain features missing columns: %s", sorted(missing))
        return 0

    items = []
    for _, r in df.iterrows():
        payload = {
            "metric": str(r["metric"]),
            "value": float(r["value"]) if pd.notna(r["value"]) else None,
        }
        items.append({
            "domain": "onchain",
            "symbol": str(r["symbol"]),
            "timeframe": "1d",  # <-- fixed bucket; metric stays in payload
            "ts": _epoch_s_from_ts(r["timestamp"]),
            "payload": payload,
        })

    if not items:
        return 0

    store = get_store()
    # inside _write_onchain_features_to_store(...)
    store = get_store()
    await store.batch_write(items)
    record_rows_written("onchain", len(items))
    return len(items)


async def _write_social_features_to_store(df: pd.DataFrame) -> int:
    """Write social features to Redis under domain 'social'.
    Expects columns: ['ts','user' or 'author','text' or ('content'/'selftext'),'sentiment_score'] with tz-aware 'ts'.
    Returns number of rows written.
    """
    if df is None or df.empty:
        return 0

    df = df.copy()
    if "user" not in df.columns and "author" in df.columns:
        df["user"] = df["author"]
    if "text" not in df.columns:
        if "content" in df.columns:
            df["text"] = df["content"]
        elif "selftext" in df.columns:
            df["text"] = df["selftext"]

    required = {"ts", "user", "text", "sentiment_score"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("social features missing columns: %s", sorted(missing))
        return 0

    items = []
    for _, r in df.iterrows():
        payload = {
            "user": None if pd.isna(r.get("user")) else str(r.get("user")),
            "text": None if pd.isna(r.get("text")) else str(r.get("text")),
            "sentiment": None if pd.isna(r.get("sentiment_score")) else float(r.get("sentiment_score")),
        }
        items.append({
            "domain": "social",
            "symbol": str(r.get("symbol", r.get("topic", "twitter"))),
            "timeframe": str(r.get("timeframe", "1m")),
            "ts": _epoch_s_from_ts(r["ts"]),  # <-- was r["ts"]
            "payload": payload,
        })
    if not items:
        return 0

    # inside _write_social_features_to_store(...)
    store = get_store()
    await store.batch_write(items)
    record_rows_written("social", len(items))
    return len(items)

async def _write_news_features_to_store(df: pd.DataFrame) -> int:
    """
    Write news features to Redis under domain 'news'.
    Expected columns (tolerant):
      - timestamp: 'ts' (preferred) or 'published_at' (string/datetime)
      - identity: 'topic' or 'source' (used as symbol; default 'news')
      - payload: 'title', 'text' or 'summary', 'url', optional 'sentiment_score'
    """
    if df is None or df.empty:
        return 0

    df = df.copy()

    # Normalize timestamp column -> 'ts' tz-aware UTC
    if "ts" not in df.columns:
        if "published_at" in df.columns:
            df["ts"] = df["published_at"]
        else:
            # nothing to write without a timestamp
            logger.warning("news features missing 'ts'/'published_at'; nothing written")
            return 0

    try:
        df = _ensure_utc_col(df, "ts")
    except Exception as norm_err:
        logger.warning("UTC normalize failed for news; falling back via epoch: %s", norm_err)
        df = _force_epoch_then_utc(df, "ts")

    # Derive symbol/topic and payload text fields
    if "topic" not in df.columns:
        if "source" in df.columns:
            df["topic"] = df["source"]
        else:
            df["topic"] = "news"

    if "text" not in df.columns:
        if "summary" in df.columns:
            df["text"] = df["summary"]
        elif "content" in df.columns:
            df["text"] = df["content"]

    items = []
    for _, r in df.iterrows():
        payload = {
            "title": None if pd.isna(r.get("title")) else str(r.get("title")),
            "text": None if pd.isna(r.get("text")) else str(r.get("text")),
            "url": None if pd.isna(r.get("url")) else str(r.get("url")),
            "sentiment": None if pd.isna(r.get("sentiment_score")) else float(r.get("sentiment_score")),
            "source": None if pd.isna(r.get("source")) else str(r.get("source")),
        }

        items.append({
            "domain": "news",
            "symbol": str(r.get("topic") or "news"),
            "timeframe": str(r.get("timeframe", "1m")),
            "ts": _epoch_s_from_ts(r["ts"]),
            "payload": payload,
        })

    if not items:
        return 0

    store = get_store()
    await store.batch_write(items)
    record_rows_written("news", len(items))
    return len(items)

# --- Admin guard via header token ---
_admin_hdr = APIKeyHeader(name="X-Admin-Token", auto_error=False)
_api_key_hdr = APIKeyHeader(name="X-API-Key", auto_error=False)
_auth_hdr = APIKeyHeader(name="Authorization", auto_error=False)

def require_admin(
    admin_token: Optional[str] = Security(_admin_hdr),
    api_key: Optional[str] = Security(_api_key_hdr),
    auth: Optional[str] = Security(_auth_hdr),
):
    expected = getattr(settings, "ADMIN_TOKEN", None)
    # Enforce token presence in non-dev environments
    if not expected:
        raise HTTPException(status_code=503, detail="Admin token not configured")

    # Determine provided token from supported headers
    provided = admin_token or api_key
    if not provided and auth:
        parts = auth.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            provided = parts[1].strip()

    if provided == expected:
        return True
    raise HTTPException(status_code=401, detail="Admin token required")

# Admin sub-router protected globally
admin = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])

@admin.post("/backfill/market")
async def admin_backfill_market(
    exchange: str = "binance",
    symbol: str = Query(..., description="e.g. BTC/USDT"),
    timeframe: str = Query(..., description="e.g. 1m"),
    lookback_minutes: int = Query(120, ge=1, le=7*24*60),
):
    """
    One-shot backfill for (exchange, symbol, timeframe) over the last N minutes.
    """
    result = await backfill_market_once(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        lookback_minutes=lookback_minutes,
    )
    return result


@admin.post("/features/ttl-sweep")
async def admin_ttl_sweep(
    pattern: str = Query("features:market:*"),
    ttl_default: Optional[int] = Query(None, description="Seconds to apply when no TTL is set"),
    max_keys: Optional[int] = Query(None, description="Stop after scanning this many keys"),
):
    """
    One-shot TTL sweep: ensures keys under pattern have expirations.
    """
    result = await ttl_sweep_once(pattern=pattern, ttl_default=ttl_default, max_keys=max_keys)
    return result

router.include_router(admin)

# ---------------------------
# Storage check (admin)
# ---------------------------

@admin.get("/storage/check")
async def storage_check(
    probe: bool = Query(False, description="Write a temporary probe file and remove it"),
    domain: Optional[str] = Query(None, description="market|onchain|social|news; if omitted, checks all"),
    cleanup: bool = Query(True, description="Remove the probe file after writing"),
):
    """
    Resolve fsspec filesystem(s) for configured data lake paths and optionally
    perform a write/delete probe to validate credentials/connectivity.
    """
    import json as _json

    storage_options = {}
    if getattr(settings, "FSSPEC_STORAGE_OPTIONS", None):
        try:
            storage_options = _json.loads(settings.FSSPEC_STORAGE_OPTIONS)
        except Exception:
            storage_options = {}

    paths = {
        "market": getattr(settings, "market_path", "data_lake/market"),
        "onchain": getattr(settings, "onchain_path", "data_lake/onchain"),
        "social": getattr(settings, "social_path", "data_lake/social"),
        "news": getattr(settings, "news_path", "data_lake/news"),
    }
    if domain:
        domain = domain.lower()
        if domain not in paths:
            raise HTTPException(status_code=422, detail="Invalid domain")
        items = [(domain, paths[domain])]
    else:
        items = list(paths.items())

    out = {"checks": []}
    for dom, base in items:
        try:
            fs, root = fsspec.core.url_to_fs(base, **storage_options)
            proto = getattr(fs, "protocol", None)
            if isinstance(proto, (list, tuple)):
                proto = "+".join(proto)
            info = {"domain": dom, "path": base, "protocol": proto or "unknown", "ok": True}
            if probe:
                fname = f"_probe_{int(time.time()*1000)}.txt"
                p = os.path.join(root, fname)
                try:
                    with fs.open(p, "wb") as f:
                        f.write(b"ok")
                    if cleanup:
                        fs.rm(p)
                    info["probe"] = {"wrote": True, "removed": bool(cleanup)}
                except Exception as e:
                    info["probe"] = {"wrote": False, "error": str(e)}
                    info["ok"] = False
            out["checks"].append(info)
        except Exception as e:
            out["checks"].append({"domain": dom, "path": base, "ok": False, "error": str(e)})

    return out
