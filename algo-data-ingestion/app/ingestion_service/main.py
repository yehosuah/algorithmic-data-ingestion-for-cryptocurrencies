from fastapi import FastAPI, Request
import asyncio
from prometheus_client import make_asgi_app
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
from contextlib import asynccontextmanager
import os
import redis.asyncio as redis

from app.ingestion_service.config import settings
from app.ingestion_service.manifests import get_manifest_registry, parse_model_specs
from app.features.jobs.backfill import backfill_market_once, ttl_sweep_once
from app.common.async_infra import get_http, close_http
from app.features.ingestion.news_client import NewsClient
from app.features.ingestion.ccxt_client import CCXTClient
from app.features.ingestion.social_client import SocialClient
from app.features.ingestion.onchain_client import OnchainClient
from app.features.store import redis_store as _feature_store  # noqa: F401
from app.ingestion_service import ml_utils
from prometheus_client import Gauge
from app.ingestion_service.utils import _METRICS_REGISTRY

# Ensure redis_store gets imported so its Counter/Histogram register with our custom registry
try:
    from app.features.store import redis_store as _feature_store  # noqa: F401
except Exception as e:
    import logging
    logging.warning("Feature store import failed (metrics may be missing): %s", e)

# Optional: one “always present” metric so /metrics is never empty
SERVICE_INFO = Gauge(
    "service_info",
    "Service metadata",
    labelnames=("service", "version"),
    registry=_METRICS_REGISTRY,
)
SERVICE_INFO.labels(service="raw-data-ingestion", version="1.0.0").set(1)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- PROMETHEUS: register default collectors into our custom registry ---  # NEW
    try:
        # These classes exist in recent prometheus_client versions
        from prometheus_client import GCCollector, ProcessCollector, PlatformCollector  # NEW
        GCCollector(registry=_METRICS_REGISTRY)                                       # NEW
        ProcessCollector(registry=_METRICS_REGISTRY)                                  # NEW
        PlatformCollector(registry=_METRICS_REGISTRY)                                 # NEW
    except Exception:
        # Older client fallback (optional) – ok to ignore if not available
        pass

    # --- PROMETHEUS: eager-import feature metrics so they’re registered now --- # NEW
    from app.features.store import redis_store as _fs                                  # NEW
    # Touch the counters so the HELP/TYPE and metric families appear immediately     # NEW
    for dom in ("bootstrap", "market"):                                               # NEW
        _fs.FEATURE_WRITES_TOTAL.labels(domain=dom).inc(0)                            # NEW
        _fs.FEATURE_READS_TOTAL.labels(domain=dom).inc(0)                             # NEW
        _fs.FEATURE_HITS_TOTAL.labels(domain=dom).inc(0)                              # NEW
        _fs.FEATURE_MISSES_TOTAL.labels(domain=dom).inc(0)                            # NEW
    # --------------------------------------------------------------------------  # NEW
    app.state.http = get_http()                         # NEW shared AsyncClient
    app.state.redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))  # NEW shared Redis
    # Create shared NewsClient instance
    app.state.news_client = NewsClient()
    # Create shared CCXT client (async)
    app.state.ccxt_client = CCXTClient(
        exchange_name=getattr(settings, "exchange_name", "binance"),
        api_key=getattr(settings, "exchange_api_key", None),
        secret=getattr(settings, "exchange_api_secret", None),
    )
    app.state.onchain_client = OnchainClient()
    app.state.social_client = SocialClient()
    # Preload manifest artifacts for deployable models (if configured)
    try:
        manifest_specs = parse_model_specs(getattr(settings, "DEPLOYABLE_MODELS", None))
        if manifest_specs:
            models_root = Path(getattr(settings, "MODELS_ROOT", "models")).expanduser().resolve()
            registry = get_manifest_registry()
            loaded_specs = registry.preload(models_root=models_root, specs=manifest_specs, clear=True)
            app.state.manifest_registry = registry
            loaded_labels = ", ".join(spec.label for spec in loaded_specs)
            logging.info("Preloaded %d deployable manifest(s): %s", len(loaded_specs), loaded_labels)
        else:
            logging.info("No deployable models configured; manifest preload skipped")
    except Exception as exc:
        logging.error("Failed to preload manifest artifacts: %s", exc, exc_info=True)
        raise
    try:
        # Optionally warm ML in background (avoid blocking startup)
        if settings.ML_SENTIMENT_ENABLED:
            logging.info("ML sentiment enabled; pipeline will load on first use")
        yield
    finally:
        # Graceful shutdown
        await app.state.redis.close()
        await app.state.social_client.aclose()
        await app.state.onchain_client.aclose()
        await app.state.ccxt_client.aclose()
        await app.state.news_client.aclose()
        await close_http()
        try:
            ml_utils.shutdown()
        except Exception:
            pass

app = FastAPI(lifespan=lifespan)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials = True,
)

# Expose Prometheus metrics at /metrics
app.mount("/metrics", make_asgi_app(registry=_METRICS_REGISTRY))

@app.get("/")
async def root():
    return {"service": "raw-data-ingestion", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.on_event("startup")
async def _start_jobs():
    app.state._bg_tasks = []

    if settings.BACKFILL_ENABLED:
        async def _backfill_loop():
            while True:
                try:
                    for sym in [s.strip() for s in settings.BACKFILL_SYMBOLS.split(",") if s.strip()]:
                        for tf in [t.strip() for t in settings.BACKFILL_TIMEFRAMES.split(",") if t.strip()]:
                            await backfill_market_once(
                                exchange=settings.BACKFILL_EXCHANGE,
                                symbol=sym,
                                timeframe=tf,
                                lookback_minutes=settings.BACKFILL_LOOKBACK_MIN,
                            )
                except Exception as e:
                    logger.warning("Backfill loop error: %s", e, exc_info=True)
                await asyncio.sleep(settings.BACKFILL_INTERVAL_SEC)

        app.state._bg_tasks.append(asyncio.create_task(_backfill_loop()))

    if settings.TTL_SWEEP_ENABLED:
        async def _ttl_loop():
            while True:
                try:
                    await ttl_sweep_once(
                        pattern="features:market:*",
                        ttl_default=getattr(settings, "FEATURE_TTL_SEC", None),
                        max_keys=None,
                    )
                except Exception as e:
                    logger.warning("TTL sweep loop error: %s", e, exc_info=True)
                await asyncio.sleep(settings.TTL_SWEEP_INTERVAL_SEC)

        app.state._bg_tasks.append(asyncio.create_task(_ttl_loop()))

@app.on_event("shutdown")
async def _stop_jobs():
    tasks = getattr(app.state, "_bg_tasks", [])
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

@app.on_event("startup")
async def on_startup():
    logging.info("Starting Raw Data Ingestion Service")

@app.on_event("shutdown")
async def on_shutdown():
    logging.info("Shutting down Raw Data Ingestion Service")

def get_news(request: Request) -> NewsClient:
    return request.app.state.news_client

def get_ccxt(request: Request) -> CCXTClient:
    return request.app.state.ccxt_client

def get_onchain(request: Request) -> OnchainClient:
    return request.app.state.onchain_client

def get_social(request: Request) -> SocialClient:
    return request.app.state.social_client

def get_http_dep(request: Request):
    return request.app.state.http

def get_redis_dep(request: Request):
    return request.app.state.redis

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.ingest_host,
        port=settings.ingest_port,
        reload=True
    )

from .routes import router
app.include_router(router, prefix="/ingest")
if settings.ML_SENTIMENT_ENABLED:
    try:
        from .ml_routes import router as ml_router
        app.include_router(ml_router)
    except Exception as e:
        logging.warning("ML routes not mounted: %s", e)
