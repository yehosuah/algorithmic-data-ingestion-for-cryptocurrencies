# Algo Data Ingestion (Docker App)

End-to-end ingestion for **market**, **on-chain**, **social**, and **news** data with a Redis feature store, admin backfills/TTL sweeps, a scheduler, and monitoring (Prometheus + Grafana).

- **API:** FastAPI (`ingestion-api`)
- **Store:** Redis (+ redis_exporter)
- **Scheduler:** APScheduler job runner (calls admin endpoints)
- **Monitoring:** Prometheus (scrapes API, scheduler, Redis), Grafana dashboards
- **Parquet sink:** Data lake under `/app/data_lake/...`

---

## TL;DR

1) Copy env and set a token:
```bash
cp .env.example .env
# edit .env -> set ADMIN_TOKEN to a strong random string
```

2) Boot (build lean image without heavy ML deps):
```bash
docker compose up -d --build
```

3) Quick checks:
- API health:
  ```bash
  curl -s http://localhost:8000/health
  ```
- API metrics:
  ```bash
  curl -s http://localhost:8000/metrics | head
  ```
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default: admin/admin)

> Notes
> - La imagen se construye “lean” por defecto (sin instalar `torch`/`transformers`) para acelerar el build. Si necesitas ML dentro del contenedor, puedes construir con `--build-arg INSTALL_ML=1`.
> - Sin llaves API, social/news y on-chain pueden devolver `no_data` o errores 401 del proveedor. Con llaves, funcionará scraping real.

---

## Services (docker-compose)

- **ingestion-api** – FastAPI app exposing ingest, features, admin; exposes `/health` and `/metrics`.
- **redis** – Feature store.
- **redis-exporter** – Redis metrics for Prometheus.
- **scheduler** – Calls admin endpoints on cron (market backfill, TTL sweep); exposes `/metrics` on port 9002.
- **prometheus** – Scrapes API, scheduler, redis-exporter.
- **grafana** – Dashboards in `monitoring/grafana/dashboards`.

---

## Environment

Create `.env` at the repo root.

### `.env.example`
```bash
# Core
ADMIN_TOKEN=change_me_please

# API host/port
INGEST_HOST=0.0.0.0
INGEST_PORT=8000

# CORS (JSON list)
CORS_ORIGINS=["*"]

# Exchange config (optional)
EXCHANGE_NAME=binance
EXCHANGE_API_KEY=
EXCHANGE_API_SECRET=

# Feature TTL in seconds (optional; admin TTL sweep can enforce)
FEATURE_TTL_SEC=3600

# Background backfill loop (inside ingestion-api; optional)
BACKFILL_ENABLED=0
BACKFILL_EXCHANGE=binance
BACKFILL_SYMBOLS=BTC/USDT
BACKFILL_TIMEFRAMES=1m
BACKFILL_LOOKBACK_MIN=15
BACKFILL_INTERVAL_SEC=300

# Background TTL sweep loop (inside ingestion-api; optional)
TTL_SWEEP_ENABLED=0
TTL_SWEEP_INTERVAL_SEC=900

# Scheduler
API_BASE_URL=http://ingestion-api:8000
RUN_ON_START=1
SCHED_TZ=UTC
SCHED_METRICS_PORT=9002
MARKET_JOBS=[{"exchange":"binance","symbol":"BTC/USDT","timeframe":"1m","lookback_minutes":15,"cron":"*/5 * * * *"}]
TTL_SWEEP_CRON=*/15 * * * *
TTL_SWEEP_PATTERN=features:market:*
TTL_SWEEP_TTL=3600

# External keys (Phase 3)
GLASSNODE_API_KEY=
TWITTER_BEARER_TOKEN=
NEWS_API_KEY=
```

---

## API Surface

### Service health & docs
- `GET /health` → `{"status":"ok"}`
- `GET /metrics` → Prometheus exposition
- `GET /openapi.json` / `GET /docs`

### Ingest
- `POST /ingest/market/{exchange}`  
  **Body:** `{"symbol":"BTC/USDT","granularity":"1m","limit":100}`  
  Writes OHLCV features into Redis and Parquet.
- `POST /ingest/onchain/{source}` (e.g. `glassnode`, `covalent`)  
  **Body (examples):**  
  `{"symbol":"BTC","metric":"active_addresses","days":1}`  
  `{"chain_id":1,"address":"0x0000000000000000000000000000000000000000"}`
- `POST /ingest/social/{platform}` (e.g. `twitter`)  
  **Body:** `{"query":"bitcoin","since":null,"until":null,"max_results":5}`
- `POST /ingest/news/{source}` (e.g. `newsapi`, `rss`)  
  **Body:**  
  API: `{"source_type":"api","category":"business"}`  
  RSS: `{"source_type":"rss","feed_url":"https://..."}`

**Typical success:**
```json
{"status":"ok","path":"/app/data_lake/.../part-*.parquet","features_written":123}
```

**No keys / no results:**
```json
{"status":"no_data","path":null,"features_written":0}
```

**Upstream error (example):**
```json
{"detail":"ingest_onchain failed: Client error '401 Unauthorized' for url '...'"}
```

### Feature retrieval (point lookups)
- `GET /ingest/features/market`  
  `symbol=BTC/USDT&timeframe=1m&ts=1724140800&ts=...`
- `GET /ingest/features/onchain`  
  `symbol=BTC&metric=active_addresses&ts=...`
- `GET /ingest/features/social`  
  `topic=twitter&timeframe=1m&ts=...`

### Feature retrieval (range)
- `GET /ingest/features/market/range`  
  `symbol=BTC/USDT&timeframe=1m&start={epoch}&end={epoch}&limit=100`
- `GET /ingest/features/social/range`  
  `topic=twitter&timeframe=1m&start={epoch}&end={epoch}&limit=100`
- `GET /ingest/features/onchain/range`  
  `symbol=BTC&metric=active_addresses&start={epoch}&end={epoch}&limit=100`

### Admin (requires `X-Admin-Token`)
- `POST /ingest/admin/backfill/market`  
  `exchange=binance&symbol=BTC/USDT&timeframe=1m&lookback_minutes=15`  
  → `{"symbol":"BTC/USDT","timeframe":"1m","exchange":"binance","expected":N,"missing_before":M,"written":W}`
- `POST /ingest/admin/features/ttl-sweep`  
  `pattern=features:market:*&ttl_default=3600&max_keys=1000`  
  → `{"pattern":"...","scanned":N,"ttl_set":M}`

---

## Feature Storage Layout (Redis)

- **Per-point keys:**  
  `features:{domain}:{id_or_symbol}:{timeframe}:{ts_epoch}`
  - `features:market:BTC-USDT:1m:1724140800`
  - `features:social:twitter:1m:1724140800`
  - `features:onchain:BTC:active_addresses:1724100000`
- **Index keys:**  
  `features:{domain}:{id_or_symbol}:{timeframe}:_idx`  
  Used for range queries and TTL sweep traversal.

> TTL can be enforced with `FEATURE_TTL_SEC` or via the admin TTL sweep’s `ttl_default`.

---

## Scheduler

**Behavior**
- On boot (when `RUN_ON_START=1`): executes one backfill per `MARKET_JOBS` and a TTL sweep.
- On schedule: runs backfills via cron in `MARKET_JOBS` and TTL sweeps via `TTL_SWEEP_CRON`.

**Env (key vars)**
- `API_BASE_URL` (default internal: `http://ingestion-api:8000`)
- `RUN_ON_START`, `SCHED_TZ`, `SCHED_METRICS_PORT`
- `MARKET_JOBS` e.g.  
  `[{"exchange":"binance","symbol":"BTC/USDT","timeframe":"1m","lookback_minutes":15,"cron":"*/5 * * * *"}]`
- `TTL_SWEEP_CRON`, `TTL_SWEEP_PATTERN`, `TTL_SWEEP_TTL`

**Metrics**
- Exposed on port `9002` (host-mapped in compose):  
  ```bash
  curl -s http://localhost:9002/metrics | head
  ```

**Manual smoke (inside scheduler container)**
```bash
docker compose exec scheduler sh -lc 'curl -s http://ingestion-api:8000/health'

docker compose exec scheduler sh -lc \
  'curl -s -X POST -H "X-Admin-Token: $ADMIN_TOKEN" \
  "http://ingestion-api:8000/ingest/admin/backfill/market?exchange=binance&symbol=BTC%2FUSDT&timeframe=1m&lookback_minutes=1"'

docker compose exec scheduler sh -lc \
  'curl -s -X POST -H "X-Admin-Token: $ADMIN_TOKEN" \
  "http://ingestion-api:8000/ingest/admin/features/ttl-sweep?pattern=features%3Amarket%3A%2A&ttl_default=3600&max_keys=25"'
```

---

## Monitoring

### Prometheus
- Scrapes:
  - `ingestion-api` → `http://ingestion-api:8000/metrics`
  - `scheduler` → `http://scheduler:9002/metrics`
  - `redis-exporter` → `http://redis-exporter:9121/metrics`
- Config: `monitoring/prometheus/prometheus.yml`

Sanity:
```bash
docker compose exec prometheus sh -lc \
  'wget -qO- http://localhost:9090/api/v1/targets | jq ".data.activeTargets[].labels.job" | sort -u'
```

### Grafana
- Access: http://localhost:3000 (admin/admin on first run)
- Datasource: Prometheus at `http://prometheus:9090`
- Import dashboards from: `monitoring/grafana/dashboards/*.json`

Common metrics:
- API: `service_info{service="raw-data-ingestion"}` + feature store counters/histograms
- Scheduler: APScheduler & process metrics on `:9002`
- Redis: via redis_exporter

---

## Example Calls (host → API)

**Health & OpenAPI**
```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/openapi.json | jq '.paths | keys'
```

**Market ingest**
```bash
curl -s -X POST "http://localhost:8000/ingest/market/binance" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC/USDT","granularity":"1m","limit":2}'
```

**On-chain (Glassnode)**
```bash
curl -s -X POST "http://localhost:8000/ingest/onchain/glassnode" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC","metric":"active_addresses","days":1}'
```

**Social (Twitter)**
```bash
curl -s -X POST "http://localhost:8000/ingest/social/twitter" \
  -H "Content-Type: application/json" \
  -d '{"query":"bitcoin","max_results":5}'
```

**News**
```bash
curl -s -X POST "http://localhost:8000/ingest/news/newsapi" \
  -H "Content-Type: application/json" \
  -d '{"source_type":"api","category":"business"}'
```

**Admin: backfill & TTL sweep**
```bash
curl -s -X POST -H "X-Admin-Token: $ADMIN_TOKEN" \
  "http://localhost:8000/ingest/admin/backfill/market?exchange=binance&symbol=BTC%2FUSDT&timeframe=1m&lookback_minutes=15"

curl -s -X POST -H "X-Admin-Token: $ADMIN_TOKEN" \
  "http://localhost:8000/ingest/admin/features/ttl-sweep?pattern=features%3Amarket%3A%2A&ttl_default=3600&max_keys=100"
```

**Feature retrieval (range)**
```bash
# Market
curl -s "http://localhost:8000/ingest/features/market/range?symbol=BTC/USDT&timeframe=1m&start=1724140800&end=1724142000&limit=10"

# Social
curl -s "http://localhost:8000/ingest/features/social/range?topic=twitter&timeframe=1m&start=1724140800&end=1724142000&limit=10"

# Onchain
curl -s "http://localhost:8000/ingest/features/onchain/range?symbol=BTC&metric=active_addresses&start=1724140800&end=1724142000&limit=10"
```

---

## Tests

Phase 2 includes e2e tests for market/onchain/social ingest and feature range endpoints. News tests are deferred until API keys are available.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

---

## Troubleshooting

- **Scheduler “All connection attempts failed” on boot** – Likely a race; scheduler retries. Ensure `API_BASE_URL=http://ingestion-api:8000`.
- **404 on admin** – Use `/ingest/admin/...` prefix and include `X-Admin-Token`.
- **401 from Glassnode** – Expected without `GLASSNODE_API_KEY`. Social/news may return `no_data` when unauthenticated.
- **Weird `jq`/zsh errors** – Avoid inline comments in multi-line commands; prefer one command per line. Use `\curl` to bypass aliases.

---

## Directory Map

- `app/ingestion_service/main.py` – FastAPI app (health, metrics mount, lifespan).
- `app/ingestion_service/routes.py` – Ingest + features + admin endpoints.
- `app/ingestion_service/schemas.py` – Request/response models.
- `app/ingestion_service/utils.py` – Metrics registry and helpers.
- `app/features/store/redis_store.py` – Feature store + Prometheus metrics.
- `app/features/ingestion/*_client.py` – External clients (market/onchain/social/news).
- `app/features/jobs/backfill.py` – Admin backfill & TTL sweep.
- `app/scheduler/main.py` – APScheduler runner.
- `monitoring/prometheus/prometheus.yml` – Prometheus scrape config.
- `monitoring/grafana/dashboards/*.json` – Grafana dashboards.

---

## Security

Admin endpoints require `X-Admin-Token`. Keep `.env` out of version control, use a strong random token, and rotate if shared across environments.

---

## Roadmap to Phase 3

- Plug in real keys (Glassnode, Twitter/X, News API), enable production fetches.
- Add tests for the news route.
- Expand scheduler jobs to on-chain/social/news once keys are present.
- Optional: retention policies (TTL per domain), richer Grafana dashboards.

---

## Data Scraping (via Docker Compose)

El archivo `docker-compose.yml` ya incluye un servicio `scheduler` con trabajos de backfill configurados para:

- `binance BTC/USDT 1m` cada 5 minutos (lookback 60 min)
- `binance BTC/USDT 5m` cada 15 minutos (lookback 360 min)
- `binance ETH/USDT 1m` cada 5 minutos (lookback 60 min)

Además, un trabajo de TTL sweep cada 15 minutos para aplicar expiraciones en Redis (`TTL_SWEEP_*`).

Cómo corre el scraping:
- Al iniciar el stack (`docker compose up -d --build`), el scheduler espera a que el API esté disponible y ejecuta los trabajos una vez (por `RUN_ON_START=1`) y luego según el cron.
- El API expone los endpoints admin bajo `/ingest/admin/*` protegidos por `X-Admin-Token` (tomado de `.env`).
- Los datos crudos (OHLCV normalizado) se escriben en Parquet bajo `./data_lake/market/...` y las features a Redis.

Verificación rápida de backfill:
```bash
# Logs del scheduler
docker compose logs -f scheduler

# Archivos parquet generados (en el host)
find data_lake/market -type f -name '*.parquet' | head

# Claves de features en Redis (dentro del contenedor redis)
docker compose exec redis redis-cli --scan --pattern 'features:market:*' | head
```

Habilitar ML en la imagen (opcional):
```bash
docker compose build --build-arg INSTALL_ML=1 ingestion-api
docker compose up -d ingestion-api
```

Añadir más trabajos de mercado:
- Edita `docker-compose.yml`, variable de entorno `MARKET_JOBS` del `scheduler` y agrega entradas JSON con `{exchange, symbol, timeframe, lookback_minutes, cron}`.

---

## Parquet Ingest (scheduler)

Además de los backfills hacia Redis, el `scheduler` puede ejecutar ingestiones periódicas que escriben OHLCV normalizado a Parquet en el data lake usando el endpoint `POST /ingest/market/{exchange}`.

- Variable: `MARKET_INGEST_JOBS` (JSON list)
- Esquema por item: `{ "exchange": "binance", "symbol": "BTC/USDT", "timeframe": "1m", "limit": 500, "cron": "*/10 * * * *" }`

Ejemplo en `docker-compose.yml`:
```
MARKET_INGEST_JOBS=[
  {"exchange":"binance","symbol":"BTC/USDT","timeframe":"1m","limit":500,"cron":"*/10 * * * *"},
  {"exchange":"binance","symbol":"ETH/USDT","timeframe":"1m","limit":500,"cron":"*/10 * * * *"}
]
```

Verificación rápida:
```bash
# Logs del scheduler
docker compose logs -f scheduler

# Nuevos archivos parquet en el host
find data_lake/market -type f -name '*.parquet' | head
```

---

## ML Inference (Opt-in)

Para habilitar inferencia de sentimiento con modelos reales (DistilBERT por defecto) y el endpoint `/ml/sentiment/predict`:

1) Construir la imagen del API con ML:
   - En `docker-compose.yml` cambiar `INSTALL_ML: 0` a `INSTALL_ML: 1` para `ingestion-api`.
2) Activar los flags en entorno:
   - `ML_SENTIMENT_ENABLED=1`
   - `SENTIMENT_MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english`
   - `ML_MAX_WORKERS=4` (opcional)
   - `HF_HOME=/app/.cache/huggingface` (cache de modelos)
   - (opcional) `SOCIAL_SENTIMENT_ENRICH=1` para enriquecer ingest social con `sentiment_label` y `sentiment_score`.
3) (Recomendado) Montar un volumen para cache de modelos:
```
volumes:
  - hf-cache:/app/.cache/huggingface
```
4) Reconstruir y levantar:
```
docker compose build ingestion-api
docker compose up -d ingestion-api
```

Probar inferencia:
```
curl -s -X POST http://localhost:8000/ml/sentiment/predict \
  -H 'Content-Type: application/json' \
  -d '{"texts":["btc to the moon","market looks bad"]}' | jq
```

Métricas:
- `ml_infer_requests_total{model=...}`
- `ml_infer_duration_seconds{model=...}`
- `ml_infer_errors_total{model=...,type=...}`

---

## Dataset Builders

The repo includes simple scripts to produce training datasets from the data lake.

1) Market dataset (features + labels)
```bash
python scripts/build_market_dataset.py \
  --exchange binance \
  --symbol BTC/USDT \
  --timeframe 1m \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --out datasets/market_btcusdt_1m_2025.parquet
```
This reads Parquet under `MARKET_PATH`, computes our standard market features, and adds labels (next‑bar return and direction).

2) RSS → Parquet (one shot)
```bash
python scripts/rss_to_parquet.py --feed https://news.google.com/rss/search?q=bitcoin
```
Writes normalized RSS entries to `NEWS_PATH/rss/...` using the Parquet writer (supports S3/GCS if configured).

3) Training matrix (market + aggregates from RSS/Reddit)
```bash
python scripts/build_training_matrix.py \
  --exchange binance \
  --symbol BTC/USDT \
  --timeframe 1min \
  --include-rss \
  --out datasets/training_matrix_btcusdt_1m.parquet
```
This merges market features with aggregated counts and mean sentiment for RSS/Reddit (if present), and builds labels.

Notes
- Scripts read from the data lake paths configured in env (local or S3/GCS). For S3/GCS, ensure credentials/env are set as described above.
- The training matrix script expects that some RSS or Reddit Parquet exists; you can generate it via the `rss_to_parquet.py` script or schedule ingest jobs.

---

## Object Storage (S3/GCS) via fsspec (Opt-in)

The Parquet writer supports any fsspec backend. To write directly to object storage:

1) Optional deps (already in the Docker image): `s3fs`, `gcsfs`.
2) Point your data lake paths to S3 or GCS and set credentials.

S3 example
```bash
# .env (host) or docker-compose env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
DATA_LAKE_PATH=s3://my-bucket/algo-data-lake
MARKET_PATH=s3://my-bucket/algo-data-lake/market
# Optional advanced fsspec options (JSON string)
FSSPEC_STORAGE_OPTIONS='{"client_kwargs": {"region_name": "us-east-1"}}'
```

GCS example
```bash
# Mount your service account JSON in the container
# docker-compose.yml
#   volumes:
#     - ./secrets:/secrets:ro

# .env
GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-sa.json
DATA_LAKE_PATH=gs://my-bucket/algo-data-lake
MARKET_PATH=gs://my-bucket/algo-data-lake/market
```

Notes
- Use absolute paths for the base `DATA_LAKE_PATH` and domain paths.
- `FSSPEC_STORAGE_OPTIONS` (JSON) is passed to fsspec’s `url_to_fs` for advanced configuration.
- For S3, IAM roles or instance profiles also work (omit explicit keys).

Storage check endpoint (admin)
```bash
# Resolve backends and run a write/delete probe (local/S3/GCS depending on env)
curl -s -H "X-Admin-Token: $ADMIN_TOKEN" \
  "http://localhost:8000/ingest/admin/storage/check?probe=1&domain=market" | jq
```
