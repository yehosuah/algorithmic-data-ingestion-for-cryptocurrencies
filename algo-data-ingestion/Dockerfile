# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Build args to optionally install heavy ML deps in production builds
ARG INSTALL_ML=0

# System dependencies for pyarrow/snappy and builds
# Switch APT sources to HTTPS to avoid 403 mirrors in some environments
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libsnappy-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Core Python dependencies
# NOTE: transformers + torch are large; needed for sentiment endpoints.
# If you don't need them immediately, you can remove them to slim the image.
# Core Python deps (lean build). Heavy ML deps are optional via INSTALL_ML.
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic>=2.6 \
    pydantic-settings>=2.2 \
    pandas \
    pyarrow \
    python-snappy \
    httpx \
    tweepy \
    tenacity \
    ratelimit \
    prometheus-client \
    ccxt \
    redis \
    scikit-learn==1.3.2 \
    numpy \
    numba \
    feedparser \
    APScheduler==3.10.4 \
    pytz \
    tzlocal \
    s3fs \
    gcsfs \
    xgboost \
    sentence-transformers || true

# Optionally add transformers/torch if requested
RUN if [ "$INSTALL_ML" = "1" ]; then \
      pip install --no-cache-dir transformers && \
      pip install --no-cache-dir torch || true ; \
    fi

# Stage inference models inside the image so runtime code can operate even when
# bind mounts are unavailable or flaky.
COPY models /opt/models

# Copy application code
COPY . .

EXPOSE 8000

# Simple container healthcheck against the FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health | grep -q '"status":"ok"' || exit 1

# Run the ingestion API
CMD ["uvicorn", "app.ingestion_service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
