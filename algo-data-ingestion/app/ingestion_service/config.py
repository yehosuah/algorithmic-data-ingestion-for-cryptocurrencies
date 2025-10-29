from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
from typing import Optional, List

load_dotenv()  # this will read your .env into os.environ


class Settings(BaseSettings):
    twitter_api_key: Optional[str] = Field(None, env="TWITTER_API_KEY")
    twitter_api_secret: Optional[str] = Field(None, env="TWITTER_API_SECRET")
    twitter_bearer: Optional[str] = Field(None, env="TWITTER_BEARER")
    reddit_client_id: Optional[str] = Field(None, env="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(None, env="REDDIT_CLIENT_SECRET")
    reddit_user_agent: Optional[str] = Field(None, env="REDDIT_USER_AGENT")
    data_lake_path: str = Field("data_lake", env="DATA_LAKE_PATH")
    news_api_key: Optional[str] = Field(None, env="NEWS_API_KEY")
    glassnode_api_key: Optional[str] = Field(None, env="GLASSNODE_API_KEY")
    covalent_api_key: Optional[str] = Field(None, env="COVALENT_API_KEY")

    # Data lake paths
    MARKET_PATH: str = Field("data_lake/market", env="MARKET_PATH")
    ONCHAIN_PATH: str = Field("data_lake/onchain", env="ONCHAIN_PATH")
    SOCIAL_PATH: str = Field("data_lake/social", env="SOCIAL_PATH")
    NEWS_PATH: str = Field("data_lake/news", env="NEWS_PATH")

    # Redis Feature Store settings
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")

    # Feature Store TTL (in seconds)
    feature_ttl_seconds: int = Field(60, env="FEATURE_TTL_SECONDS")

    market_path: str = Field("data_lake/market", env="MARKET_PATH")
    onchain_path: str = Field("data_lake/onchain", env="ONCHAIN_PATH")
    social_path: str = Field("data_lake/social", env="SOCIAL_PATH")
    news_path: str = Field("data_lake/news", env="NEWS_PATH")

    #fastAPI
    ingest_host: str = Field("0.0.0.0", env="INGEST_HOST")
    ingest_port: int = Field(8000, env="INGEST_PORT")
    #CORS
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    #Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    #Metrics/Prometheus
    metrics_path: str = Field("/metrics", env="METRICS_PATH")

    BACKFILL_ENABLED: bool = Field(default=False)
    BACKFILL_EXCHANGE: str = Field(default="binance")
    BACKFILL_SYMBOLS: str = Field(default="BTC/USDT")  # comma-separated
    BACKFILL_TIMEFRAMES: str = Field(default="1m")     # comma-separated
    BACKFILL_LOOKBACK_MIN: int = Field(default=120)
    BACKFILL_INTERVAL_SEC: int = Field(default=300)

    TTL_SWEEP_ENABLED: bool = Field(default=False)
    TTL_SWEEP_INTERVAL_SEC: int = Field(default=600)
    ADMIN_TOKEN: Optional[str] = None

    # ML / Sentiment
    ML_SENTIMENT_ENABLED: bool = Field(default=False, env="ML_SENTIMENT_ENABLED")
    SENTIMENT_MODEL_ID: str = Field(
        default="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        env="SENTIMENT_MODEL_ID",
    )
    ML_MAX_WORKERS: int = Field(default=4, env="ML_MAX_WORKERS")
    SOCIAL_SENTIMENT_ENRICH: bool = Field(default=True, env="SOCIAL_SENTIMENT_ENRICH")
    HF_HOME: Optional[str] = Field(default=None, env="HF_HOME")
    # fsspec storage options (JSON string), e.g. '{"anon": false, "client_kwargs": {"region_name": "us-east-1"}}'
    FSSPEC_STORAGE_OPTIONS: Optional[str] = Field(default=None, env="FSSPEC_STORAGE_OPTIONS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
settings = Settings()
