import logging
import os
import time
import json
import fsspec
import pandas as pd
from typing import Dict
from prometheus_client import Counter, Histogram, CollectorRegistry
from app.ingestion_service.parquet_schemas import MARKET_SCHEMA, ONCHAIN_SCHEMA, SOCIAL_SCHEMA, NEWS_SCHEMA
from app.common.time_norm import add_dt_partition
from app.ingestion_service.config import settings

_METRICS_REGISTRY = CollectorRegistry()
PARQUET_WRITES_TOTAL = Counter(
    'parquet_writes_total',
    'Total successful Parquet writes',
    registry=_METRICS_REGISTRY
)
PARQUET_WRITE_ERRORS = Counter(
    'parquet_write_errors_total',
    'Total failed Parquet writes',
    registry=_METRICS_REGISTRY
)
PARQUET_WRITE_LATENCY = Histogram(
    'parquet_write_latency_seconds',
    'Parquet write latency in seconds',
    registry=_METRICS_REGISTRY
)

logger = logging.getLogger(__name__)

# --- Helpers to enforce UTC + dt partition ---
def _dataset_type(base_path: str) -> str:
    bp = (base_path or "").lower()
    if "market" in bp:
        return "market"
    if "onchain" in bp:
        return "onchain"
    if "social" in bp:
        return "social"
    if "news" in bp:
        return "news"
    return "unknown"

def _ts_col_for(dataset: str) -> str:
    return {
        "market": "timestamp",
        "onchain": "timestamp",
        "social": "ts",
        "news": "published_at",
    }.get(dataset, "timestamp")

def _sanitize_part(val) -> str:
    if val is None:
        return "unknown"
    s = str(val)
    # avoid path separators/spaces in partition names
    return s.replace("/", "-").replace(" ", "_")


# Inline schema validator
def validate_schema(
    df: pd.DataFrame,
    schema: Dict[str, str],
    coerce: bool = False
) -> None:
    """
    Ensures df contains the specified columns with the exact dtypes.
    If coerce=True, attempts to cast columns to the expected dtype.
    Raises ValueError on missing columns or irreconcilable dtypes.
    """
    missing = [col for col in schema if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col, expected_dtype in schema.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            if coerce:
                try:
                    df[col] = df[col].astype(expected_dtype)
                except Exception:
                    raise ValueError(
                        f"Failed to coerce column '{col}' from {actual_dtype} to {expected_dtype}"
                    )
            else:
                raise ValueError(
                    f"Wrong dtype for column '{col}': actual={actual_dtype}, expected={expected_dtype}"
                )


def write_to_parquet(df, base_path, partitions, filename=None):
    if df.empty:
        logger.warning("Empty DataFrame, skipping Parquet write")
        return None

    # Enforce presence of dt partition derived from tz-aware UTC timestamp
    dataset = _dataset_type(base_path)
    ts_col = _ts_col_for(dataset)
    if "dt" not in df.columns:
        add_dt_partition(df, ts_col=ts_col)
    if "dt" not in df.columns:
        raise ValueError("Normalization error: missing 'dt' column after partition derivation")

    # Optional sanity: ensure timestamp column is tz-aware UTC when present
    if ts_col in df.columns:
        if "UTC" not in str(df[ts_col].dtype):
            raise ValueError(f"Normalization error: {ts_col} must be tz-aware UTC, found {df[ts_col].dtype}")

    # Ensure a single dt per write (split upstream if needed)
    unique_dt = df["dt"].dropna().unique().tolist()
    if len(unique_dt) != 1:
        results = []
        for dt_value in unique_dt:
            sub_df = df[df["dt"] == dt_value].copy()
            parts_dict = dict(partitions or {})
            parts_dict["dt"] = dt_value
            results.append(write_to_parquet(sub_df, base_path, parts_dict, filename=None))
        return results[-1] if results else None
    dt_value = unique_dt[0]

    # Schema validation before write
    if "market" in base_path:
        validate_schema(df, MARKET_SCHEMA, coerce=True)
    elif "onchain" in base_path:
        validate_schema(df, ONCHAIN_SCHEMA, coerce=True)
    elif "social" in base_path:
        validate_schema(df, SOCIAL_SCHEMA, coerce=True)
    elif "news" in base_path:
        validate_schema(df, NEWS_SCHEMA, coerce=True)

    # Resolve filesystem and root path (support custom storage options)
    storage_options = {}
    if settings.FSSPEC_STORAGE_OPTIONS:
        try:
            storage_options = json.loads(settings.FSSPEC_STORAGE_OPTIONS)
        except Exception:
            storage_options = {}
    fs, root = fsspec.core.url_to_fs(base_path, **storage_options)
    # Ensure fs is an instance (tests may return a class)
    if isinstance(fs, type):
        fs = fs(root)

    # Build directory path with sanitized partitions and required dt
    parts_dict = dict(partitions or {})
    parts_dict.setdefault("dt", dt_value)
    parts = [f"{k}={_sanitize_part(v)}" for k, v in parts_dict.items() if v is not None]
    dir_path = os.path.join(root, *parts)
    fs.makedirs(dir_path, exist_ok=True)

    # Generate filename
    fname = filename or f"part-{int(time.time() * 1000)}.parquet"
    full_path = os.path.join(dir_path, fname)
    temp_path = full_path + ".tmp"
    start = time.time()
    try:
        # Sort by timestamp if present for deterministic files
        if ts_col in df.columns:
            try:
                df = df.sort_values(by=[ts_col])
            except Exception:
                pass
        # Atomic write to temp file
        with fs.open(temp_path, 'wb') as f:
            df.to_parquet(f, compression="snappy", index=False, engine="pyarrow")
        # Move temp to final path (robust across backends)
        moved = False
        try:
            fs.mv(temp_path, full_path, recursive=True)  # works for many fs
            moved = True
        except TypeError:
            try:
                fs.mv(temp_path, full_path)
                moved = True
            except Exception:
                moved = False
        if not moved:
            try:
                # Fallback: copy then remove
                with fs.open(temp_path, 'rb') as src, fs.open(full_path, 'wb') as dst:
                    dst.write(src.read())
                fs.rm(temp_path)
                moved = True
            except Exception:
                moved = False
        if not moved:
            raise RuntimeError(f"Failed to move temp file to final destination: {full_path}")
        duration = time.time() - start
        PARQUET_WRITES_TOTAL.inc()
        PARQUET_WRITE_LATENCY.observe(duration)
        return full_path
    except Exception as e:
        PARQUET_WRITE_ERRORS.inc()
        logger.error(f"Failed writing Parquet to {full_path}: {e}")
        raise
