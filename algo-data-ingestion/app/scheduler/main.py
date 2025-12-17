# app/scheduler/main.py
from __future__ import annotations

import os
import json
import time
import logging
import asyncio
import errno
import threading
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
import joblib
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from prometheus_client import start_http_server, Counter, Histogram, Gauge
from redis import asyncio as aioredis
from zoneinfo import ZoneInfo
import shutil

from labels.label_generator import (
    generate_cost_adjusted_lookback_feature,
    generate_directional_lookback_feature,
    generate_meta_lookback_feature,
    generate_continuous_return_lookback_feature,
)
from regimes.regime_builder import (
    assign_event_flag,
    assign_liquidity_regime,
    assign_spread_regime,
    assign_vol_regime,
)

# Ensure inference metrics register on the global Prometheus registry so they surface on /metrics.
if "USE_INGEST_METRICS_REGISTRY" not in os.environ:
    os.environ["USE_INGEST_METRICS_REGISTRY"] = "0"

from app.features.factory.market_factory import build_market_features
from app.ingestion_service.manifests import get_manifest_registry, prepare_decision_payload
from training.feature_eng import augment_market_features
from training.blender import build_blender_features
from training.infer import (
    load_base_predictor,
    load_tcn_predictor,
    predict_base,
    predict_tcn,
    _register_metric_thresholds,
)
from training.calibration_store import load_calibrator, LoadedCalibrator
from training.calibration_utils import apply_posthoc_calibration
from app.monitoring.model_metrics import observe_gate_coverage, record_gate_coverage_sample
from app.monitoring.probability_sampler import record_probability_samples
from collections import defaultdict
from features.feature_engineering import FEATURE_REGISTRY

_MODEL_SYMBOL_COVERAGE: Dict[str, Dict[str, float]] = defaultdict(dict)


def _inference_gate_threshold(artifacts: Any) -> Optional[float]:
    try:
        gate_cfg = artifacts.gate_config.get("inference") or {}
    except AttributeError:
        return None
    value = gate_cfg.get("prob_gate_min")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_gate_coverage(
    model_label: str,
    symbol: str,
    *,
    passed: int,
    total: int,
    mode: str = "inference",
    gate_threshold: Optional[float] = None,
) -> None:
    total = max(0, int(total))
    passed = max(0, min(total, int(passed)))
    coverage = float(passed) / float(total) if total else 0.0
    """
    Publish gate coverage for a model/symbol pair and aggregate per-model coverage.
    """
    per_symbol = _MODEL_SYMBOL_COVERAGE[model_label]
    per_symbol[symbol] = float(coverage)
    log.info(
        "Gate coverage updated model=%s symbol=%s coverage=%.6f passed=%d total=%d threshold=%s mode=%s",
        model_label,
        symbol,
        coverage,
        passed,
        total,
        f"{gate_threshold:.4f}" if gate_threshold is not None else "<none>",
        mode,
    )
    observe_gate_coverage(f"{model_label}:{symbol}", mode, coverage)
    if total > 0:
        record_gate_coverage_sample(f"{model_label}:{symbol}", mode, passed, total)
        record_gate_coverage_sample(model_label, mode, passed, total)
    aggregate = sum(per_symbol.values()) / max(len(per_symbol), 1)
    observe_gate_coverage(model_label, mode, aggregate)

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [scheduler] %(message)s")
log = logging.getLogger("scheduler")

# ------------------------------------------------------------------------------
# Env
# ------------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://ingestion-api:8000").rstrip("/")
ADMIN_TOKEN: str = os.getenv("ADMIN_TOKEN", "changeme")  # keep out of logs!
RUN_ON_START: bool = os.getenv("RUN_ON_START", "1") not in ("0", "false", "False", "")
SCHED_TZ: str = os.getenv("SCHED_TZ", "UTC")
SCHED_METRICS_PORT: int = int(os.getenv("SCHED_METRICS_PORT", "9002"))
INFER_JOBS_RAW: str = os.getenv("INFER_JOBS", "[]")
MODELS_ROOT: Path = Path(os.getenv("MODELS_ROOT", "models")).expanduser().resolve()
DATA_LAKE_ROOT: Path = Path(os.getenv("DATA_LAKE_ROOT", "/app/data_lake/market")).expanduser().resolve()
DECISION_QUEUE_URL: str = os.getenv("DECISION_QUEUE_URL") or os.getenv("REDIS_URL", "redis://redis:6379/0")
DECISION_QUEUE_KEY: str = os.getenv("DECISION_QUEUE_KEY", "trading:decisions")
DEFAULT_DECISION_PAYLOAD_ITEMS: int = max(1, int(os.getenv("DECISION_PAYLOAD_ITEMS", "3")))
DEFAULT_INFER_STRIDE: int = max(1, int(os.getenv("INFER_DEFAULT_STRIDE", "30")))
DEFAULT_HISTORY_MARGIN_MIN: int = max(0, int(os.getenv("INFER_HISTORY_MARGIN_MIN", "120")))


def _env_override_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    log.warning("Invalid boolean override for %s=%r (ignoring)", name, raw)
    return None

# Market jobs JSON: list of {"exchange","symbol","timeframe","lookback_minutes","cron"}
def _get_market_jobs() -> List[Dict[str, Any]]:
    raw = os.getenv("MARKET_JOBS", "[]")
    try:
        jobs = json.loads(raw) if raw.strip() else []
        if not isinstance(jobs, list):
            raise ValueError("MARKET_JOBS must be a JSON list")
        return jobs
    except Exception as e:
        log.warning("Failed to parse MARKET_JOBS (%s). Using empty list.", e)
        return []

# Market ingest-to-parquet jobs JSON: list of {"exchange","symbol","timeframe","limit","cron"}
def _get_market_ingest_jobs() -> List[Dict[str, Any]]:
    raw = os.getenv("MARKET_INGEST_JOBS", "[]")
    try:
        jobs = json.loads(raw) if raw.strip() else []
        if not isinstance(jobs, list):
            raise ValueError("MARKET_INGEST_JOBS must be a JSON list")
        return jobs
    except Exception as e:
        log.warning("Failed to parse MARKET_INGEST_JOBS (%s). Using empty list.", e)
        return []

# TTL sweep config
TTL_SWEEP_CRON: str = os.getenv("TTL_SWEEP_CRON", "*/15 * * * *")
TTL_SWEEP_PATTERN: str = os.getenv("TTL_SWEEP_PATTERN", "features:market:*")
TTL_SWEEP_TTL: int = int(os.getenv("TTL_SWEEP_TTL", "3600"))
TTL_SWEEP_MAX_KEYS: Optional[int] = int(os.getenv("TTL_SWEEP_MAX_KEYS", "0")) or None  # optional

# ------------------------------------------------------------------------------
# Prometheus metrics
# ------------------------------------------------------------------------------
# How often jobs get invoked
JOB_RUNS = Counter(
    "scheduler_job_runs_total",
    "Total number of scheduler job invocations",
    ["job_id"],
)

# Success/failure counts
JOB_SUCCESS = Counter(
    "scheduler_job_success_total",
    "Total successful job completions",
    ["job_id"],
)
JOB_FAILURE = Counter(
    "scheduler_job_failure_total",
    "Total failed job completions",
    ["job_id", "reason"],
)

# How long each job takes
JOB_DURATION = Histogram(
    "scheduler_job_duration_seconds",
    "Duration of scheduler jobs in seconds",
    ["job_id"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)

# Last run timestamps (unix epoch)
JOB_LAST_RUN_TS = Gauge(
    "scheduler_job_last_run_timestamp",
    "Unix timestamp of the last time the job started",
    ["job_id"],
)
JOB_LAST_SUCCESS_TS = Gauge(
    "scheduler_job_last_success_timestamp",
    "Unix timestamp of the last time the job succeeded",
    ["job_id"],
)

DECISIONS_ENQUEUED = Counter(
    "scheduler_decision_messages_enqueued_total",
    "Number of decision messages emitted to downstream queue",
    ["job_id", "model"],
)
LAST_DECISION_TS = Gauge(
    "scheduler_decision_last_timestamp",
    "Unix timestamp of the latest decision published per job/model",
    ["job_id", "model"],
)

# ------------------------------------------------------------------------------
# Inference job configuration & caching
# ------------------------------------------------------------------------------
def _coerce_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    if isinstance(val, (int, float)):
        return val != 0
    s = str(val).strip().lower()
    return s not in ("", "0", "false", "no", "off")


def _safe_symbol_path(symbol: str) -> str:
    sym = (symbol or "").strip()
    if not sym:
        return ""
    return sym.replace(" ", "").replace("/", "-").replace(":", "-").upper()


def _decision_namespace(symbol: str, timeframe: str, policy_id: str, model_label: str) -> str:
    sym = _safe_symbol_path(symbol) or symbol
    tf = (timeframe or "").strip().lower()
    policy = (policy_id or "").strip() or "primary"
    model = (model_label or "").strip()
    return f"{sym}:{tf}:{policy}:{model}"


@dataclass
class InferenceJob:
    job_id: str
    exchange: str
    symbol: str
    symbol_path: str
    timeframe: str
    lookback_minutes: int
    history_minutes: int
    base_model: Optional[str]
    base_path: Optional[str]
    tcn_model: Optional[str]
    tcn_path: Optional[str]
    blender_model: Optional[str]
    blender_path: Optional[str]
    model_key: Optional[str]
    policy_id: str
    policy_contract: Optional[str]
    shadow_mode: bool
    stride: int
    include_features: bool
    max_decision_items: int
    queue_url: str
    queue_key: str
    cron: str

    def data_dir(self) -> Path:
        return DATA_LAKE_ROOT / f"exchange={self.exchange}" / f"symbol={self.symbol_path}"


def _parse_inference_jobs(raw: str) -> List[InferenceJob]:
    if not raw or not raw.strip():
        return []
    try:
        payload = json.loads(raw)
    except Exception as exc:
        log.warning("Failed to parse INFER_JOBS (%s). Scheduler inference disabled.", exc)
        return []
    if not isinstance(payload, list):
        log.warning("INFER_JOBS must be a JSON list; received %r", type(payload))
        return []

    jobs: List[InferenceJob] = []
    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict):
            log.warning("INFER_JOBS[%d] must be an object; skipping.", idx)
            continue
        try:
            exchange = str(entry["exchange"]).strip()
            symbol = str(entry["symbol"]).strip()
        except KeyError as missing:
            log.warning("INFER_JOBS[%d] missing required key %s; skipping.", idx, missing)
            continue

        timeframe = str(entry.get("timeframe", "1m")).strip() or "1m"
        lookback = max(1, int(entry.get("lookback_minutes", 120)))
        history_raw = entry.get("history_minutes", 0)
        history_minutes = int(history_raw) if history_raw not in (None, "", "None") else 0
        stride = max(1, int(entry.get("stride", DEFAULT_INFER_STRIDE)))
        include_features = _coerce_bool(entry.get("include_features", False))
        raw_max_items = entry.get("max_decision_items")
        if raw_max_items in (None, "", "None"):
            max_decision_items = DEFAULT_DECISION_PAYLOAD_ITEMS
        else:
            try:
                max_decision_items = max(1, int(raw_max_items))
            except (TypeError, ValueError):
                log.warning(
                    "INFER_JOBS[%d] has invalid max_decision_items=%r; defaulting to %d",
                    idx,
                    raw_max_items,
                    DEFAULT_DECISION_PAYLOAD_ITEMS,
                )
                max_decision_items = DEFAULT_DECISION_PAYLOAD_ITEMS

        policy_id = str(entry.get("policy_id") or "primary").strip() or "primary"
        policy_contract = str(entry.get("policy_contract") or entry.get("policy_path") or "").strip() or None
        base_model = str(entry.get("base_model") or "").strip() or None
        base_path = str(entry.get("base_path") or "").strip() or (base_model or None)
        tcn_model = str(entry.get("tcn_model") or "").strip() or None
        tcn_path = str(entry.get("tcn_path") or "").strip() or (tcn_model or None)
        blender_model = str(entry.get("blender_model") or "").strip() or None
        blender_path = str(entry.get("blender_path") or "").strip() or (blender_model or None)
        model_key = str(entry.get("model_key") or base_model or tcn_model or blender_model or "").strip() or None
        if not base_model and not tcn_model and not blender_model:
            log.warning(
                "INFER_JOBS[%d] requires at least one of base_model, tcn_model, or blender_model; skipping.",
                idx,
            )
            continue
        shadow_mode = _coerce_bool(entry.get("shadow_mode", False))

        queue_url = str(entry.get("queue_url") or DECISION_QUEUE_URL).strip()
        queue_key = str(entry.get("queue_key") or DECISION_QUEUE_KEY).strip() or DECISION_QUEUE_KEY
        cron_expr = str(entry.get("cron") or "*/1 * * * *").strip() or "*/1 * * * *"

        symbol_path = _safe_symbol_path(symbol)
        job_identifier = str(entry.get("job_id") or f"infer:{exchange}:{symbol_path}:{timeframe}")
        effective_history = history_minutes if history_minutes > 0 else max(
            lookback + DEFAULT_HISTORY_MARGIN_MIN,
            stride * 4,
        )

        jobs.append(
            InferenceJob(
                job_id=job_identifier,
                exchange=exchange,
                symbol=symbol,
                symbol_path=symbol_path,
                timeframe=timeframe,
                lookback_minutes=lookback,
                history_minutes=effective_history,
                base_model=base_model,
                base_path=base_path,
                tcn_model=tcn_model,
                tcn_path=tcn_path,
                blender_model=blender_model,
                blender_path=blender_path,
                model_key=model_key,
                policy_id=policy_id,
                policy_contract=policy_contract,
                shadow_mode=shadow_mode,
                stride=stride,
                include_features=include_features,
                max_decision_items=max_decision_items,
                queue_url=queue_url,
                queue_key=queue_key,
                cron=cron_expr,
            )
        )

    return jobs


INFERENCE_JOBS: List[InferenceJob] = _parse_inference_jobs(INFER_JOBS_RAW)

_QUEUE_CLIENTS: Dict[str, aioredis.Redis] = {}
_BASE_MODEL_CACHE: Dict[str, Tuple[Any, object, List[str]]] = {}
_TCN_MODEL_CACHE: Dict[str, Tuple[Any, object, object, List[str], object, int]] = {}
_BLENDER_MODEL_CACHE: Dict[str, Tuple[Any, object, List[str], Optional[LoadedCalibrator]]] = {}
_LAST_EMITTED: Dict[Tuple[str, str, str], pd.Timestamp] = {}
_LOCAL_MODEL_ROOT = Path("/tmp/scheduler-models")
_LOCAL_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
_MODEL_COPY_LOCKS: Dict[str, threading.Lock] = {}
_MODEL_COPY_LOCK_GUARD = threading.Lock()


def _copy_lock_for(label: str) -> threading.Lock:
    with _MODEL_COPY_LOCK_GUARD:
        lock = _MODEL_COPY_LOCKS.get(label)
        if lock is None:
            lock = threading.Lock()
            _MODEL_COPY_LOCKS[label] = lock
        return lock


def _ensure_local_model_dir(label: str, source: Path) -> Path:
    dest = _LOCAL_MODEL_ROOT / label
    src = source.expanduser().resolve()
    lock = _copy_lock_for(str(dest))
    with lock:
        if dest.exists():
            return dest
        tmp_dest = dest.with_suffix(".tmp")
        if tmp_dest.exists():
            shutil.rmtree(tmp_dest, ignore_errors=True)
        attempts = 3
        last_exc: Optional[BaseException] = None
        transient_failure = False
        for attempt in range(attempts):
            try:
                shutil.copytree(src, tmp_dest, dirs_exist_ok=True)
                break
            except shutil.Error as exc:
                last_exc = exc
                transient_failure = False
                def _is_transient_shutil_error(err: shutil.Error) -> bool:
                    details = getattr(err, "args", ())
                    if not details:
                        return False
                    records = details[0] if isinstance(details[0], list) else details
                    for record in records:
                        if not isinstance(record, (list, tuple)) or len(record) < 3:
                            return False
                        message = str(record[2])
                        if "Errno 35" not in message and "Resource deadlock avoided" not in message:
                            return False
                    return True

                if _is_transient_shutil_error(exc):
                    transient_failure = True
                    if tmp_dest.exists():
                        shutil.rmtree(tmp_dest, ignore_errors=True)
                    if attempt < attempts - 1:
                        wait = 0.1 * (2 ** attempt)
                        log.warning(
                            "Retrying copy of model %s due to transient error (%s); retry in %.2fs",
                            label,
                            exc,
                            wait,
                        )
                        time.sleep(wait)
                    continue
                if tmp_dest.exists():
                    shutil.rmtree(tmp_dest, ignore_errors=True)
                raise
            except OSError as exc:
                last_exc = exc
                transient = getattr(exc, "errno", None) in (errno.EDEADLK, errno.EBUSY)
                if transient:
                    transient_failure = True
                    if tmp_dest.exists():
                        shutil.rmtree(tmp_dest, ignore_errors=True)
                    if attempt < attempts - 1:
                        wait = 0.1 * (2 ** attempt)
                        log.warning(
                            "Retrying copy of model %s due to transient error (%s); retry in %.2fs",
                            label,
                            exc,
                            wait,
                        )
                        time.sleep(wait)
                    continue
                if tmp_dest.exists():
                    shutil.rmtree(tmp_dest, ignore_errors=True)
                raise
        else:
            if tmp_dest.exists():
                shutil.rmtree(tmp_dest, ignore_errors=True)
            if transient_failure or (
                last_exc is not None and getattr(last_exc, "errno", None) in (errno.EDEADLK, errno.EBUSY)
            ):
                log.warning(
                    "Falling back to direct model path for %s due to persistent copy error: %s",
                    label,
                    last_exc,
                )
                return src
            raise RuntimeError(f"Failed to copy model directory for {label} after retries")
        try:
            tmp_dest.rename(dest)
        except FileExistsError:
            shutil.rmtree(tmp_dest, ignore_errors=True)
        except OSError as exc:
            shutil.rmtree(tmp_dest, ignore_errors=True)
            raise exc
        return dest


def _timeframe_to_minutes(timeframe: str) -> int:
    tf = (timeframe or "1m").strip().lower()
    if tf.endswith("m"):
        return max(1, int(float(tf[:-1] or 1)))
    if tf.endswith("h"):
        return max(1, int(float(tf[:-1] or 1) * 60))
    if tf.endswith("d"):
        return max(1, int(float(tf[:-1] or 1) * 60 * 24))
    raise ValueError(f"Unsupported timeframe value: {timeframe!r}")


def _preload_inference_manifests(jobs: List[InferenceJob]) -> None:
    if not jobs:
        return
    registry = get_manifest_registry()
    specs: List[Tuple[str, str]] = []
    for job in jobs:
        if job.base_model:
            specs.append((job.base_model, job.base_path or job.base_model))
        if job.tcn_model:
            specs.append((job.tcn_model, job.tcn_path or job.tcn_model))
        if job.blender_model:
            specs.append((job.blender_model, job.blender_path or job.blender_model))
    unique: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for label, rel_path in specs:
        if not label or not rel_path:
            continue
        key = (label, rel_path)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, rel_path))
    if not unique:
        return
    registry.preload(models_root=MODELS_ROOT, specs=unique, clear=False)
    loaded_labels = ", ".join(label for label, _ in unique)
    log.info("Preloaded manifest artifacts for inference jobs: %s", loaded_labels)


def _get_queue_client(url: str) -> aioredis.Redis:
    client = _QUEUE_CLIENTS.get(url)
    if client is None:
        client = aioredis.from_url(url, encoding="utf-8", decode_responses=True)
        _QUEUE_CLIENTS[url] = client
    return client


def _filter_new_payload_items(
    job_id: str,
    model_label: str,
    policy_id: str,
    items: List[Dict[str, Any]],
    *,
    min_ts: Optional[pd.Timestamp] = None,
) -> List[Dict[str, Any]]:
    if not items:
        return []
    key = (job_id, model_label, policy_id)
    last_ts = _LAST_EMITTED.get(key)
    fresh: List[Tuple[pd.Timestamp, Dict[str, Any]]] = []
    for item in items:
        ts_raw = item.get("timestamp")
        ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        if ts is pd.NaT:
            continue
        if min_ts is not None and ts <= min_ts:
            continue
        if last_ts is None or ts > last_ts:
            fresh.append((ts, item))
    if not fresh:
        return []
    fresh.sort(key=lambda pair: pair[0])
    latest = fresh[-1][0]
    _LAST_EMITTED[key] = latest
    try:
        LAST_DECISION_TS.labels(job_id=job_id, model=model_label).set(latest.timestamp())
    except Exception:
        pass
    return [item for _, item in fresh]


async def _enqueue_payload(job: InferenceJob, payload: Dict[str, Any], now: datetime) -> int:
    model_label = str(payload.get("model") or "").strip()
    if not model_label:
        return 0
    items = payload.get("items") or []
    min_ts = None
    total_items = len(items)
    missing_ts = 0
    if total_items:
        for item in items:
            ts_raw = item.get("timestamp")
            ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
            if ts is pd.NaT:
                missing_ts += 1
    if job.lookback_minutes:
        min_ts_dt = now - timedelta(minutes=job.lookback_minutes)
        min_ts = pd.Timestamp(min_ts_dt)
        if min_ts.tzinfo is None:
            min_ts = min_ts.tz_localize("UTC")
        else:
            min_ts = min_ts.tz_convert("UTC")
    new_items = _filter_new_payload_items(job.job_id, model_label, job.policy_id, items, min_ts=min_ts)
    new_count = len(new_items)
    min_ts_str = min_ts.isoformat() if isinstance(min_ts, pd.Timestamp) else "<none>"
    log.info(
        "enqueue_stats job=%s model=%s items=%d new_items=%d missing_ts=%d min_ts=%s",
        job.job_id,
        model_label,
        total_items,
        new_count,
        missing_ts,
        min_ts_str,
    )
    limit = max(1, int(job.max_decision_items or DEFAULT_DECISION_PAYLOAD_ITEMS))
    selected_items = new_items[-limit:] if new_items else []
    selected_count = len(selected_items)
    trimmed = new_count - selected_count
    if trimmed > 0:
        log.debug(
            "Trimming %d decision(s) for job=%s model=%s limit=%d new_items=%d",
            trimmed,
            job.job_id,
            model_label,
            limit,
            new_count,
        )
    if not selected_items:
        return 0

    client = _get_queue_client(job.queue_url)
    prob_column = payload.get("prob_column")
    gate_column = payload.get("gate_column")
    artifact_dir = payload.get("artifact_dir")
    policy_id = job.policy_id or "primary"
    policy_contract = job.policy_contract
    model_key_value = job.model_key or model_label
    shadow_mode = bool(job.shadow_mode)

    messages: List[str] = []
    for item in selected_items:
        ts = pd.to_datetime(item.get("timestamp"), utc=True, errors="coerce")
        if ts is pd.NaT:
            continue
        namespace = _decision_namespace(job.symbol, job.timeframe, policy_id, model_label)
        message: Dict[str, Any] = {
            "job_id": job.job_id,
            "model": model_label,
            "exchange": job.exchange,
            "symbol": job.symbol,
            "timeframe": job.timeframe,
            "policy_id": policy_id,
            "model_key": model_key_value,
            "timestamp": ts.isoformat(),
            "probability": float(item.get("probability", 0.0) or 0.0),
            "gate_pass": bool(item.get("gate_pass")),
            "shadow_mode": shadow_mode,
            "decision_namespace": namespace,
        }
        if prob_column:
            message["prob_column"] = prob_column
        if gate_column:
            message["gate_column"] = gate_column
        if artifact_dir:
            message["artifact_dir"] = artifact_dir
        if policy_contract:
            message["policy_contract"] = policy_contract
        if job.include_features and "features" in item:
            message["features"] = item["features"]
        messages.append(json.dumps(message, default=str))

    if not messages:
        log.info(
            "enqueue_stats_no_messages job=%s model=%s new_items=%d dropped_ts=%d",
            job.job_id,
            model_label,
            selected_count,
            missing_ts,
        )
        return 0

    await client.rpush(job.queue_key, *messages)
    DECISIONS_ENQUEUED.labels(job_id=job.job_id, model=model_label).inc(len(messages))
    log.info(
        "Enqueued %d decisions for job=%s model=%s queue=%s",
        len(messages),
        job.job_id,
        model_label,
        job.queue_key,
    )
    return len(messages)


def _load_recent_ohlcv(job: InferenceJob, now: datetime, history_minutes: int) -> pd.DataFrame:
    data_dir = job.data_dir()
    if not data_dir.exists():
        log.warning("Market data directory not found for job %s: %s", job.job_id, data_dir)
        return pd.DataFrame()

    cutoff = now - timedelta(minutes=history_minutes)
    earliest_dt = (cutoff - timedelta(days=1)).date()

    # Read the newest parquet files first and stop once we have enough bars to cover the requested
    # history. This avoids scanning the entire dt=* tree when many small files exist.
    tf_minutes = max(1, _timeframe_to_minutes(job.timeframe))
    bars_needed = max(1, math.ceil(history_minutes / tf_minutes))
    frames: List[pd.DataFrame] = []
    rows_loaded = 0
    # Group dt partitions by date value so duplicate dt=YYYY-MM-DD folders (caused by upstream
    # partitioning bugs or backfills) don't cause us to stop early on a stale subset.
    dt_dirs_by_value: Dict[date, List[Path]] = defaultdict(list)
    for dt_dir in (p for p in data_dir.rglob("dt=*") if p.is_dir()):
        try:
            dt_value = datetime.strptime(dt_dir.name.split("=", 1)[1], "%Y-%m-%d").date()
        except Exception:
            continue
        if dt_value > now.date() or dt_value < earliest_dt:
            continue
        dt_dirs_by_value[dt_value].append(dt_dir)

    dt_values = sorted(dt_dirs_by_value.keys(), reverse=True)
    global_ts_min: Optional[pd.Timestamp] = None
    global_ts_max: Optional[pd.Timestamp] = None
    done = False

    def _dir_recency_key(dt_dir: Path) -> float:
        # Directory mtime advances when new parquet parts are written.
        try:
            return float(dt_dir.stat().st_mtime)
        except Exception:
            return 0.0

    for dt_value in dt_values:
        if rows_loaded >= bars_needed and dt_value < earliest_dt:
            break
        dt_dirs = dt_dirs_by_value.get(dt_value, [])
        dt_dirs = sorted(dt_dirs, key=_dir_recency_key, reverse=True)
        for dt_dir in dt_dirs:
            # Read newest file first within each dt partition.
            first_file = True
            for pq_path in sorted(dt_dir.glob("*.parquet"), reverse=True):
                try:
                    df = pd.read_parquet(pq_path)
                except Exception as exc:
                    log.warning("Failed to read parquet %s: %s", pq_path, exc)
                    continue
                if df is None or df.empty:
                    continue
                frames.append(df)
                rows_loaded += len(df)

                ts_min: Optional[pd.Timestamp] = None
                ts_max: Optional[pd.Timestamp] = None
                try:
                    ts_series = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                    ts_min = ts_series.min()
                    ts_max = ts_series.max()
                except Exception:
                    ts_min = None
                    ts_max = None

                if ts_min is not None and pd.notna(ts_min):
                    global_ts_min = ts_min if global_ts_min is None else min(global_ts_min, ts_min)
                if ts_max is not None and pd.notna(ts_max):
                    global_ts_max = ts_max if global_ts_max is None else max(global_ts_max, ts_max)

                have_window = global_ts_max is not None and pd.notna(global_ts_max) and global_ts_max >= cutoff
                have_history = global_ts_min is not None and pd.notna(global_ts_min) and global_ts_min <= cutoff
                # If the newest file in this dt partition is entirely older than the cutoff and we still
                # don't have any bars within the window, skip scanning older files and try the next dt dir.
                if (
                    first_file
                    and not have_window
                    and ts_max is not None
                    and pd.notna(ts_max)
                    and ts_max < cutoff
                ):
                    break
                first_file = False
                if rows_loaded >= bars_needed and have_window and have_history:
                    done = True
                    break
            if done:
                break
        if done:
            break

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "timestamp" not in df.columns:
        raise KeyError(f"OHLCV frame missing 'timestamp' column for job {job.job_id}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    df = df[df["timestamp"] <= now]
    df = df[df["timestamp"] >= cutoff]
    return df.reset_index(drop=True)


def _build_feature_frame(job: InferenceJob, ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        return pd.DataFrame()
    features = build_market_features(ohlcv)
    features = augment_market_features(features, inplace=False)
    if (
        not features.empty
        and "timestamp" in features.columns
        and "timestamp" in ohlcv.columns
        and "close" in ohlcv.columns
    ):
        try:
            close_frame = ohlcv.copy()
            close_frame["timestamp"] = pd.to_datetime(close_frame["timestamp"], utc=True)
            # Deduplicate on timestamp to avoid pandas reindex errors when mapping.
            dedup = close_frame.drop_duplicates(subset=["timestamp"], keep="last")
            close_series = dedup.set_index("timestamp")["close"].astype(float)
            ts = pd.to_datetime(features["timestamp"], utc=True)
            features["close"] = ts.map(close_series)
            features["price"] = features["close"]
        except Exception:
            log.debug("Failed to attach close price to feature frame for job %s", job.job_id, exc_info=True)
    features = features.sort_values("timestamp").reset_index(drop=True)
    return features


def _label_horizon_bars(timeframe: str, default_minutes: int = 15) -> int:
    try:
        tf_minutes = max(1, _timeframe_to_minutes(timeframe))
    except Exception:
        tf_minutes = default_minutes
    return max(1, int(round(float(default_minutes) / float(tf_minutes))))


_SPECIAL_FEATURES = {
    "rvol5",
    "rvol20",
    "sym_btc",
    "sym_eth",
    "sym_sol",
    "directional_15m",
    "cost_adjusted_15m",
    "meta_15m_feat_log_return_1m",
    "net_return_15m",
    "vol_regime",
    "liquidity_regime",
    "spread_regime",
    "event_flag",
}


def _ensure_required_features(frame: pd.DataFrame, required: List[str], *, job_id: str, timeframe: str = "1m") -> pd.DataFrame:
    """
    Ensure all manifest-declared features exist before scoring.

    - Compute any missing FEATURE_REGISTRY entries on the fly.
    - Populate manifest-only aliases/label/regime columns so the model sees the shape it was trained on.
    - Backfill any remaining missing columns with 0.0 to avoid silent model drift.
    """
    if not required:
        return frame
    out = frame
    missing_registry = [col for col in required if col in FEATURE_REGISTRY and col not in out.columns]
    if missing_registry:
        out = out.copy()
        for col in missing_registry:
            fn = FEATURE_REGISTRY.get(col)
            try:
                out[col] = fn(out)
            except Exception as exc:
                log.warning("Failed to compute feature %s; filling with 0.0 (job %s): %s", col, job_id, exc)
                out[col] = 0.0
    special_missing = [col for col in required if col not in out.columns and col in _SPECIAL_FEATURES]
    if special_missing:
        out = out.copy()
        horizon_bars = _label_horizon_bars(timeframe, default_minutes=15)
        if "rvol5" in special_missing and "rvol_5" in out.columns:
            out["rvol5"] = pd.to_numeric(out["rvol_5"], errors="coerce")
        if "rvol20" in special_missing and "rvol_20" in out.columns:
            out["rvol20"] = pd.to_numeric(out["rvol_20"], errors="coerce")
        if "symbol" in out.columns:
            sym_series = out["symbol"].astype(str)
            sym_map = {
                "sym_btc": "BTC/USDT",
                "sym_eth": "ETH/USDT",
                "sym_sol": "SOL/USDT",
            }
            for col, sym in sym_map.items():
                if col in required and col not in out.columns:
                    out[col] = (sym_series == sym).astype(float)
        needs_label_like = any(
            col in special_missing
            for col in ("directional_15m", "cost_adjusted_15m", "meta_15m_feat_log_return_1m", "net_return_15m")
        )
        if needs_label_like:
            if "feat_log_return_1m" not in out.columns and "feat_log_return_1m" in FEATURE_REGISTRY:
                try:
                    out["feat_log_return_1m"] = FEATURE_REGISTRY["feat_log_return_1m"](out)
                except Exception:
                    out["feat_log_return_1m"] = 0.0
            try:
                if "directional_15m" in special_missing:
                    out["directional_15m"] = generate_directional_lookback_feature(out, horizon_bars)
            except Exception:
                out["directional_15m"] = 0.0
            try:
                if "cost_adjusted_15m" in special_missing:
                    out["cost_adjusted_15m"] = generate_cost_adjusted_lookback_feature(out, horizon_bars)
            except Exception:
                out["cost_adjusted_15m"] = 0.0
            try:
                if "meta_15m_feat_log_return_1m" in special_missing:
                    out["meta_15m_feat_log_return_1m"] = generate_meta_lookback_feature(
                        out,
                        horizon_bars,
                        base_signal_col="feat_log_return_1m",
                        edge_threshold=0.0,
                    )
            except Exception:
                out["meta_15m_feat_log_return_1m"] = 0.0
            try:
                if "net_return_15m" in special_missing:
                    out["net_return_15m"] = generate_continuous_return_lookback_feature(out, horizon_bars)
            except Exception:
                out["net_return_15m"] = 0.0
        if any(col in special_missing for col in ("vol_regime", "liquidity_regime", "spread_regime", "event_flag")):
            try:
                if "vol_regime" in special_missing:
                    out["vol_regime"] = assign_vol_regime(out)
            except Exception:
                out["vol_regime"] = 0.0
            try:
                if "liquidity_regime" in special_missing:
                    out["liquidity_regime"] = assign_liquidity_regime(out)
            except Exception:
                out["liquidity_regime"] = 0.0
            try:
                if "spread_regime" in special_missing:
                    out["spread_regime"] = assign_spread_regime(out)
            except Exception:
                out["spread_regime"] = 0.0
            try:
                if "event_flag" in special_missing:
                    out["event_flag"] = assign_event_flag(out, return_col="feat_log_return_1m")
            except Exception:
                out["event_flag"] = 0.0
        for col in special_missing:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    missing_any = [col for col in required if col not in out.columns]
    if missing_any:
        out = out.copy()
        for col in missing_any:
            out[col] = 0.0
        log.warning(
            "Backfilled %d missing manifest features with 0.0 (job %s): %s",
            len(missing_any),
            job_id,
            missing_any[:10],
        )
    return out


def _get_base_context(label: str) -> Tuple[Any, object, List[str]]:
    cached = _BASE_MODEL_CACHE.get(label)
    if cached is not None:
        return cached
    registry = get_manifest_registry()
    artifacts = registry.ensure_loaded(label)
    _register_metric_thresholds(artifacts)
    model_dir = _ensure_local_model_dir(label, registry.get_path(label))
    prob_col = artifacts.prob_column or "base_prob"
    apply_calibration = getattr(artifacts, "apply_calibration", True)
    override = _env_override_bool("INFER_APPLY_CALIBRATION")
    if override is not None:
        if bool(override) != bool(apply_calibration):
            log.info(
                "Overriding apply_calibration for model=%s from %s -> %s via INFER_APPLY_CALIBRATION",
                label,
                apply_calibration,
                override,
            )
        apply_calibration = bool(override)
    calibrator, feature_columns = load_base_predictor(
        model_dir,
        prob_column=prob_col,
        apply_calibration=apply_calibration,
    )
    ctx = (artifacts, calibrator, list(feature_columns))
    _BASE_MODEL_CACHE[label] = ctx
    return ctx


def _get_tcn_context(label: str) -> Tuple[Any, object, object, List[str], object, int]:
    cached = _TCN_MODEL_CACHE.get(label)
    if cached is not None:
        return cached
    registry = get_manifest_registry()
    artifacts = registry.ensure_loaded(label)
    _register_metric_thresholds(artifacts)
    model_dir = _ensure_local_model_dir(label, registry.get_path(label))
    prob_col = artifacts.prob_column or "tcn_prob"
    model, calibrator, series_cols, scaler, window = load_tcn_predictor(model_dir, prob_column=prob_col)
    ctx = (artifacts, model, calibrator, list(series_cols), scaler, int(window))
    _TCN_MODEL_CACHE[label] = ctx
    return ctx


def _get_blender_context(label: str) -> Tuple[Any, object, List[str], Optional[LoadedCalibrator]]:
    cached = _BLENDER_MODEL_CACHE.get(label)
    if cached is not None:
        return cached
    registry = get_manifest_registry()
    artifacts = registry.ensure_loaded(label)
    _register_metric_thresholds(artifacts)
    model_dir = _ensure_local_model_dir(label, registry.get_path(label))
    feature_path = model_dir / "blender_features.txt"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing blender feature list for {label}: {feature_path}")
    candidate_columns = [
        line.strip() for line in feature_path.read_text().splitlines() if line.strip()
    ]
    if not candidate_columns:
        raise ValueError(f"No blender feature columns declared in {feature_path}")
    model_path = model_dir / "blender.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing blender model artifact for {label}: {model_path}")
    model = joblib.load(model_path)
    prob_col = artifacts.prob_column or "blender_prob"
    calibrator = load_calibrator(model_dir, prob_col)
    ctx = (artifacts, model, list(candidate_columns), calibrator)
    _BLENDER_MODEL_CACHE[label] = ctx
    return ctx


def _required_history_minutes(
    job: InferenceJob,
    *,
    tcn_ctx: Optional[Tuple[Any, object, object, List[str], object, int]] = None,
) -> int:
    tf_minutes = _timeframe_to_minutes(job.timeframe)
    history_minutes = max(
        job.history_minutes,
        job.lookback_minutes + DEFAULT_HISTORY_MARGIN_MIN,
        120 * tf_minutes + DEFAULT_HISTORY_MARGIN_MIN,
    )
    ctx = tcn_ctx
    if ctx is None and job.tcn_model:
        try:
            ctx = _get_tcn_context(job.tcn_model)
        except Exception as exc:
            log.warning(
                "Unable to resolve TCN context for job %s during history calculation: %s",
                job.job_id,
                exc,
            )
            ctx = None
    if ctx is not None:
        _, _, _, _, _, window = ctx
        history_minutes = max(history_minutes, window * tf_minutes + DEFAULT_HISTORY_MARGIN_MIN)
    return history_minutes


def _run_inference_sync(job: InferenceJob, now: datetime) -> List[Dict[str, Any]]:
    base_ctx = _get_base_context(job.base_model) if job.base_model else None
    tcn_ctx = _get_tcn_context(job.tcn_model) if job.tcn_model else None
    blender_ctx = _get_blender_context(job.blender_model) if job.blender_model else None

    history_minutes = _required_history_minutes(job, tcn_ctx=tcn_ctx)

    ohlcv = _load_recent_ohlcv(job, now, history_minutes)
    if ohlcv.empty:
        log.info("No OHLCV data available for inference job %s (history=%d minutes)", job.job_id, history_minutes)
        return []

    features = _build_feature_frame(job, ohlcv)
    if features.empty:
        log.info("Feature frame empty for inference job %s", job.job_id)
        return []

    payloads: List[Dict[str, Any]] = []
    working = features.copy()

    if base_ctx is not None:
        artifacts, calibrator, feature_cols = base_ctx
        working = _ensure_required_features(working, feature_cols, job_id=job.job_id, timeframe=job.timeframe)
        prob_series = predict_base(working, calibrator, feature_cols)
        prob_col = artifacts.prob_column or "base_prob"
        working = working.copy()
        working[prob_col] = prob_series
        try:
            record_probability_samples(
                model_label=job.base_model or artifacts.model_label,
                prob_column=prob_col,
                df=working,
                prob_series=prob_series,
                source="scheduler",
                symbol=job.symbol,
                timeframe=job.timeframe,
                job_id=job.job_id,
                extra={"exchange": job.exchange, "model_kind": "base"},
            )
        except Exception:
            log.debug("Probability sampling failed for base model %s", job.base_model, exc_info=True)
        base_payload = prepare_decision_payload(
            job.base_model,
            working,
            prob_series=prob_series,
            include_features=job.include_features,
            update_metrics=False,
        )
        items = base_payload.get("items") or []
        total = len(items)
        passed = sum(1 for item in items if item.get("gate_pass"))
        gate_threshold = _inference_gate_threshold(artifacts)
        _record_gate_coverage(
            job.base_model,
            job.symbol,
            passed=passed,
            total=total,
            mode="inference",
            gate_threshold=gate_threshold,
        )
        payloads.append(base_payload)

    if tcn_ctx is not None:
        artifacts_tcn, model, calibrator_tcn, series_cols, scaler, window = tcn_ctx
        missing = [col for col in series_cols if col not in working.columns]
        if missing:
            raise KeyError(f"Missing required TCN feature columns {missing!r} for job {job.job_id}")
        if len(working) <= window:
            log.warning(
                "Skipping TCN inference for job %s (need >%d rows, have %d). Market data likely stale.",
                job.job_id,
                window,
                len(working),
            )
            gate_threshold = _inference_gate_threshold(artifacts_tcn)
            _record_gate_coverage(
                job.tcn_model,
                job.symbol,
                passed=0,
                total=0,
                mode="inference",
                gate_threshold=gate_threshold,
            )
        else:
            prob_frame = predict_tcn(
                working,
                model,
                calibrator_tcn,
                series_cols,
                scaler,
                window,
                stride=job.stride,
            )
            if prob_frame.empty:
                log.info("TCN probabilities empty for inference job %s", job.job_id)
                gate_threshold = _inference_gate_threshold(artifacts_tcn)
                _record_gate_coverage(
                    job.tcn_model,
                    job.symbol,
                    passed=0,
                    total=0,
                    mode="inference",
                    gate_threshold=gate_threshold,
                )
            else:
                merged = pd.merge(working, prob_frame, on="timestamp", how="inner")
                if merged.empty:
                    raise ValueError(f"No overlap between features and TCN probabilities for job {job.job_id}")
                prob_col = artifacts_tcn.prob_column or "tcn_prob"
                if prob_col != "tcn_prob" and "tcn_prob" in merged.columns and prob_col not in merged.columns:
                    merged[prob_col] = merged["tcn_prob"]
                prob_series_tcn = merged[prob_col].astype(float)
                try:
                    record_probability_samples(
                        model_label=job.tcn_model or artifacts_tcn.model_label,
                        prob_column=prob_col,
                        df=merged,
                        prob_series=prob_series_tcn,
                        source="scheduler",
                        symbol=job.symbol,
                        timeframe=job.timeframe,
                        job_id=job.job_id,
                        extra={"exchange": job.exchange, "model_kind": "tcn", "stride": job.stride},
                    )
                except Exception:
                    log.debug("Probability sampling failed for TCN model %s", job.tcn_model, exc_info=True)
                tcn_payload = prepare_decision_payload(
                    job.tcn_model,
                    merged,
                    prob_series=prob_series_tcn,
                    include_features=job.include_features,
                    update_metrics=False,
                )
                items = tcn_payload.get("items") or []
                total = len(items)
                passed = sum(1 for item in items if item.get("gate_pass"))
                gate_threshold = _inference_gate_threshold(artifacts_tcn)
                _record_gate_coverage(
                    job.tcn_model,
                    job.symbol,
                    passed=passed,
                    total=total,
                    mode="inference",
                    gate_threshold=gate_threshold,
                )
                payloads.append(tcn_payload)
                try:
                    working = working.merge(prob_frame, on="timestamp", how="left")
                except Exception as exc:
                    log.warning("Unable to merge TCN probabilities into working frame for job %s: %s", job.job_id, exc)

    if blender_ctx is not None:
        artifacts_bl, blender_model, candidate_cols, calibrator_bl = blender_ctx
        try:
            feature_frame = working.copy()
            X, cols = build_blender_features(
                feature_frame,
                candidate_cols=candidate_cols,
                use_rss_features=True,
            )
        except Exception as exc:
            log.warning("Skipping blender inference for job %s (%s): %s", job.job_id, job.blender_model, exc)
            gate_threshold = _inference_gate_threshold(artifacts_bl)
            _record_gate_coverage(
                job.blender_model,
                job.symbol,
                passed=0,
                total=0,
                mode="inference",
                gate_threshold=gate_threshold,
            )
        else:
            if X.empty:
                log.info("Blender features empty for job %s", job.job_id)
                gate_threshold = _inference_gate_threshold(artifacts_bl)
                _record_gate_coverage(
                    job.blender_model,
                    job.symbol,
                    passed=0,
                    total=0,
                    mode="inference",
                    gate_threshold=gate_threshold,
                )
            else:
                try:
                    probs = blender_model.predict_proba(X.values)
                except Exception as exc:
                    log.warning("Blender model inference failed for job %s: %s", job.job_id, exc)
                    gate_threshold = _inference_gate_threshold(artifacts_bl)
                    _record_gate_coverage(
                        job.blender_model,
                        job.symbol,
                        passed=0,
                        total=0,
                        mode="inference",
                        gate_threshold=gate_threshold,
                    )
                else:
                    if probs.shape[1] < 2:
                        raise ValueError(f"Blender model {job.blender_model} must output binary probabilities")
                    prob_col = artifacts_bl.prob_column or "blender_prob"
                    prob_series_bl = pd.Series(probs[:, 1], index=X.index, name=prob_col).astype(float)
                    if calibrator_bl is not None:
                        calibrated = apply_posthoc_calibration(
                            prob_series_bl.to_numpy(),
                            method=calibrator_bl.method,
                            estimator=calibrator_bl.estimator,
                        )
                        prob_series_bl = pd.Series(calibrated, index=prob_series_bl.index, name=prob_col)
                    scored = feature_frame.loc[X.index].copy()
                    scored[prob_col] = prob_series_bl
                    blender_payload = prepare_decision_payload(
                        job.blender_model,
                        scored,
                        prob_series=prob_series_bl,
                        include_features=job.include_features,
                        update_metrics=False,
                    )
                    items = blender_payload.get("items") or []
                    total = len(items)
                    passed = sum(1 for item in items if item.get("gate_pass"))
                    gate_threshold = _inference_gate_threshold(artifacts_bl)
                    _record_gate_coverage(
                        job.blender_model,
                        job.symbol,
                        passed=passed,
                        total=total,
                        mode="inference",
                        gate_threshold=gate_threshold,
                    )
                    payloads.append(blender_payload)

    return payloads


async def _execute_inference_job(job: InferenceJob) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    payloads = await asyncio.to_thread(_run_inference_sync, job, now)
    total_enqueued = 0
    per_model: Dict[str, int] = {}
    for payload in payloads:
        count = await _enqueue_payload(job, payload, now)
        model_label = str(payload.get("model") or "")
        if count and model_label:
            per_model[model_label] = count
            total_enqueued += count
    return {"enqueued": total_enqueued, "models": per_model}


async def run_inference_job(job: InferenceJob) -> None:
    await _run_with_metrics(job.job_id, _execute_inference_job, job)

# API health (1 = up, 0 = down)
API_HEALTH = Gauge(
    "scheduler_api_up",
    "Whether the ingestion API health endpoint responded OK (1) or not (0)",
)

# ------------------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------------------
def _auth_headers() -> Dict[str, str]:
    # Never log the token value.
    return {"X-Admin-Token": ADMIN_TOKEN}

async def _get_health() -> bool:
    url = f"{API_BASE_URL}/health"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
            r.raise_for_status()
        API_HEALTH.set(1)
        log.info('API /health OK at %s', API_BASE_URL)
        return True
    except Exception as e:
        API_HEALTH.set(0)
        log.warning("API not ready yet (%s); retrying...", e)
        return False

async def wait_for_api_ready(max_tries: int = 20, delay_s: float = 2.0) -> None:
    for _ in range(max_tries):
        if await _get_health():
            return
        await asyncio.sleep(delay_s)
    # If still not ready, continue — jobs will 404/ConnectError and we’ll record failures.

async def _post_admin(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST to an admin endpoint using query params.
    NOTE: our API expects parameters in the query string (not JSON body).
    """
    url = f"{API_BASE_URL}{path}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=_auth_headers(), params=params)
        # 2xx -> OK, else raises
        resp.raise_for_status()
        # content may be json
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text}

async def _post_json(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """POST JSON body to a path (no admin header), return JSON response."""
    url = f"{API_BASE_URL}{path}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text}

# ------------------------------------------------------------------------------
# Job wrappers (with metrics)
# ------------------------------------------------------------------------------
async def _run_with_metrics(job_id: str, coro_fn, *args, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Wrap a coroutine job with standardized metrics and logging.
    """
    JOB_RUNS.labels(job_id=job_id).inc()
    JOB_LAST_RUN_TS.labels(job_id=job_id).set(time.time())

    start = time.perf_counter()
    try:
        result = await coro_fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        JOB_DURATION.labels(job_id=job_id).observe(elapsed)
        JOB_SUCCESS.labels(job_id=job_id).inc()
        JOB_LAST_SUCCESS_TS.labels(job_id=job_id).set(time.time())
        return result
    except httpx.HTTPStatusError as e:
        elapsed = time.perf_counter() - start
        JOB_DURATION.labels(job_id=job_id).observe(elapsed)
        reason = f"http_{e.response.status_code}"
        JOB_FAILURE.labels(job_id=job_id, reason=reason).inc()
        log.error("%s failed: %s\n%s", job_id, e, "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/%s" % e.response.status_code if hasattr(e, "response") else "")
        return None
    except Exception as e:
        elapsed = time.perf_counter() - start
        JOB_DURATION.labels(job_id=job_id).observe(elapsed)
        JOB_FAILURE.labels(job_id=job_id, reason=type(e).__name__).inc()
        log.exception("%s failed: %s", job_id, e)
        return None

# ------------------------------------------------------------------------------
# Concrete jobs
# ------------------------------------------------------------------------------
async def _market_backfill(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Endpoint expects: exchange, symbol, timeframe, lookback_minutes
    return await _post_admin("/ingest/admin/backfill/market", payload)

async def _ttl_sweep(params: Dict[str, Any]) -> Dict[str, Any]:
    # Endpoint expects: pattern, ttl_default (and optional max_keys)
    return await _post_admin("/ingest/admin/features/ttl-sweep", params)

async def run_market_backfill_job(exchange: str, symbol: str, timeframe: str, lookback_minutes: int) -> None:
    job_id = f"backfill:{exchange}:{symbol}:{timeframe}"
    payload = {
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "lookback_minutes": lookback_minutes,
    }
    out = await _run_with_metrics(job_id, _market_backfill, payload)
    if out is not None:
        log.info("Market backfill ok %s -> %s", job_id, out)

async def run_ttl_sweep_job(pattern: str, ttl_default: int, max_keys: Optional[int]) -> None:
    job_id = "ttl_sweep"
    params = {"pattern": pattern, "ttl_default": ttl_default}
    if max_keys:
        params["max_keys"] = max_keys
    out = await _run_with_metrics(job_id, _ttl_sweep, params)
    if out is not None:
        log.info("TTL sweep ok -> %s", out)

async def _market_ingest(exchange: str, symbol: str, timeframe: str, limit: int) -> Dict[str, Any]:
    """
    Calls the ingest endpoint which fetches OHLCV and writes Parquet.
    Endpoint: POST /ingest/market/{exchange}
    Body: { symbol, granularity=timeframe, limit }
    """
    path = f"/ingest/market/{exchange}"
    body = {"symbol": symbol, "granularity": timeframe, "limit": limit}
    return await _post_json(path, body)

async def run_market_ingest_job(exchange: str, symbol: str, timeframe: str, limit: int) -> None:
    job_id = f"ingest:{exchange}:{symbol}:{timeframe}"
    out = await _run_with_metrics(job_id, _market_ingest, exchange, symbol, timeframe, limit)
    if out is not None:
        log.info("Market ingest ok %s -> %s", job_id, out)


async def _warm_start_inference_data(jobs: List[InferenceJob]) -> None:
    if not RUN_ON_START or not jobs:
        return
    for job in jobs:
        try:
            history_minutes = await asyncio.to_thread(_required_history_minutes, job)
            tf_minutes = max(1, _timeframe_to_minutes(job.timeframe))
            bars_needed = max(1, math.ceil(history_minutes / tf_minutes))
            ingest_limit = bars_needed + max(job.stride, 5)
            ingest_limit = max(ingest_limit, math.ceil(job.lookback_minutes / tf_minutes) if job.lookback_minutes else ingest_limit)
            ingest_limit = min(1000, max(1, ingest_limit))

            log.info(
                "Warm-starting market ingest for %s (limit=%d bars, timeframe=%s)",
                job.job_id,
                ingest_limit,
                job.timeframe,
            )
            await run_market_ingest_job(
                exchange=job.exchange,
                symbol=job.symbol,
                timeframe=job.timeframe,
                limit=ingest_limit,
            )
            log.info(
                "Warm-starting market backfill for %s (history=%d minutes)",
                job.job_id,
                history_minutes,
            )
            await run_market_backfill_job(
                exchange=job.exchange,
                symbol=job.symbol,
                timeframe=job.timeframe,
                lookback_minutes=history_minutes,
            )
        except Exception as exc:
            log.warning(
                "Warm-start backfill failed for job %s (%s): %s",
                job.job_id,
                job.symbol,
                exc,
            )

# ------------------------------------------------------------------------------
# Bootstrapping + scheduling
# ------------------------------------------------------------------------------
def _tz() -> ZoneInfo:
    try:
        return ZoneInfo(SCHED_TZ)
    except Exception:
        log.warning("Unknown timezone %s, falling back to UTC", SCHED_TZ)
        return ZoneInfo("UTC")


def _add_inference_jobs(sched: AsyncIOScheduler, jobs: List[InferenceJob]) -> None:
    for job in jobs:
        trigger = CronTrigger.from_crontab(job.cron, timezone=_tz())
        sched.add_job(
            run_inference_job,
            trigger=trigger,
            id=job.job_id,
            kwargs={"job": job},
            max_instances=1,
            replace_existing=True,
        )
        log.info(
            "Scheduled inference job %s (%s %s %s, stride=%d, cron=%s)",
            job.job_id,
            job.exchange,
            job.symbol,
            job.timeframe,
            job.stride,
            job.cron,
        )
        if RUN_ON_START:
            sched.add_job(
                run_inference_job,
                trigger="date",
                run_date=None,
                id=f"boot:{job.job_id}",
                kwargs={"job": job},
                replace_existing=True,
            )


def _add_market_jobs(sched: AsyncIOScheduler, jobs: List[Dict[str, Any]]) -> None:
    for j in jobs:
        exchange = j["exchange"]
        symbol = j["symbol"]
        timeframe = j["timeframe"]
        lookback_minutes = int(j.get("lookback_minutes", 15))
        cron = j["cron"]
        job_id = f"backfill:{exchange}:{symbol}:{timeframe}"

        # Cron job
        sched.add_job(
            run_market_backfill_job,
            trigger=CronTrigger.from_crontab(cron, timezone=_tz()),
            id=job_id,
            kwargs={
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback_minutes": lookback_minutes,
            },
            max_instances=1,  # keep it simple; dedupe/locks could be added later
            replace_existing=True,
        )

        # One-shot on boot (optional)
        if RUN_ON_START:
            sched.add_job(
                run_market_backfill_job,
                trigger="date",
                run_date=None,  # now
                id=f"boot:{exchange}:{symbol}:{timeframe}",
                kwargs={
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "lookback_minutes": lookback_minutes,
                },
                replace_existing=True,
            )

def _add_market_ingest_jobs(sched: AsyncIOScheduler, jobs: List[Dict[str, Any]]) -> None:
    for j in jobs:
        exchange = j["exchange"]
        symbol = j["symbol"]
        timeframe = j.get("timeframe") or j.get("granularity")
        if timeframe is None:
            raise ValueError(f"MARKET_INGEST_JOBS entry for {symbol} missing 'timeframe'/'granularity'")
        limit = int(j.get("limit", 500))
        cron = j["cron"]
        job_id = f"ingest:{exchange}:{symbol}:{timeframe}"

        # Cron job
        sched.add_job(
            run_market_ingest_job,
            trigger=CronTrigger.from_crontab(cron, timezone=_tz()),
            id=job_id,
            kwargs={
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
                "limit": limit,
            },
            max_instances=1,
            replace_existing=True,
        )

        # One-shot on boot (optional)
        if RUN_ON_START:
            sched.add_job(
                run_market_ingest_job,
                trigger="date",
                run_date=None,  # now
                id=f"boot-ingest:{exchange}:{symbol}:{timeframe}",
                kwargs={
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "limit": limit,
                },
                replace_existing=True,
            )

def _add_ttl_sweep_job(sched: AsyncIOScheduler) -> None:
    # Cron job
    sched.add_job(
        run_ttl_sweep_job,
        trigger=CronTrigger.from_crontab(TTL_SWEEP_CRON, timezone=_tz()),
        id="ttl_sweep",
        kwargs={
            "pattern": TTL_SWEEP_PATTERN,
            "ttl_default": TTL_SWEEP_TTL,
            "max_keys": TTL_SWEEP_MAX_KEYS,
        },
        max_instances=1,
        replace_existing=True,
    )
    # One-shot on boot (optional)
    if RUN_ON_START:
        sched.add_job(
            run_ttl_sweep_job,
            trigger="date",
            run_date=None,  # now
            id="boot:ttl_sweep",
            kwargs={
                "pattern": TTL_SWEEP_PATTERN,
                "ttl_default": TTL_SWEEP_TTL,
                "max_keys": TTL_SWEEP_MAX_KEYS,
            },
            replace_existing=True,
        )

async def _amain() -> None:
    # Start metrics HTTP server
    start_http_server(SCHED_METRICS_PORT)
    log.info("Scheduler metrics on :%d", SCHED_METRICS_PORT)
    log.info("Using API_BASE_URL=%s", API_BASE_URL)

    # Optionally wait a bit for API to come up
    await wait_for_api_ready()

    # Build scheduler
    sched = AsyncIOScheduler(timezone=_tz())
    sched.start()
    log.info("Scheduler started.")

    if INFERENCE_JOBS:
        try:
            _preload_inference_manifests(INFERENCE_JOBS)
            await _warm_start_inference_data(INFERENCE_JOBS)
        except Exception as exc:
            log.exception("Failed to preload inference manifests: %s", exc)
            raise
        _add_inference_jobs(sched, INFERENCE_JOBS)
    else:
        log.info("No inference jobs configured.")

    # Add jobs
    _add_market_jobs(sched, _get_market_jobs())
    _add_market_ingest_jobs(sched, _get_market_ingest_jobs())
    _add_ttl_sweep_job(sched)

    # Park forever
    while True:
        await asyncio.sleep(3600)

def main() -> None:
    try:
        asyncio.run(_amain())
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler exiting...")

if __name__ == "__main__":
    main()
