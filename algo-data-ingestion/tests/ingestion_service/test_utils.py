import pandas as pd
import os
from types import SimpleNamespace
import pytest
from app.ingestion_service.utils import write_to_parquet, PARQUET_WRITES_TOTAL, PARQUET_WRITE_ERRORS, PARQUET_WRITE_LATENCY
from app.ingestion_service.parquet_schemas import MARKET_SCHEMA
from app.ingestion_service.utils import validate_schema


def test_write_to_parquet_skips_empty(tmp_path, caplog, monkeypatch):
    # Empty DataFrame should return None and log info
    df = pd.DataFrame()
    result = write_to_parquet(df, str(tmp_path), {"year": 2025})
    assert result is None
    assert "Empty DataFrame, skipping Parquet write" in caplog.text


def test_write_to_parquet_success(tmp_path, monkeypatch):
    # Prepare a normalized Market-like DataFrame (UTC timestamp + required cols)
    ts = pd.to_datetime(["2025-08-01T00:00:00Z", "2025-08-01T00:01:00Z"], utc=True)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [1.0, 2.0],
        "high": [1.1, 2.1],
        "low": [0.9, 1.9],
        "close": [1.05, 2.05],
        "volume": [10.0, 20.0],
        "symbol": pd.Series(["BTCUSDT", "BTCUSDT"], dtype="string"),
        "exchange": pd.Series(["binance", "binance"], dtype="string"),
        "timeframe": pd.Series(["1m", "1m"], dtype="string"),
    })
    # Monkeypatch metrics to count calls
    writes = {"n": 0}
    latency = {"obs": []}
    monkeypatch.setattr('app.ingestion_service.utils.PARQUET_WRITES_TOTAL', SimpleNamespace(inc=lambda: writes.__setitem__('n', writes.get('n', 0) + 1)))
    monkeypatch.setattr('app.ingestion_service.utils.PARQUET_WRITE_LATENCY', SimpleNamespace(observe=lambda x: latency['obs'].append(x)))
    # Invoke write (base path must include "market" to select correct ts column)
    base = str(tmp_path / "market")
    partitions = {"exchange": "binance", "symbol": "BTCUSDT"}
    path = write_to_parquet(df, base, partitions)
    # File should exist
    assert os.path.exists(path)
    # Path includes partitions and dt
    assert "exchange=binance" in path and "symbol=BTCUSDT" in path and "dt=" in path
    # Metrics should have been called
    assert writes["n"] == 1
    assert len(latency["obs"]) == 1


def test_write_to_parquet_error(tmp_path, monkeypatch):
    # Valid, normalized Market DF (so we pass normalization and reach fs.open)
    ts = pd.to_datetime(["2025-08-01T00:00:00Z"], utc=True)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [1.0],
        "high": [1.1],
        "low": [0.9],
        "close": [1.05],
        "volume": [10.0],
        "symbol": pd.Series(["BTCUSDT"], dtype="string"),
        "exchange": pd.Series(["binance"], dtype="string"),
        "timeframe": pd.Series(["1m"], dtype="string"),
    })

    # Make fs.open raise
    import fsspec
    class BadFS:
        def __init__(self, root):
            self._root = root
        def makedirs(self, path, exist_ok=True):
            pass
        def open(self, path, mode):
            raise IOError("write error")
        def mv(self, src, dst, rename_if_exists=True):
            pass
    monkeypatch.setattr('fsspec.core.url_to_fs', lambda url: (BadFS(str(tmp_path)), str(tmp_path)))

    # Monkeypatch error counter
    errors = {"n": 0}
    monkeypatch.setattr('app.ingestion_service.utils.PARQUET_WRITE_ERRORS', SimpleNamespace(inc=lambda: errors.__setitem__('n', errors.get('n', 0) + 1)))

    # Expect exception (base MUST include 'market' for correct ts column selection)
    with pytest.raises(IOError):
        base = str(tmp_path / "market")
        write_to_parquet(df, base, {"exchange": "binance", "symbol": "BTCUSDT"})

    assert errors["n"] == 1


# Schema validation tests
def test_write_to_parquet_schema_missing_columns(tmp_path):
    df = pd.DataFrame({"symbol": ["BTC"], "open": [1.0], "high": [2.0]})
    with pytest.raises(ValueError) as exc:
        validate_schema(df, MARKET_SCHEMA, coerce=False)
    msg = str(exc.value)
    assert "Missing columns" in msg
    assert "timestamp" in msg


def test_write_to_parquet_schema_wrong_dtype(tmp_path):
    df = pd.DataFrame({
        "timestamp": ["2025-08-01T00:00:00Z"],  # wrong dtype (string, not UTC datetime)
        "symbol": pd.Series(["BTCUSDT"], dtype="string"),
        "exchange": pd.Series(["binance"], dtype="string"),
        "timeframe": pd.Series(["1m"], dtype="string"),
        "open": [1.0],
        "high": [2.0],
        "low": [0.5],
        "close": [1.5],
        "volume": [100.0],
        "dt": pd.Series(["2025-08-01"], dtype="string"),
    })
    with pytest.raises(ValueError) as exc:
        validate_schema(df, MARKET_SCHEMA, coerce=False)
    msg = str(exc.value)
    assert "timestamp" in msg and "dtype" in msg


def test_write_to_parquet_schema_coerce_int_to_float(tmp_path):
    # DataFrame with ints for OHLCV that should be coerced to float64 and a UTC timestamp
    ts = pd.to_datetime(["2025-08-01T00:00:00Z"], utc=True)
    df = pd.DataFrame({
        "timestamp": ts,
        "symbol": pd.Series(["BTCUSDT"], dtype="string"),
        "exchange": pd.Series(["binance"], dtype="string"),
        "timeframe": pd.Series(["1m"], dtype="string"),
        "open": [1],
        "high": [2],
        "low": [1],
        "close": [1],
        "volume": [100],
    })
    base = str(tmp_path / "market")
    path = write_to_parquet(df, base, {"exchange": "binance", "symbol": "BTCUSDT"})
    assert os.path.exists(path)
