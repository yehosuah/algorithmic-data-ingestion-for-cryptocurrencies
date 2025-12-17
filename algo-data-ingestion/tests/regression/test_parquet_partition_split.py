from __future__ import annotations

import os

import pandas as pd

from app.ingestion_service.utils import write_to_parquet


def test_write_to_parquet_midnight_split_keeps_day_partition_consistent(tmp_path):
    ts = pd.to_datetime(["2025-12-12T23:59:00Z", "2025-12-13T00:00:00Z"], utc=True)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
            "volume": [10.0, 20.0],
            "symbol": pd.Series(["BTC/USDT", "BTC/USDT"], dtype="string"),
            "exchange": pd.Series(["binance", "binance"], dtype="string"),
            "timeframe": pd.Series(["1m", "1m"], dtype="string"),
        }
    )

    base = str(tmp_path / "market")
    # Simulate route-level partitions derived from the first row (previous day).
    partitions = {"exchange": "binance", "symbol": "BTC/USDT", "year": 2025, "month": 12, "day": 12}
    last_path = write_to_parquet(df, base, partitions)

    assert os.path.exists(last_path)
    # The newest dt write should land under day=13/dt=2025-12-13 (not day=12/dt=2025-12-13).
    assert "day=13" in str(last_path)
    assert "dt=2025-12-13" in str(last_path)

    wrong = (
        tmp_path
        / "market"
        / "exchange=binance"
        / "symbol=BTC-USDT"
        / "year=2025"
        / "month=12"
        / "day=12"
        / "dt=2025-12-13"
    )
    assert not wrong.exists()

    expected_prev = (
        tmp_path
        / "market"
        / "exchange=binance"
        / "symbol=BTC-USDT"
        / "year=2025"
        / "month=12"
        / "day=12"
        / "dt=2025-12-12"
    )
    expected_next = (
        tmp_path
        / "market"
        / "exchange=binance"
        / "symbol=BTC-USDT"
        / "year=2025"
        / "month=12"
        / "day=13"
        / "dt=2025-12-13"
    )
    assert expected_prev.exists()
    assert expected_next.exists()

