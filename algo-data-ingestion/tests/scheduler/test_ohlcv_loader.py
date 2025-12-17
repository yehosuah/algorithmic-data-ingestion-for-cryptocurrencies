from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from app.scheduler import main as scheduler_main


@dataclass
class DummyJob:
    job_id: str
    timeframe: str
    _data_dir: Path

    def data_dir(self) -> Path:
        return self._data_dir


def _write_parquet(dir_path: Path, *, start: str, end: str, freq: str = "1min") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    df = pd.DataFrame({"timestamp": ts})
    df.to_parquet(dir_path / f"part-{int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)}.parquet", index=False)


def test_load_recent_ohlcv_handles_duplicate_dt_dirs(tmp_path):
    base = tmp_path / "market" / "exchange=binance" / "symbol=BTC-USDT"

    # Create the "newer" dt dir first (later timestamps).
    dt_new = base / "year=2025" / "month=12" / "day=13" / "dt=2025-12-13"
    _write_parquet(dt_new, start="2025-12-13T05:22:00Z", end="2025-12-13T11:21:00Z")

    # Create the "older" dt dir second so its directory mtime is later (worst-case ordering).
    dt_old = base / "year=2025" / "month=12" / "day=12" / "dt=2025-12-13"
    _write_parquet(dt_old, start="2025-12-13T00:00:00Z", end="2025-12-13T05:58:00Z")

    job = DummyJob(job_id="test_job", timeframe="1m", _data_dir=base)
    now = datetime(2025, 12, 13, 11, 30, tzinfo=timezone.utc)
    history_minutes = 150  # cutoff = 09:00Z
    cutoff = now - timedelta(minutes=history_minutes)

    ohlcv = scheduler_main._load_recent_ohlcv(job, now, history_minutes)
    assert not ohlcv.empty
    assert ohlcv["timestamp"].min() >= cutoff
    assert ohlcv["timestamp"].max() <= now
