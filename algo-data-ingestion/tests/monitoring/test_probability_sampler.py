import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.monitoring.probability_sampler import ProbabilitySampler, ProbabilitySampleConfig


def _build_frame(n: int = 8) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTC/USDT"] * n,
            "timeframe": ["1m"] * n,
            "feature_a": range(n),
        }
    )


def test_probability_sampler_writes_json(tmp_path) -> None:
    config = ProbabilitySampleConfig(
        enabled=True,
        file_root=tmp_path,
        max_rows=4,
        redis_url=None,
        redis_stream="probability:samples",
        redis_maxlen=0,
    )
    sampler = ProbabilitySampler(config)
    df = _build_frame(10)
    probs = pd.Series([0.1 * i for i in range(10)], index=df.index, name="base_prob")

    sampler.record(
        model_label="base_xgb",
        prob_column="base_prob",
        df=df,
        prob_series=probs,
        source="test",
        symbol="BTC/USDT",
        timeframe="1m",
        job_id="job-1",
        extra={"runner": "unit"},
    )

    out_path = tmp_path / "base_xgb_base_prob.jsonl"
    assert out_path.exists()
    lines = [json.loads(line) for line in out_path.read_text().strip().splitlines()]
    assert len(lines) == config.max_rows  # bounded tail
    assert lines[-1]["probability"] == probs.iloc[-1]
    assert lines[-1]["runner"] == "unit"
    assert lines[-1]["job_id"] == "job-1"
    for entry in lines:
        assert entry["model"] == "base_xgb"
        assert entry["prob_column"] == "base_prob"
        assert entry["source"] == "test"
        assert entry["symbol"] == "BTC/USDT"
        assert entry["timeframe"] == "1m"
        assert entry["timestamp"].endswith("+00:00")


def test_probability_sampler_disabled(tmp_path) -> None:
    config = ProbabilitySampleConfig(
        enabled=False,
        file_root=tmp_path,
        max_rows=2,
        redis_url=None,
        redis_stream="probability:samples",
        redis_maxlen=0,
    )
    sampler = ProbabilitySampler(config)
    df = _build_frame(2)
    sampler.record(
        model_label="base_xgb",
        prob_column="base_prob",
        df=df,
        prob_series=pd.Series([0.1, 0.2], index=df.index),
        source="test",
    )
    out_path = tmp_path / "base_xgb_base_prob.jsonl"
    assert not out_path.exists()
