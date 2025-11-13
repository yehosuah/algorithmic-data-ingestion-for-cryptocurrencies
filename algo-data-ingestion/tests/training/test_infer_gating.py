import json
from pathlib import Path

import pandas as pd
import pytest

from app.monitoring.model_metrics import (
    MODEL_GATE_COVERAGE_RATIO,
    MODEL_PROBABILITY_SIGMA,
    MODEL_PROBABILITY_SIGMA_THRESHOLD,
    MODEL_RSS_MINUTE_SPIKE_SHARE,
    MODEL_RSS_MINUTE_SPIKE_THRESHOLD,
)
from training.infer import (
    DEFAULT_GATE_CONFIG,
    apply_manifest_gates,
    compute_gate_mask,
    load_gate_config,
    load_manifest_artifacts,
    score_base_with_manifest,
)


def test_load_gate_config_reads_manifest(tmp_path):
    manifest = {
        "gates": {
            "spread_column": "hl_spread",
            "prob_column": "base_prob",
            "training": {"hl_spread_z_max": 0.1},
            "inference": {"prob_gate_min": 0.92},
        }
    }
    manifest_path = Path(tmp_path) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    cfg = load_gate_config(manifest_path.parent)

    assert cfg["inference"]["prob_gate_min"] == 0.92
    assert cfg["training"]["hl_spread_z_max"] == 0.1


def test_load_gate_config_defaults_when_missing(tmp_path):
    cfg = load_gate_config(Path(tmp_path))
    assert cfg["inference"]["prob_gate_min"] == DEFAULT_GATE_CONFIG["inference"]["prob_gate_min"]
    assert cfg["training"]["hl_spread_z_max"] == DEFAULT_GATE_CONFIG["training"]["hl_spread_z_max"]


def test_compute_gate_mask_inference_defaults():
    df = pd.DataFrame(
        {
            "hl_spread": [6e-4, 8e-4, 6e-4],
            "hl_spread_z": [-0.4, -0.2, -0.3],
            "rvol_20": [6e-5, 6e-5, 9e-5],
            "base_prob": [0.82, 0.82, 0.71],
        },
        index=pd.Index([0, 1, 2], name="idx"),
    )

    mask = compute_gate_mask(df)

    assert mask.tolist() == [True, False, False]


def test_compute_gate_mask_training_mode():
    df = pd.DataFrame(
        {
            "hl_spread_z": [0.1, 0.3],
            "rvol_20": [1.5e-4, 2.5e-4],
            "base_prob": [0.5, 0.5],
        }
    )

    mask = compute_gate_mask(df, mode="training")

    assert mask.tolist() == [True, False]


def test_compute_gate_mask_symbol_thresholds():
    df = pd.DataFrame(
        {
            "symbol": ["BTC/USDT", "SOL/USDT", "ETH/USDT"],
            "sym_spread_ratio": [1.05, 1.32, 1.28],
            "hl_spread": [5e-4, 6e-4, 5.5e-4],
            "hl_spread_z": [0.0, 0.0, 0.0],
            "rvol_20": [1e-4, 1e-4, 1e-4],
            "base_prob": [0.7, 0.7, 0.7],
        }
    )
    gate_cfg = {
        "spread_column": "hl_spread",
        "prob_column": "base_prob",
        "training": DEFAULT_GATE_CONFIG["training"],
        "inference": {
            "hl_spread_max": 0.001,
            "hl_spread_z_max": 0.5,
            "rvol20_max": 0.0003,
            "prob_gate_min": 0.6,
            "min_hold_bars": 10,
            "long_only": True,
            "sym_spread_ratio_max": {
                "default": 1.1,
                "ETH/USDT": 1.3,
                "SOL/USDT": 1.35,
            },
            "sym_rvol_ratio_max": None,
            "liquidity_rank_max": None,
        },
    }

    mask = compute_gate_mask(df, gate_cfg)

    assert mask.tolist() == [True, True, True]


def test_apply_manifest_gates_updates_metrics(tmp_path):
    model_dir = Path(tmp_path) / "model"
    model_dir.mkdir()
    manifest = {
        "gates": {
            "spread_column": "hl_spread",
            "prob_column": "base_prob",
            "training": {
                "hl_spread_max": None,
                "hl_spread_z_max": 0.25,
                "rvol20_max": 0.0002,
                "prob_gate_min": None,
                "min_hold_bars": 5,
                "long_only": False,
            },
            "inference": {
                "hl_spread_max": 0.0007,
                "hl_spread_z_max": -0.25,
                "rvol20_max": 8e-05,
                "prob_gate_min": 0.72,
                "min_hold_bars": 10,
                "long_only": False,
            },
        },
        "report_path": "report.json",
        "metadata": {
            "prob_sigma_guardrail": {"threshold": 0.03},
        },
    }
    report = {
        "rss_audit": {
            "minute_indicator_column": "rss_count_minute",
            "min_minute_spike_share": 5e-4,
            "passed": True,
        },
        "prob_sigma_guardrail": {"threshold": 0.03},
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    (model_dir / "report.json").write_text(json.dumps(report))

    df = pd.DataFrame(
        {
            "hl_spread": [5e-4, 8e-4, 6e-4, 6e-4],
            "hl_spread_z": [-0.3, -0.2, -0.4, -0.5],
            "rvol_20": [6e-5, 9e-5, 7e-5, 6e-5],
            "timestamp": pd.to_datetime(
                ["2025-01-01", "2025-01-15", "2025-02-01", "2025-02-15"], utc=True
            ),
            "rss_count_minute": [1, 0, 2, 0],
        }
    )
    prob_series = pd.Series([0.75, 0.8, 0.7, 0.9], index=df.index)

    # Ensure metrics start fresh
    MODEL_GATE_COVERAGE_RATIO.clear()
    MODEL_RSS_MINUTE_SPIKE_SHARE.clear()
    MODEL_RSS_MINUTE_SPIKE_THRESHOLD.clear()
    MODEL_PROBABILITY_SIGMA.clear()
    MODEL_PROBABILITY_SIGMA_THRESHOLD.clear()

    artifacts = load_manifest_artifacts(model_dir, model_label="calmon-base")
    mask, _ = apply_manifest_gates(
        df,
        artifacts,
        prob_series=prob_series,
        update_metrics=True,
    )

    assert mask.tolist() == [True, False, False, True]

    coverage = MODEL_GATE_COVERAGE_RATIO.labels(model="calmon-base", mode="inference")._value.get()
    assert coverage == pytest.approx(0.5)
    rss_share = MODEL_RSS_MINUTE_SPIKE_SHARE.labels(model="calmon-base")._value.get()
    assert rss_share == pytest.approx(0.5)
    rss_threshold = MODEL_RSS_MINUTE_SPIKE_THRESHOLD.labels(model="calmon-base")._value.get()
    assert rss_threshold == pytest.approx(5e-4)
    prob_sigma = MODEL_PROBABILITY_SIGMA.labels(model="calmon-base")._value.get()
    assert prob_sigma == pytest.approx(0.025, rel=1e-6)
    prob_sigma_threshold = MODEL_PROBABILITY_SIGMA_THRESHOLD.labels(model="calmon-base")._value.get()
    assert prob_sigma_threshold == pytest.approx(0.03)


def test_score_base_with_manifest_uses_gate(monkeypatch, tmp_path):
    model_dir = Path(tmp_path) / "base_model"
    model_dir.mkdir()
    manifest = {
        "gates": {
            "spread_column": "hl_spread",
            "prob_column": "base_prob",
            "inference": {
                "hl_spread_max": 0.0007,
                "hl_spread_z_max": -0.25,
                "rvol20_max": 8e-05,
                "prob_gate_min": 0.72,
                "min_hold_bars": 10,
                "long_only": False,
            },
            "training": DEFAULT_GATE_CONFIG["training"],
        },
        "report_path": "report.json",
    }
    report = {
        "rss_audit": {
            "minute_indicator_column": "rss_count_minute",
            "min_minute_spike_share": 5e-4,
            "passed": True,
        }
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    (model_dir / "report.json").write_text(json.dumps(report))

    df = pd.DataFrame(
        {
            "hl_spread": [5e-4, 9e-4],
            "hl_spread_z": [-0.3, -0.1],
            "rvol_20": [6e-5, 5e-5],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True),
            "rss_count_minute": [1, 0],
        }
    )
    prob_series = pd.Series([0.8, 0.6], index=df.index, name="base_prob")

    monkeypatch.setattr(
        "training.infer.load_base_predictor",
        lambda *args, **kwargs: ("calib", ["hl_spread", "hl_spread_z", "rvol_20"]),
    )
    monkeypatch.setattr(
        "training.infer.predict_base",
        lambda *_: prob_series,
    )

    scored = score_base_with_manifest(
        df,
        model_dir,
        update_metrics=False,
        model_label="patched-base",
    )

    assert list(scored["gate_pass"]) == [True, False]
    assert "base_prob" in scored.columns
    assert scored["base_prob"].tolist() == pytest.approx([0.8, 0.6])
