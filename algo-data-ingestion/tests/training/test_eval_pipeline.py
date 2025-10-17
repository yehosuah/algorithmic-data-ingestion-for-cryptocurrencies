import pandas as pd
import pytest

from training.eval_pipeline import compare_oos_frames


def test_compare_oos_frames_reports_diffs():
    timestamps = pd.date_range("2025-01-01", periods=3, freq="T", tz="UTC")
    current = pd.DataFrame(
        {
            "timestamp": timestamps,
            "prob_calibrated": [0.1, 0.2, 0.3],
            "prob_uncalibrated": [0.1, 0.21, 0.32],
            "ret_next": [0.01, -0.02, 0.03],
            "label": [0, 1, 0],
            "gate_training": [True, False, True],
            "gate_inference": [True, True, False],
        }
    )
    baseline = pd.DataFrame(
        {
            "timestamp": timestamps,
            "prob_calibrated": [0.1, 0.25, 0.28],
            "prob_uncalibrated": [0.12, 0.20, 0.30],
            "ret_next": [0.01, -0.03, 0.035],
            "label": [0, 0, 0],
            "gate_training": [True, True, True],
            "gate_inference": [True, False, False],
        }
    )

    metrics = compare_oos_frames(current, baseline)

    assert metrics["row_count_current"] == 3
    assert metrics["row_count_baseline"] == 3
    assert metrics["overlap_rows"] == 3
    assert metrics["label_mismatch_rate"] == pytest.approx(1 / 3)
    assert metrics["gate_training_mismatch_rate"] == pytest.approx(1 / 3)
    assert metrics["gate_inference_mismatch_rate"] == pytest.approx(1 / 3)
    assert metrics["prob_calibrated_mae"] == pytest.approx((0.0 + 0.05 + 0.02) / 3)
    assert "prob_calibrated_corr" in metrics


def test_compare_oos_frames_handles_empty_baseline():
    current = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="T", tz="UTC"),
            "prob_calibrated": [0.1, 0.2],
        }
    )
    baseline = pd.DataFrame(columns=["timestamp", "prob_calibrated"])

    metrics = compare_oos_frames(current, baseline)

    assert metrics["row_count_baseline"] == 0
    assert metrics["overlap_rows"] == 0
