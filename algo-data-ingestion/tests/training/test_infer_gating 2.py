import json
from pathlib import Path

import pandas as pd

from training.infer import load_gate_config, compute_gate_mask, DEFAULT_GATE_CONFIG


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
