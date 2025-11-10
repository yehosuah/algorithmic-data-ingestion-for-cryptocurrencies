#!/usr/bin/env python3
"""
Replay a historical dataset through the blender manifest so `training.infer.apply_manifest_gates`
runs and emits Prometheus metrics for the gate coverage gauges.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
from prometheus_client import start_http_server

# Ensure the inference metrics register on the default/global Prometheus registry so
# the standalone HTTP server we spawn can expose them.
os.environ.setdefault("USE_INGEST_METRICS_REGISTRY", "0")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion_service.manifests import get_manifest_registry
from app.ingestion_service.scoring import get_scoring_service


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a parquet dataset through the blender manifest and expose gate metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-root",
        default="models",
        help="Directory that contains deployable manifests.",
    )
    parser.add_argument(
        "--model-label",
        default="blender_h120_v6",
        help="Manifest label to load from the registry.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional relative/absolute path to the manifest directory; defaults to model-label.",
    )
    parser.add_argument(
        "--dataset",
        default="datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet",
        help="Parquet dataset that already contains the blender feature columns.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=2000,
        help="Row offset inside the dataset to begin the replay from.",
    )
    parser.add_argument(
        "--row-count",
        type=int,
        default=2000,
        help="Number of consecutive rows to replay. Use -1 to consume the remainder of the dataset.",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9002,
        help="Port used by the embedded Prometheus HTTP exporter.",
    )
    return parser.parse_args(argv)


def _load_frame(dataset_path: Path, start_row: int, row_count: int) -> pd.DataFrame:
    frame = pd.read_parquet(dataset_path)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    start = max(0, int(start_row))
    if row_count is None or row_count < 0:
        end = len(frame)
    else:
        end = min(len(frame), start + int(row_count))
    if start >= end:
        raise ValueError(
            f"Dataset slice is empty (start={start}, end={end}, total_rows={len(frame)})"
        )
    return frame.iloc[start:end].reset_index(drop=True)


def _score_blender_slice(
    df: pd.DataFrame,
    model_label: str,
    models_root: Path,
    manifest_path: Path,
) -> Tuple[int, int]:
    registry = get_manifest_registry()
    registry.preload(
        models_root=models_root,
        specs=[(model_label, str(manifest_path))],
        clear=True,
    )
    scoring = get_scoring_service()
    payload = scoring.score_batch(
        model_label,
        df,
        include_features=False,
        update_metrics=True,
    )
    items = payload.get("items") or []
    passed = sum(1 for item in items if item.get("gate_pass"))
    return passed, len(items)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    models_root = Path(args.models_root).expanduser().resolve()
    manifest_rel = args.model_path or args.model_label
    manifest_path = Path(manifest_rel)
    if not manifest_path.is_absolute():
        manifest_path = (models_root / manifest_path).resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest directory not found: {manifest_path}")

    start_http_server(int(args.metrics_port))
    print(f"[blender-replay] metrics server running on :{args.metrics_port}", flush=True)

    frame = _load_frame(dataset_path, args.start_row, args.row_count)
    passed, total = _score_blender_slice(
        frame,
        args.model_label,
        models_root,
        manifest_path,
    )
    coverage = float(passed) / float(total) if total else 0.0
    ts_start = frame["timestamp"].iloc[0]
    ts_end = frame["timestamp"].iloc[-1]
    print(
        f"[blender-replay] scored {total} rows "
        f"({ts_start} → {ts_end}), gate_pass={passed}, coverage={coverage:.4f}",
        flush=True,
    )

    # Keep the process alive so Prometheus continues to scrape the registry.
    try:
        while True:
            time.sleep(30.0)
    except KeyboardInterrupt:
        print("[blender-replay] shutting down", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
