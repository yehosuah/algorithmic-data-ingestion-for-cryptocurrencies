#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from .hparam_search import random_search
from .time_series_cv import TimeSeriesSplitConfig


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run hyperparameter search with time-series CV.")
    ap.add_argument("--model", required=True, help="Model name in registry (xgb, tcn, transformer, deeplob, blender, stacking_meta, regime_blender).")
    ap.add_argument("--contract", required=True, help="Path to canonical contract file.")
    ap.add_argument("--cv-config", required=True, help="Path to CV config YAML.")
    ap.add_argument("--hparam-space", required=True, help="Path to YAML defining per-model search spaces.")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--seq-stride", type=int, default=1, help="Stride for sequence builders to reduce window count for resource safety.")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows loaded from dataset (uses most recent rows).")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--min-hold-bars", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    space = _load_yaml(args.hparam_space)
    if args.model not in space:
        raise KeyError(f"Model {args.model} not found in search space file.")
    search_space = space[args.model] or {}
    cv_cfg_raw = _load_yaml(args.cv_config)
    cv_cfg: TimeSeriesSplitConfig = {
        "n_splits": int(cv_cfg_raw.get("n_splits", 4)),
        "train_window": cv_cfg_raw.get("train_window"),
        "val_window": cv_cfg_raw.get("val_window"),
        "test_window": cv_cfg_raw.get("test_window"),
        "min_gap": cv_cfg_raw.get("min_gap"),
        "expanding": bool(cv_cfg_raw.get("expanding", cv_cfg_raw.get("train_window") is None)),
        "step": cv_cfg_raw.get("step"),
    }
    random_search(
        args.model,
        search_space,
        n_trials=args.n_trials,
        contract_path=args.contract,
        cv_config=cv_cfg,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        horizon=args.horizon,
        seq_stride=args.seq_stride,
        max_rows=args.max_rows,
        cost_bps=args.cost_bps,
        long_only=bool(args.long_only),
        min_hold_bars=args.min_hold_bars,
        seed=args.seed,
    )
    summary_path = Path(args.output_dir) / "summary_sorted.csv"
    print(f"[HParamSearch] Completed {args.n_trials} trials for {args.model}. Summary at {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
