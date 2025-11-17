#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from training.experiment_utils import (
    append_leaderboard,
    evaluate_predictions,
    generate_cv_predictions,
    prepare_canonical_data,
)
from training.model_registry import MODEL_REGISTRY
# Ensure models register
from training import tcn_model  # noqa: F401
from training import model as xgb_model  # noqa: F401
from training import transformer_model  # noqa: F401
from training import deeplob_model  # noqa: F401

SEQUENCE_MODELS = {"tcn", "transformer", "deeplob"}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Architecture search for sequence/tabular baselines.")
    ap.add_argument("--contract", default="configs/canonical_training_contract_market_multi_3symbol_1m.yaml")
    ap.add_argument("--models", nargs="+", default=["tcn", "xgb", "transformer", "deeplob"])
    ap.add_argument("--output_dir", default="experiments/arch_search/")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--seq_stride", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--n_folds", type=int, default=4)
    ap.add_argument("--embargo_minutes", type=int, default=60)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    ap.add_argument("--spread_col", default=None)
    ap.add_argument("--leaderboard", default="experiments/leaderboard_arch_search.csv")
    ap.add_argument("--long_only", action="store_true")
    ap.add_argument("--min_hold_bars", type=int, default=1)
    args = ap.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ArchSearch] Loading contract {args.contract}")
    bundle = prepare_canonical_data(
        args.contract,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        horizon=args.horizon,
        test_size=args.test_size,
    )

    for name in args.models:
        if name not in MODEL_REGISTRY:
            print(f"[ArchSearch] Unknown model {name}; skipping")
            continue
        use_seq = name in SEQUENCE_MODELS
        print(f"[ArchSearch] Training {name} (sequences={use_seq})")
        run = generate_cv_predictions(
            name,
            bundle,
            n_folds=args.n_folds,
            embargo_minutes=args.embargo_minutes,
            model_config={},
            use_sequences=use_seq,
        )
        model_dir = output_dir / name
        run.model.save(str(model_dir))

        metrics = evaluate_predictions(
            bundle,
            run.test_preds,
            cost_bps=args.cost_bps,
            spread_col=args.spread_col,
            long_only=args.long_only,
            min_hold_bars=args.min_hold_bars,
        )
        (output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
        row = {
            "model_name": name,
            "base_models": None,
            "config_path": args.contract,
            "pnl_net": metrics.get("pnl_net"),
            "sharpe": metrics.get("sharpe"),
            "hit_rate": metrics.get("hit_rate"),
            "max_drawdown": metrics.get("max_drawdown"),
            "regime_pnl": metrics.get("regime_pnl"),
            "output_dir": str(model_dir),
        }
        append_leaderboard(row, Path(args.leaderboard))
        print(f"[ArchSearch] {name} metrics: {json.dumps(metrics, indent=2)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
