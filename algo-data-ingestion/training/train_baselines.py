#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from training.experiment_utils import (
    append_leaderboard,
    evaluate_predictions,
    generate_cv_predictions,
    prepare_canonical_data,
)
from training.blender import BlenderModel
from training.model_registry import MODEL_REGISTRY
# Ensure model classes register themselves
from training import tcn_model  # noqa: F401
from training import model as xgb_model  # noqa: F401
from training import transformer_model  # noqa: F401
from training import deeplob_model  # noqa: F401


SEQUENCE_MODELS = {"tcn", "transformer", "deeplob"}


def _train_blender(bundle, base_runs: Dict[str, any], output_dir: Path, args) -> pd.Series:
    train_idx = bundle.df.index[: bundle.test_start]
    test_idx = bundle.df.index[bundle.test_start :]
    train_pred_df = pd.DataFrame({k: v.oof_preds for k, v in base_runs.items()}).reindex(train_idx)
    train_pred_df = train_pred_df.dropna()
    common_train_idx = train_pred_df.index
    if common_train_idx.empty:
        raise RuntimeError("No overlapping OOS predictions to train blender.")
    y_train = bundle.df.loc[common_train_idx, bundle.label_col].to_numpy()
    blender = BlenderModel(base_model_names=list(base_runs.keys()), config={})
    blender.fit({k: train_pred_df[k].to_numpy() for k in train_pred_df.columns}, y_train)

    test_pred_df = pd.DataFrame({k: v.test_preds for k, v in base_runs.items()}).reindex(test_idx)
    test_pred_df = test_pred_df.dropna()
    common_test_idx = test_pred_df.index
    if common_test_idx.empty:
        raise RuntimeError("No overlapping test predictions for blender.")
    blended = blender.predict_proba({k: test_pred_df[k].to_numpy() for k in test_pred_df.columns})
    pred_series = pd.Series(blended, index=common_test_idx)
    blender_dir = output_dir / "blender"
    blender.save(str(blender_dir))
    return pred_series


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train baseline models (TCN, XGB, Blender) on canonical contract.")
    ap.add_argument("--contract", default="configs/canonical_training_contract_market_multi_3symbol_1m.yaml")
    ap.add_argument("--models", nargs="+", default=["tcn", "xgb", "blender"], help="Models to train/evaluate")
    ap.add_argument("--output_dir", default="experiments/baselines/")
    ap.add_argument("--seq_len", type=int, default=32)
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

    print(f"[Baselines] Loading contract {args.contract}")
    bundle = prepare_canonical_data(
        args.contract,
        seq_len=args.seq_len,
        horizon=args.horizon,
        test_size=args.test_size,
    )

    runs: Dict[str, any] = {}
    for name in args.models:
        if name == "blender":
            continue
        if name not in MODEL_REGISTRY:
            print(f"[Baselines] Skipping unknown model {name}")
            continue
        use_seq = name in SEQUENCE_MODELS
        print(f"[Baselines] Training {name} (sequences={use_seq})")
        run = generate_cv_predictions(
            name,
            bundle,
            n_folds=args.n_folds,
            embargo_minutes=args.embargo_minutes,
            model_config={},
            use_sequences=use_seq,
        )
        runs[name] = run
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
        print(f"[Baselines] {name} metrics: {json.dumps(metrics, indent=2)}")

    if "blender" in args.models:
        print("[Baselines] Training blender on OOS predictions")
        if not runs:
            raise RuntimeError("No base model runs available to train blender.")
        blended_series = _train_blender(bundle, runs, output_dir, args)
        metrics = evaluate_predictions(
            bundle,
            blended_series,
            cost_bps=args.cost_bps,
            spread_col=args.spread_col,
            long_only=args.long_only,
            min_hold_bars=args.min_hold_bars,
        )
        (output_dir / "blender_metrics.json").write_text(json.dumps(metrics, indent=2))
        row = {
            "model_name": "blender",
            "base_models": list(runs.keys()),
            "config_path": args.contract,
            "pnl_net": metrics.get("pnl_net"),
            "sharpe": metrics.get("sharpe"),
            "hit_rate": metrics.get("hit_rate"),
            "max_drawdown": metrics.get("max_drawdown"),
            "regime_pnl": metrics.get("regime_pnl"),
            "output_dir": str(output_dir / "blender"),
        }
        append_leaderboard(row, Path(args.leaderboard))
        print(f"[Baselines] blender metrics: {json.dumps(metrics, indent=2)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
