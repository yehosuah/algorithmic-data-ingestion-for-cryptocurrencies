#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from training.experiment_utils import (
    append_leaderboard,
    compute_regime_id,
    evaluate_predictions,
    generate_cv_predictions,
    prepare_canonical_data,
)
from training.meta_dataset_builder import build_meta_dataset
from training.blender import BlenderModel
from training.regime_blender import RegimeBlender
from training.stacking_meta_model import StackingMetaModel
from training.model_registry import MODEL_REGISTRY
# Ensure model classes register
from training import tcn_model  # noqa: F401
from training import model as xgb_model  # noqa: F401
from training import transformer_model  # noqa: F401
from training import deeplob_model  # noqa: F401

SEQUENCE_MODELS = {"tcn", "transformer", "deeplob"}


def _align_preds(runs: Dict[str, any], indices: pd.Index) -> pd.DataFrame:
    df = pd.DataFrame({k: v.oof_preds for k, v in runs.items()}).reindex(indices)
    return df.dropna()


def _align_test_preds(runs: Dict[str, any], indices: pd.Index) -> pd.DataFrame:
    df = pd.DataFrame({k: v.test_preds for k, v in runs.items()}).reindex(indices)
    return df.dropna()


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train advanced blenders and regime-aware ensembles.")
    ap.add_argument("--contract", default="configs/canonical_training_contract_market_multi_3symbol_1m.yaml")
    ap.add_argument("--base_models", nargs="+", default=["tcn", "xgb", "transformer"])
    ap.add_argument("--blenders", nargs="+", default=["blender", "stacking_meta", "regime_blender"])
    ap.add_argument("--output_dir", default="experiments/blenders/")
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

    bundle = prepare_canonical_data(
        args.contract,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        horizon=args.horizon,
        test_size=args.test_size,
    )
    base_runs: Dict[str, any] = {}
    for name in args.base_models:
        if name not in MODEL_REGISTRY:
            print(f"[Blenders] Unknown base model {name}; skipping")
            continue
        use_seq = name in SEQUENCE_MODELS
        run = generate_cv_predictions(
            name,
            bundle,
            n_folds=args.n_folds,
            embargo_minutes=args.embargo_minutes,
            model_config={},
            use_sequences=use_seq,
        )
        base_runs[name] = run
        run.model.save(str(output_dir / name))

    train_idx = bundle.df.index[: bundle.test_start]
    test_idx = bundle.df.index[bundle.test_start :]
    base_train_df = _align_preds(base_runs, train_idx)
    base_test_df = _align_test_preds(base_runs, test_idx)
    if base_train_df.empty or base_test_df.empty:
        raise RuntimeError("Base model predictions did not overlap; cannot fit blenders.")
    regime_train = compute_regime_id(bundle.df.loc[base_train_df.index], bundle.regime_cols)
    regime_test = compute_regime_id(bundle.df.loc[base_test_df.index], bundle.regime_cols)
    y_train = bundle.df.loc[base_train_df.index, bundle.label_col].to_numpy()

    if "blender" in args.blenders:
        blender = BlenderModel(base_model_names=list(base_runs.keys()), config={})
        blender.fit({c: base_train_df[c].to_numpy() for c in base_train_df.columns}, y_train)
        blended = blender.predict_proba({c: base_test_df[c].to_numpy() for c in base_test_df.columns})
        blended_series = pd.Series(blended, index=base_test_df.index)
        metrics = evaluate_predictions(
            bundle,
            blended_series,
            cost_bps=args.cost_bps,
            spread_col=args.spread_col,
            long_only=args.long_only,
            min_hold_bars=args.min_hold_bars,
        )
        blender.save(str(output_dir / "blender"))
        (output_dir / "blender_metrics.json").write_text(json.dumps(metrics, indent=2))
        append_leaderboard(
            {
                "model_name": "blender",
                "base_models": list(base_runs.keys()),
                "config_path": args.contract,
                "pnl_net": metrics.get("pnl_net"),
                "sharpe": metrics.get("sharpe"),
                "hit_rate": metrics.get("hit_rate"),
                "max_drawdown": metrics.get("max_drawdown"),
                "regime_pnl": metrics.get("regime_pnl"),
                "output_dir": str(output_dir / "blender"),
            },
            Path(args.leaderboard),
        )
        print(f"[Blenders] blender metrics: {json.dumps(metrics, indent=2)}")

    if "stacking_meta" in args.blenders:
        X_meta, y_meta = build_meta_dataset(
            {c: base_train_df[c].to_numpy() for c in base_train_df.columns},
            y_train,
            regimes=regime_train.values,
        )
        stacker = StackingMetaModel(config={})
        stacker.fit(X_meta, y_meta)
        meta_test = pd.DataFrame({c: base_test_df[c].to_numpy() for c in base_test_df.columns}, index=base_test_df.index)
        if bundle.regime_cols:
            reg_ohe = pd.get_dummies(regime_test.astype(str), prefix="regime")
            reg_ohe.index = meta_test.index
            meta_test = pd.concat([meta_test, reg_ohe], axis=1)
        preds = stacker.predict_proba(meta_test)
        pred_series = pd.Series(preds, index=meta_test.index)
        metrics = evaluate_predictions(
            bundle,
            pred_series,
            cost_bps=args.cost_bps,
            spread_col=args.spread_col,
            long_only=args.long_only,
            min_hold_bars=args.min_hold_bars,
        )
        stacker.save(str(output_dir / "stacking_meta"))
        (output_dir / "stacking_meta_metrics.json").write_text(json.dumps(metrics, indent=2))
        append_leaderboard(
            {
                "model_name": "stacking_meta",
                "base_models": list(base_runs.keys()),
                "config_path": args.contract,
                "pnl_net": metrics.get("pnl_net"),
                "sharpe": metrics.get("sharpe"),
                "hit_rate": metrics.get("hit_rate"),
                "max_drawdown": metrics.get("max_drawdown"),
                "regime_pnl": metrics.get("regime_pnl"),
                "output_dir": str(output_dir / "stacking_meta"),
            },
            Path(args.leaderboard),
        )
        print(f"[Blenders] stacking_meta metrics: {json.dumps(metrics, indent=2)}")

    if "regime_blender" in args.blenders:
        rb = RegimeBlender(base_model_names=list(base_runs.keys()), config={})
        rb.fit(
            {c: base_train_df[c].to_numpy() for c in base_train_df.columns},
            y_train,
            regime_ids=regime_train.values,
        )
        preds = rb.predict_proba(
            {c: base_test_df[c].to_numpy() for c in base_test_df.columns},
            regime_ids=regime_test.values,
        )
        pred_series = pd.Series(preds, index=base_test_df.index)
        metrics = evaluate_predictions(
            bundle,
            pred_series,
            cost_bps=args.cost_bps,
            spread_col=args.spread_col,
            long_only=args.long_only,
            min_hold_bars=args.min_hold_bars,
        )
        rb.save(str(output_dir / "regime_blender"))
        (output_dir / "regime_blender_metrics.json").write_text(json.dumps(metrics, indent=2))
        append_leaderboard(
            {
                "model_name": "regime_blender",
                "base_models": list(base_runs.keys()),
                "config_path": args.contract,
                "pnl_net": metrics.get("pnl_net"),
                "sharpe": metrics.get("sharpe"),
                "hit_rate": metrics.get("hit_rate"),
                "max_drawdown": metrics.get("max_drawdown"),
                "regime_pnl": metrics.get("regime_pnl"),
                "output_dir": str(output_dir / "regime_blender"),
            },
            Path(args.leaderboard),
        )
        print(f"[Blenders] regime_blender metrics: {json.dumps(metrics, indent=2)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
