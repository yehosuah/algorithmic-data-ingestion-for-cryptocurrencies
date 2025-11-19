#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from training.hparam_search import objective_single_model
from training.time_series_cv import TimeSeriesSplitConfig


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def _hash_cfg(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:8]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run sampling & weighting comparison experiments using best configs.")
    ap.add_argument("--contract", required=True)
    ap.add_argument("--best-configs", required=True, help="Path to best_model_configs.yaml from 7.3")
    ap.add_argument("--experiment-config", required=True, help="YAML defining experiment combos")
    ap.add_argument("--cv-config", default="configs/cv_config.yaml")
    ap.add_argument("--output-file", default="experiments/sampling_weighting_comparison.csv")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--seq_stride", type=int, default=10)
    args = ap.parse_args(argv)

    cv_cfg_raw = _load_yaml(args.cv_config)
    cv_cfg: TimeSeriesSplitConfig = {
        "n_splits": int(cv_cfg_raw.get("n_splits", 3)),
        "train_window": cv_cfg_raw.get("train_window"),
        "val_window": cv_cfg_raw.get("val_window"),
        "test_window": cv_cfg_raw.get("test_window"),
        "min_gap": cv_cfg_raw.get("min_gap"),
        "expanding": bool(cv_cfg_raw.get("expanding", True)),
        "step": cv_cfg_raw.get("step"),
    }

    best_cfgs = _load_yaml(args.best_configs)
    exp_cfg = _load_yaml(args.experiment_config)
    experiments = exp_cfg.get("experiments", [])
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for exp in experiments:
        model = exp["model_name"]
        if model not in best_cfgs:
            print(f"[SamplingExp] Skipping {model}: not in best configs")
            continue
        hparams = best_cfgs[model].get("params", {})
        sampling_policy = exp.get("sampling_policy")
        sampling_config_path = exp.get("sampling_config")
        sampling_config = _load_yaml(sampling_config_path) if isinstance(sampling_config_path, str) else exp.get("sampling_config", {})
        if sampling_policy and sampling_policy in sampling_config:
            sampling_config = sampling_config.get(sampling_policy, sampling_config)
        weight_policy = exp.get("weight_policy")
        weight_config_path = exp.get("weight_config")
        weight_config = _load_yaml(weight_config_path) if isinstance(weight_config_path, str) else exp.get("weight_config", {})
        if weight_policy and weight_policy in weight_config:
            weight_config = weight_config.get(weight_policy, weight_config)

        cfg_hash = _hash_cfg({"model": model, "sampling_policy": sampling_policy, "weight_policy": weight_policy, "hparams": hparams})
        print(f"[SamplingExp] Running {model} sampling={sampling_policy} weight={weight_policy} hash={cfg_hash}")
        res = objective_single_model(
            model,
            hparams,
            args.contract,
            cv_cfg,
            seq_len=args.seq_len,
            seq_stride=args.seq_stride,
            max_rows=args.max_rows,
            sampling_policy=sampling_policy,
            sampling_config=sampling_config,
            weight_policy=weight_policy,
            weight_config=weight_config,
        )
        row = {
            "model_name": model,
            "sampling_policy": sampling_policy,
            "weight_policy": weight_policy,
            "config_hash": cfg_hash,
            "mean_pnl_net_cv": res.get("mean_pnl_net_cv"),
            "mean_sharpe_cv": res.get("mean_sharpe_cv"),
            "mean_hit_rate_cv": res.get("mean_hit_rate_cv"),
            "regime_pnl_variance_cv": res.get("regime_pnl_variance_cv"),
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"[SamplingExp] Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
