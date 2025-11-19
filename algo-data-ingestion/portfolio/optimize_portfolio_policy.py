from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yaml

from training.data import load_training_dataset, load_canonical_contract

from .predictions import generate_oos_predictions_for_models
from .simulator import run_portfolio_simulation


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r") as fh:
        return yaml.safe_load(fh) or {}


def _iter_thresholds(grid: Dict[str, Iterable[float]]) -> Iterable[Dict[str, float]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield {k: float(v) for k, v in zip(keys, combo)}


def _iter_weights(grid: Dict[str, Iterable[float]]) -> Iterable[Dict[str, float]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield {k: float(v) for k, v in zip(keys, combo)}


def _prepare_threshold_map(models: List[str], thr_cfg: Dict[str, float]) -> Dict[str, dict]:
    return {m: {"global": thr_cfg} for m in models}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Portfolio-layer threshold/weight search using canonical dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--contract", required=True, help="Canonical contract path.")
    ap.add_argument("--best-model-configs", required=True, help="YAML with best hyperparameters per model.")
    ap.add_argument("--risk-config", required=True, help="Portfolio risk limits YAML.")
    ap.add_argument("--model-weights-config", required=True, help="Base model weight YAML.")
    ap.add_argument("--policy-search-config", required=True, help="Grid search YAML.")
    ap.add_argument("--output-dir", required=True, help="Directory to write results.")
    ap.add_argument("--max-combinations", type=int, default=None, help="Optional cap overriding config.")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional tail-cap to speed up iteration.")
    ap.add_argument("--n-folds", type=int, default=None, help="Override folds for prediction generation.")
    ap.add_argument("--embargo-minutes", type=int, default=60)
    args = ap.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = load_canonical_contract(args.contract)
    df = load_training_dataset(contract)
    if args.max_rows:
        df = df.tail(int(args.max_rows)).reset_index(drop=True)
    search_cfg = _load_yaml(args.policy_search_config)
    weight_cfg = _load_yaml(args.model_weights_config)
    risk_cfg = _load_yaml(args.risk_config)
    models = list(search_cfg.get("models") or weight_cfg.get("models") or [])
    if not models:
        raise ValueError("No models specified in search or weights config")

    preds = generate_oos_predictions_for_models(
        contract_path=args.contract,
        model_configs_path=args.best_model_configs,
        model_names=models,
        sampling_policy=search_cfg.get("sampling_policy"),
        weight_policy=search_cfg.get("weight_policy"),
        max_rows=args.max_rows,
        n_folds=args.n_folds,
        embargo_minutes=args.embargo_minutes,
    )

    thr_grid = search_cfg.get("thresholds_grid") or {}
    weights_grid = search_cfg.get("weights_grid") or {}
    max_combos = args.max_combinations or search_cfg.get("max_combinations") or 0

    thr_candidates = list(_iter_thresholds(thr_grid)) or [{}]
    weight_candidates = list(_iter_weights(weights_grid)) or [{}]
    combos = list(itertools.product(thr_candidates, weight_candidates))
    random.shuffle(combos)
    if max_combos and len(combos) > max_combos:
        combos = combos[: int(max_combos)]

    results: List[Dict[str, object]] = []
    for idx, (thr_cfg, weight_map) in enumerate(combos, start=1):
        thresholds = _prepare_threshold_map(models, thr_cfg)
        run_risk_cfg = dict(risk_cfg)
        run_risk_cfg["model_weights"] = weight_map
        run_risk_cfg["ensemble_mode"] = weight_cfg.get("mode", "weighted_sum")

        sim = run_portfolio_simulation(
            df=df,
            model_signals=preds,
            thresholds=thresholds,
            risk_config=run_risk_cfg,
            sampling_policy=search_cfg.get("sampling_policy"),
            weight_policy=search_cfg.get("weight_policy"),
        )
        metrics = sim["metrics"]
        result = {
            "policy_id": f"policy_{idx:04d}",
            "thresholds": json.dumps(thr_cfg),
            "weights": json.dumps(weight_map),
            "models": ",".join(models),
        }
        result.update(metrics)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_path = output_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
