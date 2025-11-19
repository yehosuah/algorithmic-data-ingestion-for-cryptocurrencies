from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml

from training.data import load_canonical_contract, load_training_dataset
from training.model import extract_features_labels, train_xgb, calibrate, save_artifacts

from .predictions import generate_oos_predictions_for_models
from .simulator import run_portfolio_simulation


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r") as fh:
        return yaml.safe_load(fh) or {}


def finalize_portfolio_policies(
    contract_path: str,
    best_model_configs_path: str,
    final_policies_path: str,
    risk_limits_path: str,
    output_dir: str,
    *,
    max_rows: int | None = None,
    oos_fraction: float = 0.1,
) -> None:
    """
    Evaluate final candidate policies on a fresh OOS window and persist diagnostics.

    The function trains lightweight artifacts for XGB (deployability) and runs
    portfolio simulation over a held-out OOS slice.
    """
    contract = load_canonical_contract(contract_path)
    df = load_training_dataset(contract)
    if max_rows:
        df = df.tail(int(max_rows)).reset_index(drop=True)
    if "regime_id" in df.columns:
        df = df.drop(columns=["regime_id"])
    obj_cols = [c for c in df.columns if df[c].dtype == object and c != "symbol"]
    if obj_cols:
        df = df.drop(columns=obj_cols)
    policies = _load_yaml(final_policies_path)
    risk_cfg = _load_yaml(risk_limits_path)

    # Train / OOS split
    split_idx = int(len(df) * (1.0 - float(oos_fraction)))
    split_idx = max(10, min(len(df) - 1, split_idx))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_oos = df.iloc[split_idx:].reset_index(drop=True)

    # Collect required model names across policies to avoid duplicate computation
    model_names: List[str] = []
    for pol in policies.values():
        for m in pol.get("models", []):
            if m not in model_names:
                model_names.append(m)

    preds = generate_oos_predictions_for_models(
        contract_path=contract_path,
        model_configs_path=best_model_configs_path,
        model_names=model_names,
        max_rows=len(df_oos),
    )

    out_root = Path(output_dir)
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir = (out_root / "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, dict] = {}
    artifact_map: Dict[str, str] = {}

    # Save lightweight XGB artifact for deployability
    if "xgb" in model_names:
        X, y, feat_cols = extract_features_labels(df_train)
        params = dict(_load_yaml(best_model_configs_path).get("xgb", {}).get("params") or {})
        params.pop("early_stopping_rounds", None)
        split = max(1, int(len(X) * 0.9))
        booster = train_xgb(X.iloc[:split], y.iloc[:split], params=params, early_stopping_rounds=0)
        calib = calibrate(booster, X.iloc[split:], y.iloc[split:])
        xgb_out = models_dir / "final_xgb_primary"
        save_artifacts(xgb_out, booster, calib, feat_cols, threshold=0.5, report={}, gate_config=None)
        artifact_map["xgb"] = str(xgb_out)

    for label, policy in policies.items():
        models = policy.get("models") or list(preds.keys())
        weights = policy.get("weights") or {}
        thresholds = {m: policy.get("thresholds", policy.get("threshold", {})) for m in models}
        risk_cfg_run = dict(risk_cfg)
        risk_cfg_run["model_weights"] = weights
        sim = run_portfolio_simulation(
            df=df_oos,
            model_signals={m: preds[m] for m in models},
            thresholds=thresholds,
            risk_config=risk_cfg_run,
        )
        metrics = sim["metrics"]
        summary[label] = metrics
        (metrics_dir / f"{label}.json").write_text(json.dumps(metrics, indent=2))

    (out_root / "portfolio_final_metrics.json").write_text(json.dumps(summary, indent=2))
    (out_root / "artifact_map.json").write_text(json.dumps(artifact_map, indent=2))


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Finalize portfolio policies by evaluating on a fresh OOS window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--contract", required=True)
    ap.add_argument("--best-model-configs", required=True)
    ap.add_argument("--final-policies", required=True)
    ap.add_argument("--risk-limits", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-rows", type=int, default=None, help="Optional tail cap for fast finalization.")
    ap.add_argument("--oos-fraction", type=float, default=0.1, help="Fraction reserved for final OOS evaluation.")
    args = ap.parse_args(argv)
    finalize_portfolio_policies(
        contract_path=args.contract,
        best_model_configs_path=args.best_model_configs,
        final_policies_path=args.final_policies,
        risk_limits_path=args.risk_limits,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        oos_fraction=args.oos_fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
