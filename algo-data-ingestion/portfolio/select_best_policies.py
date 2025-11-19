from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r") as fh:
        return yaml.safe_load(fh) or {}


def _decode_json_field(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return {}
    return val or {}


def select_best_policies(
    results_path: str | Path,
    risk_limits_path: str | Path,
    optimize_for: str = "sharpe",
    top_k: int = 3,
) -> Dict[str, dict]:
    results = pd.read_csv(results_path)
    risk_cfg = _load_yaml(risk_limits_path)
    if results.empty:
        raise ValueError("No portfolio policy results to select from.")

    filtered = results.copy()
    if "max_drawdown" in filtered.columns and "max_drawdown" in risk_cfg:
        filtered = filtered[filtered["max_drawdown"] <= float(risk_cfg["max_drawdown"])]
    if "turnover" in filtered.columns and "max_turnover_per_day" in risk_cfg:
        filtered = filtered[filtered["turnover"] <= float(risk_cfg["max_turnover_per_day"])]

    if filtered.empty:
        raise ValueError("No feasible policies after applying risk constraints.")

    opt_col = optimize_for if optimize_for in filtered.columns else "sharpe"
    filtered = filtered.sort_values(opt_col, ascending=False)
    selected = filtered.head(max(1, int(top_k)))

    payload: Dict[str, dict] = {}
    labels = ["primary", "conservative", "aggressive"]
    for label, (_, row) in zip(labels, selected.iterrows()):
        payload[label] = {
            "id": row.get("policy_id", label),
            "weights": _decode_json_field(row.get("weights")),
            "thresholds": {"global": _decode_json_field(row.get("thresholds"))},
        }
        models = row.get("models")
        if isinstance(models, str):
            payload[label]["models"] = [m for m in models.split(",") if m]
        elif isinstance(models, list):
            payload[label]["models"] = models
        else:
            payload[label]["models"] = []
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Select feasible portfolio policies given risk limits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--results", "--results-file", dest="results", required=True, help="Path to results.csv from optimize_portfolio_policy.")
    ap.add_argument("--risk-limits", "--risk-config", dest="risk_limits", required=True, help="Risk limits YAML used for filtering.")
    ap.add_argument("--output", required=True, help="Destination YAML for final policies.")
    ap.add_argument("--optimize-for", default="sharpe", help="Primary metric for sorting.")
    ap.add_argument("--top-k", type=int, default=3, help="Number of policies to keep.")
    args = ap.parse_args(argv)

    payload = select_best_policies(
        results_path=args.results,
        risk_limits_path=args.risk_limits,
        optimize_for=args.optimize_for,
        top_k=args.top_k,
    )
    if not payload:
        raise RuntimeError("No policies selected; check results and risk constraints.")
    for name, pol in payload.items():
        models = pol.get("models") or []
        weights = pol.get("weights") or {}
        if not models:
            raise RuntimeError(f"Policy '{name}' missing models list")
        missing_weight = [m for m in models if m not in weights]
        if missing_weight:
            raise RuntimeError(f"Policy '{name}' missing weights for models: {missing_weight}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
