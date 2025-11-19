from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def _check_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")


def promote(
    best_scenario_json: Path,
    sweep_config: Path,
    output_policies: Path,
    output_deployment_contract: Path,
    base_output_dir: Path,
    *,
    summary_file: Path | None = None,
    min_sharpe: float = 0.0,
    min_trade_count: int = 100,
    min_fraction_time_in_position: float = 0.01,
) -> None:
    sweep = _load_yaml(sweep_config)
    scenarios = sweep.get("scenarios") or []
    scenario_id = None
    if summary_file is not None:
        import pandas as pd

        df = pd.read_csv(summary_file)
        if df.empty:
            raise ValueError("summary_file is empty; cannot promote.")
        allowed_ids = {s.get("id") for s in scenarios}
        df = df[df["scenario_id"].isin(allowed_ids)]
        if df.empty:
            raise ValueError("No scenarios in summary_file match sweep config.")
        mask = (df["primary_sharpe"].fillna(0) >= min_sharpe)
        if "trade_count" in df.columns:
            mask &= df["trade_count"].fillna(0) >= min_trade_count
        if "fraction_time_in_position" in df.columns:
            mask &= df["fraction_time_in_position"].fillna(0) >= min_fraction_time_in_position
        filtered = df[mask].sort_values(["primary_sharpe", "primary_pnl_net"], ascending=[False, False])
        if filtered.empty:
            raise ValueError("No scenarios meet promotion thresholds.")
        scenario_id = filtered.iloc[0]["scenario_id"]
    else:
        best = json.loads(best_scenario_json.read_text())
        scenario_id = best.get("scenario_id")

    if not scenario_id:
        raise ValueError("best_scenario_json missing scenario_id")
    if scenario_id not in {s.get("id") for s in scenarios}:
        raise ValueError(f"Scenario '{scenario_id}' not found in sweep config")

    scen_dir = base_output_dir / scenario_id
    src_policies = scen_dir / "final_policies.yaml"
    src_models_dir = scen_dir / "portfolio_final" / "models"
    src_metrics_dir = scen_dir / "portfolio_final" / "metrics"
    _check_exists(src_policies, "final policies")
    _check_exists(src_models_dir, "models dir")
    _check_exists(src_metrics_dir, "metrics dir")

    # Promote policies
    output_policies.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_policies, output_policies)

    # Build deployment contract with promoted artifacts
    contract = _load_yaml(output_deployment_contract)
    contract["portfolio_policies"] = str(output_policies)
    contract["models"] = {}
    for model_dir in src_models_dir.iterdir():
        if not model_dir.is_dir():
            continue
        contract["models"][model_dir.name.replace("final_", "")] = str(model_dir)

    output_deployment_contract.parent.mkdir(parents=True, exist_ok=True)
    with output_deployment_contract.open("w") as fh:
        yaml.safe_dump(contract, fh, sort_keys=False)

    # Consistency check
    for path in [
        Path(contract.get("dataset_contract", "")),
        Path(contract.get("best_model_configs", "")),
        Path(contract.get("risk_limits", "")),
        Path(contract.get("portfolio_policies", "")),
    ]:
        _check_exists(path, path.name)
    for _, path in (contract.get("models") or {}).items():
        _check_exists(Path(path), f"model artifact {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Promote best scenario outputs to global configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--best-scenario-json", required=False)
    ap.add_argument("--sweep-config", required=True)
    ap.add_argument("--output-policies-config", required=True)
    ap.add_argument("--output-deployment-contract", required=True)
    ap.add_argument("--base-output-dir", required=True)
    ap.add_argument("--summary-file", required=False, help="Optional summary.csv to select best automatically.")
    ap.add_argument("--criteria-config", required=False, help="Optional YAML with promotion thresholds.")
    ap.add_argument("--min-sharpe", type=float, default=0.0)
    ap.add_argument("--min-trade-count", type=int, default=100)
    ap.add_argument("--min-fraction-time-in-position", type=float, default=0.01)
    args = ap.parse_args(argv)

    criteria = {}
    if args.criteria_config:
        criteria = _load_yaml(Path(args.criteria_config))
    min_sharpe = criteria.get("min_sharpe", args.min_sharpe)
    min_trade_count = criteria.get("min_trade_count", args.min_trade_count)
    min_fraction_time_in_position = criteria.get("min_fraction_time_in_position", args.min_fraction_time_in_position)

    promote(
        best_scenario_json=Path(args.best_scenario_json) if args.best_scenario_json else Path(""),
        sweep_config=Path(args.sweep_config),
        output_policies=Path(args.output_policies_config),
        output_deployment_contract=Path(args.output_deployment_contract),
        base_output_dir=Path(args.base_output_dir),
        summary_file=Path(args.summary_file) if args.summary_file else None,
        min_sharpe=min_sharpe,
        min_trade_count=min_trade_count,
        min_fraction_time_in_position=min_fraction_time_in_position,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
