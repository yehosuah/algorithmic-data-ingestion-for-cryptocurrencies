from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from portfolio.predictions import generate_oos_predictions_for_models
from portfolio.optimize_portfolio_policy import main as optimize_policy_main
from portfolio.select_best_policies import select_best_policies
from portfolio.finalize_portfolio_models import finalize_portfolio_policies


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def _append_summary_row(summary_path: Path, row: Dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        df = pd.read_csv(summary_path)
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(summary_path, index=False)


def _log_failure(log_path: Path, scenario_id: str, exc: Exception) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(f"{scenario_id}: {exc}\n")


def run_scenario(
    scenario: Dict[str, object],
    *,
    contract: Path,
    best_model_configs: Path,
    risk_limits: Path,
    base_output_dir: Path,
    max_rows_cap: int | None = None,
) -> Dict[str, object]:
    scenario_id = str(scenario["id"])
    models = scenario.get("models") or ["xgb"]
    max_rows = scenario.get("max_rows")
    if max_rows_cap:
        max_rows = min(int(max_rows_cap), int(max_rows or max_rows_cap))
    n_folds = scenario.get("n_folds")
    oos_fraction = scenario.get("oos_fraction", 0.1)
    policy_search_config = Path(scenario.get("policy_search_config") or "configs/portfolio_policy_search.yaml")
    model_weights_config = Path(scenario.get("model_weights_config") or "configs/portfolio_model_weights.yaml")

    scen_dir = base_output_dir / scenario_id
    policy_dir = scen_dir / "portfolio_policy"
    final_dir = scen_dir / "portfolio_final"
    scen_dir.mkdir(parents=True, exist_ok=True)

    base_risk = _load_yaml(risk_limits)
    # Override risk config per scenario (costs/long-only)
    risk_override = dict(base_risk)
    if scenario.get("transaction_cost_bps") is not None:
        risk_override["transaction_cost_bps"] = float(scenario["transaction_cost_bps"])
    if scenario.get("long_only") is not None:
        risk_override["long_only"] = bool(scenario["long_only"])
    scen_risk_path = scen_dir / "risk_limits.yaml"
    with scen_risk_path.open("w") as fh:
        yaml.safe_dump(risk_override, fh, sort_keys=False)

    # Policy search via CLI main (reuse logic, not subprocess)
    optimize_policy_main(
        [
            "--contract",
            str(contract),
            "--best-model-configs",
            str(best_model_configs),
            "--risk-config",
            str(scen_risk_path),
            "--model-weights-config",
            str(model_weights_config),
            "--policy-search-config",
            str(policy_search_config),
            "--output-dir",
            str(policy_dir),
            "--max-rows",
            str(max_rows or 0),
            "--n-folds",
            str(n_folds or 0),
        ]
    )

    results_csv = policy_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv missing for scenario {scenario_id}")

    # 3) Selection (scenario-local)
    final_policies_path = scen_dir / "final_policies.yaml"
    payload = select_best_policies(
        results_path=results_csv,
        risk_limits_path=scen_risk_path,
        optimize_for=scenario.get("optimize_for", "sharpe"),
        top_k=3,
    )
    yaml.safe_dump(payload, final_policies_path.open("w"), sort_keys=False)

    # 4) Finalize
    finalize_portfolio_policies(
        contract_path=str(contract),
        best_model_configs_path=str(best_model_configs),
        final_policies_path=str(final_policies_path),
        risk_limits_path=str(scen_risk_path),
        output_dir=str(final_dir),
        max_rows=int(max_rows) if max_rows else None,
        oos_fraction=float(oos_fraction),
    )

    metrics_path = Path(final_dir) / "portfolio_final_metrics.json"
    metrics_payload = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    primary_metrics = metrics_payload.get("primary", {})
    return {
        "scenario_id": scenario_id,
        "models": ",".join(models),
        "max_rows": max_rows,
        "oos_fraction": oos_fraction,
        "primary_pnl_net": primary_metrics.get("pnl_net"),
        "primary_sharpe": primary_metrics.get("sharpe"),
        "primary_max_drawdown": primary_metrics.get("max_drawdown"),
        "primary_turnover": primary_metrics.get("turnover"),
        "trade_count": primary_metrics.get("trade_count"),
        "fraction_time_in_position": primary_metrics.get("fraction_time_in_position"),
        "avg_gross_exposure": primary_metrics.get("avg_gross_exposure"),
        "transaction_cost_bps": risk_override.get("transaction_cost_bps"),
        "long_only": risk_override.get("long_only", False),
        "primary_metrics_path": str(metrics_path),
        "final_policies_path": str(final_policies_path),
        "final_dir": str(final_dir),
    }


def run_production_xgb_sweeps() -> int:
    """
    Convenience wrapper to launch the full-scale XGB sweeps used for promotion.
    """
    return main(
        [
            "--contract",
            "configs/canonical_training_contract_market_multi_3symbol_1m.yaml",
            "--best-model-configs",
            "configs/best_model_configs.yaml",
            "--risk-limits",
            "configs/portfolio_risk_limits.yaml",
            "--sweep-config",
            "configs/perf_sweep_scenarios.yaml",
            "--base-output-dir",
            "experiments/perf_sweeps",
            "--scenario-ids",
            "medium_xgb_realistic_cost,medium_xgb_low_cost,medium_xgb_high_cost,long_xgb_realistic_cost",
        ]
    )


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run performance sweeps across predefined scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--contract", required=True)
    ap.add_argument("--best-model-configs", required=True)
    ap.add_argument("--risk-limits", required=True)
    ap.add_argument("--sweep-config", required=True)
    ap.add_argument("--base-output-dir", required=True)
    ap.add_argument("--max-rows-cap", type=int, default=None, help="Optional cap applied to all scenarios for faster runs.")
    ap.add_argument("--scenario-ids", type=str, default=None, help="Optional comma-separated subset of scenarios to run.")
    args = ap.parse_args(argv)

    contract = Path(args.contract)
    best_model_configs = Path(args.best_model_configs)
    risk_limits = Path(args.risk_limits)
    sweep = _load_yaml(Path(args.sweep_config))
    scenarios = sweep.get("scenarios") or []
    if args.scenario_ids:
        allowed = {s.strip() for s in args.scenario_ids.split(",") if s.strip()}
        scenarios = [s for s in scenarios if s.get("id") in allowed]
    if not scenarios:
        raise ValueError("No scenarios defined in sweep config.")

    base_output_dir = Path(args.base_output_dir)
    summary_path = base_output_dir / "summary.csv"
    failure_log = base_output_dir / "failures.log"

    for scenario in scenarios:
        try:
            row = run_scenario(
                scenario,
                contract=contract,
                best_model_configs=best_model_configs,
                risk_limits=risk_limits,
                base_output_dir=base_output_dir,
                max_rows_cap=args.max_rows_cap,
            )
            _append_summary_row(summary_path, row)
        except Exception as exc:  # pragma: no cover - defensive sweep harness
            _log_failure(failure_log, str(scenario.get("id")), exc)
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
