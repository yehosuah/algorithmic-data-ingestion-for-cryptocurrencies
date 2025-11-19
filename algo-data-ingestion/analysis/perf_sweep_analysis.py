from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Analyze performance sweep summary and pick best scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input",
        "--summary",
        dest="summary",
        required=True,
        help="Path to experiments/perf_sweeps/summary.csv",
    )
    ap.add_argument(
        "--risk-limits",
        default="configs/portfolio_risk_limits.yaml",
        help="Risk limits YAML to enforce; optional for filtering.",
    )
    ap.add_argument(
        "--output-md",
        "--report-md",
        dest="report_md",
        required=True,
        help="Destination markdown report.",
    )
    ap.add_argument(
        "--output-json",
        "--best-json",
        dest="best_json",
        required=True,
        help="Destination JSON for best scenario.",
    )
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--min-trade-count", type=int, default=100, help="Coverage guardrail for flagging.")
    ap.add_argument(
        "--min-fraction-time-in-position",
        type=float,
        default=0.01,
        help="Coverage guardrail for flagging.",
    )
    args = ap.parse_args(argv)

    df = pd.read_csv(args.summary)
    risk = _load_yaml(Path(args.risk_limits)) if args.risk_limits else {}
    if df.empty:
        raise ValueError("Summary file is empty; run sweeps first.")

    # Apply basic feasibility filters
    if risk:
        if "primary_max_drawdown" in df.columns and "max_drawdown" in risk:
            df = df[df["primary_max_drawdown"] <= float(risk["max_drawdown"])]
        if "primary_turnover" in df.columns and "max_turnover_per_day" in risk:
            df = df[df["primary_turnover"] <= float(risk["max_turnover_per_day"])]

    df = df[df["primary_sharpe"].notna()]
    deg_mask = pd.Series(False, index=df.index)
    if "trade_count" in df.columns:
        deg_mask |= df["trade_count"].fillna(0) < args.min_trade_count
    if "fraction_time_in_position" in df.columns:
        deg_mask |= df["fraction_time_in_position"].fillna(0) < args.min_fraction_time_in_position
    degenerate = df[deg_mask]
    filtered_df = df[~deg_mask] if (~deg_mask).any() else df
    filtered_df = filtered_df.sort_values(["primary_sharpe", "primary_pnl_net"], ascending=[False, False])
    if filtered_df.empty:
        raise ValueError("No feasible scenarios after filtering.")

    top = filtered_df.head(max(1, int(args.top_n)))
    md_lines = [
        "# Performance Sweep Summary",
        "",
        f"Coverage thresholds: trade_count >= {args.min_trade_count}, "
        f"fraction_time_in_position >= {args.min_fraction_time_in_position:.3f}.",
    ]
    if not degenerate.empty:
        md_lines.append("## Degenerate (low coverage) scenarios flagged")
        md_lines.append(degenerate.to_csv(index=False))
    md_lines.append("## Top scenarios")
    md_lines.append(top.to_csv(index=False))
    Path(args.report_md).write_text("\n".join(md_lines))

    best_row = top.iloc[0].to_dict()
    best_payload = {
        "scenario_id": best_row["scenario_id"],
        "primary_policy_id": "primary",
        "metrics": {
            "pnl_net": best_row.get("primary_pnl_net"),
            "sharpe": best_row.get("primary_sharpe"),
            "max_drawdown": best_row.get("primary_max_drawdown"),
            "turnover": best_row.get("primary_turnover"),
            "trade_count": best_row.get("trade_count"),
            "fraction_time_in_position": best_row.get("fraction_time_in_position"),
            "avg_gross_exposure": best_row.get("avg_gross_exposure"),
            "transaction_cost_bps": best_row.get("transaction_cost_bps"),
        },
        "long_only": best_row.get("long_only"),
    }
    Path(args.best_json).write_text(json.dumps(best_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
