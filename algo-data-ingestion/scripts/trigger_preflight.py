#!/usr/bin/env python3
"""
Coverage/trade-count preflight for trigger promotions.

Exits non-zero if predicted coverage or trade count is near zero to prevent deadlocks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import yaml

from analysis.trigger_optimizer import (
    load_dataset,
    ensure_probabilities,
    simulate_trades,
    TriggerConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Trigger coverage/trade preflight")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("configs/final_trigger_policy.yaml"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--prob-column", type=str, default="base_prob")
    parser.add_argument("--gate-column", type=str, default="gate_pass")
    parser.add_argument("--price-column", type=str, default="close")
    parser.add_argument("--spread-column", type=str, default=None)
    parser.add_argument("--min-coverage", type=float, default=0.01)
    parser.add_argument("--min-trades", type=int, default=5)
    args = parser.parse_args()

    pol = yaml.safe_load(args.policy.read_text())
    active = pol.get(pol.get("meta", {}).get("active_policy", "primary"), pol.get("primary", {}))

    cfg = TriggerConfig(
        entry_threshold=float(active.get("entry_threshold", 0.5)),
        exit_threshold=float(active.get("exit_threshold", active.get("entry_threshold", 0.5))),
        exit_prob_drop=float(active.get("exit_prob_drop", 0.15)),
        min_hold_bars=int(active.get("min_hold_bars", 1)),
        bar_seconds=60,
        long_only=True,
        max_hold_seconds=int(active["max_hold_minutes"] * 60) if active.get("max_hold_minutes") else None,
        stop_loss_pct=active.get("stop_loss_pct"),
        take_profit_pct=active.get("take_profit_pct"),
        max_spread_bps=active.get("max_spread_bps"),
    )

    df = load_dataset(args.contract, max_rows=args.max_rows)
    df = ensure_probabilities(df, args.prob_column, args.gate_column, args.model_dir)
    res = simulate_trades(
        df,
        cfg,
        prob_column=args.prob_column,
        gate_column=args.gate_column,
        price_column=args.price_column,
        spread_column=args.spread_column,
    )
    print(
        f"Preflight: coverage={res.coverage:.4f} trades={res.trades} sharpe={res.sharpe:.3f} pnl={res.pnl:.4f}"
    )
    if res.coverage < args.min_coverage or res.trades < args.min_trades:
        print(
            f"Preflight failed (coverage<{args.min_coverage} or trades<{args.min_trades}).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
