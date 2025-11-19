from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.edge_diagnostics import run_edge_diagnostics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run edge diagnostics for XGB/TCN/Transformer models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--contract", required=True)
    ap.add_argument("--best-model-configs", required=True)
    ap.add_argument("--model", default="xgb", choices=["xgb", "tcn", "transformer"])
    ap.add_argument("--train-fraction", type=float, default=0.7)
    ap.add_argument("--low-cost-bps", type=float, default=0.25)
    ap.add_argument("--regime-col", default="regime_id")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_edge_diagnostics(
        contract_path=args.contract,
        best_model_configs_path=args.best_model_configs,
        model_name=args.model,
        regime_col=args.regime_col,
        train_fraction=args.train_fraction,
        low_cost_bps=args.low_cost_bps,
    )
    json_path = out_dir / f"edge_diagnostics_{args.model}.json"
    json_path.write_text(json.dumps(result, indent=2))

    # Simple markdown summary
    global_sig = result["global"]["metrics_signal"]
    global_pnl = result["global"]["metrics_pnl"]
    md_lines = [
        f"# Edge diagnostics for {args.model}",
        "",
        "## Global signal metrics",
        f"- AUC: {global_sig.get('auc')}",
        f"- Brier: {global_sig.get('brier')}",
        f"- Log loss: {global_sig.get('log_loss')}",
        f"- IC: {global_sig.get('ic')}",
        "",
        "## Global PnL metrics (loose gate)",
        f"- PnL net: {global_pnl.get('pnl_net')}",
        f"- Sharpe: {global_pnl.get('sharpe')}",
        f"- Max drawdown: {global_pnl.get('max_drawdown')}",
        f"- Toggle count: {global_pnl.get('toggle_count')}",
        "",
        "## Per-regime",
    ]
    for reg, payload in result.get("by_regime", {}).items():
        if payload.get("insufficient_data"):
            md_lines.append(f"- {reg}: insufficient data (n={payload.get('count')})")
            continue
        sig = payload["metrics_signal"]
        pnl = payload["metrics_pnl"]
        md_lines.append(f"- {reg}: AUC={sig.get('auc')} IC={sig.get('ic')} PnL={pnl.get('pnl_net')} Sharpe={pnl.get('sharpe')}")
    (out_dir / f"edge_diagnostics_{args.model}.md").write_text("\n".join(md_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
