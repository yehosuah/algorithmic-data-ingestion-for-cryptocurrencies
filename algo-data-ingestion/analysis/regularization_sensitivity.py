#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


TARGET_PARAMS = [
    "dropout",
    "weight_decay",
    "max_depth",
    "reg_lambda",
    "reg_alpha",
]


def _summarize(df: pd.DataFrame, param: str) -> Dict:
    col = f"param_{param}"
    if col not in df.columns:
        return {}
    grouped = df.groupby(col).agg(
        mean_pnl=("mean_pnl_net_cv", "mean"),
        mean_sharpe=("mean_sharpe_cv", "mean"),
        count=("trial_id", "count"),
    ).reset_index()
    grouped = grouped.sort_values("mean_sharpe", ascending=False)
    top = grouped.head(3).to_dict(orient="records")
    recommendation = None
    if len(grouped):
        best_row = grouped.iloc[0]
        recommendation = f"{param}≈{best_row[col]:g} yields sharpe≈{best_row['mean_sharpe']:.3f}"
    return {"param": param, "top": top, "recommendation": recommendation}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Regularization sensitivity analysis from hparam search results.")
    ap.add_argument("--results", required=True, help="Path to results.csv from hparam search.")
    ap.add_argument("--output-dir", default=None, help="Directory to write reports (defaults to results parent).")
    args = ap.parse_args(argv)

    res_path = Path(args.results)
    if not res_path.exists():
        raise FileNotFoundError(f"results.csv not found: {res_path}")
    df = pd.read_csv(res_path)
    out_dir = Path(args.output_dir) if args.output_dir else res_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    findings = []
    for p in TARGET_PARAMS:
        summary = _summarize(df, p)
        if summary:
            findings.append(summary)

    best_rows = df.sort_values(["mean_sharpe_cv", "mean_pnl_net_cv"], ascending=False).head(5)
    sharpe_std = float(df["mean_sharpe_cv"].std()) if "mean_sharpe_cv" in df.columns else None
    notes = []
    for f in findings:
        if f.get("recommendation"):
            notes.append(f.get("recommendation"))
    if sharpe_std:
        notes.append(f"Sharpe dispersion across trials ≈ {sharpe_std:.3f} (higher may indicate instability).")

    report = {
        "regularization_profiles": findings,
        "top_trials": best_rows[["trial_id", "mean_pnl_net_cv", "mean_sharpe_cv"]].to_dict(orient="records"),
        "source": str(res_path),
        "notes": notes,
    }
    (out_dir / "regularization_summary.json").write_text(json.dumps(report, indent=2))

    md_lines = ["# Regularization Sensitivity", f"Source: `{res_path}`", ""]
    for item in findings:
        md_lines.append(f"## {item['param']}")
        for row in item.get("top", []):
            md_lines.append(f"- {item['param']}={row['param_'+item['param']] if 'param_'+item['param'] in row else row.get(item['param'],'?')} | sharpe={row['mean_sharpe']:.4f} | pnl={row['mean_pnl']:.6f} | n={int(row['count'])}")
        if item.get("recommendation"):
            md_lines.append(f"- Recommendation: {item['recommendation']}")
    md_lines.append("")
    md_lines.append("## Top Trials")
    for _, r in best_rows.iterrows():
        md_lines.append(f"- {r['trial_id']}: sharpe={r['mean_sharpe_cv']:.4f}, pnl={r['mean_pnl_net_cv']:.6f}")
    if notes:
        md_lines.append("")
        md_lines.append("## Notes")
        md_lines.extend([f"- {n}" for n in notes])
    (out_dir / "regularization_summary.md").write_text("\n".join(md_lines))
    print(f"[Regularization] Saved reports to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
