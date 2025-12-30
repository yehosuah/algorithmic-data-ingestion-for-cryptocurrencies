# Market Alignment — ETH/USDT

## Evidence (baseline)

- Evidence bundle: `reports/log_forensics/evidence/20251225T181022Z`
- Executed-exit forensics: `reports/expectancy_fix/baseline_forensics.md`
- Market alignment summary: `reports/expectancy_fix/baseline_alignment/alignment_summary.md`
- Trade list (with MFE/MAE): `reports/expectancy_fix/trades_ETH_USDT.csv`

## Findings (baseline window)

1) We lose big mostly when exits are driven by **prob_trailing / prob_floor** (plus rare but very large stop events).
   - Loss mass concentrates in prob_trailing and prob_floor; max_loss is a single large tail event (stop_loss) (`reports/expectancy_fix/baseline_forensics.md`).

2) We clip winners mostly via a **too-tight take-profit**.
   - take_profit trades have avg realized return ≈ 0.0017 vs avg MFE ≈ 0.0042, indicating “upside starvation” (`reports/expectancy_fix/baseline_forensics.md`).

3) Parameter/regime mismatch: exits often occur on **noise** while the market still has positive drift after exit.
   - Multiple exit reasons (prob_trailing, prob_floor, gate_close, time_limit) show negative exit return but positive post-exit max drift in alignment stats (`reports/expectancy_fix/baseline_alignment/alignment_summary.md`).

