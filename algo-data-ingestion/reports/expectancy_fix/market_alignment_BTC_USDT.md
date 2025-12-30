# Market Alignment — BTC/USDT

## Evidence (baseline)

- Evidence bundle: `reports/log_forensics/evidence/20251225T181022Z`
- Executed-exit forensics: `reports/expectancy_fix/baseline_forensics.md`
- Market alignment summary: `reports/expectancy_fix/baseline_alignment/alignment_summary.md`
- Trade list (with MFE/MAE): `reports/expectancy_fix/trades_BTC_USDT.csv`

## Findings (baseline window)

1) We lose big mostly when **prob_floor** triggers exits.
   - Baseline loss attribution: prob_floor is ~66% of BTC loss mass (`reports/expectancy_fix/baseline_forensics.md`).

2) We clip winners mostly by exiting into **post-exit rebound** (mistimed exits), not via take-profit.
   - No take-profit exits in BTC baseline, but regret_fraction ≈ 0.43 (MFE materially > realized) (`reports/expectancy_fix/baseline_forensics.md`).

3) Parameter misalignment: the model exit gate is too sensitive to **temporary probability dips** versus market continuation.
   - Alignment shows `prob_floor` exits have negative exit_return but positive post-exit drift (post_exit_max_return_mean ≈ +0.0031), i.e., selling before rebound (`reports/expectancy_fix/baseline_alignment/alignment_summary.md`).

