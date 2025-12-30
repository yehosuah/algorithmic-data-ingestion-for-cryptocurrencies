# Market Alignment — SOL/USDT

## Evidence (baseline)

- Evidence bundle: `reports/log_forensics/evidence/20251225T181022Z`
- Executed-exit forensics: `reports/expectancy_fix/baseline_forensics.md`
- Market alignment summary: `reports/expectancy_fix/baseline_alignment/alignment_summary.md`
- Trade list (with MFE/MAE): `reports/expectancy_fix/trades_SOL_USDT.csv`

## Findings (baseline window)

1) We lose big mostly on **prob_floor** and **gate_close** exits (tail loss present but less frequent).
   - Loss attribution shows prob_floor dominates SOL losses; gate_close is the next largest driver (`reports/expectancy_fix/baseline_forensics.md`).

2) We clip winners mostly via a **too-tight take-profit** (strongest “upside starvation” of the three symbols).
   - take_profit regret_fraction is high (≈ 0.63): mean MFE is far larger than realized take_profit return (`reports/expectancy_fix/baseline_forensics.md`).

3) Parameter misalignment: the exit policy sells into **mean reversion**.
   - Alignment shows `prob_floor` exits have negative exit_return but positive post-exit drift (post_exit_max_return_mean ≈ +0.0026), consistent with exiting on transient weakness (`reports/expectancy_fix/baseline_alignment/alignment_summary.md`).

