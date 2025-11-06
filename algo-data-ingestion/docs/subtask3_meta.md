# Subtask 3b – Meta-Label Attempt

_Last updated: 2025-11-05 14:56 UTC_

## Goal
Train a logistic meta-label filter on horizon-120 signals using the relaxed-gate base/TCN probabilities and the RSS-enriched matrix.

## Observations (2025-10-30)
- Even with the forward replay matrix (`datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`, 40 201 rows) layered on top of the year-wide build, filtering for rows that satisfy deployable gates still collapses towards a single class—base manifest supplies only 12 gate hits (8 trades), TCN horizons deliver sparse but non-zero coverage (`gate_hits 4/31/2`), and the blender dominates coverage with 6 346 toggles.
- Blender stride experiments (`models/blender_h120_stride1_v2`) confirm that collapsing the smoothing window pushes turnover down to 134 toggles while keeping equity at 4.48, but the deployable gate still sits near 15.8 %; the meta layer continues to inherit the blender’s class imbalance.
- Meta models fitted on the relaxed gate still hover around equity ≈1.0; there is no separation to exploit until the primary models expand coverage beyond the new floor across additional months.
- `scripts/train_meta_label.py` retains relaxed gate defaults and stride control; artifacts (`models/meta_h120_v2`) remain exploratory and are not deployable.
- The scheduler/trading dry run intentionally skips meta outputs; keep `TRADING_MODELS` pointed at base/TCN/blender manifests only to avoid draining a queue with low-quality meta signals.

## Next Steps
1. Wait until base/TCN/blender deployable gates sustain higher coverage (beyond the 5e-4 floor) on forward windows; otherwise meta labels will remain degenerate.
2. Experiment with asymmetric barriers (`pt_mult ≤ 1.2`, `sl_mult ≥ 2.0`, `max_hold ≥ 240`) once coverage stabilises to increase class balance without overfitting noise.
3. Hold meta rollout until the blender delivers consistent equity ≥1.2 across additional months; otherwise rely on deterministic manifest gates plus the blender.
4. Update the trading runbook once meta coverage improves so operators know when to flip `TRADING_MODELS` to include meta outputs.

_Current status: Research-only artifacts. Do not deploy meta gate yet._
