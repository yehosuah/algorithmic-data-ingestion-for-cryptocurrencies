# Subtask 3b – Meta-Label Attempt

_Last updated: 2025-10-29 15:53 UTC_

## Goal
Train a logistic meta-label filter on horizon-120 signals using the relaxed-gate base/TCN probabilities and the RSS-enriched matrix.

## Observations (2025-10-29)
- Even with the forward replay matrix (`datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`) layered on top of the year-wide build, filtering for rows that satisfy deployable gates still collapses towards a single class—base manifest now supplies only 12 gate hits (8 trades) and all TCN probabilities remain at zero coverage under the strict mask.
- Meta models fitted on the relaxed gate still hover around equity ≈1.0; there is no separation to exploit until the primary models regain coverage under the deployable constraints.
- `scripts/train_meta_label.py` retains relaxed gate defaults and stride control; artifacts (`models/meta_h120_v2`) remain exploratory and are not deployable.

## Next Steps
1. Wait until base/TCN/blender deployable gates are retuned to deliver non-zero coverage on forward windows; otherwise meta labels will remain degenerate.
2. Experiment with asymmetric barriers (`pt_mult ≤ 1.2`, `sl_mult ≥ 2.0`, `max_hold ≥ 240`) once coverage stabilises to increase class balance without overfitting noise.
3. Hold meta rollout until the blender delivers consistent equity ≥1.2 across additional months; otherwise rely on deterministic manifest gates plus the blender.

_Current status: Research-only artifacts. Do not deploy meta gate yet._
