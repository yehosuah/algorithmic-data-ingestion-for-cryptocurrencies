# Subtask 3b – Meta-Label Attempt

_Last updated: 2025-10-21 02:50 UTC_

## Goal
Train a logistic meta-label filter on horizon-120 signals using the relaxed-gate base/TCN probabilities and the RSS-enriched matrix.

## Observations (2025-10-21)
- With the extended matrix (`datasets/blender_matrix_2024-09_to_2025-09_rss_latest.parquet`), barrier labels populate (positive share ≈0.54), but filtering for rows with stable base/TCN/RSS coverage still collapses to a single class once we enforce deployable gates.
- Meta models fitted on the relaxed gate show equity ≈1.0: the meta filter has no separation because the primary probabilities already saturate under the strict inference mask.
- `scripts/train_meta_label.py` now supports the relaxed gate defaults and stride control; artifacts (`models/meta_h120_v2`) remain exploratory and are not deployable.

## Next Steps
1. Extend the blender matrix to include Oct–Nov 2025 so the relaxed gate produces more overlap between base/TCN signals and RSS spikes.
2. Experiment with asymmetric barriers (`pt_mult ≤ 1.2`, `sl_mult ≥ 2.0`, `max_hold ≥ 240`) to increase class balance without overfitting noise.
3. Hold meta rollout until the blender delivers consistent equity ≥1.2 across additional months; otherwise rely on deterministic manifest gates plus the blender.

_Current status: Research-only artifacts. Do not deploy meta gate yet._
