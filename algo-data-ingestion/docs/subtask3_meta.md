# Subtask 3b – Meta-Label Attempt

_Last updated: 2025-11-30 18:55 UTC_

> Update 2025-11-30: Documented the BTC/ETH/SOL rollout plus kill/safe switch enforcement, HMAC-signed trading audits, the Redis intent ledger + reconciliation loop, runtime risk/deadlock policies, and scheduler shadow-mode controls in this drop.

> Update 2025-11-29: Added the trigger optimizer + preflight lane (analysis/trigger_optimizer.py, configs/trigger_search_space*.yaml, configs/final_trigger_policy.yaml, scripts/trigger_preflight.py), shared trading decision logic with spread/hold/SL/TP guards, enriched market ingest/backfill/scheduler to compute augmented features and attach prices for inference/Redis payloads, and aligned dry-run paths to MODELS_ROOT=/opt/models with guard-aware TRADING_MODELS defaults.
> Update 2025-11-13: Added references to the sanitized multi-symbol feed + symbol-gate generator and the feature parity helpers (`export_feature_slice.py`, `compare_feature_stats.py`) so any future meta work aligns with the gates/metrics enforced downstream.

## Goal
Train a logistic meta-label filter on horizon-120 signals using the relaxed-gate base/TCN probabilities and the RSS-enriched matrix.

## Observations (2025-10-30)
- Even with the forward replay matrix (`datasets/blender_matrix_2025-10_to_2025-11_with_preds.parquet`, 40 201 rows) layered on top of the year-wide build, filtering for rows that satisfy deployable gates still collapses towards a single class—base manifest supplies only 12 gate hits (8 trades), TCN horizons deliver sparse but non-zero coverage (`gate_hits 4/31/2`), and the blender dominates coverage with 6 346 toggles.
- Blender stride experiments (`models/blender_h120_stride1_v2`) confirm that collapsing the smoothing window pushes turnover down to 134 toggles while keeping equity at 4.48, but the deployable gate still sits near 15.8 %; the meta layer continues to inherit the blender’s class imbalance.
- Meta models fitted on the relaxed gate still hover around equity ≈1.0; there is no separation to exploit until the primary models expand coverage beyond the new floor across additional months.
- `scripts/train_meta_label.py` retains relaxed gate defaults and stride control; artifacts (`models/meta_h120_v2`) remain exploratory and are not deployable.
- The scheduler/trading dry run intentionally skips meta outputs; keep `TRADING_MODELS` pointed at base/TCN/blender manifests only to avoid draining a queue with low-quality meta signals.
- All upstream models now rely on the sanitized multi-symbol parquet (`training.data.sanitize_market_dataset` → `datasets/market_multi_3symbol_1m.parquet`) and the shared gate payload (`release/symbol_gates/market_multi_3symbol_1m.json`), so any meta experiments must reuse that dataset/JSON combo to mirror scheduler/trading gates.

## Next Steps
1. Wait until base/TCN/blender deployable gates sustain higher coverage (beyond the 5e-4 floor) on forward windows; otherwise meta labels will remain degenerate.
2. Experiment with asymmetric barriers (`pt_mult ≤ 1.2`, `sl_mult ≥ 2.0`, `max_hold ≥ 240`) once coverage stabilises to increase class balance without overfitting noise.
3. Hold meta rollout until the blender delivers consistent equity ≥1.2 across additional months; otherwise rely on deterministic manifest gates plus the blender.
4. Update the trading runbook once meta coverage improves so operators know when to flip `TRADING_MODELS` to include meta outputs.
- When revisiting, export a parity slice (`scripts/export_feature_slice.py`) and capture the drift JSON via `scripts/compare_feature_stats.py --train datasets/market_multi_3symbol_1m.parquet --live /tmp/features_debug.parquet --out release/calibration/latest/meta_parity.json` so reviewers see how the meta dataset aligns with production features before gating.

_Current status: Research-only artifacts. Do not deploy meta gate yet._
