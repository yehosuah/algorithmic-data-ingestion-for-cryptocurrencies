# Subtask 3b – Meta-Label Attempt

## Goal
Train a logistic meta-label filter on horizon-120 signals using the refreshed base/TCN probabilities.

## Observations
- Triple-barrier generation (`training.meta.triple_barrier_events`) applied to 120-bar horizons produced **zero positive labels** across the training matrix (`positive ratio = 0.0`), regardless of reasonable `pt_mult`/`sl_mult` settings.
- With a single class, `LogisticRegression` cannot fit; the job aborted with `ValueError: This solver needs samples of at least 2 classes`.

## Next Steps
- Define horizon-appropriate barrier logic (e.g., lower `pt_mult/sl_mult`, longer `max_hold`, or event sampling) to reintroduce positive labels.
- Alternatively, consider meta-labeling directly on the blender outputs once a suitable event definition is established.

_No model artifacts were produced for the meta stage during this run._
