# Step 2B - the leakage audit

**Notebook:** `notebooks/step2b_audit.ipynb` (CPU, inputs `agender-code` and `agender-step1`, about 5-10 min).

## Threat model

The operator releases the 24 cleaned LAT attributes (`lat.feature_columns`) of every subscriber. The partner
already knows age and gender of some subscribers and wants them for the others. In `splits/`:

| Part | Share | Role |
|---|---|---|
| train | 60 % | the partner trains its attackers |
| validation | 10 % | the partner chooses settings and its strongest attacker |
| calibration | 10 % | the partner builds conformal sets |
| test | 20 % | the subscribers whose age and gender the partner does not know |

## Attackers

Logistic Regression (C chosen from 0.1, 1, 10), LightGBM (7, 15 or 31 leaves, early stopping), CatBoost
(depth 6, early stopping), MLP (64-32, two weight-decay values), and the baseline multi-task network
(`agender/mtl.py`). All choices by validation log-loss. The encoding (`agender/features.py`) is fitted on
training rows.
The attacker "chosen on validation" is the one with the best validation AUC: this is the one the partner
would use, since it cannot see test labels.

## Conformal sets

Split conformal prediction with the score 1 - p(true class), threshold from the calibration part at the
finite-sample level ceil((n+1)(1-alpha))/n; alpha = 0.05, 0.10, 0.20. Plain sets (guarantee on average) and
class-conditional sets (guarantee inside every age group and every gender). A single-class set = the
subscriber is pinned down at that guarantee. Reported: coverage, mean set size, share of single-class sets,
share of sets with at most two classes, per group.

## Output (`agender_outputs/step2b/`)

`lat_scores.csv`, `lat_conformal.csv`, `lat_risk.csv` (one row per test subscriber, repeat and task, for the
chosen attacker at 90 %), `lat_intervals.csv` (bootstrap 95 % intervals), and the same for `copenhagen_*`
(gender only, no multi-task network). `step2b_summary.json`.
