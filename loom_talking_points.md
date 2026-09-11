# Loom talking points (2–3 min) — merged solution

Talking points, not a script to read verbatim. ~2.5 min total.

## 1. Objective (~15s)
- Predict `posted_rate` for 12,000 unlabeled loads (Nov/Dec 2025), trained on
  48,000 labeled loads (Jan–Oct 2025), plus a fixed-lane December scenario.

## 2. Key findings from exploring the data (~25s)
- `distance` is the dominant driver of `posted_rate` — modeled the target as
  rate **per mile** rather than raw rate, which normalizes distance out
  before the model even starts and was the single biggest accuracy win.
- Equipment type (Reefer > Flatbed > Dry Van) shifts price per mile
  meaningfully; market_index/quote_signal matter, but less than their names
  suggest — quantified with feature importance, not assumed.

## 3. Data-quality issues and how you addressed them (~35s)
- `weight` has a sign-flip data-entry error (~0.6% of rows, values as low as
  -47,500 lb) — confirmed by checking that `abs(weight)` for those rows
  matches the normal positive-weight distribution almost exactly. Fixed by
  using only the corrected value as a feature, not the raw signed one.
- Separately, `weight`/`market_index` have some genuine missing values
  (under 1–2%) — handled with an explicit missingness flag plus native NaN
  support, a different mechanism from the sign-flip fix above.
- `validation.csv` has 8 cities never seen in training. Addressed with
  lat/lon + geo-distance for generalization, and actually **measured** the
  residual risk with an experiment: held 6 cities out of training entirely
  and compared holdout accuracy on rows touching them vs. known-city rows —
  a real, moderate accuracy gap, not catastrophic, but worth stating
  honestly rather than assuming the fix fully solves it.

## 4. Reasoning behind the chosen model (~25s)
- CatBoostRegressor: handles the categorical route/equipment/city fields
  natively (no one-hot matrix), supports missing values natively, and
  captures nonlinear interactions.
- Worth mentioning: feature importance shows the model's four distance-
  related columns (distance, distance_log, geo_distance, distance_ratio)
  combined outweigh equipment, even though equipment tops the raw importance
  list — a reminder to read correlated-feature importance carefully.

## 5. Training and validation approach (~35s)
- Time-based holdout — train Jan–Sep, validate on October — because the
  real task is forecasting unseen future months, not interpolating.
- Compared against a random 85/15 split on the same model: random split is
  measurably more optimistic (lower MAE/MAPE), confirming it would've been
  the wrong number to report.
- Added an unseen-city experiment (see above) specifically because a review
  of an earlier draft pointed out that the time-based holdout alone doesn't
  actually test performance on cities the model has never seen.
- Two models trained: full-feature model for `validation_predictions.csv`,
  reduced-feature model (no market_index/quote_signal — not available for
  December) for the December chart. Unlike an earlier draft, kept these as
  two separate models rather than feeding one model an all-NaN pattern it
  never saw in training.

## 6. Code walkthrough (~30s)
- `src/features.py` — shared cleaning (weight fix) and feature engineering
  (rate-per-mile target setup happens in train.py; cyclical-only date
  encoding here so Nov/Dec dates don't fall outside the training range).
- `src/train.py` — runs the split comparisons and the unseen-city
  experiment, then saves the two final CatBoost models.
- `src/predict.py` — loads those models, writes both submission files.
- Point out `report.docx` Section 2 ("What changed, and why") — it explains
  this is a merged, reviewed version of two earlier drafts, not a first pass.

## Don't forget to show
- `scorer_results/candidate_december.png` — note the clean weekly cycle
  ($811–$835) is what the model actually learned from day-of-week effects,
  and that it can't reflect month-to-month market swings since December has
  no market_index/quote_signal data to work with.
