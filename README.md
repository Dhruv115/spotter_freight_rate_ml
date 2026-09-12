# Freight Rate Prediction — Spotter ML Engineer Assessment (merged solution)

Predicts `posted_rate` for freight loads with a CatBoostRegressor trained on
**log(rate per mile)**. See `report.docx` / `report.pdf` for the full
write-up: data-quality findings, an unseen-city validation experiment, model
comparison, and the December 2025 forecast chart. `report.docx` Section 2
("What changed, and why") explains how this version merges and improves on
two earlier drafts.

## Structure

```
data/
  train_test.csv, validation.csv          NOT included in this repo 
                                            Place your local
                                           copies here before running.
  validation_predictions_template.csv     included (12,000 load_ids, no data)
  december_chart_inputs.csv               included — completed with
                                           predicted_rate after predict.py
src/
  features.py     shared cleaning + feature engineering
  train.py        trains + validates both models, runs the validation
                   experiments in the report, saves models to models/
  predict.py      loads the saved models, writes the submission files
models/                     model_full.cbm, model_reduced.cbm, city_lookup.joblib
outputs/                    holdout metrics, feature importance, report figures
scorer_results/              candidate_december.png (created by score.py)
score.py                     scorer provided with the assessment
requirements.txt
validation_predictions.csv   final submission file (created by predict.py)
report.docx / report.pdf     written report
build_report.js              generates report.docx (not needed to reproduce
                              predictions)
```

## Run instructions

```bash
python -m pip install -r requirements.txt

# 0. Put train_test.csv and validation.csv in data/ (see GITHUB_SETUP.md)

# 1. Clean the data, run the validation experiments, save the two final models
python src/train.py

# 2. Generate validation_predictions.csv and fill in
#    data/december_chart_inputs.csv
python src/predict.py

# 3. Run the provided scorer (validates both files, regenerates the chart)
python score.py --predictions validation_predictions.csv \
                 --december-predictions data/december_chart_inputs.csv
```

`train.py` takes under a minute on a single CPU core.

## Approach 

A `CatBoostRegressor` is trained on `log(posted_rate / distance)` — i.e. it
predicts **rate per mile**, converted back to a dollar prediction by
multiplying by `distance`. This single choice (adopted from one of the two
draft solutions this report merges) is the largest accuracy improvement
found across either draft. 
Validation uses a **time-based holdout** (train
Jan–Sep 2025, validate on October) because the real task is forecasting the
unseen Nov/Dec period, not interpolating within data the model has already
partly seen. 
An additional experiment holds specific cities out of training
entirely to directly measure — not just assume — how much accuracy degrades
on cities never seen during training, since `validation.csv` contains 8 such
cities.
Two models are trained: a full-feature model for
`validation_predictions.csv`, and a reduced-feature model (no
`market_index`/`quote_signal`, which `december_chart_inputs.csv` doesn't
provide) for the December chart. Full details in `report.docx`.
