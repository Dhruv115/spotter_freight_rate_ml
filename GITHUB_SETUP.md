# GitHub submission steps

1. Create a repository (e.g. `spotter-freight-rate-ml`). Spotter's assessment
   materials are proprietary — check with them before making it public; a
   private repo shared with the reviewer is the safer default.
2. Upload everything in this folder **except** `data/train_test.csv` and
   `data/validation.csv` (see `.gitignore` — already excluded). Those two
   files are the assessment's confidential input data, not something you
   should be redistributing to a public/shared GitHub repo. `train.py` and
   `predict.py` expect them at `data/train_test.csv` and `data/validation.csv`
   respectively — a reviewer re-running your code just needs to drop the
   originals back into `data/` first.
3. `data/validation_predictions_template.csv` and
   `data/december_chart_inputs.csv` (the *input* template, i.e. before you
   fill in `predicted_rate`) are small and don't contain sensitive
   development data — fine to include, and needed for `predict.py` to run.
4. Include: `README.md`, `GITHUB_SETUP.md`, `requirements.txt`, `src/`,
   `score.py`, `validation_predictions.csv`, `data/december_chart_inputs.csv`
   (completed, with `predicted_rate` filled in), `report.docx`/`report.pdf`,
   `scorer_results/candidate_december.png`, `outputs/metrics.json`,
   `outputs/feature_importance_full.csv`. Model artifacts in `models/`
   (`model_full.cbm`, `model_reduced.cbm`) are optional but nice to include
   since they let a reviewer run `predict.py` without retraining.
5. In the submission form, provide the GitHub URL, `validation_predictions.csv`,
   the report, and the Loom URL as requested by Spotter.
