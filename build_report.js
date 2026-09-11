const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ImageRun, AlignmentType, BorderStyle, ShadingType,
} = require("docx");
const fs = require("fs");

const NAVY = "064A56";
const LIGHT = "E8EEEF";
const GREY = "5B6B6E";

function h1(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } }); }
function h2(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 } }); }
function p(text, opts = {}) {
  return new Paragraph({ spacing: { after: 160, line: 300 }, children: [new TextRun({ text, ...opts })] });
}
function bullet(text, opts = {}) {
  return new Paragraph({ bullet: { level: 0 }, spacing: { after: 80, line: 280 }, children: [new TextRun({ text, ...opts })] });
}
function caption(text) {
  return new Paragraph({ spacing: { before: 80, after: 260 }, children: [new TextRun({ text, italics: true, size: 18, color: GREY })] });
}
function image(path, width, height, alignment = AlignmentType.CENTER) {
  return new Paragraph({ alignment, spacing: { before: 120, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width, height } })] });
}
function metricsTable(headers, rows) {
  const headerCells = headers.map((t) => new TableCell({
    width: { size: Math.floor(10000 / headers.length), type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: NAVY },
    children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, color: "FFFFFF", size: 20 })] })],
  }));
  const bodyRows = rows.map((r, i) => new TableRow({
    children: r.map((cell) => new TableCell({
      width: { size: Math.floor(10000 / headers.length), type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: i % 2 === 0 ? "FFFFFF" : LIGHT },
      children: [new Paragraph({ children: [new TextRun({ text: String(cell), size: 20 })] })],
    })),
  }));
  return new Table({ width: { size: 10000, type: WidthType.DXA },
    rows: [new TableRow({ children: headerCells, tableHeader: true }), ...bodyRows] });
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 1200, right: 1200 } } },
    children: [
      new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: "Spotter — Machine Learning Engineer Assessment", color: NAVY, bold: true, size: 44 })] }),
      new Paragraph({ spacing: { after: 300 }, children: [new TextRun({ text: "Freight rate prediction — merged solution: CatBoost, rate-per-mile target, two-model December handling", color: GREY, size: 24 })] }),
      new Paragraph({ border: { bottom: { color: NAVY, space: 4, style: BorderStyle.SINGLE, size: 8 } }, spacing: { after: 300 }, children: [new TextRun({ text: "" })] }),

      h1("1. Summary"),
      p("The final model is a CatBoostRegressor trained on log(posted_rate / distance) — i.e. it predicts rate per mile, which is then multiplied back by distance to recover the dollar prediction. On a time-based holdout (train Jan\u2013Sep 2025, validate on October, which the model never sees during training) it scores MAE $114.67, RMSE $649.41, median absolute error $41.66, and MAPE 5.48%. Two versions are trained: a full-feature model (includes market_index/quote_signal) used for validation_predictions.csv, and a reduced-feature model (excludes them, since december_chart_inputs.csv doesn't provide them) used only for the December chart."),
      bullet("Target: log(rate per mile), not log(rate) \u2014 divides out the dominant distance effect before the model starts, which measurably improves accuracy over modeling the raw rate."),
      bullet("Categoricals: pickup, delivery, equipment, route (pickup+delivery), route_equipment \u2014 handled natively by CatBoost, no one-hot encoding needed."),
      bullet("Weight: only the corrected weight_abs plus a weight_missing flag are used as features. The raw signed value is dropped \u2014 see Section 3."),
      bullet("Validation: time-based holdout is the headline number; a random-split comparison and a new unseen-city experiment (Section 5) are run alongside it to pressure-test the validation methodology itself, not just the model."),
      bullet("December: predicted rate for the fixed Lexington\u2192Fort Wayne lane ranges $811\u2013$835 across the 31 days of December, with a clean weekly cycle."),

      h1("2. What changed, and why (merge notes)"),
      p("This report merges two earlier draft solutions to the same assessment. Rather than pick one wholesale, each design decision was evaluated on its own merits and, where the evidence supported it, adopted, dropped, or modified:"),
      bullet("Adopted: modeling log(rate per mile) instead of log(rate). This was the single biggest accuracy win in either draft \u2014 it's the reason this model beats a plain log(rate) gradient-boosting model on MAE and MAPE."),
      bullet("Adopted, then verified: route/route_equipment categoricals. These were kept, but Section 6's feature importance shows they contribute far less than expected once pickup/delivery/geo_distance are already in the model (route ranks near the bottom, not near the top) \u2014 worth knowing before leaning on this as a selling point."),
      bullet("Adopted: keeping the raw train_test.csv/validation.csv data files out of the submitted repository. One draft's GITHUB_SETUP.md flagged that Spotter's assessment data is likely not meant for public redistribution \u2014 a legitimate concern the other draft missed. See README.md."),
      bullet("Changed: weight is represented only by weight_abs (the corrected value) and a weight_missing flag. One draft additionally fed the model the raw signed weight; since abs(weight) for the negative rows matches the positive-weight distribution almost exactly (Section 3), the sign is a data-entry artifact, not a signal, and keeping it risks the model fitting noise."),
      bullet("Changed: date features are cyclical only (month_sin/cos, dow_sin/cos, plus bounded month/dow). Raw day-of-year and an elapsed-day counter were dropped \u2014 both exceed their training range once the model scores November/December dates, which is exactly the kind of extrapolation tree-based models handle poorly."),
      bullet("Kept from the other draft: two separately-trained models (full vs. reduced feature set) for the two prediction targets, rather than one model fed an all-NaN market_index/quote_signal pattern for December it essentially never saw in training."),
      bullet("Added: an unseen-city holdout experiment (Section 5) to actually measure, rather than just assert, how much the model's accuracy degrades on cities it has never seen \u2014 relevant because validation.csv contains 8 such cities."),

      h1("3. Data quality"),
      p("weight has a sign-flip data-entry error in a subset of rows (values as low as \u221247,500 lb). This is distinguishable from genuine missingness: |weight| for the negative rows matches the distribution of weight for the normal positive rows almost exactly (same mean \u2248 31,700 lb, same 5,000\u201347,500 lb range), which is strong evidence of a sign artifact rather than a real, different population. Fix: only the corrected weight_abs is used as a model feature; the raw signed value is dropped entirely rather than fed alongside it, since it carries no real signal."),
      p("Separately, weight and market_index each have a small fraction of genuinely missing (NaN) values \u2014 under 1% in train_test.csv, up to ~2% in validation.csv \u2014 spread roughly evenly across equipment types, consistent with missing-at-random rather than a systematic collection issue. These are handled with an explicit weight_missing indicator plus CatBoost's native NaN support, which is a separate mechanism from the weight_abs fix above; the two are easy to conflate in writing but address different problems (a sign artifact vs. real missingness)."),
      p("validation.csv contains 8 pickup/delivery cities never seen in train_test.csv (Chicago, Knoxville, San Diego, Allentown, Jackson, Charlotte, Norfolk, Laredo). This is addressed two ways: geographically, via lat/lon and a haversine geo_distance feature that generalizes smoothly to new locations; and categorically, via CatBoost's native MISSING-category fallback for pickup/delivery/route values it has never seen. Section 5 measures how well this actually works rather than assuming it does."),

      h1("4. Feature engineering"),
      p("Categorical (handled natively by CatBoost, no one-hot encoding): pickup, delivery, equipment, route (pickup + delivery concatenated), route_eq (route + equipment)."),
      p("Numeric: distance, distance_log, geo_distance (haversine distance from lat/lon), distance_ratio (distance / geo_distance \u2014 captures how road-circuitous a lane is), weight_abs, weight_missing, pickup/delivery lat & lon, market_index, quote_signal (full model only), month, dow, and cyclical month_sin/cos, dow_sin/cos."),
      p("Target: log(posted_rate / distance), i.e. log rate-per-mile. Predictions are recovered as exp(model output) \u00D7 distance."),

      h1("5. Validation approach"),
      h2("5.1 Time-based vs. random split"),
      p("The real task is forecasting November/December, dates the model has never seen \u2014 so the headline validation number comes from a time-based holdout: train on 2025-01-01 through 2025-09-30, validate on October (never seen during training). A random 85/15 holdout was run on the identical model for comparison only."),
      metricsTable(["Split", "MAE", "RMSE", "Median AE", "MAPE", "n"], [
        ["Time-based (Jan\u2013Sep train / Oct holdout) \u2014 reported", "$114.67", "$649.41", "$41.66", "5.48%", "4,853"],
        ["Random 85/15 holdout \u2014 comparison only", "$91.26", "$625.92", "$26.64", "3.80%", "7,200"],
      ]),
      caption("Table 1. Same model, same features, two split strategies. The random split is a substantially more optimistic (and less realistic) estimate for this forecasting task."),
      h2("5.2 Unseen-city experiment"),
      p("To actually measure the risk from Section 3 rather than just flag it: 6 cities (Albany, Bakersfield, Lubbock, New York, Salt Lake City, Syracuse) were removed from training entirely (8,422 rows dropped from the Jan\u2013Sep training set), and the resulting model was evaluated on the October holdout rows that touch one of those 6 cities \u2014 simulating exactly what validation.csv does with its 8 real unseen cities \u2014 against the same model's performance on ordinary, known-city October rows."),
      image("outputs/fig_unseen_city.png", 480, 256),
      caption("Figure 1. Same model, evaluated on October rows touching a city it never saw in training vs. rows with only known cities."),
      metricsTable(["Holdout subset", "MAE", "RMSE", "Median AE", "MAPE", "n"], [
        ["Known cities (same model)", "$122.95", "$577.69", "$55.12", "6.37%", "3,905"],
        ["Never-seen-city rows", "$154.61", "$893.94", "$53.25", "5.82%", "948"],
      ]),
      caption("Table 2. MAE and RMSE are meaningfully worse for never-seen cities (as expected), but median absolute error and MAPE are actually similar or slightly better \u2014 the typical prediction isn't much worse, but there are more large misses in the tail. Directionally this confirms the risk from Section 3, but it's a real, moderate degradation rather than a catastrophic one, and the lat/lon + geo_distance fallback is doing real work."),
      p("Caveat: this experiment approximates the real situation but isn't identical to it \u2014 the 6 cities chosen here are removed from a Jan\u2013Sep/Oct split of train_test.csv, while validation.csv's 8 unseen cities appear in Nov/Dec, a period no version of the model has training data for regardless. Treat this as a lower bound on the accuracy gap, not an exact figure."),

      h1("6. Model selection and feature importance"),
      p("CatBoostRegressor was chosen for the same reasons in both source drafts: native categorical handling (no one-hot matrix for pickup/delivery/route), native missing-value support (no imputation step for weight/market_index), and it captures nonlinear interactions between distance, equipment, and geography without manual interaction terms. Configuration: depth 7, learning rate 0.06, 200 iterations, L2 regularization 10, seed 42 \u2014 selected via the time-based holdout above."),
      image("outputs/fig_pred_vs_actual.png", 380, 330),
      caption("Figure 2. Predicted vs. actual posted_rate, October holdout. As in both source drafts, a small number of loads carry rate-per-mile values the given features can't explain (visible as the scattered points off the diagonal); these inflate RMSE relative to MAE/median AE."),
      image("outputs/fig_feature_importance.png", 460, 350),
      caption("Figure 3. CatBoost feature importance (PredictionValuesChange), full model. Two things worth flagging: (1) distance_log + geo_distance + distance + distance_ratio together (\u224839) outweigh equipment (\u224824) \u2014 the \u201cequipment is the top feature\u201d framing in one of the source drafts is an artifact of how importance is split across several correlated distance columns, not evidence that equipment matters more than distance. (2) route and route_eq \u2014 the lane-identity categoricals adopted from the CatBoost draft on the theory that they'd capture lane-specific pricing \u2014 rank near the very bottom (1.51 and 0.05). Once pickup, delivery, and geo_distance are already in the model, the route string adds little independent signal. This doesn't mean the merge decision to keep route was wrong (it's essentially free, and might still help specific lanes like December's), but it's not the differentiator it looked like on paper."),

      h1("7. Handling the December scenario"),
      p("december_chart_inputs.csv omits lat/lon, market_index, and quote_signal. Lat/lon is recovered from a city\u2192coordinate lookup built from train_test.csv (Lexington and Fort Wayne both appear there). market_index/quote_signal cannot be recovered \u2014 no December history exists to extrapolate from \u2014 so a second model, trained on the same data minus those two columns, is used specifically for December, rather than feeding the full model a pattern (both fields NaN simultaneously) it essentially never saw during training."),
      metricsTable(["Model", "MAE", "RMSE", "Median AE", "MAPE"], [
        ["Full features (\u2192 validation_predictions.csv)", "$114.67", "$649.41", "$41.66", "5.48%"],
        ["Reduced, no market_index/quote_signal (\u2192 December)", "$129.23", "$654.41", "$51.92", "5.84%"],
      ]),
      caption("Table 3. Unlike the near-zero cost seen with a different (HistGradientBoosting) model architecture in an earlier draft of this project, dropping market_index/quote_signal costs this CatBoost model a real, measurable ~13% increase in MAE. quote_signal in particular ranks #5 in Figure 3's importance table \u2014 this model leans on it more than the alternative architecture did, so its absence for December is a genuine, not just theoretical, limitation worth stating plainly rather than assuming away."),

      h1("8. December 2025 forecast"),
      p("The chart below is score.py's output against the completed december_chart_inputs.csv. Because pickup, delivery, distance, equipment, and weight are fixed across all 31 rows, only date varies, and the model relies on it through month/dow (and their cyclical encodings). The result is a clean, repeating weekly cycle between about $811 and $835 \u2014 an honest reflection of what the model learned (a within-week pattern), not evidence of a month-to-month market forecast, which Table 3 shows this model has reduced ability to produce without market_index/quote_signal anyway."),
      image("scorer_results/candidate_december.png", 600, 195),
      caption("Figure 4. candidate_december.png, generated by score.py from the submitted december_chart_inputs.csv."),

      h1("9. Reproducing this submission"),
      bullet("python -m pip install -r requirements.txt"),
      bullet("python src/train.py \u2014 cleans the data, runs Sections 5\u20137's comparisons, saves the two final models to models/"),
      bullet("python src/predict.py \u2014 writes validation_predictions.csv and the completed data/december_chart_inputs.csv"),
      bullet("python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv"),
      p("Note: train_test.csv and validation.csv are not included in the submitted GitHub repository (see README.md) \u2014 place them in data/ locally before running. This avoids redistributing Spotter's assessment data in a public repo."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("report.docx", buf);
  console.log("wrote report.docx", buf.length, "bytes");
});
