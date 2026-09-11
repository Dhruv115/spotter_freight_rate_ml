"""
Generate validation_predictions.csv and the completed december_chart_inputs.csv
using the two CatBoost models saved by train.py.

Usage:
    python src/predict.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from features import FULL_FEATURES, REDUCED_FEATURES, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def load_model(path):
    m = CatBoostRegressor()
    m.load_model(str(path))
    return m


def main():
    model_full = load_model(MODEL_DIR / "model_full.cbm")
    model_reduced = load_model(MODEL_DIR / "model_reduced.cbm")
    city_lookup = joblib.load(MODEL_DIR / "city_lookup.joblib")

    # ---------------- validation.csv -> validation_predictions.csv ----------------
    val_raw = pd.read_csv(DATA_DIR / "validation.csv")
    val = prepare(val_raw)  # already has lat/lon, market_index, quote_signal
    pred_rpm = np.exp(model_full.predict(val[FULL_FEATURES]))
    pred = np.maximum(pred_rpm * val["distance"].to_numpy(), 0.01)

    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    out = template[["load_id"]].merge(
        pd.DataFrame({"load_id": val["load_id"], "predicted_rate": pred}), on="load_id", how="left")
    assert out["predicted_rate"].notna().all() and len(out) == 12000
    out.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"Wrote validation_predictions.csv ({len(out):,} rows). "
          f"range ${out['predicted_rate'].min():.2f}-${out['predicted_rate'].max():.2f}, "
          f"mean ${out['predicted_rate'].mean():.2f}")

    # ---------------- december_chart_inputs.csv (reduced model) ----------------
    dec_raw = pd.read_csv(DATA_DIR / "december_chart_inputs.csv")
    dec = prepare(dec_raw.drop(columns=["predicted_rate"]), city_lookup=city_lookup)
    pred_rpm_dec = np.exp(model_reduced.predict(dec[REDUCED_FEATURES]))
    pred_dec = np.maximum(pred_rpm_dec * dec["distance"].to_numpy(), 0.01)

    dec_out = dec_raw.copy()
    dec_out["predicted_rate"] = pred_dec
    dec_out.to_csv(DATA_DIR / "december_chart_inputs.csv", index=False)
    print(f"Wrote data/december_chart_inputs.csv. range ${dec_out['predicted_rate'].min():.2f}"
          f"-${dec_out['predicted_rate'].max():.2f}, mean ${dec_out['predicted_rate'].mean():.2f}")


if __name__ == "__main__":
    main()
