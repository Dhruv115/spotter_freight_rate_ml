"""
Train and validate the merged freight-rate model.

Usage:
    python src/train.py

What this does, and why (see report.docx for the full writeup):
  1. Cleans the data (weight sign-flip fix -- keep abs value only).
  2. Compares a time-based holdout (train Jan-Sep, validate Oct) against a
     random 85/15 holdout, on the identical model, to show the random split
     is optimistic for this forecasting task.
  3. Runs an "unseen-city" experiment: holds a handful of cities out of
     training entirely and evaluates on them, to directly measure (not just
     assert) how much the route/pickup/delivery categoricals degrade for
     cities never seen in training -- exactly the situation validation.csv
     puts us in for 8 real cities.
  4. Trains two final CatBoost models on 100% of train_test.csv:
       - model_full.cbm     (includes market_index/quote_signal) -> used for
                              validation_predictions.csv
       - model_reduced.cbm  (excludes them, since december_chart_inputs.csv
                              doesn't provide them) -> used for the December
                              chart
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

sys.path.insert(0, str(Path(__file__).parent))
from features import CAT_FEATURES, FULL_FEATURES, REDUCED_FEATURES, build_city_lookup, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
for d in (MODEL_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)

RANDOM_SEED = 42
CATBOOST_PARAMS = dict(iterations=200, depth=7, learning_rate=0.06, l2_leaf_reg=10.0,
                        loss_function="RMSE", random_seed=RANDOM_SEED, verbose=False, thread_count=4)


def evaluate(actual, pred, label):
    metrics = {
        "MAE": round(float(mean_absolute_error(actual, pred)), 2),
        "RMSE": round(float(np.sqrt(np.mean((actual - pred) ** 2))), 2),
        "median_AE": round(float(np.median(np.abs(actual - pred))), 2),
        "MAPE_%": round(100 * float(mean_absolute_percentage_error(actual, pred)), 2),
        "n": int(len(actual)),
    }
    print(f"  [{label:38s}] " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
    return metrics


def fit_catboost(train_df, feature_list):
    X = train_df[feature_list]
    y = np.log(train_df["posted_rate"] / train_df["distance"])
    cat_idx = [X.columns.get_loc(c) for c in CAT_FEATURES]
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(X, y, cat_features=cat_idx, verbose=False)
    return model


def predict_dollars(model, df, feature_list):
    pred_rpm = np.exp(model.predict(df[feature_list]))
    return pred_rpm * df["distance"].to_numpy()


def main():
    t_start = time.time()
    raw = pd.read_csv(DATA_DIR / "train_test.csv")
    city_lookup = build_city_lookup(raw)
    df = prepare(raw)
    print(f"Loaded {len(df):,} rows, {df['date'].min().date()} to {df['date'].max().date()}")

    cutoff = pd.Timestamp("2025-10-01")
    train_t, hold_t = df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy()
    print(f"Time-based split: train={len(train_t):,}, holdout={len(hold_t):,} (Oct 2025)")

    results = {}

    # ---------------------------------------------------------------
    # 1) Time-based vs. random split, full features
    # ---------------------------------------------------------------
    print("\n=== Split-strategy comparison (full features) ===")
    m_time = fit_catboost(train_t, FULL_FEATURES)
    pred_time = predict_dollars(m_time, hold_t, FULL_FEATURES)
    metrics_time = evaluate(hold_t["posted_rate"].to_numpy(), pred_time, "Time-based holdout (Oct)")

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = df.sample(frac=1.0, random_state=RANDOM_SEED)
    cut = int(0.85 * len(shuffled))
    train_r, hold_r = shuffled.iloc[:cut], shuffled.iloc[cut:]
    m_rand = fit_catboost(train_r, FULL_FEATURES)
    pred_rand = predict_dollars(m_rand, hold_r, FULL_FEATURES)
    metrics_rand = evaluate(hold_r["posted_rate"].to_numpy(), pred_rand, "Random 85/15 holdout")
    results["split_strategy_comparison"] = {"time_based": metrics_time, "random": metrics_rand}

    hold_out_df = hold_t[["load_id", "date", "equipment", "pickup", "delivery", "distance", "posted_rate"]].copy()
    hold_out_df["predicted_rate"] = pred_time
    hold_out_df.to_csv(OUT_DIR / "holdout_predictions.csv", index=False)

    # ---------------------------------------------------------------
    # 2) Unseen-city experiment: hold a handful of cities out of
    #    training ENTIRELY, evaluate on Oct rows that touch them.
    #    Quantifies (rather than just asserts) how much the route /
    #    pickup / delivery categoricals degrade for cities the model
    #    has literally never seen -- the exact situation validation.csv
    #    creates for 8 real cities.
    # ---------------------------------------------------------------
    print("\n=== Unseen-city experiment (full features) ===")
    all_cities = sorted(set(df["pickup"]) | set(df["delivery"]))
    held_out_cities = set(pd.Series(all_cities).sample(6, random_state=RANDOM_SEED))
    print(f"  Cities excluded from training entirely: {sorted(held_out_cities)}")

    touches_held_city = train_t["pickup"].isin(held_out_cities) | train_t["delivery"].isin(held_out_cities)
    train_no_city = train_t[~touches_held_city]
    m_no_city = fit_catboost(train_no_city, FULL_FEATURES)

    hold_touches = hold_t["pickup"].isin(held_out_cities) | hold_t["delivery"].isin(held_out_cities)
    hold_unseen = hold_t[hold_touches]
    hold_known = hold_t[~hold_touches]

    pred_unseen = predict_dollars(m_no_city, hold_unseen, FULL_FEATURES)
    pred_known = predict_dollars(m_no_city, hold_known, FULL_FEATURES)
    metrics_unseen = evaluate(hold_unseen["posted_rate"].to_numpy(), pred_unseen, "Oct rows touching a NEVER-seen city")
    metrics_known = evaluate(hold_known["posted_rate"].to_numpy(), pred_known, "Oct rows, all-known cities (same model)")
    results["unseen_city_experiment"] = {
        "held_out_cities": sorted(held_out_cities),
        "rows_touching_unseen_city_in_training_removed": int(touches_held_city.sum()),
        "never_seen_city_holdout": metrics_unseen,
        "known_city_holdout_same_model": metrics_known,
    }

    # ---------------------------------------------------------------
    # 3) Reduced feature set (Model B, used for December)
    # ---------------------------------------------------------------
    print("\n=== Reduced feature set: no market_index/quote_signal (time-based holdout) ===")
    m_reduced_eval = fit_catboost(train_t, REDUCED_FEATURES)
    pred_reduced = predict_dollars(m_reduced_eval, hold_t, REDUCED_FEATURES)
    metrics_reduced = evaluate(hold_t["posted_rate"].to_numpy(), pred_reduced, "Reduced features, time-based holdout")
    results["model_B_reduced_features"] = metrics_reduced

    # ---------------------------------------------------------------
    # Final fit: retrain both models on 100% of train_test.csv
    # ---------------------------------------------------------------
    print("\n=== Refitting final models on 100% of train_test.csv ===")
    final_full = fit_catboost(df, FULL_FEATURES)
    final_reduced = fit_catboost(df, REDUCED_FEATURES)

    final_full.save_model(str(MODEL_DIR / "model_full.cbm"))
    final_reduced.save_model(str(MODEL_DIR / "model_reduced.cbm"))
    import joblib
    joblib.dump(city_lookup, MODEL_DIR / "city_lookup.joblib")

    fi_full = pd.DataFrame({"feature": FULL_FEATURES, "importance": final_full.get_feature_importance()}) \
        .sort_values("importance", ascending=False)
    fi_full.to_csv(OUT_DIR / "feature_importance_full.csv", index=False)
    print("\nFeature importance (full model, CatBoost PredictionValuesChange):")
    print(fi_full.to_string(index=False))

    results["final_model_feature_importance_full"] = fi_full.to_dict("records")

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone in {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()
