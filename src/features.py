"""
Feature engineering for the merged solution. Combines:
  - distance-normalized target (rate per mile) -- from the CatBoost version,
    measurably improves holdout MAE over modeling raw log(rate)
  - route / route_equipment categoricals -- from the CatBoost version,
    captures lane-specific pricing for lanes seen in training
  - lat/lon as a continuous fallback -- from the HGB version, generalizes to
    the 8 cities in validation.csv that never appear in train_test.csv
  - weight sign-flip fix (abs only, no raw signed value) -- see report,
    Section 3: the sign carries no signal, it's a data-entry artifact
  - cyclical-only date encoding (no raw day-of-year / elapsed-day-count) --
    avoids tree models extrapolating past the training date range when
    scoring Nov/Dec dates they never saw during training
"""
from __future__ import annotations

import numpy as np
import pandas as pd 

CAT_FEATURES = ["pickup", "delivery", "equipment", "route", "route_eq"]

NUMERIC_FULL = [
    "distance", "distance_log", "geo_distance", "distance_ratio",
    "weight_abs", "weight_missing",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "market_index", "quote_signal",
    "month", "dow", "month_sin", "month_cos", "dow_sin", "dow_cos",
]
NUMERIC_REDUCED = [c for c in NUMERIC_FULL if c not in ("market_index", "quote_signal")]

FULL_FEATURES = CAT_FEATURES + NUMERIC_FULL
REDUCED_FEATURES = CAT_FEATURES + NUMERIC_REDUCED


def build_city_lookup(*frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for df in frames:
        if {"pickup", "pickup_lat", "pickup_lon"}.issubset(df.columns):
            rows.append(df[["pickup", "pickup_lat", "pickup_lon"]].rename(
                columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}))
        if {"delivery", "delivery_lat", "delivery_lon"}.issubset(df.columns):
            rows.append(df[["delivery", "delivery_lat", "delivery_lon"]].rename(
                columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}))
    lookup = pd.concat(rows, ignore_index=True).drop_duplicates(subset="city")
    return lookup.set_index("city")[["lat", "lon"]]


def _haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def prepare(df: pd.DataFrame, city_lookup: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clean + engineer features. Works for train_test / validation (which
    already have lat/lon) and december_chart_inputs (which doesn't -- pass
    city_lookup to fill it in from known cities)."""
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])

    if city_lookup is not None and "pickup_lat" not in x.columns:
        x["pickup_lat"] = x["pickup"].map(city_lookup["lat"])
        x["pickup_lon"] = x["pickup"].map(city_lookup["lon"])
        x["delivery_lat"] = x["delivery"].map(city_lookup["lat"])
        x["delivery_lon"] = x["delivery"].map(city_lookup["lon"])
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index", "quote_signal"]:
        if c not in x.columns:
            x[c] = np.nan

    # --- weight: fix the sign-flip artifact and keep only the corrected value
    # plus an explicit missingness flag. The raw signed value is NOT kept as
    # a feature -- abs(negative weight) matches the positive weight
    # distribution almost exactly, so the sign carries no real signal and
    # only risks the model learning spurious splits on it.
    x["weight_missing"] = x["weight"].isna().astype(int)
    x["weight_abs"] = x["weight"].abs()

    # --- distance family
    x["distance_log"] = np.log1p(x["distance"])
    x["geo_distance"] = _haversine(x["pickup_lat"], x["pickup_lon"], x["delivery_lat"], x["delivery_lon"])
    x["distance_ratio"] = x["distance"] / (x["geo_distance"] + 1.0)

    # --- date: cyclical only (bounded, generalizes to unseen Nov/Dec
    # without needing to extrapolate a tree split past its training range)
    x["month"] = x["date"].dt.month
    x["dow"] = x["date"].dt.dayofweek
    x["month_sin"] = np.sin(2 * np.pi * x["month"] / 12)
    x["month_cos"] = np.cos(2 * np.pi * x["month"] / 12)
    x["dow_sin"] = np.sin(2 * np.pi * x["dow"] / 7)
    x["dow_cos"] = np.cos(2 * np.pi * x["dow"] / 7)

    # --- route categoricals (captures lane-specific pricing for lanes seen
    # in training; falls back gracefully via CatBoost's MISSING handling for
    # the 8 cities that only appear in validation.csv)
    x["route"] = x["pickup"].astype(str) + "__" + x["delivery"].astype(str)
    x["route_eq"] = x["route"] + "__" + x["equipment"].astype(str)

    for c in CAT_FEATURES:
        x[c] = x[c].fillna("MISSING").astype(str)

    return x
