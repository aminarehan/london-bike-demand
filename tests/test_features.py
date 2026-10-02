"""Checks on the feature engineering: distances are right and no lag can see the future."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import add_features, feature_columns, haversine_km  # noqa: E402


def toy_hourly(hours=200):
    """Two stations with departures that simply count up hour by hour, plus weather and location."""
    idx = pd.date_range("2026-05-01", periods=hours, freq="h")
    rows = []
    for station, offset in (("001", 0), ("002", 1000)):
        for i, h in enumerate(idx):
            rows.append({"station": station, "hour": h, "departures": offset + i, "arrivals": offset + i,
                         "lat": 51.5, "lon": -0.12, "docks": 20, "km_from_centre": 1.0, "area": 0,
                         "temperature_c": 12.0, "precipitation_mm": 0.0, "wind_kmh": 10.0})
    return pd.DataFrame(rows)


def test_haversine_known_distance():
    # Charing Cross to King's Cross is about 2.6 km as the crow flies.
    assert np.isclose(haversine_km(51.5074, -0.1278, 51.5308, -0.1238), 2.6, atol=0.3)


def test_lags_only_look_backwards():
    df = add_features(toy_hourly())
    # Departures rise by exactly 1 an hour, so each lag must equal the current value minus its offset.
    assert (df.departures - df.departures_lag_1h == 1).all()
    assert (df.departures - df.departures_lag_24h == 24).all()
    assert (df.departures - df.departures_lag_168h == 168).all()


def test_lags_never_cross_stations():
    df = add_features(toy_hourly())
    first = df.groupby("station").departures_lag_1h.min()
    assert first["002"] >= 1000          # station 002's history never borrows station 001's values


def test_feature_columns_present():
    df = add_features(toy_hourly())
    for target in ("departures", "arrivals"):
        assert set(feature_columns(target)) <= set(df.columns)
