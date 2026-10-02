"""Checks on the feature engineering: distances are right and no lag can see the future."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import add_features, feature_columns, haversine_km, precision_at_10  # noqa: E402


def toy_hourly(hours=200):
    """Two stations with departures that simply count up hour by hour, plus weather and location."""
    idx = pd.date_range("2026-05-01", periods=hours, freq="h")
    rows = []
    for station, offset in (("001", 0), ("002", 1000)):
        for i, h in enumerate(idx):
            rows.append({"station": station, "hour": h, "departures": offset + i, "arrivals": offset + i,
                         "net_flow": 0, "lat": 51.5, "lon": -0.12, "docks": 20, "km_from_centre": 1.0,
                         "area": 0, "temperature_prev": 12.0, "precipitation_prev": 0.0, "wind_prev": 10.0,
                         "wet_prev": 0})
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
    for target in ("departures", "arrivals", "net_flow"):
        assert set(feature_columns(target)) <= set(df.columns)


def test_models_never_see_current_hour_weather():
    for target in ("departures", "arrivals", "net_flow"):
        cols = feature_columns(target)
        assert not {"precipitation_mm", "temperature_c", "wind_kmh", "is_wet"} & set(cols)


def test_precision_at_10_perfect_and_reversed_rankings():
    hour = pd.Timestamp("2026-05-19 08:00")
    frame = pd.DataFrame({"station": [f"{i:03d}" for i in range(20)], "hour": hour, "hour_of_day": 8,
                          "net_flow": np.arange(20) - 10})
    assert precision_at_10(frame.assign(score=frame.net_flow), "score") == 1.0
    assert precision_at_10(frame.assign(score=-frame.net_flow), "score") == 0.0
