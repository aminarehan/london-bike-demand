"""Forecast next-hour bike departures and arrivals per docking station.

Builds features, compares a seasonal baseline, a linear model and gradient boosting on a time-based split,
logs every run to MLflow, and saves the best models plus test-period predictions for the dashboard.

Run from the project folder:  python src/train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROCESSED = Path("data/processed")
MODELS = Path("models")
TEST_START = pd.Timestamp("2026-05-11")          # last three weeks are held out
BANK_HOLIDAYS = pd.to_datetime(["2026-04-03", "2026-04-06", "2026-05-04", "2026-05-25"])
CENTRE = (51.5074, -0.1278)                       # Charing Cross
RAIN_MM = 0.2                                     # an hour with more than this counts as wet

CALENDAR = ["hour_of_day", "day_of_week", "is_weekend", "is_bank_holiday"]
WEATHER = ["temperature_c", "precipitation_mm", "wind_kmh", "is_wet"]
STATION = ["lat", "lon", "docks", "km_from_centre", "area"]


def haversine_km(lat, lon, lat0, lon0):
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def load() -> pd.DataFrame:
    hourly = pd.read_parquet(PROCESSED / "station_hourly")
    stations = pd.read_csv(PROCESSED / "stations.csv", dtype={"station": str})
    weather = pd.read_csv(PROCESSED / "weather_hourly.csv", parse_dates=["hour"])

    stations["km_from_centre"] = haversine_km(stations.lat, stations.lon, *CENTRE)
    # Group stations into areas by location (the City, the West End, the East...). Areas behave differently.
    stations["area"] = KMeans(n_clusters=8, n_init=10, random_state=0).fit_predict(stations[["lat", "lon"]])

    # Inner join drops the handful of stations TfL no longer lists (about 0.6% of hires).
    return hourly.merge(stations, on="station").merge(weather, on="hour")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station", "hour"]).reset_index(drop=True)
    df["hour_of_day"] = df.hour.dt.hour
    df["day_of_week"] = df.hour.dt.dayofweek
    df["is_weekend"] = (df.day_of_week >= 5).astype(int)
    df["is_bank_holiday"] = df.hour.dt.normalize().isin(BANK_HOLIDAYS).astype(int)
    df["is_wet"] = (df.precipitation_mm > RAIN_MM).astype(int)

    # Recent history at the same station. Every lag looks strictly backwards, so nothing leaks from the future.
    for target in ("departures", "arrivals"):
        by_station = df.groupby("station")[target]
        df[f"{target}_lag_1h"] = by_station.shift(1)
        df[f"{target}_lag_24h"] = by_station.shift(24)
        df[f"{target}_lag_168h"] = by_station.shift(168)          # same hour last week
        df[f"{target}_mean_24h"] = by_station.transform(lambda s: s.shift(1).rolling(24).mean())
    return df.dropna().reset_index(drop=True)                      # the first week has no week-ago lag


def feature_columns(target: str) -> list[str]:
    lags = [f"{target}_lag_1h", f"{target}_lag_24h", f"{target}_lag_168h", f"{target}_mean_24h"]
    return CALENDAR + WEATHER + STATION + lags


def models(target: str):
    categorical = ["hour_of_day", "day_of_week", "area"]
    numeric = [c for c in feature_columns(target) if c not in categorical]
    linear = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                           ("num", StandardScaler(), numeric)]),
        Ridge(alpha=1.0))
    # Poisson loss suits counts: predictions stay non-negative and errors scale with the size of the count.
    boosted = HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.08,
                                            max_leaf_nodes=63, categorical_features=categorical,
                                            random_state=0)
    return {"ridge": linear, "gradient_boosting": boosted}


def scores(y_true, y_pred, hours) -> dict:
    peak = hours.isin([7, 8, 9, 17, 18, 19]).to_numpy()
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae_peak": mean_absolute_error(y_true[peak], y_pred[peak])}


def main():
    import mlflow  # only needed for training, so the dashboard can import this module without it

    df = add_features(load())
    train, test = df[df.hour < TEST_START], df[df.hour >= TEST_START]
    print(f"train rows: {len(train):,} | test rows: {len(test):,} | stations: {df.station.nunique()}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("london-bike-demand")
    MODELS.mkdir(exist_ok=True)
    results, predictions = {}, test[["station", "name", "lat", "lon", "docks", "hour",
                                     "departures", "arrivals", "precipitation_mm"]].copy()

    for target in ("departures", "arrivals"):
        cols = feature_columns(target)
        X_train, X_test = train[cols], test[cols]
        y_train, y_test = train[target].to_numpy(), test[target].to_numpy()

        # Seasonal-naive baseline: "the same as this hour last week". Any model must beat it to be worth using.
        with mlflow.start_run(run_name=f"{target}-baseline"):
            m = scores(y_test, test[f"{target}_lag_168h"].to_numpy(), test.hour_of_day)
            mlflow.log_params({"target": target, "model": "same_hour_last_week"})
            mlflow.log_metrics(m)
            results[f"{target}/baseline"] = m

        best_name, best_mae, best_model = None, np.inf, None
        for name, model in models(target).items():
            with mlflow.start_run(run_name=f"{target}-{name}"):
                model.fit(X_train, y_train)
                pred = np.clip(model.predict(X_test), 0, None)
                m = scores(y_test, pred, test.hour_of_day)
                mlflow.log_params({"target": target, "model": name, "features": len(cols),
                                   "train_rows": len(train), "test_start": str(TEST_START.date())})
                mlflow.log_metrics(m)
                results[f"{target}/{name}"] = m
                if m["mae"] < best_mae:
                    best_name, best_mae, best_model = name, m["mae"], model

        joblib.dump(best_model, MODELS / f"{target}.joblib")
        predictions[f"pred_{target}"] = np.clip(best_model.predict(X_test), 0, None)
        print(f"{target}: best model = {best_name}")

    predictions["pred_net_flow"] = predictions.pred_arrivals - predictions.pred_departures
    predictions.to_parquet(PROCESSED / "predictions.parquet", index=False)
    test.to_parquet(PROCESSED / "test_features.parquet", index=False)   # lets the dashboard ask "what if it rains?"

    # How much does rain change demand? Compare wet and dry hours at the same time of day and type of day.
    dep = df.groupby(["is_weekend", "hour_of_day", "is_wet"]).departures.mean().unstack()
    rain_effect = float((dep[1] / dep[0] - 1).mean())

    # The same question asked of the model: re-predict the test period as if every hour had steady rain.
    wet = test.copy()
    wet["precipitation_mm"], wet["is_wet"] = 2.0, 1
    dep_model = joblib.load(MODELS / "departures.joblib")
    dry_pred = dep_model.predict(test[feature_columns("departures")])
    wet_pred = dep_model.predict(wet[feature_columns("departures")])
    model_rain_effect = float(wet_pred.sum() / dry_pred.sum() - 1)

    summary = {"results": results, "rain_effect_observed": rain_effect, "rain_effect_model": model_rain_effect,
               "train_rows": len(train), "test_rows": len(test), "stations": int(df.station.nunique())}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/results.json").write_text(json.dumps(summary, indent=2))

    print("\n" + pd.DataFrame(results).T.round(3).to_string())
    print(f"\nrain changes departures by {rain_effect:+.0%} (observed, like-for-like hours) "
          f"and {model_rain_effect:+.0%} (model, steady 2 mm/h rain)")


if __name__ == "__main__":
    main()
