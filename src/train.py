"""Forecast next-hour bike departures, arrivals and net flow per docking station.

Builds leakage-free features, compares two baselines with a linear model and gradient boosting over two
time-based test folds, logs every run (and the final models) to MLflow, and saves the final fold's
predictions for the dashboard.

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
# Two consecutive three-week test windows. Each model trains only on the data before its window.
FOLDS = [(pd.Timestamp("2026-04-20"), pd.Timestamp("2026-05-11")),
         (pd.Timestamp("2026-05-11"), pd.Timestamp("2026-06-01"))]
BANK_HOLIDAYS = pd.to_datetime(["2026-04-03", "2026-04-06", "2026-05-04", "2026-05-25"])
CENTRE = (51.5074, -0.1278)                       # Charing Cross
RAIN_MM = 0.2                                     # an hour with more than this counts as wet
PEAK_HOURS = [7, 8, 9, 17, 18, 19]
TARGETS = ("departures", "arrivals", "net_flow")

CALENDAR = ["hour_of_day", "day_of_week", "is_weekend", "is_bank_holiday"]
# Weather from the hour *before* the forecast hour: a live system knows that, but not the coming hour's rain.
WEATHER = ["temperature_prev", "precipitation_prev", "wind_prev", "wet_prev"]
STATION = ["lat", "lon", "docks", "km_from_centre", "area"]
CATEGORICAL = ["hour_of_day", "day_of_week", "area"]


def haversine_km(lat, lon, lat0, lon0):
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def load() -> pd.DataFrame:
    hourly = pd.read_parquet(PROCESSED / "station_hourly")
    stations = pd.read_csv(PROCESSED / "stations.csv", dtype={"station": str})
    weather = pd.read_csv(PROCESSED / "weather_hourly.csv", parse_dates=["hour"]).sort_values("hour")

    stations["km_from_centre"] = haversine_km(stations.lat, stations.lon, *CENTRE)
    # Group stations into areas by location (the City, the West End, the East...). Areas behave differently.
    stations["area"] = KMeans(n_clusters=8, n_init=10, random_state=0).fit_predict(stations[["lat", "lon"]])

    # Current-hour weather is kept for analysis only; the models see the previous hour's.
    weather["is_wet"] = (weather.precipitation_mm > RAIN_MM).astype(int)
    weather["temperature_prev"] = weather.temperature_c.shift(1)
    weather["precipitation_prev"] = weather.precipitation_mm.shift(1)
    weather["wind_prev"] = weather.wind_kmh.shift(1)
    weather["wet_prev"] = weather.is_wet.shift(1)

    # Inner join drops the handful of stations TfL no longer lists (about 0.6% of hires).
    return hourly.merge(stations, on="station").merge(weather, on="hour")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station", "hour"]).reset_index(drop=True)
    df["hour_of_day"] = df.hour.dt.hour
    df["day_of_week"] = df.hour.dt.dayofweek
    df["is_weekend"] = (df.day_of_week >= 5).astype(int)
    df["is_bank_holiday"] = df.hour.dt.normalize().isin(BANK_HOLIDAYS).astype(int)

    # Recent history at the same station. Every lag looks strictly backwards, so nothing leaks from the future.
    for target in TARGETS:
        by_station = df.groupby("station")[target]
        df[f"{target}_lag_1h"] = by_station.shift(1)
        df[f"{target}_lag_24h"] = by_station.shift(24)
        df[f"{target}_lag_168h"] = by_station.shift(168)          # same hour last week
        df[f"{target}_mean_24h"] = by_station.transform(lambda s: s.shift(1).rolling(24).mean())
    return df.dropna().reset_index(drop=True)                      # the first week has no week-ago lag


def lag_columns(target: str) -> list[str]:
    return [f"{target}_lag_1h", f"{target}_lag_24h", f"{target}_lag_168h", f"{target}_mean_24h"]


def feature_columns(target: str) -> list[str]:
    # Net flow depends on both sides, so its model sees departure and arrival history as well as its own.
    sources = ("departures", "arrivals", "net_flow") if target == "net_flow" else (target,)
    return CALENDAR + WEATHER + STATION + [c for s in sources for c in lag_columns(s)]


def models(target: str):
    numeric = [c for c in feature_columns(target) if c not in CATEGORICAL]
    linear = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                           ("num", StandardScaler(), numeric)]),
        Ridge(alpha=1.0))
    # Counts get a Poisson loss (non-negative, error relative to how busy a station is). Net flow can be
    # negative, so it uses ordinary squared error.
    boosted = HistGradientBoostingRegressor(loss="squared_error" if target == "net_flow" else "poisson",
                                            max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                            categorical_features=CATEGORICAL, random_state=0)
    return {"ridge": linear, "gradient_boosting": boosted}


def historical_profile(train: pd.DataFrame, target: str) -> pd.DataFrame:
    """The fair baseline: this station's average for this hour on this type of day, from training data only."""
    return (train.groupby(["station", "hour_of_day", "is_weekend"])[target].mean()
            .rename(f"{target}_profile").reset_index())


def precision_at_10(frame: pd.DataFrame, score: str) -> float:
    """At each peak hour, how many of the 10 stations ranked most likely to drain really were in the actual top 10."""
    peak = frame[frame.hour_of_day.isin(PEAK_HOURS)]
    hits = [len(set(g.nsmallest(10, "net_flow").station) & set(g.nsmallest(10, score).station)) / 10
            for _, g in peak.groupby("hour")]
    return float(np.mean(hits))


def scores(frame: pd.DataFrame, target: str, pred: np.ndarray) -> dict:
    y = frame[target].to_numpy()
    peak = frame.hour_of_day.isin(PEAK_HOURS).to_numpy()
    out = {"mae": mean_absolute_error(y, pred),
           "rmse": float(np.sqrt(mean_squared_error(y, pred))),
           "mae_peak": mean_absolute_error(y[peak], pred[peak])}
    if target == "net_flow":
        out["precision_at_10"] = precision_at_10(frame.assign(score=pred), "score")
    else:
        # WAPE: total absolute error as a share of total demand. Unlike MAE it says how big the error is
        # relative to how busy the stations are.
        out["wape"] = float(np.abs(y - pred).sum() / y.sum())
        city = frame.assign(pred=pred).groupby("hour")[[target, "pred"]].sum()
        out["wape_citywide"] = float((city[target] - city.pred).abs().sum() / city[target].sum())
    return out


def rain_effect_with_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Wet vs dry hours, like for like (same hour of day, weekday or weekend), with a 95% interval from
    resampling whole days. Uses observed current-hour weather: this is analysis, not forecasting."""
    city = df.groupby("hour").agg(departures=("departures", "sum"), is_wet=("is_wet", "first"),
                                  hour_of_day=("hour_of_day", "first"), is_weekend=("is_weekend", "first"))
    city["day"] = city.index.normalize()

    def effect(frame):
        cell = frame.groupby(["is_weekend", "hour_of_day", "is_wet"]).departures.mean().unstack()
        if 1 not in cell or 0 not in cell:
            return np.nan
        return float(np.nanmean(cell[1] / cell[0] - 1))

    rng = np.random.default_rng(seed)
    days = city.day.unique()
    by_day = {d: g for d, g in city.groupby("day")}
    boot = [effect(pd.concat([by_day[d] for d in rng.choice(days, len(days))])) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"estimate": effect(city), "ci_low": float(lo), "ci_high": float(hi),
            "wet_hours": int(city.is_wet.sum()), "hours": len(city)}


def evaluate_fold(df, start, end, mlflow, fold, final):
    train, test = df[df.hour < start], df[(df.hour >= start) & (df.hour < end)].copy()
    results, fitted = {}, {}
    for target in TARGETS:
        cols = feature_columns(target)
        test = test.merge(historical_profile(train, target), on=["station", "hour_of_day", "is_weekend"], how="left")
        test[f"{target}_profile"] = test[f"{target}_profile"].fillna(0)

        baselines = {"baseline_last_week": test[f"{target}_lag_168h"].to_numpy(),
                     "baseline_profile": test[f"{target}_profile"].to_numpy()}
        if target == "net_flow":   # the original approach: forecast arrivals minus forecast departures
            baselines["arrivals_minus_departures"] = (fitted["arrivals"][1] - fitted["departures"][1])
        for name, pred in baselines.items():
            with mlflow.start_run(run_name=f"{target}-{name}-fold{fold}"):
                m = scores(test, target, pred)
                mlflow.log_params({"target": target, "model": name, "fold": fold})
                mlflow.log_metrics(m)
                results[f"{target}/{name}"] = m

        best = (None, np.inf, None, None)
        for name, model in models(target).items():
            with mlflow.start_run(run_name=f"{target}-{name}-fold{fold}"):
                model.fit(train[cols], train[target])
                pred = model.predict(test[cols])
                pred = pred if target == "net_flow" else np.clip(pred, 0, None)
                m = scores(test, target, pred)
                mlflow.log_params({"target": target, "model": name, "fold": fold, "features": len(cols),
                                   "train_rows": len(train), "test_start": str(start.date())})
                mlflow.log_metrics(m)
                results[f"{target}/{name}"] = m
                if final:
                    # Store the fitted model with its run, so any result in MLflow can be traced to its model.
                    path = MODELS / f"{target}_{name}.joblib"
                    joblib.dump(model, path)
                    mlflow.log_artifact(str(path), artifact_path="model")
                if m["mae"] < best[1]:
                    best = (name, m["mae"], model, pred)
        fitted[target] = (best[2], best[3], best[0])
    return results, fitted, test


def main():
    import mlflow  # only needed for training, so the dashboard can import this module without it

    df = add_features(load())
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("london-bike-demand")
    MODELS.mkdir(exist_ok=True)

    all_results = {}
    for fold, (start, end) in enumerate(FOLDS, 1):
        final = fold == len(FOLDS)
        results, fitted, test = evaluate_fold(df, start, end, mlflow, fold, final)
        all_results[f"fold{fold}"] = results
        print(f"fold {fold} ({start.date()} to {(end - pd.Timedelta(days=1)).date()}): "
              + ", ".join(f"{t} -> {fitted[t][2]}" for t in TARGETS))

    # The final fold's models and predictions feed the dashboard.
    for target in TARGETS:
        joblib.dump(fitted[target][0], MODELS / f"{target}.joblib")
    predictions = test[["station", "name", "lat", "lon", "docks", "hour", "departures", "arrivals", "net_flow"]].copy()
    for target in TARGETS:
        predictions[f"pred_{target}"] = fitted[target][1]
    predictions.to_parquet(PROCESSED / "predictions.parquet", index=False)
    test.to_parquet(PROCESSED / "test_features.parquet", index=False)   # lets the dashboard ask "what if it rains?"

    # Noise floor: even with perfectly known rates, hires at one station in one hour vary at random (Poisson).
    # The MAE a perfect model would still make shows how much of the remaining error is reducible at all.
    rng = np.random.default_rng(0)
    rates = np.clip(predictions.pred_departures.to_numpy(), 1e-6, None)
    noise_floor = float(np.mean([np.abs(rng.poisson(rates) - rates).mean() for _ in range(20)]))

    # Rain: observed effect with an interval, and the model's answer if the previous hour had been wet.
    rain = rain_effect_with_ci(df)
    wet = test.copy()
    wet["precipitation_prev"], wet["wet_prev"] = 2.0, 1
    dep = fitted["departures"][0]
    rain["model_effect"] = float(dep.predict(wet[feature_columns("departures")]).sum()
                                 / dep.predict(test[feature_columns("departures")]).sum() - 1)

    folds = list(all_results.values())
    mean = {k: {m: float(np.mean([f[k][m] for f in folds])) for m in folds[0][k]} for k in folds[0]}
    summary = {"mean_over_folds": mean, "folds": all_results, "rain": rain, "noise_floor_mae": noise_floor,
               "test_period": [str(FOLDS[-1][0].date()), str((FOLDS[-1][1] - pd.Timedelta(days=1)).date())],
               "stations": int(df.station.nunique())}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/results.json").write_text(json.dumps(summary, indent=2))

    print("\nMean over both folds:\n" + pd.DataFrame(mean).T.round(3).to_string())
    print(f"\nrain: {rain['estimate']:+.0%} observed (95% CI {rain['ci_low']:+.0%} to {rain['ci_high']:+.0%}, "
          f"{rain['wet_hours']} wet hours) | model, after a wet hour: {rain['model_effect']:+.0%}")
    print(f"noise floor: a perfect model would still have departures MAE of about {noise_floor:.2f}")


if __name__ == "__main__":
    main()
