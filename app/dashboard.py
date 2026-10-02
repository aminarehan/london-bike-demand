"""Interactive map of forecast bike demand across London's docking stations.

Run from the project folder:  streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train import feature_columns  # noqa: E402  (same feature list the models were trained on)

st.set_page_config(page_title="London Bike Demand", layout="wide")


@st.cache_data
def load():
    preds = pd.read_parquet(ROOT / "data/processed/predictions.parquet")
    features = pd.read_parquet(ROOT / "data/processed/test_features.parquet")
    results = json.loads((ROOT / "reports/results.json").read_text())
    return preds, features, results


@st.cache_resource
def load_models():
    return {t: joblib.load(ROOT / f"models/{t}.joblib") for t in ("departures", "arrivals", "net_flow")}


preds, features, results = load()
models = load_models()

st.title("London Bike-Share Demand Forecaster")
st.caption("Next-hour departures, arrivals and net flow at every Santander Cycles docking station, forecast with "
           "gradient boosting on 2.4 million TfL journeys. Test period: 11–31 May 2026 (unseen during training). "
           "Forecasts use only information available an hour ahead, including the previous hour's weather.")

with st.sidebar:
    day = st.date_input("Day", value=pd.Timestamp("2026-05-19").date(),
                        min_value=preds.hour.min().date(), max_value=preds.hour.max().date())
    hour_of_day = st.slider("Hour", 0, 23, 8)
    rain = st.toggle("What if the last hour was rainy? (2 mm/h)")

hour = pd.Timestamp(day) + pd.Timedelta(hours=hour_of_day)
view = preds[preds.hour == hour].copy()

if rain:
    # Re-run the models on this hour's real features, with the previous hour's weather swapped for rain.
    wet = features[features.hour == hour].copy()
    wet["precipitation_prev"], wet["wet_prev"] = 2.0, 1
    for target in ("departures", "arrivals", "net_flow"):
        pred = models[target].predict(wet[feature_columns(target)])
        view[f"pred_{target}"] = pred if target == "net_flow" else pred.clip(0)

mean = results["mean_over_folds"]
model, profile = mean["departures/gradient_boosting"], mean["departures/baseline_profile"]
rain_stats = results["rain"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast departures this hour", f"{view.pred_departures.sum():,.0f}", help="Sum across all stations")
c2.metric("Actual departures", f"{view.departures.sum():,.0f}")
c3.metric("City-wide hourly error (WAPE)", f"{model['wape_citywide']:.0%}",
          f"{model['wape_citywide'] - profile['wape_citywide']:+.0%} vs typical-pattern baseline", delta_color="inverse",
          help="Average of two three-week test periods. Per station-hour the error is close to the random noise floor.")
c4.metric("Rain effect on demand", f"{rain_stats['estimate']:+.0%}",
          help=f"Wet vs dry hours, like for like. 95% interval {rain_stats['ci_low']:+.0%} to "
               f"{rain_stats['ci_high']:+.0%} ({rain_stats['wet_hours']} wet hours)")

# Red = forecast to lose bikes (emptying), blue = forecast to gain bikes (filling). Size = forecast departures.
# The map only gets plain columns: timestamps in the layer data stop pydeck drawing anything.
points = pd.DataFrame({
    "name": view.name,
    "lon": view.lon,
    "lat": view.lat,
    "out": view.pred_departures.round(1),
    "net": view.pred_net_flow.round(1),
    "radius": 40 + view.pred_departures * 15,
    "colour": view.pred_net_flow.apply(lambda f: [215, 48, 39, 210] if f < -1 else
                                       [33, 102, 172, 210] if f > 1 else [120, 120, 120, 150]),
})
st.pydeck_chart(pdk.Deck(
    layers=[pdk.Layer("ScatterplotLayer", points, get_position=["lon", "lat"], get_fill_color="colour",
                      get_radius="radius", radius_min_pixels=3, pickable=True)],
    initial_view_state=pdk.ViewState(latitude=51.507, longitude=-0.125, zoom=11.3),
    tooltip={"text": "{name}\nForecast departures: {out}\nForecast net flow: {net}"},
    map_provider="carto", map_style="light"))
st.caption("Red: forecast to empty out · Blue: forecast to fill up · Grey: roughly balanced · Size: forecast departures")

left, right = st.columns(2)
with left:
    st.subheader("Stations forecast to lose the most bikes")
    st.caption(f"Ranked by forecast net outflow. At peak hours, {mean['net_flow/gradient_boosting']['precision_at_10']:.0%} "
               "of the top 10 were in the actual top 10, similar to ranking by each station's usual pattern.")
    short = view.nsmallest(10, "pred_net_flow")[["name", "docks", "pred_departures", "pred_arrivals", "pred_net_flow",
                                                  "departures", "arrivals"]]
    st.dataframe(short.round(1).rename(columns={
        "name": "Station", "docks": "Docks", "pred_departures": "Forecast out", "pred_arrivals": "Forecast in",
        "pred_net_flow": "Forecast net", "departures": "Actual out", "arrivals": "Actual in"}),
        hide_index=True, use_container_width=True)
with right:
    st.subheader(f"City-wide departures on {day:%a %d %b}")
    daily = preds[preds.hour.dt.date == day].groupby("hour")[["departures", "pred_departures"]].sum()
    st.line_chart(daily.rename(columns={"departures": "Actual", "pred_departures": "Forecast"}),
                  color=["#9aa0a6", "#d7301f"])   # grey actual, red forecast
