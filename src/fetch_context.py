"""Download the context the model needs alongside the journeys: station locations and London weather.

Run from the project folder:  python src/fetch_context.py
"""
import pandas as pd
import requests

STATIONS_OUT = "data/processed/stations.csv"
WEATHER_OUT = "data/processed/weather_hourly.csv"
START, END = "2026-03-01", "2026-05-31"   # matches the journey files in data/raw


def fetch_stations() -> pd.DataFrame:
    """Every docking station from TfL's BikePoint API (open, no key needed)."""
    points = requests.get("https://api.tfl.gov.uk/BikePoint", timeout=60).json()
    rows = []
    for p in points:
        props = {a["key"]: a["value"] for a in p["additionalProperties"]}
        rows.append({
            # The journey files identify stations by this terminal number, not by the API id.
            "station": props["TerminalName"],
            "name": p["commonName"],
            "lat": p["lat"],
            "lon": p["lon"],
            "docks": int(props["NbDocks"]),
        })
    return pd.DataFrame(rows)


def fetch_weather() -> pd.DataFrame:
    """Hourly central-London weather from the Open-Meteo archive (free, no key needed)."""
    resp = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params={
        "latitude": 51.5074, "longitude": -0.1278,
        "start_date": START, "end_date": END,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "Europe/London",
    }).json()["hourly"]
    return pd.DataFrame({
        "hour": pd.to_datetime(resp["time"]),
        "temperature_c": resp["temperature_2m"],
        "precipitation_mm": resp["precipitation"],
        "wind_kmh": resp["wind_speed_10m"],
    })


if __name__ == "__main__":
    stations = fetch_stations()
    stations.to_csv(STATIONS_OUT, index=False)
    weather = fetch_weather()
    weather.to_csv(WEATHER_OUT, index=False)
    print(f"stations: {len(stations)} | weather hours: {len(weather)} "
          f"| wet hours: {(weather.precipitation_mm > 0.2).mean():.0%}")
