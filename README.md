# London Bike-Share Demand Forecaster

Forecasts next-hour bike **departures and arrivals at every Santander Cycles docking station in London**, and flags
the stations about to run short of bikes. Built on 2.4 million real TfL journeys (March–May 2026) with PySpark,
scikit-learn and MLflow, and served as an interactive map in Streamlit.

**Why it matters:** a bike-share scheme loses customers every time someone finds an empty dock. Knowing an hour
ahead where demand will spike, and which stations will drain, lets an operator move bikes before it happens, and
helps people choose green transport with confidence.

## Results

Held-out test period: 11–31 May 2026 (three weeks the models never saw), 797 stations, 401,688 station-hours.

| Model | Departures MAE | Arrivals MAE | Peak-hour departures MAE |
|---|---|---|---|
| Baseline: same hour last week | 1.18 | 1.17 | 1.75 |
| Ridge regression | 1.04 | 1.03 | 1.48 |
| **Gradient boosting (Poisson loss)** | **0.96** | **0.95** | **1.38** |

Gradient boosting cuts error by **19% against the seasonal baseline overall, and 21% at peak hours** (7–9 am, 5–7 pm).
City-wide, it forecast 1,905 departures for 8 am on Tuesday 19 May against 1,818 actual.

**Findings**
- **Rain cuts demand by about a fifth.** Wet hours see 21% fewer departures than dry hours at the same time of
  day and day type. The model independently estimates 14% fewer under steady rain.
- **Commuters are less put off than leisure riders.** Across the test period, steady rain lowers forecast demand by
  12% at the 8 am weekday peak but 15% on weekend afternoons.
- **Morning hubs drain fast.** The busiest station-hour saw 195 bikes leave and 3 return in a single hour (8 am),
  the pattern the "stations most likely to run short" table is built to catch.
- **E-bikes are now 19% of journeys.**

## How it works

```
data/raw/*.csv ──► PySpark ──► station × hour table ──► features ──► models ──► MLflow
 (2.4M journeys)   clean,      (1.78M rows, zeros       calendar,     baseline,   every run
                   aggregate    for quiet hours)         weather,      ridge,      logged
                                                         location,     gradient
                                                         lags          boosting ──► Streamlit map
```

1. **`src/build_hourly_demand.py` (PySpark).** Reads every journey file, drops broken and implausible trips
   (under a minute or over a day), and aggregates departures and arrivals per station per hour. It builds the full
   station × hour grid so hours with no hires are recorded as zero rather than going missing.
2. **`src/fetch_context.py`.** Station locations and dock counts from the TfL BikePoint API; hourly London
   weather from Open-Meteo. Both are open and need no key.
3. **`src/train.py`.** Features: hour, day of week, weekends and bank holidays; temperature, rain and wind;
   distance from the centre and a k-means **area cluster** of station locations; and the station's own recent
   history (last hour, same hour yesterday, same hour last week, 24-hour mean). Every lag looks strictly backwards,
   and the test set is the **final three weeks**, never a random split, so nothing leaks from the future. Each model
   run is logged to **MLflow** with its parameters and metrics.
4. **`app/dashboard.py` (Streamlit + pydeck).** Pick a day and hour to see every station sized by forecast demand
   and coloured by whether it is filling or emptying, the stations most likely to run short, forecast against actual
   for the whole day, and a **"what if it rains?"** toggle that re-runs the models with the weather changed.

**Design choices**
- **Poisson loss** for gradient boosting, because the targets are counts: predictions stay non-negative and the
  error is judged relative to how busy a station is.
- **A seasonal-naive baseline** ("same hour last week") gives the models something honest to beat.
- **Peak-hour MAE** is reported separately, because rush hours are when a wrong forecast costs most.

**Limitations and next steps**
- Weather comes from one point in central London; per-area weather would sharpen the outer stations.
- "Run short" is judged from forecast net flow, not live bike counts. Adding TfL's live dock availability would turn
  it into a true "empty in the next hour" alert.
- Three months of spring data. A full year would capture summer peaks and winter lows.

## Run it

Requires Python 3.12 and Java 17 (for Spark).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download March–May 2026 journeys from TfL open data (about 400 MB)
for f in 439JourneyDataExtract01Mar2026-15Mar2026 440JourneyDataExtract15Mar2026-31Mar2026 \
         441JourneyDataExtract01Apr2026-15Apr2026 442JourneyDataExtract16Apr2026-30Apr2026 \
         443JourneyDataExtract01May2026-16May2026 444JourneyDataExtract17May2026-31May2026; do
  curl -L -o "data/raw/$f.csv" "https://s3-eu-west-1.amazonaws.com/cycling.data.tfl.gov.uk/usage-stats/$f.csv"
done

# 2. Build, train, explore
python src/build_hourly_demand.py
python src/fetch_context.py
python src/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db     # compare runs at http://localhost:5000
streamlit run app/dashboard.py

# Tests
pytest -q
```

Or run the dashboard in Docker once the models are built:

```bash
docker build -t london-bike-demand .
docker run -p 8501:8501 london-bike-demand
```

## Data

- Journeys: [TfL cycle hire usage data](https://cycling.data.tfl.gov.uk/), Powered by TfL Open Data
- Stations: [TfL Unified API, BikePoint](https://api.tfl.gov.uk/BikePoint)
- Weather: [Open-Meteo historical weather API](https://open-meteo.com/)
