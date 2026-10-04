# London Bike-Share Demand Forecaster

Forecasts next-hour bike **departures, arrivals and net flow at every Santander Cycles docking station in London**,
using only information available an hour ahead. Built on 2.4 million real TfL journeys (March–May 2026) with
PySpark, scikit-learn and MLflow, and served as an interactive map in Streamlit.

**Why it matters:** a bike-share scheme loses customers every time someone finds an empty dock. Knowing an hour
ahead where demand will spike lets an operator plan bike redistribution and staffing, and helps people choose green
transport with confidence.

## Results

Evaluated on **two consecutive three-week test windows** (20 April–10 May and 11–31 May 2026). Each model trains
only on the data before its window. Figures are the mean of the two windows, across 797 stations.

**Departures** (arrivals behave almost identically):

| Model | MAE per station-hour | WAPE per station-hour | City-wide hourly WAPE |
|---|---|---|---|
| Baseline: same hour last week | 1.19 | 80% | 16.0% |
| Baseline: station's usual for this hour and day type | 0.99 | 67% | 22.8% |
| Ridge regression | 1.04 | 70% | 16.0% |
| **Gradient boosting (Poisson loss)** | **0.97** | **65%** | **11.2%** |

**What these numbers mean**
- **City-wide, the model halves the error** of the typical-pattern baseline (11.2% vs 22.8% WAPE), because it reacts
  to the things a fixed pattern cannot: recent demand, the previous hour's weather, holidays.
- **At a single station in a single hour, accuracy is limited by chance.** Stations average only 1.47 hires an hour
  and 48% of station-hours have none. Even a model that knew the true hourly rates exactly would still have an MAE
  of about **0.80**, because individual hires arrive at random (Poisson noise). The model's 0.97 sits close to that
  floor, and only about 2.5% below the typical-pattern baseline's 0.99. The useful signal is in the aggregate.
- **Ranking the stations that will lose the most bikes** is harder: at peak hours, 29% of the model's top 10 were in
  the actual top 10. That is no better than ranking by each station's usual pattern (29%), so the dashboard presents
  it as a ranking by forecast outflow, not as a reliable "will run empty" alert.

**Findings**
- **Rain cuts demand by about a fifth.** Wet hours see **21% fewer departures** than dry hours at the same time of
  day and day type (95% interval: −29% to −12%, from resampling whole days; 54 wet hours out of 2,040).
  Forecasting only from the previous hour's weather, the model predicts **9% fewer** departures after a wet hour.
- **Morning hubs drain fast.** The busiest station-hour saw 195 bikes leave and 3 return in an hour at 8 am. Waterloo
  and King's Cross top the outflow ranking on weekday mornings, while the City and West End fill up.
- **E-bikes are now 19% of journeys.**

## How it works

```
data/raw/*.csv ──► PySpark ──► station × hour table ──► features ──► models ──► MLflow
 (2.4M journeys)   clean,      (1.78M rows, zeros       calendar,     baselines,  every run
                   aggregate    for quiet hours)         past weather, ridge,      + models
                                                         location,     gradient
                                                         lags          boosting ──► Streamlit map
```

1. **`src/build_hourly_demand.py` (PySpark).** Reads every journey file, drops broken and implausible trips
   (under a minute or over a day), and aggregates departures and arrivals per station per hour. It builds the full
   station × hour grid so hours with no hires are recorded as zero. Three months fits in pandas; Spark is used so
   the same pipeline scales to years of journeys without changes.
2. **`src/fetch_context.py`.** Station locations and dock counts from the TfL BikePoint API; hourly London
   weather from Open-Meteo. Both are open and need no key.
3. **`src/train.py`.** Features: hour, day of week, weekends and bank holidays; **the previous hour's** temperature,
   rain and wind; distance from the centre and a k-means **area cluster** of station locations; and the station's
   recent history (last hour, same hour yesterday, same hour last week, 24-hour mean). Three targets are modelled:
   departures, arrivals and net flow. Every run, and the final models, are logged to **MLflow**.
4. **`app/dashboard.py` (Streamlit + pydeck).** Pick a day and hour to see every station sized by forecast demand
   and coloured by forecast net flow, the stations forecast to lose the most bikes, forecast against actual for the
   whole day, and a what-if toggle that re-runs the models as if the previous hour had been rainy.

**Design choices**
- **No look-ahead.** Lags only look backwards, models see the *previous* hour's weather rather than the hour being
  forecast, and test windows always come after the training data. Tests check the lags and the weather columns.
- **Two baselines.** "Same hour last week" is easy to beat; "this station's usual for this hour and day type" is the
  honest benchmark.
- **Poisson loss** for the count targets, so predictions stay non-negative and errors are judged relative to how busy
  a station is. Net flow, which can be negative, uses squared error.
- **Metrics chosen for the question:** WAPE for relative error, city-wide WAPE for planning-level accuracy,
  peak-hour MAE because rush hours matter most, and precision@10 for the outflow ranking.

**Limitations and next steps**
- Weather comes from one point in central London and uses observations rather than forecasts.
- The outflow ranking does not beat the typical pattern. Live dock availability from TfL would turn it into a true
  "empty within the hour" alert, which is the most valuable next step.
- Forecasting at area level or for the next few hours would reduce the per-station noise.
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
mlflow ui --backend-store-uri sqlite:///mlflow.db     # compare runs and download models at http://localhost:5000
streamlit run app/dashboard.py

# Tests
pytest -q
```

The Docker image serves the dashboard from the trained artifacts, so build it after `src/train.py` has run:

```bash
docker build -t london-bike-demand .
docker run -p 8501:8501 london-bike-demand
```

## Data

- Journeys: [TfL cycle hire usage data](https://cycling.data.tfl.gov.uk/), Powered by TfL Open Data
- Stations: [TfL Unified API, BikePoint](https://api.tfl.gov.uk/BikePoint)
- Weather: [Open-Meteo historical weather API](https://open-meteo.com/)
