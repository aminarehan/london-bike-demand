"""Turn raw TfL journey CSVs into hourly departures and arrivals per docking station.

Run from the project folder:  python src/build_hourly_demand.py
"""
from pyspark.sql import SparkSession, functions as F

RAW = "data/raw/*.csv"
OUT = "data/processed/station_hourly"

spark = (SparkSession.builder.master("local[*]")      # all CPU cores on this machine
         .appName("hourly-demand")
         .config("spark.driver.memory", "6g")
         .config("spark.sql.session.timeZone", "Europe/London")
         .config("spark.ui.showConsoleProgress", "false")   # no progress bars in the log
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# Every column is read as text so station IDs keep their leading zeros ("001213").
trips = spark.read.csv(RAW, header=True)
raw_count = trips.count()

trips = (trips
    .select(
        F.col("Start station number").alias("start_station"),
        F.col("End station number").alias("end_station"),
        F.to_timestamp("Start date", "yyyy-MM-dd HH:mm").alias("start_time"),
        F.to_timestamp("End date", "yyyy-MM-dd HH:mm").alias("end_time"),
        (F.col("Total duration (ms)").cast("long") / 60000).alias("duration_min"),
        F.col("Bike model").alias("bike_model"),
    )
    .dropna(subset=["start_station", "end_station", "start_time", "end_time"])
    # Under a minute is usually a re-dock; over a day is a lost or faulty bike.
    .filter((F.col("duration_min") >= 1) & (F.col("duration_min") <= 24 * 60))
    .dropDuplicates())
trips.cache()
clean_count = trips.count()

departures = (trips.groupBy(F.col("start_station").alias("station"),
                            F.date_trunc("hour", "start_time").alias("hour"))
              .agg(F.count("*").alias("departures")))
arrivals = (trips.groupBy(F.col("end_station").alias("station"),
                          F.date_trunc("hour", "end_time").alias("hour"))
            .agg(F.count("*").alias("arrivals")))

# Every station x every hour, so quiet hours are recorded as 0 instead of going missing.
bounds = trips.agg(F.date_trunc("hour", F.min("start_time")).alias("lo"),
                   F.date_trunc("hour", F.max("start_time")).alias("hi")).first()
hours = spark.createDataFrame([(bounds.lo, bounds.hi)], ["lo", "hi"]).select(
    F.explode(F.sequence("lo", "hi", F.expr("interval 1 hour"))).alias("hour"))
stations = departures.select("station").union(arrivals.select("station")).distinct()

hourly = (stations.crossJoin(hours)
    .join(departures, ["station", "hour"], "left")
    .join(arrivals, ["station", "hour"], "left")
    .fillna(0, subset=["departures", "arrivals"])
    .withColumn("net_flow", F.col("arrivals") - F.col("departures")))

hourly.repartition(1).write.mode("overwrite").parquet(OUT)

n_stations, n_hours, n_rows = stations.count(), hours.count(), hourly.count()
print(f"raw journeys: {raw_count:,} | clean: {clean_count:,} ({raw_count - clean_count:,} dropped)")
print(f"stations: {n_stations:,} | hours: {n_hours:,} | station-hours: {n_rows:,}")
print("e-bike share of journeys:",
      round(trips.filter(F.col("bike_model").contains("EBIKE")).count() / clean_count, 3))
hourly.orderBy(F.desc("departures")).show(5, truncate=False)
spark.stop()
