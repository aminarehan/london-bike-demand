# Serves the dashboard. Build the data and models first (see README), then:
#   docker build -t london-bike-demand .
#   docker run -p 8501:8501 london-bike-demand
FROM python:3.12-slim

WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY reports/ reports/
COPY data/processed/predictions.parquet data/processed/test_features.parquet data/processed/

EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
