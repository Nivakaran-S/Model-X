"""
models/weather-prediction/dags/weather_prediction_dag.py
Airflow DAG for daily weather prediction pipeline
Runs at 4:00 AM IST daily
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add paths for imports
PIPELINE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Load environment variables from root .env
try:
    from dotenv import load_dotenv
    # Path: dags/ -> weather-prediction/ -> models/ -> root/
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[MLflow] ✓ Loaded env from {env_path}")
except ImportError:
    pass


# Default arguments
default_args = {
    "owner": "modelx",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def ingest_data(**context):
    """Task: Ingest latest weather data from Tutiempo."""
    from components.data_ingestion import DataIngestion
    from entity.config_entity import DataIngestionConfig
    
    config = DataIngestionConfig(months_to_fetch=3)  # Last 3 months for daily
    ingestion = DataIngestion(config)
    
    # Try to load existing data first, only re-scrape weekly
    try:
        df = ingestion.load_existing()
        latest_date = df["date"].max()
        days_old = (datetime.now() - latest_date).days
        
        if days_old < 7:
            print(f"Using existing data ({days_old} days old)")
            return str(Path(config.raw_data_dir) / f"weather_history_*.csv")
    except FileNotFoundError:
        pass
    
    # Full ingestion
    data_path = ingestion.ingest_all()
    context["ti"].xcom_push(key="data_path", value=data_path)
    return data_path


def train_models(**context):
    """Task: Train LSTM models for each station."""
    from components.model_trainer import WeatherLSTMTrainer
    from components.data_ingestion import DataIngestion
    from entity.config_entity import WEATHER_STATIONS
    import pandas as pd
    
    # Load data
    ingestion = DataIngestion()
    df = ingestion.load_existing()
    
    trainer = WeatherLSTMTrainer(
        sequence_length=30,
        lstm_units=[64, 32]
    )
    
    results = []
    priority_stations = ["COLOMBO", "KANDY", "JAFFNA", "BATTICALOA", "RATNAPURA"]
    
    for station in priority_stations:
        if station in WEATHER_STATIONS:
            try:
                result = trainer.train(
                    df=df,
                    station_name=station,
                    epochs=50,  # Reduced for daily runs
                    batch_size=32
                )
                results.append(result)
            except Exception as e:
                print(f"[WARNING] Failed to train {station}: {e}")
    
    return results


def generate_predictions(**context):
    """Task: Generate predictions for all 25 districts."""
    from components.predictor import WeatherPredictor
    
    predictor = WeatherPredictor()
    
    # Get RiverNet data if available
    rivernet_data = None
    try:
        from src.utils.utils import tool_rivernet_status
        rivernet_data = tool_rivernet_status()
    except ImportError:
        print("[INFO] RiverNet data not available, using fallback")
    
    predictions = predictor.predict_all_districts(rivernet_data=rivernet_data)
    output_path = predictor.save_predictions(predictions)
    
    context["ti"].xcom_push(key="predictions_path", value=output_path)
    return output_path


def publish_predictions(**context):
    """Task: Publish predictions to database/API."""
    import json
    
    predictions_path = context["ti"].xcom_pull(
        task_ids="generate_predictions",
        key="predictions_path"
    )
    
    if not predictions_path:
        # Use latest
        from components.predictor import WeatherPredictor
        predictor = WeatherPredictor()
        predictions = predictor.get_latest_predictions()
    else:
        with open(predictions_path) as f:
            predictions = json.load(f)
    
    if predictions:
        # Log summary
        districts = predictions.get("districts", {})
        severity_counts = {}
        for d, p in districts.items():
            sev = p.get("severity", "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        print(f"\n{'='*50}")
        print(f"WEATHER PREDICTIONS FOR {predictions.get('prediction_date')}")
        print(f"{'='*50}")
        print(f"Districts: {len(districts)}")
        print(f"Severity breakdown: {severity_counts}")
        print(f"{'='*50}\n")
    
    return True


# Define DAG
with DAG(
    dag_id="weather_prediction_daily",
    default_args=default_args,
    description="Daily weather prediction pipeline for Sri Lanka (25 districts)",
    schedule_interval="0 4 * * *",  # 4:00 AM daily (IST is UTC+5:30)
    start_date=datetime(2024, 12, 1),
    catchup=False,
    tags=["weather", "ml", "prediction", "lstm"],
) as dag:
    
    # Task 1: Check/Ingest Data
    task_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
        provide_context=True,
    )
    
    # Task 2: Train Models (runs weekly, skips if recent)
    task_train = PythonOperator(
        task_id="train_models",
        python_callable=train_models,
        provide_context=True,
    )
    
    # Task 3: Generate Predictions
    task_predict = PythonOperator(
        task_id="generate_predictions",
        python_callable=generate_predictions,
        provide_context=True,
    )
    
    # Task 4: Publish Predictions
    task_publish = PythonOperator(
        task_id="publish_predictions",
        python_callable=publish_predictions,
        provide_context=True,
    )
    
    # Define dependencies
    task_ingest >> task_train >> task_predict >> task_publish


if __name__ == "__main__":
    # Test run
    print("Weather Prediction DAG loaded successfully")
    print(f"Schedule: Daily at 4:00 AM")
    print(f"Tasks: {[t.task_id for t in dag.tasks]}")
