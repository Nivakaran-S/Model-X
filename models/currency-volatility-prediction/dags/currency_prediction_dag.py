"""
models/currency-volatility-prediction/dags/currency_prediction_dag.py
Airflow DAG for daily USD/LKR currency prediction
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

# Load environment variables from root .env
try:
    from dotenv import load_dotenv
    # Path: dags/ -> currency-volatility-prediction/ -> models/ -> root/
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
    "retry_delay": timedelta(minutes=5),
}


def ingest_data(**context):
    """Task: Ingest currency data from yfinance."""
    from components.data_ingestion import CurrencyDataIngestion
    from entity.config_entity import DataIngestionConfig
    
    print("[CURRENCY DAG] Starting data ingestion...")
    
    config = DataIngestionConfig(history_period="2y")
    ingestion = CurrencyDataIngestion(config)
    
    # Check if we have recent data
    try:
        df = ingestion.load_existing()
        latest_date = df["date"].max()
        if isinstance(latest_date, str):
            latest_date = datetime.strptime(latest_date, "%Y-%m-%d")
        
        days_old = (datetime.now() - latest_date).days
        
        if days_old < 1:
            print(f"[CURRENCY DAG] Data is current ({days_old} days old)")
            context["ti"].xcom_push(key="data_path", value=str(ingestion.config.raw_data_dir))
            return str(ingestion.config.raw_data_dir)
    except FileNotFoundError:
        pass
    
    # Full ingestion
    data_path = ingestion.ingest_all()
    context["ti"].xcom_push(key="data_path", value=data_path)
    
    print(f"[CURRENCY DAG] ✓ Data saved to {data_path}")
    return data_path


def train_model(**context):
    """Task: Train GRU model."""
    from components.model_trainer import CurrencyGRUTrainer
    from components.data_ingestion import CurrencyDataIngestion
    from entity.config_entity import ModelTrainerConfig
    
    print("[CURRENCY DAG] Starting model training...")
    
    # Load data
    ingestion = CurrencyDataIngestion()
    df = ingestion.load_existing()
    
    print(f"[CURRENCY DAG] Loaded {len(df)} records")
    
    # Train
    config = ModelTrainerConfig(
        epochs=100,
        batch_size=16,
        early_stopping_patience=15
    )
    trainer = CurrencyGRUTrainer(config)
    
    results = trainer.train(df=df, use_mlflow=True)
    
    print(f"[CURRENCY DAG] ✓ Training complete!")
    print(f"  MAE: {results['test_mae']:.4f} LKR")
    print(f"  Direction Accuracy: {results['direction_accuracy']*100:.1f}%")
    
    context["ti"].xcom_push(key="model_path", value=results["model_path"])
    return results


def generate_prediction(**context):
    """Task: Generate next-day prediction."""
    from components.predictor import CurrencyPredictor
    from components.data_ingestion import CurrencyDataIngestion
    
    print("[CURRENCY DAG] Generating prediction...")
    
    predictor = CurrencyPredictor()
    
    try:
        # Load latest data
        ingestion = CurrencyDataIngestion()
        df = ingestion.load_existing()
        
        # Generate prediction
        prediction = predictor.predict(df)
        
    except FileNotFoundError:
        # Model not trained, use fallback
        print("[CURRENCY DAG] Model not trained, using fallback")
        prediction = predictor.generate_fallback_prediction()
    except Exception as e:
        print(f"[CURRENCY DAG] Error predicting: {e}")
        prediction = predictor.generate_fallback_prediction()
    
    # Save prediction
    output_path = predictor.save_prediction(prediction)
    
    print(f"[CURRENCY DAG] ✓ Prediction generated!")
    print(f"  Current: {prediction['current_rate']} LKR/USD")
    print(f"  Predicted: {prediction['predicted_rate']} LKR/USD")
    print(f"  Change: {prediction['expected_change_pct']:+.2f}%")
    print(f"  Direction: {prediction['direction']}")
    
    context["ti"].xcom_push(key="prediction_path", value=output_path)
    return prediction


def publish_prediction(**context):
    """Task: Log prediction summary."""
    prediction = context["ti"].xcom_pull(task_ids="generate_prediction")
    
    if prediction:
        print("\n" + "="*50)
        print(f"USD/LKR CURRENCY PREDICTION")
        print("="*50)
        print(f"Prediction for: {prediction.get('prediction_date')}")
        print(f"Current Rate: {prediction.get('current_rate')} LKR/USD")
        print(f"Predicted Rate: {prediction.get('predicted_rate')} LKR/USD")
        print(f"Expected Change: {prediction.get('expected_change_pct'):+.3f}%")
        print(f"Direction: {prediction.get('direction_emoji')} {prediction.get('direction')}")
        print(f"Volatility: {prediction.get('volatility_class')}")
        if prediction.get('is_fallback'):
            print("⚠️  Using fallback model")
        print("="*50 + "\n")
    
    return True


# Define DAG
with DAG(
    dag_id="currency_prediction_daily",
    default_args=default_args,
    description="Daily USD/LKR currency prediction using GRU neural network",
    schedule_interval="0 4 * * *",  # 4:00 AM daily (IST is UTC+5:30)
    start_date=datetime(2024, 12, 1),
    catchup=False,
    tags=["currency", "ml", "prediction", "gru", "forex"],
) as dag:
    
    # Task 1: Ingest Data
    task_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
        provide_context=True,
    )
    
    # Task 2: Train Model
    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        provide_context=True,
    )
    
    # Task 3: Generate Prediction
    task_predict = PythonOperator(
        task_id="generate_prediction",
        python_callable=generate_prediction,
        provide_context=True,
    )
    
    # Task 4: Publish Prediction
    task_publish = PythonOperator(
        task_id="publish_prediction",
        python_callable=publish_prediction,
        provide_context=True,
    )
    
    # Dependencies
    task_ingest >> task_train >> task_predict >> task_publish


if __name__ == "__main__":
    print("Currency Prediction DAG loaded successfully")
    print(f"Schedule: Daily at 4:00 AM")
    print(f"Tasks: {[t.task_id for t in dag.tasks]}")
