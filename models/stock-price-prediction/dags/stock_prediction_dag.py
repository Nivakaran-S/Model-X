"""
models/stock-price-prediction/dags/stock_prediction_dag.py
Airflow DAG for daily stock price prediction for Sri Lankan stocks
Runs at 4:15 AM IST daily (after market close, before market open)
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
    # Path: dags/ -> stock-price-prediction/ -> models/ -> root/
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
    """Task: Ingest stock data from yfinance."""
    from components.data_ingestion import StockDataIngestion
    from entity.config_entity import DataIngestionConfig
    
    print("[STOCK DAG] Starting data ingestion...")
    
    config = DataIngestionConfig(history_period="2y")
    ingestion = StockDataIngestion(config)
    
    results = ingestion.ingest_all_stocks()
    
    print(f"[STOCK DAG] ✓ Ingested {len(results)} stocks")
    
    context["ti"].xcom_push(key="data_dir", value=ingestion.config.raw_data_dir)
    return results


def train_models(**context):
    """Task: Train models for all stocks with Optuna."""
    from components.model_trainer import StockModelTrainer
    from entity.config_entity import ModelTrainerConfig, OptunaConfig
    
    print("[STOCK DAG] Starting model training with Optuna...")
    
    data_dir = context["ti"].xcom_pull(task_ids="ingest_data", key="data_dir")
    if not data_dir:
        data_dir = str(Path(__file__).parent.parent / "artifacts" / "data")
    
    # Reduced trials for daily runs (full optimization on weekends)
    is_weekend = datetime.now().weekday() >= 5
    n_trials = 30 if is_weekend else 10  # More trials on weekends
    
    optuna_config = OptunaConfig(
        n_trials=n_trials,
        timeout=600  # 10 min per stock
    )
    
    config = ModelTrainerConfig(optuna=optuna_config)
    trainer = StockModelTrainer(config)
    
    results = trainer.train_all_stocks(
        data_dir=data_dir,
        use_optuna=True,
        use_mlflow=True
    )
    
    print(f"[STOCK DAG] ✓ Trained {len(results)} stock models")
    
    for stock, result in results.items():
        arch = result.get("architecture", "?")
        mae = result.get("metrics", {}).get("test_mae", 0)
        print(f"  {stock}: {arch} (MAE: {mae:.4f})")
    
    context["ti"].xcom_push(key="training_results", value=list(results.keys()))
    return results


def generate_predictions(**context):
    """Task: Generate predictions for all stocks."""
    from components.predictor import StockPredictor
    from entity.config_entity import PredictionConfig
    
    print("[STOCK DAG] Generating predictions...")
    
    data_dir = context["ti"].xcom_pull(task_ids="ingest_data", key="data_dir")
    if not data_dir:
        data_dir = str(Path(__file__).parent.parent / "artifacts" / "data")
    
    predictor = StockPredictor()
    predictions = predictor.predict_all_stocks(data_dir)
    
    output_path = predictor.save_predictions(predictions)
    
    print(f"[STOCK DAG] ✓ Generated predictions for {len(predictions)} stocks")
    
    for stock, pred in predictions.items():
        emoji = pred.get("trend_emoji", "?")
        change = pred.get("expected_change_pct", 0)
        print(f"  {stock}: {emoji} {change:+.2f}%")
    
    context["ti"].xcom_push(key="predictions", value=predictions)
    context["ti"].xcom_push(key="output_path", value=output_path)
    return predictions


def publish_predictions(**context):
    """Task: Log prediction summary."""
    predictions = context["ti"].xcom_pull(task_ids="generate_predictions", key="predictions")
    
    if predictions:
        bullish = sum(1 for p in predictions.values() if "bullish" in p.get("trend", ""))
        bearish = sum(1 for p in predictions.values() if "bearish" in p.get("trend", ""))
        
        print("\n" + "="*60)
        print(f"CSE STOCK PREDICTIONS SUMMARY")
        print("="*60)
        print(f"Total Stocks: {len(predictions)}")
        print(f"Bullish: {bullish} | Bearish: {bearish} | Neutral: {len(predictions)-bullish-bearish}")
        print("-"*60)
        
        # Top movers
        sorted_preds = sorted(
            predictions.items(), 
            key=lambda x: abs(x[1].get("expected_change_pct", 0)),
            reverse=True
        )
        
        print("Top Movers:")
        for stock, pred in sorted_preds[:5]:
            emoji = pred.get("trend_emoji", "?")
            change = pred.get("expected_change_pct", 0)
            print(f"  {stock}: {emoji} {change:+.2f}%")
        
        print("="*60 + "\n")
    
    return True


# Define DAG
with DAG(
    dag_id="stock_prediction_daily",
    default_args=default_args,
    description="Daily CSE stock price prediction using LSTM/GRU with Optuna optimization",
    schedule_interval="15 4 * * 1-5",  # 4:15 AM Mon-Fri (CSE trading days only)
    start_date=datetime(2024, 12, 1),
    catchup=False,
    tags=["stock", "ml", "prediction", "optuna", "cse", "sri-lanka"],
) as dag:
    
    # Task 1: Ingest Data
    task_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
        provide_context=True,
    )
    
    # Task 2: Train Models
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
    
    # Dependencies
    task_ingest >> task_train >> task_predict >> task_publish


if __name__ == "__main__":
    print("Stock Prediction DAG loaded successfully")
    print(f"Schedule: Mon-Fri at 4:15 AM")
    print(f"Tasks: {[t.task_id for t in dag.tasks]}")
