"""
models/stock-price-prediction/dags/stock_prediction_dag.py
Apache Airflow DAG for daily stock price prediction training
Runs at 4:00 AM IST (22:30 UTC previous day) daily
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


def run_multi_stock_training(**context):
    """
    Task: Run the multi-stock training pipeline.
    This trains separate models for all stocks in STOCKS_TO_TRAIN.
    """
    import subprocess
    import sys
    
    print("[DAG] Starting multi-stock training pipeline...")
    
    main_py_path = Path(__file__).parent.parent / "main.py"
    
    # Run main.py which trains all stocks
    result = subprocess.run(
        [sys.executable, str(main_py_path)],
        capture_output=True,
        text=True,
        cwd=str(main_py_path.parent)
    )
    
    # Log output
    if result.stdout:
        print("[DAG] STDOUT:")
        print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
    
    if result.stderr:
        print("[DAG] STDERR:")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    
    if result.returncode != 0:
        raise Exception(f"Training failed with exit code {result.returncode}")
    
    print("[DAG] ✓ Multi-stock training completed successfully!")
    
    context["ti"].xcom_push(key="training_completed", value=True)
    return True


def log_training_summary(**context):
    """Task: Log training completion summary."""
    from src.constants.training_pipeline import STOCKS_TO_TRAIN
    
    training_completed = context["ti"].xcom_pull(
        task_ids="run_training", 
        key="training_completed"
    )
    
    print("\n" + "="*60)
    print("DAILY STOCK TRAINING SUMMARY")
    print("="*60)
    print(f"Execution Date: {context['execution_date']}")
    print(f"Stocks configured: {list(STOCKS_TO_TRAIN.keys())}")
    print(f"Training completed: {training_completed}")
    print("="*60 + "\n")
    
    return True


# Define DAG
# Schedule: 4:00 AM IST = 22:30 UTC (previous day)
# Cron: "30 22 * * *" (minute=30, hour=22 UTC daily)
with DAG(
    dag_id="stock_prediction_daily_training",
    default_args=default_args,
    description="Daily stock price prediction model training - runs at 4 AM IST",
    schedule_interval="30 22 * * *",  # 4:00 AM IST = 22:30 UTC
    start_date=datetime(2024, 12, 1),
    catchup=False,
    tags=["stock", "ml", "prediction", "lstm", "daily", "training"],
    max_active_runs=1,  # Only one training run at a time
) as dag:
    
    # Task 1: Run Multi-Stock Training
    task_train = PythonOperator(
        task_id="run_training",
        python_callable=run_multi_stock_training,
        provide_context=True,
        execution_timeout=timedelta(hours=4),  # 4 hour timeout for all stocks
    )
    
    # Task 2: Log Summary
    task_summary = PythonOperator(
        task_id="log_summary",
        python_callable=log_training_summary,
        provide_context=True,
    )
    
    # Dependencies
    task_train >> task_summary


if __name__ == "__main__":
    print("Stock Prediction Daily Training DAG")
    print(f"Schedule: 4:00 AM IST (22:30 UTC) daily")
    print(f"Tasks: {[t.task_id for t in dag.tasks]}")
