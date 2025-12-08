"""
models/anomaly-detection/dags/train_anomaly_model.py
Apache Airflow DAG for scheduled anomaly detection model training
Uses Astronomer (Astro) for deployment
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.python import PythonSensor
import os
import sys
import logging

# Add project to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Load .env from root ModelX directory for MLflow credentials
try:
    from dotenv import load_dotenv
    root_env = os.path.join(PROJECT_ROOT, '..', '..', '.env')
    if os.path.exists(root_env):
        load_dotenv(root_env)
    else:
        load_dotenv()  # Try default locations
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Configuration
BATCH_THRESHOLD = int(os.getenv("BATCH_THRESHOLD", "1000"))
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "")

# Default DAG arguments
default_args = {
    'owner': 'modelx',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def check_new_records(**context) -> bool:
    """
    Sensor function to check if enough new records exist.
    Returns True if batch threshold is met or daily run is due.
    """
    import sqlite3
    from datetime import datetime, timedelta
    
    try:
        # Get last training timestamp from XCom or default to 24h ago
        last_training = context['ti'].xcom_pull(key='last_training_timestamp')
        if not last_training:
            last_training = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        
        # Check SQLite for new records
        if SQLITE_DB_PATH and os.path.exists(SQLITE_DB_PATH):
            conn = sqlite3.connect(SQLITE_DB_PATH)
            cursor = conn.execute(
                'SELECT COUNT(*) FROM seen_hashes WHERE last_seen > ?',
                (last_training,)
            )
            new_records = cursor.fetchone()[0]
            conn.close()
            
            logger.info(f"[AnomalyDAG] New records since {last_training}: {new_records}")
            
            if new_records >= BATCH_THRESHOLD:
                logger.info(f"[AnomalyDAG] Batch threshold met ({new_records} >= {BATCH_THRESHOLD})")
                return True
        
        # Check if 24 hours have passed (daily fallback)
        if last_training:
            last_dt = datetime.fromisoformat(last_training)
            hours_since = (datetime.utcnow() - last_dt).total_seconds() / 3600
            if hours_since >= 24:
                logger.info(f"[AnomalyDAG] Daily run triggered ({hours_since:.1f}h since last run)")
                return True
        
        logger.info(f"[AnomalyDAG] Waiting for more records...")
        return False
        
    except Exception as e:
        logger.error(f"[AnomalyDAG] Error checking records: {e}")
        # Trigger anyway on error
        return True


def run_data_ingestion(**context):
    """Run data ingestion step"""
    from src.components import DataIngestion
    from src.entity import DataIngestionConfig
    
    config = DataIngestionConfig()
    ingestion = DataIngestion(config)
    artifact = ingestion.ingest()
    
    # Store artifact path in XCom
    context['ti'].xcom_push(key='ingestion_artifact', value={
        'raw_data_path': artifact.raw_data_path,
        'total_records': artifact.total_records,
        'is_data_available': artifact.is_data_available
    })
    
    if not artifact.is_data_available:
        raise ValueError("No data available for training")
    
    return artifact.raw_data_path


def run_data_validation(**context):
    """Run data validation step"""
    from src.components import DataValidation
    from src.entity import DataValidationConfig
    
    # Get ingestion output from XCom
    ingestion = context['ti'].xcom_pull(key='ingestion_artifact', task_ids='data_ingestion')
    raw_data_path = ingestion['raw_data_path']
    
    config = DataValidationConfig()
    validation = DataValidation(config)
    artifact = validation.validate(raw_data_path)
    
    # Store artifact in XCom
    context['ti'].xcom_push(key='validation_artifact', value={
        'validated_data_path': artifact.validated_data_path,
        'validation_status': artifact.validation_status,
        'valid_records': artifact.valid_records
    })
    
    return artifact.validated_data_path


def run_data_transformation(**context):
    """Run data transformation step"""
    from src.components import DataTransformation
    from src.entity import DataTransformationConfig
    
    # Get validation output from XCom
    validation = context['ti'].xcom_pull(key='validation_artifact', task_ids='data_validation')
    validated_data_path = validation['validated_data_path']
    
    config = DataTransformationConfig()
    transformation = DataTransformation(config)
    artifact = transformation.transform(validated_data_path)
    
    # Store artifact in XCom
    context['ti'].xcom_push(key='transformation_artifact', value={
        'feature_store_path': artifact.feature_store_path,
        'language_distribution': artifact.language_distribution,
        'total_records': artifact.total_records
    })
    
    return artifact.feature_store_path


def run_model_training(**context):
    """Run model training with Optuna and MLflow"""
    from src.components import ModelTrainer
    from src.entity import ModelTrainerConfig
    from datetime import datetime
    
    # Get transformation output from XCom
    transformation = context['ti'].xcom_pull(key='transformation_artifact', task_ids='data_transformation')
    feature_path = transformation['feature_store_path']
    
    config = ModelTrainerConfig()
    trainer = ModelTrainer(config)
    artifact = trainer.train(feature_path)
    
    # Store training timestamp for next run
    context['ti'].xcom_push(key='last_training_timestamp', value=datetime.utcnow().isoformat())
    
    # Store artifact in XCom
    context['ti'].xcom_push(key='training_artifact', value={
        'best_model_name': artifact.best_model_name,
        'best_model_path': artifact.best_model_path,
        'mlflow_run_id': artifact.mlflow_run_id,
        'n_anomalies': artifact.n_anomalies
    })
    
    return artifact.best_model_path


# Create DAG
with DAG(
    'anomaly_detection_training',
    default_args=default_args,
    description='Train anomaly detection models on feed data',
    schedule_interval=timedelta(hours=4),  # Check every 4 hours
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'anomaly', 'modelx'],
) as dag:
    
    # Start
    start = EmptyOperator(task_id='start')
    
    # Sensor: Check for new records
    check_records = PythonSensor(
        task_id='check_new_records',
        python_callable=check_new_records,
        timeout=3600,
        poke_interval=300,  # Check every 5 minutes
        mode='poke',
    )
    
    # Data Ingestion
    data_ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=run_data_ingestion,
    )
    
    # Data Validation
    data_validation = PythonOperator(
        task_id='data_validation',
        python_callable=run_data_validation,
    )
    
    # Data Transformation
    data_transformation = PythonOperator(
        task_id='data_transformation',
        python_callable=run_data_transformation,
    )
    
    # Model Training
    model_training = PythonOperator(
        task_id='model_training',
        python_callable=run_model_training,
    )
    
    # End
    end = EmptyOperator(task_id='end')
    
    # Pipeline flow
    start >> check_records >> data_ingestion >> data_validation >> data_transformation >> model_training >> end
