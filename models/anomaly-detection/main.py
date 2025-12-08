"""
models/anomaly-detection/main.py
Entry point for the anomaly detection training pipeline
"""
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import run_training_pipeline
from src.entity import PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)

logger = logging.getLogger("main")


def main():
    """Run the anomaly detection training pipeline"""
    logger.info("=" * 60)
    logger.info("ANOMALY DETECTION PIPELINE")
    logger.info("=" * 60)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create configuration
    config = PipelineConfig()
    
    # Run pipeline
    try:
        artifact = run_training_pipeline(config)
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE RESULTS")
        logger.info("=" * 60)
        logger.info(f"Status: {artifact.pipeline_status}")
        logger.info(f"Run ID: {artifact.pipeline_run_id}")
        logger.info(f"Duration: {artifact.pipeline_start_time} to {artifact.pipeline_end_time}")
        
        logger.info("\n--- Data Ingestion ---")
        logger.info(f"Total records: {artifact.data_ingestion.total_records}")
        logger.info(f"From SQLite: {artifact.data_ingestion.records_from_sqlite}")
        logger.info(f"From CSV: {artifact.data_ingestion.records_from_csv}")
        
        logger.info("\n--- Data Validation ---")
        logger.info(f"Valid records: {artifact.data_validation.valid_records}")
        logger.info(f"Validation status: {artifact.data_validation.validation_status}")
        
        logger.info("\n--- Data Transformation ---")
        logger.info(f"Language distribution: {artifact.data_transformation.language_distribution}")
        
        logger.info("\n--- Model Training ---")
        logger.info(f"Best model: {artifact.model_trainer.best_model_name}")
        logger.info(f"Best metrics: {artifact.model_trainer.best_model_metrics}")
        logger.info(f"MLflow run: {artifact.model_trainer.mlflow_run_id}")
        
        if artifact.model_trainer.n_anomalies:
            logger.info(f"Anomalies detected: {artifact.model_trainer.n_anomalies}")
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        return artifact
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
