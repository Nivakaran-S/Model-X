"""
models/weather-prediction/main.py
Entry point for Weather Prediction Pipeline
Can run data collection, training, or prediction independently
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Setup paths
PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("weather_prediction")


def run_data_ingestion(months: int = 12):
    """Run data ingestion for all stations."""
    from components.data_ingestion import DataIngestion
    from entity.config_entity import DataIngestionConfig
    
    logger.info(f"Starting data ingestion ({months} months)...")
    
    config = DataIngestionConfig(months_to_fetch=months)
    ingestion = DataIngestion(config)
    
    data_path = ingestion.ingest_all()
    
    df = ingestion.load_existing(data_path)
    stats = ingestion.get_data_stats(df)
    
    logger.info("Data Ingestion Complete!")
    logger.info(f"Total records: {stats['total_records']}")
    logger.info(f"Stations: {stats['stations']}")
    logger.info(f"Date range: {stats['date_range']}")
    
    return data_path


def run_training(station: str = None, epochs: int = 100):
    """Run model training."""
    from components.data_ingestion import DataIngestion
    from components.model_trainer import WeatherLSTMTrainer
    from entity.config_entity import WEATHER_STATIONS
    
    logger.info("Starting model training...")
    
    ingestion = DataIngestion()
    df = ingestion.load_existing()
    
    trainer = WeatherLSTMTrainer(
        sequence_length=30,
        lstm_units=[64, 32]
    )
    
    stations_to_train = [station] if station else list(WEATHER_STATIONS.keys())
    results = []
    
    for station_name in stations_to_train:
        try:
            logger.info(f"Training {station_name}...")
            result = trainer.train(
                df=df,
                station_name=station_name,
                epochs=epochs
            )
            results.append(result)
            logger.info(f"✓ {station_name}: MAE={result['test_mae']:.3f}")
        except Exception as e:
            logger.error(f"✗ {station_name}: {e}")
    
    logger.info(f"Training complete! Trained {len(results)} models.")
    return results


def run_prediction():
    """Run prediction for all districts."""
    from components.predictor import WeatherPredictor
    
    logger.info("Generating predictions...")
    
    predictor = WeatherPredictor()
    
    # Try to get RiverNet data
    rivernet_data = None
    try:
        sys.path.insert(0, str(PIPELINE_ROOT.parent.parent / "src"))
        from utils.utils import tool_rivernet_status
        rivernet_data = tool_rivernet_status()
        logger.info(f"RiverNet data available: {len(rivernet_data.get('rivers', []))} rivers")
    except Exception as e:
        logger.warning(f"RiverNet data unavailable: {e}")
    
    predictions = predictor.predict_all_districts(rivernet_data=rivernet_data)
    output_path = predictor.save_predictions(predictions)
    
    # Summary
    districts = predictions.get("districts", {})
    severity_counts = {"normal": 0, "advisory": 0, "warning": 0, "critical": 0}
    
    for d, p in districts.items():
        sev = p.get("severity", "normal")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    logger.info(f"\n{'='*50}")
    logger.info(f"PREDICTIONS FOR {predictions['prediction_date']}")
    logger.info(f"{'='*50}")
    logger.info(f"Districts: {len(districts)}")
    logger.info(f"Normal: {severity_counts['normal']}")
    logger.info(f"Advisory: {severity_counts['advisory']}")
    logger.info(f"Warning: {severity_counts['warning']}")
    logger.info(f"Critical: {severity_counts['critical']}")
    logger.info(f"Output: {output_path}")
    
    return predictions


def run_full_pipeline():
    """Run the full pipeline: ingest → train → predict."""
    logger.info("=" * 60)
    logger.info("WEATHER PREDICTION PIPELINE - FULL RUN")
    logger.info("=" * 60)
    
    # Step 1: Data Ingestion
    try:
        run_data_ingestion(months=3)
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        logger.info("Attempting to use existing data...")
    
    # Step 2: Training (priority stations only)
    priority_stations = ["COLOMBO", "KANDY", "JAFFNA", "BATTICALOA", "RATNAPURA"]
    for station in priority_stations:
        try:
            run_training(station=station, epochs=50)
        except Exception as e:
            logger.warning(f"Training {station} failed: {e}")
    
    # Step 3: Prediction
    predictions = run_prediction()
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Prediction Pipeline")
    parser.add_argument(
        "--mode",
        choices=["ingest", "train", "predict", "full"],
        default="predict",
        help="Pipeline mode to run"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Months of historical data to ingest"
    )
    parser.add_argument(
        "--station",
        type=str,
        default=None,
        help="Specific station to train (e.g., COLOMBO)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs"
    )
    
    args = parser.parse_args()
    
    if args.mode == "ingest":
        run_data_ingestion(months=args.months)
    elif args.mode == "train":
        run_training(station=args.station, epochs=args.epochs)
    elif args.mode == "predict":
        run_prediction()
    elif args.mode == "full":
        run_full_pipeline()
