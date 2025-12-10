"""
models/weather-prediction/main.py
Entry point for Weather Prediction Pipeline
Can run data collection, training, or prediction independently
"""
import os
import sys
import logging  # Import standard library BEFORE path manipulation
import argparse
from pathlib import Path
from datetime import datetime

# CRITICAL: Configure logging BEFORE adding src/ to path
# (src/logging/ directory would otherwise shadow the standard module)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("weather_prediction")

# Setup paths - AFTER logging is configured
PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT / "src"))


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
                epochs=epochs,
                use_mlflow=False  # Disabled due to Windows Unicode encoding issues
            )
            results.append(result)
            logger.info(f"[OK] {station_name}: MAE={result['test_mae']:.3f}")
        except Exception as e:
            logger.error(f"[FAIL] {station_name}: {e}")

    logger.info(f"Training complete! Trained {len(results)} models.")
    return results


def check_and_train_missing_models(priority_only: bool = True, epochs: int = 25):
    """
    Check for missing LSTM models and train them automatically.
    
    Args:
        priority_only: If True, only train priority stations (COLOMBO, KANDY, etc.)
                      If False, train all configured stations
        epochs: Number of epochs for training
        
    Returns:
        List of trained station names
    """
    from entity.config_entity import WEATHER_STATIONS

    models_dir = PIPELINE_ROOT / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Priority stations for minimal prediction coverage
    priority_stations = ["COLOMBO", "KANDY", "JAFFNA", "BATTICALOA", "RATNAPURA"]

    stations_to_check = priority_stations if priority_only else list(WEATHER_STATIONS.keys())
    missing_stations = []

    # Check which models are missing
    for station in stations_to_check:
        model_file = models_dir / f"lstm_{station.lower()}.h5"
        if not model_file.exists():
            missing_stations.append(station)

    if not missing_stations:
        logger.info("[AUTO-TRAIN] All required models exist.")
        return []

    logger.info(f"[AUTO-TRAIN] Missing models for: {', '.join(missing_stations)}")
    logger.info("[AUTO-TRAIN] Starting automatic training...")

    # Ensure we have data first
    data_path = PIPELINE_ROOT / "artifacts" / "data"
    existing_data = list(data_path.glob("weather_history_*.csv")) if data_path.exists() else []

    if not existing_data:
        logger.info("[AUTO-TRAIN] No training data found, ingesting...")
        try:
            run_data_ingestion(months=3)
        except Exception as e:
            logger.error(f"[AUTO-TRAIN] Data ingestion failed: {e}")
            logger.info("[AUTO-TRAIN] Cannot train without data. Please run: python main.py --mode ingest")
            return []

    # Train missing models
    trained = []
    for station in missing_stations:
        try:
            logger.info(f"[AUTO-TRAIN] Training {station}...")
            run_training(station=station, epochs=epochs)
            trained.append(station)
        except Exception as e:
            logger.warning(f"[AUTO-TRAIN] Failed to train {station}: {e}")

    logger.info(f"[AUTO-TRAIN] Auto-training complete. Trained {len(trained)} models: {', '.join(trained)}")
    return trained


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
        choices=["ingest", "train", "predict", "full", "auto-train"],
        default="predict",
        help="Pipeline mode to run (auto-train checks and trains missing models)"
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
    parser.add_argument(
        "--skip-auto-train",
        action="store_true",
        help="Skip automatic training of missing models during predict"
    )

    args = parser.parse_args()

    if args.mode == "ingest":
        run_data_ingestion(months=args.months)
    elif args.mode == "train":
        run_training(station=args.station, epochs=args.epochs)
    elif args.mode == "auto-train":
        # Explicitly auto-train missing models
        check_and_train_missing_models(priority_only=True, epochs=25)
    elif args.mode == "predict":
        # Auto-train missing models before prediction (unless skipped)
        if not args.skip_auto_train:
            check_and_train_missing_models(priority_only=True, epochs=25)
        run_prediction()
    elif args.mode == "full":
        run_full_pipeline()

