"""
models/currency-volatility-prediction/main.py
Entry point for Currency Prediction Pipeline
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
logger = logging.getLogger("currency_prediction")

# Setup paths - AFTER logging is configured
PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT / "src"))


def run_data_ingestion(period: str = "2y"):
    """Run data ingestion from yfinance."""
    from components.data_ingestion import CurrencyDataIngestion
    from entity.config_entity import DataIngestionConfig
    
    logger.info(f"Starting data ingestion ({period})...")
    
    config = DataIngestionConfig(history_period=period)
    ingestion = CurrencyDataIngestion(config)
    
    data_path = ingestion.ingest_all()
    
    df = ingestion.load_existing(data_path)
    
    logger.info("Data Ingestion Complete!")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Features: {len(df.columns)}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Latest rate: {df['close'].iloc[-1]:.2f} LKR/USD")
    
    return data_path


def run_training(epochs: int = 100):
    """Run GRU model training."""
    from components.data_ingestion import CurrencyDataIngestion
    from components.model_trainer import CurrencyGRUTrainer
    from entity.config_entity import ModelTrainerConfig
    
    logger.info("Starting model training...")
    
    # Load data
    ingestion = CurrencyDataIngestion()
    df = ingestion.load_existing()
    
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
    
    # Train
    config = ModelTrainerConfig(epochs=epochs)
    trainer = CurrencyGRUTrainer(config)
    
    results = trainer.train(df=df, use_mlflow=False)  # Disabled due to Windows Unicode encoding issues
    
    logger.info(f"\nTraining Results:")
    logger.info(f"  MAE: {results['test_mae']:.4f} LKR")
    logger.info(f"  RMSE: {results['rmse']:.4f} LKR")
    logger.info(f"  Direction Accuracy: {results['direction_accuracy']*100:.1f}%")
    logger.info(f"  Epochs: {results['epochs_trained']}")
    logger.info(f"  Model saved: {results['model_path']}")
    
    return results


def run_prediction():
    """Run prediction for next day."""
    from components.data_ingestion import CurrencyDataIngestion
    from components.predictor import CurrencyPredictor
    
    logger.info("Generating prediction...")
    
    predictor = CurrencyPredictor()
    
    try:
        ingestion = CurrencyDataIngestion()
        df = ingestion.load_existing()
        prediction = predictor.predict(df)
    except FileNotFoundError:
        logger.warning("Model not trained, using fallback")
        prediction = predictor.generate_fallback_prediction()
    except Exception as e:
        logger.error(f"Error: {e}")
        prediction = predictor.generate_fallback_prediction()
    
    output_path = predictor.save_prediction(prediction)
    
    # Display
    logger.info(f"\n{'='*50}")
    logger.info(f"USD/LKR PREDICTION FOR {prediction['prediction_date']}")
    logger.info(f"{'='*50}")
    logger.info(f"Current Rate:   {prediction['current_rate']:.2f} LKR/USD")
    logger.info(f"Predicted Rate: {prediction['predicted_rate']:.2f} LKR/USD")
    logger.info(f"Expected Change: {prediction['expected_change_pct']:+.3f}%")
    logger.info(f"Direction: {prediction['direction_emoji']} LKR {prediction['direction']}")
    logger.info(f"Volatility: {prediction['volatility_class']}")
    
    if prediction.get('weekly_trend'):
        logger.info(f"Weekly Trend: {prediction['weekly_trend']:+.2f}%")
    if prediction.get('monthly_trend'):
        logger.info(f"Monthly Trend: {prediction['monthly_trend']:+.2f}%")
    
    logger.info(f"{'='*50}")
    logger.info(f"Saved to: {output_path}")
    
    return prediction


def run_full_pipeline():
    """Run the complete pipeline: ingest → train → predict."""
    logger.info("=" * 60)
    logger.info("CURRENCY PREDICTION PIPELINE - FULL RUN")
    logger.info("=" * 60)
    
    # Step 1: Data Ingestion
    try:
        run_data_ingestion(period="2y")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return None
    
    # Step 2: Training
    try:
        run_training(epochs=100)
    except Exception as e:
        logger.error(f"Training failed: {e}")
    
    # Step 3: Prediction
    prediction = run_prediction()
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Currency Prediction Pipeline")
    parser.add_argument(
        "--mode",
        choices=["ingest", "train", "predict", "full"],
        default="predict",
        help="Pipeline mode to run"
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="Data period (1y, 2y, 5y)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs"
    )
    
    args = parser.parse_args()
    
    if args.mode == "ingest":
        run_data_ingestion(period=args.period)
    elif args.mode == "train":
        run_training(epochs=args.epochs)
    elif args.mode == "predict":
        run_prediction()
    elif args.mode == "full":
        run_full_pipeline()
