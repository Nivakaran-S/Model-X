"""
models/stock-price-prediction/main.py
Entry point for Stock Price Prediction Pipeline
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
logger = logging.getLogger("stock_prediction")

# Setup paths - AFTER logging is configured
PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT / "src"))


def run_data_ingestion():
    """Run data ingestion for all stocks."""
    from components.data_ingestion import StockDataIngestion
    from entity.config_entity import DataIngestionConfig
    
    logger.info("Starting stock data ingestion...")
    
    config = DataIngestionConfig(history_period="2y")
    ingestion = StockDataIngestion(config)
    
    results = ingestion.ingest_all_stocks()
    
    logger.info(f"\nData Ingestion Complete!")
    logger.info(f"Stocks ingested: {len(results)}")
    
    for stock, path in results.items():
        df = ingestion.load_stock_data(stock)
        if df is not None:
            logger.info(f"  {stock}: {len(df)} records, latest: {df['close'].iloc[-1]:.2f}")
    
    return results


def run_training(use_optuna: bool = True, stock: str = None):
    """Run model training."""
    from components.model_trainer import StockModelTrainer
    from components.data_ingestion import StockDataIngestion
    from entity.config_entity import ModelTrainerConfig
    
    logger.info("Starting model training...")
    
    config = ModelTrainerConfig()
    trainer = StockModelTrainer(config)
    
    ingestion = StockDataIngestion()
    data_dir = ingestion.config.raw_data_dir
    
    if stock:
        # Train single stock
        df = ingestion.load_stock_data(stock)
        if df is None:
            logger.error(f"No data found for {stock}")
            return None
        
        result = trainer.train_stock(
            df=df,
            stock_code=stock,
            use_optuna=use_optuna,
            use_mlflow=True
        )
        return {stock: result}
    else:
        # Train all stocks
        results = trainer.train_all_stocks(
            data_dir=data_dir,
            use_optuna=use_optuna,
            use_mlflow=True
        )
        return results


def run_prediction():
    """Run prediction for all stocks."""
    from components.predictor import StockPredictor
    from components.data_ingestion import StockDataIngestion
    
    logger.info("Generating predictions...")
    
    predictor = StockPredictor()
    ingestion = StockDataIngestion()
    data_dir = ingestion.config.raw_data_dir
    
    predictions = predictor.predict_all_stocks(data_dir)
    
    if not predictions:
        logger.warning("No predictions generated")
        return None
    
    output_path = predictor.save_predictions(predictions)
    
    # Display summary
    logger.info(f"\n{'='*60}")
    logger.info(f"STOCK PREDICTIONS FOR {list(predictions.values())[0]['prediction_date']}")
    logger.info(f"{'='*60}")
    
    for stock_code, pred in sorted(predictions.items()):
        emoji = pred.get("trend_emoji", "?")
        change = pred.get("expected_change_pct", 0)
        current = pred.get("current_price", 0)
        predicted = pred.get("predicted_price", 0)
        arch = pred.get("model_architecture", "?")
        
        logger.info(
            f"  {stock_code:6} {emoji} {change:+6.2f}% | "
            f"Current: {current:8.2f} → Predicted: {predicted:8.2f} | {arch}"
        )
    
    bullish = sum(1 for p in predictions.values() if "bullish" in p.get("trend", ""))
    bearish = sum(1 for p in predictions.values() if "bearish" in p.get("trend", ""))
    
    logger.info(f"{'='*60}")
    logger.info(f"Summary: {bullish} bullish, {bearish} bearish, {len(predictions)-bullish-bearish} neutral")
    logger.info(f"Saved to: {output_path}")
    
    return predictions


def run_full_pipeline(use_optuna: bool = True):
    """Run the complete pipeline: ingest → train → predict."""
    logger.info("=" * 60)
    logger.info("STOCK PREDICTION PIPELINE - FULL RUN")
    logger.info("=" * 60)
    
    # Step 1: Data Ingestion
    try:
        run_data_ingestion()
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return None
    
    # Step 2: Training
    try:
        run_training(use_optuna=use_optuna)
    except Exception as e:
        logger.error(f"Training failed: {e}")
    
    # Step 3: Prediction
    predictions = run_prediction()
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Price Prediction Pipeline")
    parser.add_argument(
        "--mode",
        choices=["ingest", "train", "predict", "full"],
        default="predict",
        help="Pipeline mode to run"
    )
    parser.add_argument(
        "--stock",
        type=str,
        default=None,
        help="Specific stock to train (e.g., JKH, COMB)"
    )
    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Disable Optuna optimization"
    )
    
    args = parser.parse_args()
    use_optuna = not args.no_optuna
    
    if args.mode == "ingest":
        run_data_ingestion()
    elif args.mode == "train":
        run_training(use_optuna=use_optuna, stock=args.stock)
    elif args.mode == "predict":
        run_prediction()
    elif args.mode == "full":
        run_full_pipeline(use_optuna=use_optuna)
