"""
Stock Price Prediction Training Script
Convenience wrapper for: python models/stock-price-prediction/main.py --mode train

Usage:
    python models/stock-price-prediction/src/pipeline/train.py [--stock JKH] [--no-optuna] [--full]
"""
import sys
import argparse
import logging  # CRITICAL: Import BEFORE path manipulation
from pathlib import Path

# Configure logging BEFORE adding src/ to path
# (src/logging/ directory would otherwise shadow the standard module)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add parent directories to path - AFTER logging is configured
PIPELINE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Price Prediction Training")
    parser.add_argument("--stock", type=str, default=None, help="Specific stock to train (e.g., JKH, COMB)")
    parser.add_argument("--no-optuna", action="store_true", help="Disable Optuna hyperparameter optimization")
    parser.add_argument("--full", action="store_true", help="Run full pipeline (ingest + train + predict)")
    
    args = parser.parse_args()
    use_optuna = not args.no_optuna
    
    # Import from main.py (after path setup)
    from main import run_training, run_full_pipeline, run_data_ingestion
    
    print("=" * 60)
    print("STOCK PRICE (CSE) PREDICTION - TRAINING PIPELINE")
    print("=" * 60)
    
    if args.full:
        run_full_pipeline(use_optuna=use_optuna)
    else:
        # Run data ingestion first if no data exists
        try:
            from components.data_ingestion import StockDataIngestion
            ingestion = StockDataIngestion()
            stocks = list(ingestion.config.stocks.keys())
            df = ingestion.load_stock_data(stocks[0])
            if df is not None:
                print(f"✓ Found existing data for {len(stocks)} stocks")
            else:
                raise FileNotFoundError()
        except (FileNotFoundError, Exception):
            print("No existing data, running ingestion first...")
            run_data_ingestion()
        
        # Run training
        run_training(use_optuna=use_optuna, stock=args.stock)
    
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
