"""
models/stock-price-prediction/src/entity/config_entity.py
Configuration entities for Sri Lanka Stock Price Prediction Pipeline
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import os


# Top 10 Sri Lankan Stocks by market cap (CSE - Colombo Stock Exchange)
# Using Yahoo Finance symbols
SRI_LANKA_STOCKS = {
    "JKH": {
        "yahoo_symbol": "JKH.N0000.CM",  # John Keells Holdings
        "name": "John Keells Holdings PLC",
        "sector": "Diversified Holdings"
    },
    "COMB": {
        "yahoo_symbol": "COMB.N0000.CM",  # Commercial Bank
        "name": "Commercial Bank of Ceylon PLC",
        "sector": "Banking"
    },
    "SAMP": {
        "yahoo_symbol": "SAMP.N0000.CM",  # Sampath Bank
        "name": "Sampath Bank PLC",
        "sector": "Banking"
    },
    "HNB": {
        "yahoo_symbol": "HNB.N0000.CM",  # Hatton National Bank
        "name": "Hatton National Bank PLC",
        "sector": "Banking"
    },
    "DIAL": {
        "yahoo_symbol": "DIAL.N0000.CM",  # Dialog Axiata
        "name": "Dialog Axiata PLC",
        "sector": "Telecommunications"
    },
    "CTC": {
        "yahoo_symbol": "CTC.N0000.CM",  # Ceylon Tobacco
        "name": "Ceylon Tobacco Company PLC",
        "sector": "Consumer Goods"
    },
    "NEST": {
        "yahoo_symbol": "NEST.N0000.CM",  # Nestle Lanka
        "name": "Nestle Lanka PLC",
        "sector": "Consumer Goods"
    },
    "CARG": {
        "yahoo_symbol": "CARG.N0000.CM",  # Cargills Ceylon
        "name": "Cargills Ceylon PLC",
        "sector": "Retail"
    },
    "HNBA": {
        "yahoo_symbol": "HNBA.N0000.CM",  # HNB Assurance
        "name": "HNB Assurance PLC",
        "sector": "Insurance"
    },
    "CARS": {
        "yahoo_symbol": "CARS.N0000.CM",  # Carson Cumberbatch
        "name": "Carson Cumberbatch PLC",
        "sector": "Diversified Holdings"
    }
}

# Model architectures to try with Optuna
MODEL_ARCHITECTURES = ["LSTM", "GRU", "BiLSTM", "BiGRU"]


@dataclass
class DataIngestionConfig:
    """Configuration for stock data ingestion"""
    
    # Stocks to train
    stocks: Dict = field(default_factory=lambda: SRI_LANKA_STOCKS)
    
    # Historical data period
    history_period: str = "2y"  # 2 years
    history_interval: str = "1d"  # Daily
    
    # Output paths
    raw_data_dir: str = field(default_factory=lambda: str(
        Path(__file__).parent.parent.parent / "artifacts" / "data"
    ))
    
    # Feature engineering
    include_technical_indicators: bool = True


@dataclass
class OptunaConfig:
    """Configuration for Optuna hyperparameter optimization"""
    
    n_trials: int = 30  # Number of trials per stock
    timeout: int = 1800  # 30 minutes max per stock
    
    # Search space
    sequence_length_range: tuple = (10, 60)  # 10-60 days lookback
    units_range: tuple = (16, 128)  # Hidden units
    layers_range: tuple = (1, 3)  # Number of layers
    dropout_range: tuple = (0.1, 0.5)
    learning_rate_range: tuple = (1e-4, 1e-2)
    batch_size_options: List[int] = field(default_factory=lambda: [8, 16, 32])
    
    # Models to test
    architectures: List[str] = field(default_factory=lambda: MODEL_ARCHITECTURES)


@dataclass
class ModelTrainerConfig:
    """Configuration for model training"""
    
    # Training parameters
    epochs: int = 100
    early_stopping_patience: int = 15
    validation_split: float = 0.2
    
    # MLflow config
    mlflow_tracking_uri: str = field(default_factory=lambda: os.getenv(
        "MLFLOW_TRACKING_URI", "https://dagshub.com/sliitguy/modelx.mlflow"
    ))
    experiment_name: str = "stock_prediction_cse"
    
    # Optuna config
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    
    # Output
    models_dir: str = field(default_factory=lambda: str(
        Path(__file__).parent.parent.parent / "artifacts" / "models"
    ))


@dataclass
class PredictionConfig:
    """Configuration for stock predictions"""
    
    # Output
    predictions_dir: str = field(default_factory=lambda: str(
        Path(__file__).parent.parent.parent / "output" / "predictions"
    ))
    
    # Best model selection
    model_selection_metric: str = "val_mae"  # or val_rmse, val_mape
    
    # Prediction targets
    predict_next_day: bool = True


@dataclass
class PipelineConfig:
    """Master configuration for stock prediction pipeline"""
    data_ingestion: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    model_trainer: ModelTrainerConfig = field(default_factory=ModelTrainerConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
