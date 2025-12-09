import os 
import numpy as np 

"""
Defining common constant variable for training pipeline
"""

# Stocks available on Yahoo Finance for training
# NOTE: CSE (Sri Lanka) tickers are NOT available on Yahoo Finance
# Using globally available tickers instead
STOCKS_TO_TRAIN = {
    "AAPL": {
        "yahoo_symbol": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology"
    },
    "GOOGL": {
        "yahoo_symbol": "GOOGL",
        "name": "Alphabet Inc.",
        "sector": "Technology"
    },
    "MSFT": {
        "yahoo_symbol": "MSFT",
        "name": "Microsoft Corporation",
        "sector": "Technology"
    },
    "AMZN": {
        "yahoo_symbol": "AMZN",
        "name": "Amazon.com Inc.",
        "sector": "Consumer Discretionary"
    },
    "META": {
        "yahoo_symbol": "META",
        "name": "Meta Platforms Inc.",
        "sector": "Technology"
    },
    "NVDA": {
        "yahoo_symbol": "NVDA",
        "name": "NVIDIA Corporation",
        "sector": "Technology"
    },
    "TSLA": {
        "yahoo_symbol": "TSLA",
        "name": "Tesla Inc.",
        "sector": "Automotive"
    },
    "JPM": {
        "yahoo_symbol": "JPM",
        "name": "JPMorgan Chase & Co.",
        "sector": "Financial Services"
    },
    "V": {
        "yahoo_symbol": "V",
        "name": "Visa Inc.",
        "sector": "Financial Services"
    },
    "JNJ": {
        "yahoo_symbol": "JNJ",
        "name": "Johnson & Johnson",
        "sector": "Healthcare"
    }
}

# Default stock for single-stock training mode
DEFAULT_STOCK = "AAPL"

# Legacy alias for backward compatibility
SRI_LANKA_STOCKS = STOCKS_TO_TRAIN
AVAILABLE_TEST_STOCKS = STOCKS_TO_TRAIN

TARGET_COLUMN="Close"
PIPELINE_NAME:str = "StockPricePrediction"
ARTIFACT_DIR:str = "Artifacts"

TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME: str = "test.csv"


SCHEMA_FILE_PATH=os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR=os.path.join("saved_models")
MODEL_FILE_NAME= "model.pkl"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME: str ="NetworkData"
DATA_INGESTION_DATABASE_NAME: str ="NIVAAI"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
FILE_NAME: str = "stock_data.csv"  # Single stock file name
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME:str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME:str="preprocessing.pkl"
"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str="transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str="transformed_object"

## Knn imputer class to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict={
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform"
}


DATA_TRANSFORMATION_TRAIN_FILE_PATH:str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH:str = "test.npy"

"""
Model Trainer related content startt with MODEL_TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME:str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR:str="trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str="model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float=0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD: float=0.05

