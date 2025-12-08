"""
models/stock-price-prediction/src/components/model_trainer.py
Multi-Architecture Stock Prediction Model Trainer with Optuna
Trains LSTM, GRU, BiLSTM, BiGRU for each stock and selects best
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
import joblib

# Load environment variables from root .env
try:
    from dotenv import load_dotenv
    root_env = Path(__file__).parent.parent.parent.parent.parent / ".env"
    if root_env.exists():
        load_dotenv(root_env)
        print(f"[MLflow] Loaded environment from {root_env}")
except ImportError:
    pass

# TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, BatchNormalization, 
        Input, Bidirectional
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    # Memory optimization for 8GB RAM
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not available")

# Optuna for hyperparameter tuning
try:
    import optuna
    from optuna.integration import TFKerasPruningCallback
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARNING] Optuna not available")

# MLflow for tracking
try:
    import mlflow
    import mlflow.keras
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from entity.config_entity import ModelTrainerConfig, OptunaConfig

logger = logging.getLogger("stock_prediction.model_trainer")


def setup_mlflow():
    """Configure MLflow with DagsHub credentials from environment."""
    if not MLFLOW_AVAILABLE:
        return False
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    if not tracking_uri:
        logger.info("[MLflow] No MLFLOW_TRACKING_URI set, using local tracking")
        return False
    
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        logger.info(f"[MLflow] ✓ Configured with DagsHub credentials for {username}")
    
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"[MLflow] ✓ Tracking URI: {tracking_uri}")
    return True


class StockModelTrainer:
    """
    Trains multiple model architectures for each stock using Optuna.
    
    Key Features:
    - Tests LSTM, GRU, BiLSTM, BiGRU for each stock
    - Optuna hyperparameter optimization
    - Selects best model per stock
    - MLflow tracking with DagsHub
    """
    
    # Features to use for training
    FEATURE_COLUMNS = [
        "close", "daily_return", "daily_range",
        "sma_5", "sma_10", "sma_20", "ema_5", "ema_10", "ema_20",
        "price_to_sma20", "price_to_sma50",
        "volatility_5", "volatility_20",
        "momentum_5", "momentum_10", "momentum_20",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_position", "bb_width",
        "volume_ratio",
        "day_sin", "day_cos", "month_sin", "month_cos"
    ]
    
    def __init__(self, config: Optional[ModelTrainerConfig] = None):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required")
        
        self.config = config or ModelTrainerConfig()
        os.makedirs(self.config.models_dir, exist_ok=True)
        
        self.best_models = {}  # {stock_code: {model, architecture, params, metrics}}
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any]:
        """
        Prepare stock data for training.
        
        Returns:
            X_train, X_test, y_train, y_test, feature_scaler, target_scaler
        """
        available_features = [f for f in self.FEATURE_COLUMNS if f in df.columns]
        
        feature_scaler = StandardScaler()
        target_scaler = MinMaxScaler()
        
        feature_data = df[available_features].values
        target_data = df[["close"]].values
        
        feature_scaled = feature_scaler.fit_transform(feature_data)
        target_scaled = target_scaler.fit_transform(target_data)
        
        X, y = [], []
        for i in range(len(feature_scaled) - sequence_length):
            X.append(feature_scaled[i:i + sequence_length])
            y.append(target_scaled[i + sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train/test split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, feature_scaler, target_scaler, available_features
    
    def build_model(
        self,
        input_shape: Tuple[int, int],
        architecture: str,
        units: int,
        layers: int,
        dropout: float,
        learning_rate: float
    ) -> Sequential:
        """
        Build model with specified architecture.
        
        Args:
            architecture: One of "LSTM", "GRU", "BiLSTM", "BiGRU"
        """
        model = Sequential()
        model.add(Input(shape=input_shape))
        
        # Select layer type
        if architecture == "LSTM":
            LayerClass = LSTM
            bidirectional = False
        elif architecture == "GRU":
            LayerClass = GRU
            bidirectional = False
        elif architecture == "BiLSTM":
            LayerClass = LSTM
            bidirectional = True
        elif architecture == "BiGRU":
            LayerClass = GRU
            bidirectional = True
        else:
            LayerClass = LSTM
            bidirectional = False
        
        # Add layers
        for i in range(layers):
            return_sequences = (i < layers - 1)
            layer = LayerClass(
                units,
                return_sequences=return_sequences,
                recurrent_dropout=dropout * 0.5
            )
            
            if bidirectional:
                layer = Bidirectional(layer)
            
            model.add(layer)
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        
        # Output layers
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1, activation="linear"))
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae", "mape"]
        )
        
        return model
    
    def create_optuna_objective(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        stock_code: str
    ) -> Callable:
        """Create Optuna objective function for a specific stock."""
        
        def objective(trial: optuna.Trial) -> float:
            # Hyperparameters to optimize
            architecture = trial.suggest_categorical(
                "architecture", 
                self.config.optuna.architectures
            )
            sequence_length = trial.suggest_int(
                "sequence_length",
                *self.config.optuna.sequence_length_range
            )
            units = trial.suggest_int(
                "units",
                *self.config.optuna.units_range
            )
            layers = trial.suggest_int(
                "layers",
                *self.config.optuna.layers_range
            )
            dropout = trial.suggest_float(
                "dropout",
                *self.config.optuna.dropout_range
            )
            learning_rate = trial.suggest_float(
                "learning_rate",
                *self.config.optuna.learning_rate_range,
                log=True
            )
            batch_size = trial.suggest_categorical(
                "batch_size",
                self.config.optuna.batch_size_options
            )
            
            # Build model
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = self.build_model(
                input_shape=input_shape,
                architecture=architecture,
                units=units,
                layers=layers,
                dropout=dropout,
                learning_rate=learning_rate
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=0
                )
            ]
            
            # Train
            try:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,  # Fewer epochs for Optuna trials
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0
                )
                
                val_loss = min(history.history["val_loss"])
                return val_loss
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float("inf")
        
        return objective
    
    def train_stock(
        self,
        df: pd.DataFrame,
        stock_code: str,
        use_optuna: bool = True,
        use_mlflow: bool = True
    ) -> Dict[str, Any]:
        """
        Train models for a single stock.
        
        Args:
            df: DataFrame with stock data
            stock_code: Stock ticker symbol
            use_optuna: Whether to use Optuna optimization
            use_mlflow: Whether to log to MLflow
        """
        logger.info(f"[TRAIN] Starting training for {stock_code}...")
        
        # Prepare data with default sequence length
        X_train, X_test, y_train, y_test, feature_scaler, target_scaler, feature_names = \
            self.prepare_data(df, sequence_length=30)
        
        logger.info(f"[TRAIN] Data prepared: X_train={X_train.shape}, X_test={X_test.shape}")
        
        # Setup MLflow
        mlflow_active = False
        if use_mlflow and MLFLOW_AVAILABLE:
            mlflow_active = setup_mlflow()
            if mlflow_active:
                mlflow.set_experiment(self.config.experiment_name)
        
        best_params = None
        best_val_loss = float("inf")
        
        if use_optuna and OPTUNA_AVAILABLE:
            # Optuna optimization
            logger.info(f"[TRAIN] Running Optuna optimization ({self.config.optuna.n_trials} trials)...")
            
            study = optuna.create_study(
                direction="minimize",
                study_name=f"stock_{stock_code}_{datetime.now().strftime('%Y%m%d')}"
            )
            
            objective = self.create_optuna_objective(
                X_train, X_test, y_train, y_test, stock_code
            )
            
            study.optimize(
                objective,
                n_trials=self.config.optuna.n_trials,
                timeout=self.config.optuna.timeout,
                show_progress_bar=True
            )
            
            best_params = study.best_params
            best_val_loss = study.best_value
            
            logger.info(f"[TRAIN] ✓ Best trial: val_loss={best_val_loss:.6f}")
            logger.info(f"[TRAIN] Best params: {best_params}")
            
        else:
            # Default parameters
            best_params = {
                "architecture": "GRU",
                "sequence_length": 30,
                "units": 64,
                "layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 16
            }
        
        # Train final model with best params
        logger.info(f"[TRAIN] Training final model with best params...")
        
        # Re-prepare data with optimal sequence length
        seq_len = best_params.get("sequence_length", 30)
        X_train, X_test, y_train, y_test, feature_scaler, target_scaler, feature_names = \
            self.prepare_data(df, sequence_length=seq_len)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        final_model = self.build_model(
            input_shape=input_shape,
            architecture=best_params["architecture"],
            units=best_params["units"],
            layers=best_params["layers"],
            dropout=best_params["dropout"],
            learning_rate=best_params["learning_rate"]
        )
        
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        run_context = None
        if mlflow_active:
            run_context = mlflow.start_run(
                run_name=f"{stock_code}_{best_params['architecture']}_{datetime.now().strftime('%Y%m%d')}"
            )
            run_context.__enter__()
            
            mlflow.log_params({
                "stock_code": stock_code,
                **best_params,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "num_features": len(feature_names)
            })
        
        try:
            history = final_model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=self.config.epochs,
                batch_size=best_params["batch_size"],
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            test_loss, test_mae, test_mape = final_model.evaluate(X_test, y_test, verbose=0)
            
            # Calculate additional metrics
            y_pred_scaled = final_model.predict(X_test, verbose=0)
            y_pred = target_scaler.inverse_transform(y_pred_scaled)
            y_actual = target_scaler.inverse_transform(y_test)
            
            rmse = np.sqrt(np.mean((y_pred - y_actual) ** 2))
            
            # Direction accuracy
            actual_direction = np.sign(np.diff(y_actual.flatten()))
            pred_direction = np.sign(y_pred[1:].flatten() - y_actual[:-1].flatten())
            direction_accuracy = np.mean(actual_direction == pred_direction)
            
            metrics = {
                "test_loss": float(test_loss),
                "test_mae": float(test_mae),
                "test_mape": float(test_mape),
                "rmse": float(rmse),
                "direction_accuracy": float(direction_accuracy),
                "epochs_trained": len(history.history["loss"]),
                "best_val_loss": float(min(history.history["val_loss"]))
            }
            
            if mlflow_active:
                mlflow.log_metrics(metrics)
                mlflow.keras.log_model(final_model, "model")
            
            logger.info(f"[TRAIN] ✓ {stock_code} training complete!")
            logger.info(f"  Architecture: {best_params['architecture']}")
            logger.info(f"  MAE: {test_mae:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  Direction Accuracy: {direction_accuracy*100:.1f}%")
            
        finally:
            if run_context:
                run_context.__exit__(None, None, None)
        
        # Save model
        model_path = os.path.join(self.config.models_dir, f"{stock_code}_model.h5")
        final_model.save(model_path)
        
        # Save scalers and config
        scaler_path = os.path.join(self.config.models_dir, f"{stock_code}_scalers.joblib")
        joblib.dump({
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "feature_names": feature_names,
            "sequence_length": seq_len
        }, scaler_path)
        
        config_path = os.path.join(self.config.models_dir, f"{stock_code}_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "stock_code": stock_code,
                "architecture": best_params["architecture"],
                "params": best_params,
                "metrics": metrics,
                "trained_at": datetime.now().isoformat()
            }, f, indent=2)
        
        self.best_models[stock_code] = {
            "model_path": model_path,
            "architecture": best_params["architecture"],
            "params": best_params,
            "metrics": metrics
        }
        
        return {
            "stock_code": stock_code,
            "model_path": model_path,
            "architecture": best_params["architecture"],
            "params": best_params,
            "metrics": metrics
        }
    
    def train_all_stocks(
        self,
        data_dir: str,
        use_optuna: bool = True,
        use_mlflow: bool = True
    ) -> Dict[str, Dict]:
        """
        Train models for all available stocks.
        
        Args:
            data_dir: Directory containing stock CSV files
            
        Returns:
            Results for all stocks
        """
        results = {}
        
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("*_data.csv"))
        
        logger.info(f"[TRAIN] Found {len(csv_files)} stocks to train")
        
        for csv_file in csv_files:
            stock_code = csv_file.stem.replace("_data", "")
            
            try:
                df = pd.read_csv(csv_file, parse_dates=["date"])
                
                if len(df) < 60:  # Need enough data
                    logger.warning(f"[TRAIN] Skipping {stock_code}: insufficient data")
                    continue
                
                result = self.train_stock(
                    df=df,
                    stock_code=stock_code,
                    use_optuna=use_optuna,
                    use_mlflow=use_mlflow
                )
                
                results[stock_code] = result
                
            except Exception as e:
                logger.error(f"[TRAIN] Error training {stock_code}: {e}")
                continue
        
        # Save summary
        summary_path = os.path.join(self.config.models_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "trained_stocks": list(results.keys()),
                "training_date": datetime.now().isoformat(),
                "models": {k: {
                    "architecture": v["architecture"],
                    "metrics": v["metrics"]
                } for k, v in results.items()}
            }, f, indent=2)
        
        logger.info(f"\n[TRAIN] ✓ All training complete!")
        logger.info(f"  Trained: {len(results)} stocks")
        logger.info(f"  Summary saved to: {summary_path}")
        
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("StockModelTrainer initialized")
    print(f"TensorFlow: {TF_AVAILABLE}")
    print(f"Optuna: {OPTUNA_AVAILABLE}")
    print(f"MLflow: {MLFLOW_AVAILABLE}")
    
    if TF_AVAILABLE:
        print(f"TensorFlow version: {tf.__version__}")
