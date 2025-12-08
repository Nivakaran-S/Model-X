"""
models/stock-price-prediction/src/components/predictor.py
Stock Price Prediction Inference Component
Generates next-day predictions for all trained stocks
"""
import os
import sys
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, List
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from entity.config_entity import PredictionConfig, SRI_LANKA_STOCKS

# TensorFlow for model loading
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger("stock_prediction.predictor")


class StockPredictor:
    """
    Generates next-day stock price predictions using trained models.
    
    Features:
    - Loads per-stock best models
    - Generates predictions with confidence
    - Classifies trend (bullish/bearish/neutral)
    """
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        os.makedirs(self.config.predictions_dir, exist_ok=True)
        
        self.models_dir = str(
            Path(__file__).parent.parent.parent / "artifacts" / "models"
        )
        
        self._loaded_models = {}
    
    def _load_model(self, stock_code: str) -> bool:
        """Load trained model for a stock."""
        if stock_code in self._loaded_models:
            return True
        
        model_path = os.path.join(self.models_dir, f"{stock_code}_model.h5")
        scaler_path = os.path.join(self.models_dir, f"{stock_code}_scalers.joblib")
        config_path = os.path.join(self.models_dir, f"{stock_code}_config.json")
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, config_path]):
            logger.warning(f"[PREDICT] Model files not found for {stock_code}")
            return False
        
        try:
            model = load_model(model_path)
            scalers = joblib.load(scaler_path)
            with open(config_path) as f:
                model_config = json.load(f)
            
            self._loaded_models[stock_code] = {
                "model": model,
                "feature_scaler": scalers["feature_scaler"],
                "target_scaler": scalers["target_scaler"],
                "feature_names": scalers["feature_names"],
                "sequence_length": scalers["sequence_length"],
                "config": model_config
            }
            
            logger.info(f"[PREDICT] ✓ Loaded {stock_code} model ({model_config['architecture']})")
            return True
            
        except Exception as e:
            logger.error(f"[PREDICT] Error loading {stock_code}: {e}")
            return False
    
    def classify_trend(self, change_pct: float) -> Dict[str, str]:
        """Classify the predicted trend."""
        if change_pct > 2:
            return {"trend": "strongly_bullish", "emoji": "🚀", "color": "green"}
        elif change_pct > 0.5:
            return {"trend": "bullish", "emoji": "📈", "color": "green"}
        elif change_pct > -0.5:
            return {"trend": "neutral", "emoji": "➡️", "color": "gray"}
        elif change_pct > -2:
            return {"trend": "bearish", "emoji": "📉", "color": "red"}
        else:
            return {"trend": "strongly_bearish", "emoji": "⬇️", "color": "red"}
    
    def predict_stock(self, stock_code: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Predict next-day price for a single stock.
        
        Args:
            stock_code: Stock ticker
            df: DataFrame with recent stock data (must have last N days)
            
        Returns:
            Prediction dictionary
        """
        if not self._load_model(stock_code):
            return self._generate_fallback_prediction(stock_code, df)
        
        model_data = self._loaded_models[stock_code]
        model = model_data["model"]
        feature_scaler = model_data["feature_scaler"]
        target_scaler = model_data["target_scaler"]
        feature_names = model_data["feature_names"]
        seq_len = model_data["sequence_length"]
        model_config = model_data["config"]
        
        # Get features
        available_features = [f for f in feature_names if f in df.columns]
        recent = df[available_features].tail(seq_len).values
        
        if len(recent) < seq_len:
            logger.warning(f"[PREDICT] Insufficient data for {stock_code}")
            return self._generate_fallback_prediction(stock_code, df)
        
        # Scale and predict
        X = feature_scaler.transform(recent)
        X = X.reshape(1, seq_len, -1)
        
        y_scaled = model.predict(X, verbose=0)
        y = target_scaler.inverse_transform(y_scaled)
        
        current_price = df["close"].iloc[-1]
        predicted_price = float(y[0, 0])
        change = predicted_price - current_price
        change_pct = (change / current_price) * 100
        
        trend = self.classify_trend(change_pct)
        
        stock_info = SRI_LANKA_STOCKS.get(stock_code, {})
        
        return {
            "stock_code": stock_code,
            "name": stock_info.get("name", stock_code),
            "sector": stock_info.get("sector", "Unknown"),
            
            "prediction_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
            
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "expected_change": round(change, 2),
            "expected_change_pct": round(change_pct, 3),
            
            "trend": trend["trend"],
            "trend_emoji": trend["emoji"],
            "trend_color": trend["color"],
            
            "model_architecture": model_config.get("architecture", "Unknown"),
            "model_metrics": model_config.get("metrics", {}),
            
            "is_fallback": False
        }
    
    def _generate_fallback_prediction(
        self, 
        stock_code: str, 
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Generate fallback prediction when model not available."""
        current_price = 100.0
        if df is not None and "close" in df.columns:
            current_price = df["close"].iloc[-1]
        
        # Random walk
        change_pct = np.random.normal(0, 1.5)
        predicted_price = current_price * (1 + change_pct / 100)
        trend = self.classify_trend(change_pct)
        
        stock_info = SRI_LANKA_STOCKS.get(stock_code, {})
        
        return {
            "stock_code": stock_code,
            "name": stock_info.get("name", stock_code),
            "sector": stock_info.get("sector", "Unknown"),
            
            "prediction_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
            
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "expected_change": round(predicted_price - current_price, 2),
            "expected_change_pct": round(change_pct, 3),
            
            "trend": trend["trend"],
            "trend_emoji": trend["emoji"],
            "trend_color": trend["color"],
            
            "model_architecture": "fallback",
            "is_fallback": True,
            "note": "Model not trained - using fallback prediction"
        }
    
    def predict_all_stocks(self, data_dir: str) -> Dict[str, Dict]:
        """
        Generate predictions for all stocks with available data.
        
        Args:
            data_dir: Directory containing stock CSV files
            
        Returns:
            Dictionary of predictions by stock code
        """
        predictions = {}
        
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("*_data.csv"))
        
        for csv_file in csv_files:
            stock_code = csv_file.stem.replace("_data", "")
            
            try:
                df = pd.read_csv(csv_file, parse_dates=["date"])
                prediction = self.predict_stock(stock_code, df)
                
                if prediction:
                    predictions[stock_code] = prediction
                    
            except Exception as e:
                logger.error(f"[PREDICT] Error predicting {stock_code}: {e}")
        
        return predictions
    
    def save_predictions(self, predictions: Dict[str, Dict]) -> str:
        """Save all predictions to JSON."""
        date_str = datetime.now().strftime("%Y%m%d")
        
        output = {
            "prediction_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
            "stocks": predictions,
            "summary": {
                "total_stocks": len(predictions),
                "bullish_count": sum(1 for p in predictions.values() if "bullish" in p.get("trend", "")),
                "bearish_count": sum(1 for p in predictions.values() if "bearish" in p.get("trend", "")),
                "neutral_count": sum(1 for p in predictions.values() if p.get("trend") == "neutral")
            }
        }
        
        output_path = os.path.join(
            self.config.predictions_dir,
            f"stock_predictions_{date_str}.json"
        )
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"[PREDICT] ✓ Saved predictions to {output_path}")
        return output_path
    
    def get_latest_predictions(self) -> Optional[Dict]:
        """Load the latest predictions file."""
        pred_dir = Path(self.config.predictions_dir)
        json_files = list(pred_dir.glob("stock_predictions_*.json"))
        
        if not json_files:
            return None
        
        latest = max(json_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest) as f:
            return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    predictor = StockPredictor()
    
    # Test fallback
    for code in ["JKH", "COMB", "DIAL"]:
        pred = predictor._generate_fallback_prediction(code)
        print(f"{code}: {pred['trend_emoji']} {pred['expected_change_pct']:+.2f}%")
