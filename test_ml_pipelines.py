"""
test_ml_pipelines.py
Test script to verify all 4 ML pipelines are working correctly
"""
import sys
import os
import io

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("[ML PIPELINE TESTING]")
print("="*70)

results = {}

# =============================================================================
# 1. ANOMALY DETECTION PIPELINE
# =============================================================================
print("\n" + "="*60)
print("[1] ANOMALY DETECTION PIPELINE")
print("="*60)

try:
    # Check if model exists
    from pathlib import Path
    model_dir = Path(__file__).parent / "models" / "anomaly-detection" / "output"
    models_found = list(model_dir.glob("*.joblib")) if model_dir.exists() else []
    
    if models_found:
        print(f"[OK] Found {len(models_found)} trained models:")
        for m in models_found[:3]:
            print(f"   - {m.name}")
        
        # Try to load and run prediction
        from models.anomaly_detection.src.utils.vectorizer import get_vectorizer
        vectorizer = get_vectorizer()
        print(f"[OK] Vectorizer loaded")
        
        import joblib
        model = joblib.load(models_found[0])
        print(f"[OK] Model loaded: {models_found[0].name}")
        
        # Test prediction
        test_text = "Breaking news: Major political announcement in Colombo"
        vector = vectorizer.vectorize(test_text, "en")
        prediction = model.predict([vector])[0]
        score = -model.decision_function([vector])[0] if hasattr(model, 'decision_function') else 0
        
        print(f"[OK] Test prediction: is_anomaly={prediction==-1}, score={score:.3f}")
        results["anomaly_detection"] = {"status": "success", "models": len(models_found)}
    else:
        print("[WARN] No trained models found. Run training first.")
        print("   Command: python models/anomaly-detection/main.py --mode train")
        results["anomaly_detection"] = {"status": "not_trained"}
        
except Exception as e:
    print(f"[FAIL] Anomaly Detection error: {e}")
    results["anomaly_detection"] = {"status": "error", "error": str(e)}

# =============================================================================
# 2. WEATHER PREDICTION PIPELINE
# =============================================================================
print("\n" + "="*60)
print("[2] WEATHER PREDICTION PIPELINE")
print("="*60)

try:
    from pathlib import Path
    weather_model_dir = Path(__file__).parent / "models" / "weather-prediction" / "artifacts" / "models"
    weather_models = list(weather_model_dir.glob("*.h5")) if weather_model_dir.exists() else []
    
    predictions_dir = Path(__file__).parent / "models" / "weather-prediction" / "output" / "predictions"
    prediction_files = list(predictions_dir.glob("*.json")) if predictions_dir.exists() else []
    
    if weather_models:
        print(f"[OK] Found {len(weather_models)} trained LSTM models:")
        for m in weather_models[:5]:
            print(f"   - {m.name}")
        
        # Check for predictions
        if prediction_files:
            import json
            latest = max(prediction_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                preds = json.load(f)
            districts = preds.get("districts", {})
            print(f"[OK] Found predictions for {len(districts)} districts")
            print(f"   Latest prediction date: {preds.get('prediction_date', 'N/A')}")
            
            # Show sample
            if districts:
                sample_district = list(districts.keys())[0]
                sample = districts[sample_district]
                print(f"   Sample ({sample_district}):")
                print(f"     - Temp: {sample.get('temp_max', 'N/A')}C - {sample.get('temp_min', 'N/A')}C")
                print(f"     - Rain: {sample.get('rainfall_mm', 'N/A')}mm")
            
            results["weather_prediction"] = {"status": "success", "models": len(weather_models), "districts": len(districts)}
        else:
            print("[WARN] No prediction files found. Run predictor.")
            results["weather_prediction"] = {"status": "models_only", "models": len(weather_models)}
    else:
        print("[WARN] No trained models found")
        print("   Command: python models/weather-prediction/main.py --mode train")
        results["weather_prediction"] = {"status": "not_trained"}
        
except Exception as e:
    print(f"[FAIL] Weather Prediction error: {e}")
    results["weather_prediction"] = {"status": "error", "error": str(e)}

# =============================================================================
# 3. CURRENCY PREDICTION PIPELINE
# =============================================================================
print("\n" + "="*60)
print("[3] CURRENCY PREDICTION PIPELINE (USD/LKR)")
print("="*60)

try:
    from pathlib import Path
    currency_model_dir = Path(__file__).parent / "models" / "currency-volatility-prediction" / "artifacts" / "models"
    currency_model = currency_model_dir / "gru_usd_lkr.h5" if currency_model_dir.exists() else None
    
    predictions_dir = Path(__file__).parent / "models" / "currency-volatility-prediction" / "output" / "predictions"
    prediction_files = list(predictions_dir.glob("*.json")) if predictions_dir.exists() else []
    
    if currency_model and currency_model.exists():
        print(f"[OK] Found GRU model: {currency_model.name}")
        
        # Check for predictions
        if prediction_files:
            import json
            latest = max(prediction_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                pred = json.load(f)
            
            print(f"[OK] Latest prediction found:")
            print(f"   - Current Rate: {pred.get('current_rate', 'N/A')} LKR")
            print(f"   - Predicted: {pred.get('predicted_rate', 'N/A')} LKR")
            print(f"   - Change: {pred.get('change_percent', 'N/A')}%")
            print(f"   - Direction: {pred.get('direction', 'N/A')}")
            
            results["currency_prediction"] = {"status": "success", "rate": pred.get("predicted_rate")}
        else:
            print("[WARN] No prediction files found")
            results["currency_prediction"] = {"status": "model_only"}
    else:
        print("[WARN] No trained model found")
        print("   Command: python models/currency-volatility-prediction/main.py --mode train")
        results["currency_prediction"] = {"status": "not_trained"}
        
except Exception as e:
    print(f"[FAIL] Currency Prediction error: {e}")
    results["currency_prediction"] = {"status": "error", "error": str(e)}

# =============================================================================
# 4. STOCK PRICE PREDICTION PIPELINE
# =============================================================================
print("\n" + "="*60)
print("[4] STOCK PRICE PREDICTION PIPELINE")
print("="*60)

try:
    from pathlib import Path
    stock_model_dir = Path(__file__).parent / "models" / "stock-price-prediction" / "artifacts" / "models"
    stock_models = list(stock_model_dir.glob("*.h5")) if stock_model_dir.exists() else []
    
    predictions_dir = Path(__file__).parent / "models" / "stock-price-prediction" / "output" / "predictions"
    prediction_files = list(predictions_dir.glob("*.json")) if predictions_dir.exists() else []
    
    if stock_models:
        print(f"[OK] Found {len(stock_models)} stock models:")
        for m in stock_models[:5]:
            print(f"   - {m.name}")
        
        # Check for predictions
        if prediction_files:
            import json
            latest = max(prediction_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                preds = json.load(f)
            
            stocks = preds.get("stocks", preds.get("predictions", {}))
            print(f"[OK] Found predictions for {len(stocks)} stocks")
            
            # Show sample
            if stocks:
                sample_stock = list(stocks.keys())[0] if isinstance(stocks, dict) else stocks[0]
                if isinstance(stocks, dict):
                    sample = stocks[sample_stock]
                    print(f"   Sample ({sample_stock}):")
                    print(f"     - Current: {sample.get('current_price', 'N/A')}")
                    print(f"     - Predicted: {sample.get('predicted_price', 'N/A')}")
            
            results["stock_prediction"] = {"status": "success", "models": len(stock_models), "stocks": len(stocks)}
        else:
            print("[WARN] No prediction files found")
            results["stock_prediction"] = {"status": "models_only", "models": len(stock_models)}
    else:
        print("[WARN] No trained models found")
        print("   Command: python models/stock-price-prediction/main.py --mode train")
        results["stock_prediction"] = {"status": "not_trained"}
        
except Exception as e:
    print(f"[FAIL] Stock Prediction error: {e}")
    results["stock_prediction"] = {"status": "error", "error": str(e)}

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("[SUMMARY] ML PIPELINE STATUS")
print("="*70)

for pipeline, result in results.items():
    status = result.get("status", "unknown")
    if status == "success":
        print(f"[OK] {pipeline}: Working")
    elif status == "not_trained":
        print(f"[WARN] {pipeline}: Not trained yet")
    elif status in ["model_only", "models_only"]:
        print(f"[WARN] {pipeline}: Model exists, no recent predictions")
    else:
        print(f"[FAIL] {pipeline}: {result.get('error', status)}")

print("="*70)
