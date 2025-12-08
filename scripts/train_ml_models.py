
import os
import sys
from pathlib import Path

# Add the anomaly detection model source to path
MODEL_DIR = Path(__file__).parent.parent / "models" / "anomaly-detection"
sys.path.append(str(MODEL_DIR))

try:
    from src.pipeline.training_pipeline import run_training_pipeline
    print("🚀 Starting ML Model Training Pipeline (Standalone)...")
    print(f"📂 Model Directory: {MODEL_DIR}")
    
    # Run the pipeline
    artifact = run_training_pipeline()
    
    print("\n✅ Training Complete!")
    print(f"📊 Model Artifacts stored in: {MODEL_DIR}/output")
    
except ImportError as e:
    print(f"❌ Error: Could not import training pipeline. {e}")
    print("Ensure you are running this from the project root.")
except Exception as e:
    print(f"❌ Training Failed: {e}")
