#!/bin/bash
set -e

echo "🚀 Starting ModelX Backend on HuggingFace Space..."

# 1. Run ML Training Pipeline (if models missing)
# We trust the script to handle logic. For Hackathon, we force run it to ensure fresh state if possible,
# or we can check if output dir is empty.
echo "🧠 Checking ML Models..."
# Create output dir if not exists
mkdir -p models/anomaly-detection/output

# Run training (standalone script)
# This will use data from 'datasets/' if available. 
# If datasets are empty, it might fail/skip, so we allow failure without stopping container.
python scripts/train_ml_models.py || echo "⚠️ ML Training warning (continuing anyway)..."

# 2. Start Request Server
# HuggingFace expects us to listen on port 7860
echo "⚡ Starting FastAPI Server on port $PORT..."
uvicorn main:app --host 0.0.0.0 --port $PORT
