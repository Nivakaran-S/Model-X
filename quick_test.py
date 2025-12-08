"""
Quick test for DataTransformation with Vectorization API
"""
import sys
from pathlib import Path

# Add proper paths FIRST
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "anomaly-detection"))

print("Testing DataTransformation with Vectorization API")
print("=" * 60)
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print()

# Now import
from src.components import DataTransformation
from src.entity import DataTransformationConfig
import tempfile

config = DataTransformationConfig()
config.output_directory = tempfile.mkdtemp()

print("Creating DataTransformation with use_agent_graph=True...")
transformer = DataTransformation(config, use_agent_graph=True)

print()
print("=" * 60)
print(f"Vectorization API URL: {transformer.vectorization_api_url}")
print(f"Vectorization API Available: {transformer.vectorization_api_available}")
print("=" * 60)

if transformer.vectorization_api_available:
    print("[SUCCESS] Vectorization API connected!")
    print()
    print("Now testing vectorization...")
    
    # Create sample texts
    sample_texts = [
        {"post_id": "test_001", "text": "Heavy rainfall expected in Colombo district tomorrow."},
        {"post_id": "test_002", "text": "Stock market showing positive trends today."}
    ]
    
    result = transformer._process_with_agent_graph(sample_texts)
    if result:
        print(f"  [OK] Processed {len(sample_texts)} texts")
        print(f"  Expert Summary: {len(result.get('expert_summary', ''))} chars")
        print(f"  {result.get('expert_summary', '')[:200]}...")
    else:
        print("  [WARN] Processing returned None")
else:
    print("[FAIL] Vectorization API NOT available")
    print("Make sure vectorization_api is running:")
    print("  python -m src.api.vectorization_api")
