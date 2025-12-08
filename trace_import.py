"""
Debug trace for import issue
"""
import sys
from pathlib import Path

print("=== IMPORT TRACE ===")
print(f"Initial sys.path:")
for i, p in enumerate(sys.path[:5]):
    print(f"  {i}: {p}")
print()

# Add paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "anomaly-detection"))

print("After adding paths:")
for i, p in enumerate(sys.path[:5]):
    print(f"  {i}: {p}")
print()

# Try importing src.graphs directly
print("Step 1: Importing src.graphs.vectorizationAgentGraph...")
try:
    from src.graphs.vectorizationAgentGraph import graph as vgraph
    print(f"  [OK] Imported successfully: {type(vgraph)}")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")

print()
print("Step 2: Importing from anomaly-detection components...")
try:
    from src.components import DataTransformation
    print(f"  [OK] DataTransformation imported")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")

print()
print("Step 3: Creating instance...")
try:
    from src.entity import DataTransformationConfig
    import tempfile
    config = DataTransformationConfig()
    config.output_directory = tempfile.mkdtemp()
    
    t = DataTransformation(config, use_agent_graph=True)
    print(f"  Agent graph available: {t.vectorizer_graph is not None}")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")

print()
print("=== DONE ===")
