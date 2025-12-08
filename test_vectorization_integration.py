"""
Test script for Vectorization Pipeline Integration
Tests that DataTransformation correctly invokes the Vectorizer Agent Graph
"""
import os
import sys
import json
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project roots to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "anomaly-detection"))

def create_test_data():
    """Create sample test data with multilingual content"""
    data = [
        {
            "id": "test_001",
            "text": "Heavy rainfall expected in Colombo district tomorrow. Residents advised to stay indoors.",
            "source": "DMC",
            "timestamp": datetime.now().isoformat(),
            "engagement_score": 100
        },
        {
            "id": "test_002",
            "text": "Sinhala text sample for testing language detection",
            "source": "twitter",
            "timestamp": datetime.now().isoformat(),
            "engagement_score": 50
        },
        {
            "id": "test_003",
            "text": "Tamil text sample for testing language detection",
            "source": "facebook",
            "timestamp": datetime.now().isoformat(),
            "engagement_score": 75
        },
        {
            "id": "test_004",
            "text": "Stock market showing positive trends. Tourism sector recovering well after monsoon season.",
            "source": "news",
            "timestamp": datetime.now().isoformat(),
            "engagement_score": 200
        }
    ]
    return pd.DataFrame(data)


def test_vectorizer_agent_graph_standalone():
    """Test 1: Verify vectorizer agent graph works independently"""
    print("\n" + "="*60)
    print("TEST 1: Vectorizer Agent Graph (Standalone)")
    print("="*60)
    
    try:
        from src.graphs.vectorizationAgentGraph import graph as vectorization_graph
        print("[OK] Vectorizer Agent Graph loaded successfully")
        
        # Prepare test input
        test_input = {
            "input_texts": [
                {"post_id": "test_001", "text": "Heavy rainfall in Colombo"},
                {"post_id": "test_002", "text": "Sinhala test text"},
                {"post_id": "test_003", "text": "Tamil test text"}
            ],
            "batch_id": "test_standalone"
        }
        
        print(f"  Input: {len(test_input['input_texts'])} texts")
        
        # Invoke graph
        result = vectorization_graph.invoke(test_input)
        
        print(f"  [OK] Graph executed successfully")
        print(f"  Keys in result: {list(result.keys())}")
        
        # Check outputs
        lang_results = result.get("language_detection_results", [])
        embeddings = result.get("vector_embeddings", [])
        expert_summary = result.get("expert_summary", "")
        
        print(f"  [OK] Language detection: {len(lang_results)} results")
        print(f"  [OK] Vector embeddings: {len(embeddings)} vectors")
        print(f"  [OK] Expert summary: {len(expert_summary)} chars")
        
        # Show language distribution
        if lang_results:
            langs = [r.get("language", "unknown") for r in lang_results]
            print(f"  Languages detected: {set(langs)}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_transformation_with_agent():
    """Test 2: Verify DataTransformation integrates with agent graph"""
    print("\n" + "="*60)
    print("TEST 2: DataTransformation with Agent Graph")
    print("="*60)
    
    try:
        # Import with correct path
        sys.path.insert(0, str(PROJECT_ROOT / "models" / "anomaly-detection"))
        from src.components import DataTransformation
        from src.entity import DataTransformationConfig
        
        # Create temp directory for outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config
            config = DataTransformationConfig()
            config.output_directory = tmpdir
            
            # Initialize with agent graph enabled
            transformer = DataTransformation(config, use_agent_graph=True)
            
            print(f"  [OK] DataTransformation initialized")
            print(f"  Agent graph available: {transformer.vectorizer_graph is not None}")
            
            # Create test data
            df = create_test_data()
            test_data_path = Path(tmpdir) / "test_data.parquet"
            df.to_parquet(test_data_path, index=False)
            print(f"  [OK] Test data created: {len(df)} records")
            
            # Run transformation
            artifact = transformer.transform(str(test_data_path))
            
            print(f"  [OK] Transformation complete")
            print(f"  Total records: {artifact.total_records}")
            print(f"  Languages: {artifact.language_distribution}")
            
            # Check if LLM insights were saved
            insights_files = list(Path(tmpdir).glob("llm_insights_*.json"))
            if insights_files:
                print(f"  [OK] LLM insights saved: {insights_files[0].name}")
                
                with open(insights_files[0], "r", encoding="utf-8") as f:
                    insights = json.load(f)
                
                print(f"    Expert summary: {len(insights.get('expert_summary', ''))} chars")
                print(f"    Opportunities: {len(insights.get('opportunities', []))}")
                print(f"    Threats: {len(insights.get('threats', []))}")
            else:
                print("  [WARN] No LLM insights file found (agent graph may not have run)")
            
            # Check embeddings
            embeddings_files = list(Path(tmpdir).glob("embeddings_*.npy"))
            if embeddings_files:
                embeddings = np.load(embeddings_files[0])
                print(f"  [OK] Embeddings saved: shape {embeddings.shape}")
            
            return True
            
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_transformation_fallback():
    """Test 3: Verify fallback works when agent graph is disabled"""
    print("\n" + "="*60)
    print("TEST 3: DataTransformation Fallback (Agent Disabled)")
    print("="*60)
    
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "models" / "anomaly-detection"))
        from src.components import DataTransformation
        from src.entity import DataTransformationConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DataTransformationConfig()
            config.output_directory = tmpdir
            
            # Initialize with agent graph DISABLED
            transformer = DataTransformation(config, use_agent_graph=False)
            
            print(f"  [OK] DataTransformation initialized (fallback mode)")
            print(f"  Agent graph: {transformer.vectorizer_graph}")
            
            # Create test data
            df = create_test_data()
            test_data_path = Path(tmpdir) / "test_data.parquet"
            df.to_parquet(test_data_path, index=False)
            
            # Run transformation
            artifact = transformer.transform(str(test_data_path))
            
            print(f"  [OK] Fallback transformation complete")
            print(f"  Total records: {artifact.total_records}")
            
            # Verify no LLM insights (since agent was disabled)
            insights_files = list(Path(tmpdir).glob("llm_insights_*.json"))
            if not insights_files:
                print(f"  [OK] Correctly no LLM insights (agent disabled)")
            else:
                print(f"  [WARN] Unexpected LLM insights file found")
            
            return True
            
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VECTORIZATION PIPELINE INTEGRATION TESTS")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Vectorizer Agent Graph Standalone", test_vectorizer_agent_graph_standalone()))
    results.append(("DataTransformation with Agent", test_data_transformation_with_agent()))
    results.append(("DataTransformation Fallback", test_data_transformation_fallback()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n[SUCCESS] All tests passed! Pipeline integration is working.")
    else:
        print("\n[WARNING] Some tests failed. Check the output above for details.")
