"""
test_vectorizer_and_anomaly.py
Test script to run the Vectorizer Agent and Anomaly Detection pipeline
Generates visualizations of the results
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

print("=" * 70)
print("  VECTORIZER AGENT & ANOMALY DETECTION TEST")
print("=" * 70)
print()

# ============================================
# STEP 1: TEST VECTORIZER AGENT
# ============================================
print("\n" + "=" * 50)
print("STEP 1: Testing Vectorizer Agent")
print("=" * 50)

# Sample multilingual test data
test_texts = [
    {"text": "The political situation in Colombo is tense with protests happening", "post_id": "EN_001"},
    {"text": "Stock market shows bullish trends in JKH and Commercial Bank", "post_id": "EN_002"},
    {"text": "Heavy rainfall expected in Southern Province causing flood warnings", "post_id": "EN_003"},
    {"text": "Economic reforms by the government receive mixed public response", "post_id": "EN_004"},
    {"text": "URGENT: Massive landslide in Ratnapura district, evacuations underway!", "post_id": "EN_005"},
    {"text": "Normal day, nothing much happening, just regular news", "post_id": "EN_006"},
    {"text": "Coffee prices remain stable in local markets", "post_id": "EN_007"},
    {"text": "BREAKING: Major corruption scandal exposed in government ministry", "post_id": "EN_008"},
    {"text": "Sri Lanka cricket team wins against India in thrilling match", "post_id": "EN_009"},
    {"text": "Warning: Tsunami alert issued for coastal areas - immediate evacuation!", "post_id": "EN_010"},
]

# Add some Sinhala text samples (using romanized for simplicity)
sinhala_texts = [
    {"text": "කොළඹ නගරයේ අද මහ වර්ෂාවක් ඇති විය", "post_id": "SI_001"},
    {"text": "ආර්ථික අර්බුදය හේතුවෙන් ජනතාව දුෂ්කරතාවන්ට මුහුණ දෙයි", "post_id": "SI_002"},
]

# Add Tamil text samples
tamil_texts = [
    {"text": "கொழும்பில் பெரும் மழை பெய்தது", "post_id": "TA_001"},
]

all_texts = test_texts + sinhala_texts + tamil_texts

print(f"📝 Testing with {len(all_texts)} sample texts")
print(f"   - English: {len(test_texts)}")
print(f"   - Sinhala: {len(sinhala_texts)}")
print(f"   - Tamil: {len(tamil_texts)}")

# Run the vectorizer agent
try:
    from src.graphs.vectorizationAgentGraph import graph as vectorizer_graph
    
    initial_state = {
        "input_texts": all_texts,
        "batch_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    
    print("\n🔄 Running Vectorizer Agent Graph...")
    result = vectorizer_graph.invoke(initial_state)
    
    print("\n✅ Vectorizer Agent Results:")
    print("-" * 40)
    
    # Language detection results
    lang_results = result.get("language_detection_results", [])
    print(f"\n📊 Language Detection:")
    lang_stats = {}
    for item in lang_results:
        lang = item.get("language", "unknown")
        lang_stats[lang] = lang_stats.get(lang, 0) + 1
        print(f"   - {item.get('post_id')}: {lang} (conf: {item.get('confidence', 0):.2f})")
    
    print(f"\n📈 Language Distribution: {lang_stats}")
    
    # Vector embeddings
    embeddings = result.get("vector_embeddings", [])
    print(f"\n🔢 Vector Embeddings Generated: {len(embeddings)}")
    if embeddings:
        sample = embeddings[0]
        print(f"   Sample vector dim: {sample.get('vector_dim', 0)}")
        print(f"   Models used: {set(e.get('model_used', '') for e in embeddings)}")
    
    # Anomaly detection results
    anomaly_results = result.get("anomaly_results", {})
    print(f"\n🔍 Anomaly Detection:")
    print(f"   Status: {anomaly_results.get('status', 'unknown')}")
    print(f"   Model: {anomaly_results.get('model_used', 'none')}")
    print(f"   Total Analyzed: {anomaly_results.get('total_analyzed', 0)}")
    print(f"   Anomalies Found: {anomaly_results.get('anomalies_found', 0)}")
    
    anomalies = anomaly_results.get("anomalies", [])
    if anomalies:
        print(f"\n⚠️ Detected Anomalies:")
        for a in anomalies:
            print(f"   - {a.get('post_id')}: score={a.get('anomaly_score', 0):.3f}")
    
    # Expert summary
    expert_summary = result.get("expert_summary", "")
    if expert_summary:
        print(f"\n📋 Expert Summary (first 500 chars):")
        print(f"   {expert_summary[:500]}...")
    
    # Domain insights
    domain_insights = result.get("domain_insights", [])
    print(f"\n💡 Domain Insights Generated: {len(domain_insights)}")
    
except Exception as e:
    print(f"❌ Vectorizer Agent Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# STEP 2: RUN ANOMALY DETECTION PIPELINE
# ============================================
print("\n\n" + "=" * 50)
print("STEP 2: Running Anomaly Detection ML Pipeline")
print("=" * 50)

MODELS_PATH = PROJECT_ROOT / "models" / "anomaly-detection"
sys.path.insert(0, str(MODELS_PATH))

try:
    from src.pipeline.training_pipeline import TrainingPipeline
    
    print("\n🔄 Running Anomaly Detection Training Pipeline...")
    pipeline = TrainingPipeline()
    artifacts = pipeline.run()
    
    print("\n✅ Training Pipeline Results:")
    print("-" * 40)
    
    if artifacts.get("model_trainer"):
        trainer_artifact = artifacts["model_trainer"]
        print(f"   Best Score: {getattr(trainer_artifact, 'best_score', 'N/A')}")
        print(f"   Best Model: {getattr(trainer_artifact, 'best_model', 'N/A')}")
        
        # Check for model files
        model_path = MODELS_PATH / "output"
        if model_path.exists():
            models = list(model_path.glob("*.joblib"))
            print(f"\n📁 Saved Models: {len(models)}")
            for m in models:
                print(f"   - {m.name}")
                
except ImportError as e:
    print(f"⚠️ Could not import training pipeline: {e}")
    print("   Running standalone model training instead...")
    
    try:
        # Try running the main.py directly
        os.chdir(MODELS_PATH)
        exec(open(MODELS_PATH / "main.py").read())
    except Exception as e2:
        print(f"❌ Standalone training error: {e2}")
except Exception as e:
    print(f"❌ Pipeline Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# STEP 3: VISUALIZATION
# ============================================
print("\n\n" + "=" * 50)
print("STEP 3: Generating Visualizations")
print("=" * 50)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    # Only visualize if we have embeddings
    if 'embeddings' in dir() and embeddings:
        # Extract vectors
        vectors = []
        labels = []
        for emb in embeddings:
            vec = emb.get("vector", [])
            if len(vec) == 768:
                vectors.append(vec)
                labels.append(emb.get("post_id", ""))
        
        if vectors:
            X = np.array(vectors)
            
            # PCA for visualization
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Language distribution scatter
            ax1 = axes[0]
            colors = {'english': 'blue', 'sinhala': 'green', 'tamil': 'orange', 'unknown': 'gray'}
            
            for i, emb in enumerate(embeddings):
                if i < len(X_2d):
                    lang = emb.get("language", "unknown")
                    ax1.scatter(X_2d[i, 0], X_2d[i, 1], c=colors.get(lang, 'gray'), 
                               s=100, alpha=0.7, label=lang if lang not in [e.get('language') for e in embeddings[:i]] else "")
            
            ax1.set_title("Text Embeddings by Language (PCA 2D)")
            ax1.set_xlabel("PC1")
            ax1.set_ylabel("PC2")
            
            # Add legend (unique labels only)
            handles, legend_labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(legend_labels, handles))
            ax1.legend(by_label.values(), by_label.keys())
            
            # Plot 2: Anomaly scores
            ax2 = axes[1]
            if anomalies:
                anomaly_ids = [a.get("post_id", "") for a in anomalies]
                
                for i, emb in enumerate(embeddings):
                    if i < len(X_2d):
                        is_anomaly = emb.get("post_id", "") in anomaly_ids
                        color = 'red' if is_anomaly else 'blue'
                        marker = 'X' if is_anomaly else 'o'
                        ax2.scatter(X_2d[i, 0], X_2d[i, 1], c=color, marker=marker,
                                   s=150 if is_anomaly else 80, alpha=0.7)
                
                ax2.scatter([], [], c='red', marker='X', s=150, label='Anomaly')
                ax2.scatter([], [], c='blue', marker='o', s=80, label='Normal')
                ax2.legend()
            else:
                ax2.scatter(X_2d[:, 0], X_2d[:, 1], c='blue', s=80, alpha=0.7)
                ax2.text(0.5, 0.5, "No anomalies detected\n(Model may not be trained)", 
                        ha='center', va='center', transform=ax2.transAxes)
            
            ax2.set_title("Anomaly Detection Results (PCA 2D)")
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            
            plt.tight_layout()
            
            # Save figure
            output_path = PROJECT_ROOT / "vectorizer_anomaly_visualization.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✅ Visualization saved to: {output_path}")
            
            plt.close()
    else:
        print("⚠️ No embeddings available for visualization")
        
except ImportError as e:
    print(f"⚠️ Matplotlib not available for visualization: {e}")
except Exception as e:
    print(f"❌ Visualization Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# STEP 4: GRAPH FLOW VISUALIZATION
# ============================================
print("\n\n" + "=" * 50)
print("STEP 4: Generating Graph Flow Diagram")
print("=" * 50)

try:
    # Create a simple ASCII graph visualization
    graph_viz = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║           VECTORIZATION AGENT GRAPH FLOW                          ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║   ┌─────────────────┐                                             ║
    ║   │   INPUT TEXTS   │ (Multilingual: EN, SI, TA)                  ║
    ║   └────────┬────────┘                                             ║
    ║            │                                                      ║
    ║            ▼                                                      ║
    ║   ┌─────────────────────────────────────────────────────┐        ║
    ║   │  STEP 1: LANGUAGE DETECTION                         │        ║
    ║   │  ├─ FastText (primary)                              │        ║
    ║   │  └─ Unicode Script Analysis (fallback)              │        ║
    ║   └────────┬────────────────────────────────────────────┘        ║
    ║            │                                                      ║
    ║            ▼                                                      ║
    ║   ┌─────────────────────────────────────────────────────┐        ║
    ║   │  STEP 2: TEXT VECTORIZATION                         │        ║
    ║   │  ├─ English  → DistilBERT (768-dim)                 │        ║
    ║   │  ├─ Sinhala  → SinhalaBERTo (768-dim)               │        ║
    ║   │  └─ Tamil    → Tamil-BERT (768-dim)                 │        ║
    ║   └────────┬────────────────────────────────────────────┘        ║
    ║            │                                                      ║
    ║            ▼                                                      ║
    ║   ┌─────────────────────────────────────────────────────┐        ║
    ║   │  STEP 3: ANOMALY DETECTION                          │        ║
    ║   │  ├─ Model: Isolation Forest / LOF                   │        ║
    ║   │  ├─ Input: 768-dim embedding vectors                │        ║
    ║   │  └─ Output: anomaly_score (0-1), is_anomaly flag    │        ║
    ║   └────────┬────────────────────────────────────────────┘        ║
    ║            │                                                      ║
    ║            ▼                                                      ║
    ║   ┌─────────────────────────────────────────────────────┐        ║
    ║   │  STEP 4: EXPERT SUMMARY (GroqLLM)                   │        ║
    ║   │  ├─ Opportunity Detection                           │        ║
    ║   │  └─ Threat Detection                                │        ║
    ║   └────────┬────────────────────────────────────────────┘        ║
    ║            │                                                      ║
    ║            ▼                                                      ║
    ║   ┌─────────────────────────────────────────────────────┐        ║
    ║   │  STEP 5: FORMAT OUTPUT                              │        ║
    ║   │  └─ domain_insights[] for Combined Agent            │        ║
    ║   └────────┬────────────────────────────────────────────┘        ║
    ║            │                                                      ║
    ║            ▼                                                      ║
    ║   ┌─────────────────┐                                             ║
    ║   │      END        │ → Passed to Feed Aggregator                 ║
    ║   └─────────────────┘                                             ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(graph_viz)
    
    # Save as text file
    graph_path = PROJECT_ROOT / "vectorizer_graph_flow.txt"
    with open(graph_path, "w", encoding="utf-8") as f:
        f.write(graph_viz)
    print(f"✅ Graph flow saved to: {graph_path}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================
# SUMMARY
# ============================================
print("\n\n" + "=" * 70)
print("  TEST SUMMARY")
print("=" * 70)

print("""
📊 VECTORIZER AGENT ARCHITECTURE:
   ├── 5-Step Sequential Pipeline
   ├── Multilingual Support: English, Sinhala, Tamil
   ├── BERT Models: DistilBERT, SinhalaBERTo, Tamil-BERT
   └── Output: 768-dimensional embeddings

🔍 ANOMALY DETECTION:
   ├── Algorithm: Isolation Forest / LOF
   ├── Training: Optuna hyperparameter optimization
   ├── MLflow: Experiment tracking (DagsHub)
   └── Integration: Real-time inference on every graph cycle

📁 OUTPUT FILES:
   ├── vectorizer_anomaly_visualization.png (if matplotlib available)
   └── vectorizer_graph_flow.txt (graph architecture)
""")

print("=" * 70)
print("  TEST COMPLETE")
print("=" * 70)
