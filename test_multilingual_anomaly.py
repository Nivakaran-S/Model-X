"""
test_multilingual_anomaly.py
Test the multilingual anomaly detection fix.
"""
import sys
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path('.').resolve()))

from src.graphs.vectorizationAgentGraph import graph
from datetime import datetime

test_texts = [
    {"text": "URGENT: Massive landslide in Ratnapura!", "post_id": "EN_001"},
    {"text": "Normal stock market day", "post_id": "EN_002"},
    {"text": "ආර්ථික අර්බුදය නිසා ජනතාව දුෂ්කරතාවන්ට මුහුණ දෙයි", "post_id": "SI_001"},
    {"text": "கொழும்பில் பெரும் மழை பெய்தது", "post_id": "TA_001"},
    {"text": "Breaking news about corruption scandal", "post_id": "EN_003"},
]

result = graph.invoke({
    "input_texts": test_texts,
    "batch_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
})

print("=" * 60)
print("MULTILINGUAL ANOMALY DETECTION TEST")
print("=" * 60)

anomaly_results = result.get("anomaly_results", {})
print(f"\nStatus: {anomaly_results.get('status')}")
print(f"Model: {anomaly_results.get('model_used')}")
print(f"Total analyzed: {anomaly_results.get('total_analyzed')}")

anomalies = anomaly_results.get("anomalies", [])
print(f"\nAnomalies found: {len(anomalies)}")
for a in anomalies:
    method = a.get("detection_method", "unknown")
    print(f"  - {a.get('post_id')}: {a.get('language')} | method: {method} | score: {a.get('anomaly_score', 0):.2f}")

lang_results = result.get("language_detection_results", [])
print(f"\nLanguage Detection:")
for lr in lang_results:
    print(f"  - {lr.get('post_id')}: {lr.get('language')} (conf: {lr.get('confidence', 0):.2f})")

# Summary
print("\n" + "=" * 60)
print("The fix ensures:")
print("  - English texts: Isolation Forest ML model")
print("  - Sinhala/Tamil: Magnitude-based heuristic (avoids false positives)")
print("=" * 60)
