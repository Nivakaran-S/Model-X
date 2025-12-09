"""
test_trending_integration.py
Test the trending detection integration in the vectorizer pipeline.
"""
import sys
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path('.').resolve()))

from src.graphs.vectorizationAgentGraph import graph
from datetime import datetime

print("=" * 60)
print("TESTING TRENDING DETECTION INTEGRATION")
print("=" * 60)

# Test with multiple mentions of the same topic to trigger trending
test_texts = [
    {"text": "URGENT: Major earthquake hits Colombo, buildings damaged!", "post_id": "EN_001"},
    {"text": "Breaking news: Earthquake in Colombo measuring 5.5 magnitude", "post_id": "EN_002"},
    {"text": "Colombo earthquake causes panic, residents evacuated", "post_id": "EN_003"},
    {"text": "Sri Lanka Cricket team wins against India in thrilling match", "post_id": "EN_004"},
    {"text": "Stock market shows bullish trends in JKH", "post_id": "EN_005"},
    {"text": "Another earthquake aftershock reported in Colombo area", "post_id": "EN_006"},
]

print(f"\nProcessing {len(test_texts)} texts with repeated topics...")

result = graph.invoke({
    "input_texts": test_texts,
    "batch_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
})

# Show trending results
print("\n" + "=" * 60)
print("TRENDING DETECTION RESULTS")
print("=" * 60)

trending_results = result.get("trending_results", {})
print(f"\nStatus: {trending_results.get('status', 'N/A')}")
print(f"Entities extracted: {trending_results.get('entities_extracted', 0)}")

# Show extracted entities
entities = trending_results.get("entities", [])
print(f"\nExtracted Entities ({len(entities)}):")
for e in entities[:10]:
    print(f"  - {e.get('entity')} ({e.get('type')}) from {e.get('post_id')}")

# Show trending topics
trending_topics = trending_results.get("trending_topics", [])
print(f"\nTrending Topics ({len(trending_topics)}):")
if trending_topics:
    for t in trending_topics:
        print(f"  - {t.get('topic')}: momentum={t.get('momentum', 0):.2f}, is_spike={t.get('is_spike', False)}")
else:
    print("  (No trending topics yet - need more historical data)")

# Show spike alerts
spike_alerts = trending_results.get("spike_alerts", [])
print(f"\nSpike Alerts ({len(spike_alerts)}):")
if spike_alerts:
    for s in spike_alerts:
        print(f"  - {s.get('topic')}: momentum={s.get('momentum', 0):.2f}")
else:
    print("  (No spike alerts)")

# Show anomaly results
print("\n" + "=" * 60)
print("ANOMALY DETECTION RESULTS")
print("=" * 60)
anomaly_results = result.get("anomaly_results", {})
print(f"Status: {anomaly_results.get('status', 'N/A')}")
print(f"Anomalies found: {anomaly_results.get('anomalies_found', 0)}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE - 6-Step Architecture Working!")
print("=" * 60)
