"""
visualize_trending.py
Creates visual graphs for trending detection results
"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path('.').resolve()))

print("=" * 60)
print("GENERATING TRENDING DETECTION VISUALIZATION")
print("=" * 60)

# Run the vectorizer to get fresh data
from src.graphs.vectorizationAgentGraph import graph

test_texts = [
    {"text": "URGENT: Major earthquake hits Colombo, buildings damaged!", "post_id": "EN_001"},
    {"text": "Breaking news: Earthquake in Colombo measuring 5.5 magnitude", "post_id": "EN_002"},
    {"text": "Colombo earthquake causes panic, residents evacuated", "post_id": "EN_003"},
    {"text": "Sri Lanka Cricket team wins against India in thrilling match", "post_id": "EN_004"},
    {"text": "Stock market shows bullish trends in JKH and Commercial Bank", "post_id": "EN_005"},
    {"text": "Another earthquake aftershock reported in Colombo area", "post_id": "EN_006"},
    {"text": "President announces relief fund for earthquake victims", "post_id": "EN_007"},
    {"text": "Heavy rainfall expected in Southern Province", "post_id": "EN_008"},
]

print(f"\nProcessing {len(test_texts)} texts...")

result = graph.invoke({
    "input_texts": test_texts,
    "batch_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
})

trending_results = result.get("trending_results", {})
anomaly_results = result.get("anomaly_results", {})

# Get trending data
trending_topics = trending_results.get("trending_topics", [])
spike_alerts = trending_results.get("spike_alerts", [])
entities = trending_results.get("entities", [])

print(f"Trending topics: {len(trending_topics)}")
print(f"Spike alerts: {len(spike_alerts)}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Vectorizer Agent: Trending Detection Dashboard', fontsize=16, fontweight='bold')

# ===== PLOT 1: Trending Topics Momentum =====
ax1 = axes[0, 0]
if trending_topics:
    topics = [t.get('topic', '')[:15] for t in trending_topics[:10]]
    momentums = [t.get('momentum', 0) for t in trending_topics[:10]]
    colors = ['#e74c3c' if m > 30 else '#f39c12' if m > 10 else '#3498db' for m in momentums]
    
    bars = ax1.barh(topics, momentums, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Momentum Score', fontsize=11)
    ax1.set_title('Top Trending Topics by Momentum', fontsize=12, fontweight='bold')
    ax1.axvline(x=3, color='orange', linestyle='--', alpha=0.7, label='Spike Threshold (3x)')
    ax1.axvline(x=2, color='green', linestyle='--', alpha=0.7, label='Trending Threshold (2x)')
    ax1.legend(loc='lower right', fontsize=8)
    
    # Add value labels
    for bar, val in zip(bars, momentums):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}x', 
                va='center', fontsize=9, fontweight='bold')
else:
    ax1.text(0.5, 0.5, 'No trending topics', ha='center', va='center', fontsize=12)
    ax1.set_title('Top Trending Topics', fontsize=12, fontweight='bold')

# ===== PLOT 2: Entity Types Distribution =====
ax2 = axes[0, 1]
if entities:
    entity_types = {}
    for e in entities:
        t = e.get('type', 'unknown')
        entity_types[t] = entity_types.get(t, 0) + 1
    
    labels = list(entity_types.keys())
    sizes = list(entity_types.values())
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f1c40f'][:len(labels)]
    explode = [0.05] * len(labels)
    
    ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', 
            colors=colors, shadow=True, startangle=90)
    ax2.set_title('Extracted Entity Types', fontsize=12, fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'No entities', ha='center', va='center', fontsize=12)
    ax2.set_title('Extracted Entity Types', fontsize=12, fontweight='bold')

# ===== PLOT 3: Spike Alerts =====
ax3 = axes[1, 0]
if spike_alerts:
    spike_topics = [s.get('topic', '')[:12] for s in spike_alerts[:8]]
    spike_moms = [s.get('momentum', 0) for s in spike_alerts[:8]]
    
    bars = ax3.bar(spike_topics, spike_moms, color='#e74c3c', edgecolor='black', linewidth=1)
    ax3.set_ylabel('Momentum', fontsize=11)
    ax3.set_title('🔥 SPIKE ALERTS (>3x Normal Volume)', fontsize=12, fontweight='bold', color='#c0392b')
    ax3.axhline(y=3, color='orange', linestyle='--', alpha=0.7)
    ax3.set_xticklabels(spike_topics, rotation=45, ha='right', fontsize=9)
    
    # Add explosion effect
    for bar, val in zip(bars, spike_moms):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.0f}x', 
                ha='center', fontsize=10, fontweight='bold', color='#c0392b')
else:
    ax3.text(0.5, 0.5, 'No spike alerts', ha='center', va='center', fontsize=12)
    ax3.set_title('Spike Alerts', fontsize=12, fontweight='bold')

# ===== PLOT 4: Pipeline Summary =====
ax4 = axes[1, 1]
ax4.axis('off')

# Create a summary box
summary_text = f"""
╔══════════════════════════════════════════════════╗
║        VECTORIZER AGENT PIPELINE SUMMARY         ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  📝 Texts Processed:        {len(test_texts):>5}                  ║
║  🌐 Entities Extracted:     {len(entities):>5}                  ║
║  📈 Trending Topics:        {len(trending_topics):>5}                  ║
║  🔥 Spike Alerts:           {len(spike_alerts):>5}                  ║
║  ⚠️  Anomalies Detected:    {anomaly_results.get('anomalies_found', 0):>5}                  ║
║                                                  ║
╠══════════════════════════════════════════════════╣
║  Top Trending:                                   ║
"""

if trending_topics:
    for i, t in enumerate(trending_topics[:3]):
        topic = t.get('topic', 'N/A')[:20]
        mom = t.get('momentum', 0)
        summary_text += f"║    {i+1}. {topic:<20} ({mom:.0f}x)     ║\n"
else:
    summary_text += "║    (No trending topics)                          ║\n"

summary_text += """╚══════════════════════════════════════════════════╝"""

ax4.text(0.5, 0.5, summary_text, family='monospace', fontsize=9,
         ha='center', va='center', 
         bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#2c3e50'))

plt.tight_layout()
plt.subplots_adjust(top=0.93)

# Save
output_path = Path('trending_detection_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✅ Visualization saved: {output_path}")

# Also save to artifacts
artifacts_dir = Path(r'C:\Users\LENOVO\.gemini\antigravity\brain\b892f63f-afbc-4c4a-bbf1-37195faf04a5')
if artifacts_dir.exists():
    artifacts_output = artifacts_dir / 'trending_visualization.png'
    plt.savefig(str(artifacts_output), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Also saved to: {artifacts_output}")

plt.close()

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
