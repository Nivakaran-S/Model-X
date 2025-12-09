"""
create_visualization.py
Creates visualization of multilingual embeddings and anomaly detection results
using actual training data.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import joblib

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("  CREATING VECTORIZER VISUALIZATION")
print("=" * 60)

# Load saved embeddings from the training pipeline
embeddings_path = Path('models/anomaly-detection/artifacts/data_transformation')

# Find the latest embeddings file
emb_files = list(embeddings_path.glob('embeddings_*.npy'))
if emb_files:
    latest_emb = sorted(emb_files)[-1]
    embeddings = np.load(latest_emb)
    print(f'Loaded embeddings: {embeddings.shape}')
else:
    print('No embeddings found')
    sys.exit(1)

# Load transformed data to get language info
import pandas as pd
data_files = list(embeddings_path.glob('transformed_*.parquet'))
if data_files:
    latest_data = sorted(data_files)[-1]
    df = pd.read_parquet(latest_data)
    languages = df['language'].values
    lang_counts = df['language'].value_counts().to_dict()
    print(f'Languages: {lang_counts}')
else:
    languages = ['english'] * len(embeddings)
    lang_counts = {'english': len(embeddings)}

# Load anomaly model and predict
model_path = Path('models/anomaly-detection/artifacts/model_trainer/isolation_forest_embeddings_only.joblib')
model = joblib.load(model_path)
predictions = model.predict(embeddings)
anomaly_mask = predictions == -1

print(f'Total samples: {len(embeddings)}')
print(f'Anomalies detected: {anomaly_mask.sum()}')
print(f'Normal samples: {(~anomaly_mask).sum()}')

# PCA for visualization
print('\nRunning PCA...')
pca = PCA(n_components=2)
X_2d = pca.fit_transform(embeddings)
print(f'Explained variance: {pca.explained_variance_ratio_.sum():.2%}')

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: By Language
ax1 = axes[0]
colors = {'english': '#3498db', 'sinhala': '#2ecc71', 'tamil': '#e74c3c', 'unknown': '#95a5a6'}

for lang in colors:
    mask = np.array(languages) == lang
    if mask.any():
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=colors[lang], label=f'{lang.capitalize()} ({mask.sum()})', 
                   alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

ax1.set_title('Text Embeddings by Language (PCA Projection)', fontsize=14, fontweight='bold')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax1.legend(loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Plot 2: Anomalies
ax2 = axes[1]
normal_mask = ~anomaly_mask

# Plot normal points first (so anomalies are on top)
ax2.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1], 
           c='#3498db', label=f'Normal ({normal_mask.sum()})', alpha=0.6, s=60,
           edgecolors='white', linewidth=0.5)

# Plot anomalies with X markers
ax2.scatter(X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1], 
           c='#e74c3c', marker='X', label=f'Anomaly ({anomaly_mask.sum()})', 
           alpha=0.9, s=120, edgecolors='black', linewidth=0.5)

ax2.set_title('Anomaly Detection Results (Isolation Forest)', fontsize=14, fontweight='bold')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax2.legend(loc='best', framealpha=0.9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'vectorizer_anomaly_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'\nVisualization saved: {output_path}')

# Also create the visualization in artifacts dir
artifacts_dir = Path(r'C:\Users\LENOVO\.gemini\antigravity\brain\b892f63f-afbc-4c4a-bbf1-37195faf04a5')
if artifacts_dir.exists():
    artifacts_output = artifacts_dir / 'vectorizer_visualization.png'
    plt.savefig(str(artifacts_output), dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Also saved to: {artifacts_output}')

plt.close()

print("\n" + "=" * 60)
print("  VISUALIZATION COMPLETE")
print("=" * 60)
