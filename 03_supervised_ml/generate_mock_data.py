"""
PixelProspector — Mock Data Generator for Member 3
====================================================
Generates synthetic outputs that Member 2 (Unsupervised ML) would normally
produce.  This allows Member 3 to develop and test live_inference.py
independently.

Files generated:
  ../02_unsupervised_ml/fcm_centroids.pkl
      ├── "game_centroids"   : ndarray (3, 6)  — Flop / Niche / Breakout
      └── "user_centroids"   : ndarray (3, 4)  — Spam / Casual / Expert

  ../02_unsupervised_ml/cluster_mean_shap.pkl
      ├── "game_shap_means"  : dict[str, ndarray(6,)]
      └── "user_shap_means"  : dict[str, ndarray(4,)]

  ../02_unsupervised_ml/training_data.pkl
      ├── "game_features"    : ndarray (N, 6)
      ├── "game_labels"      : ndarray (N,)   — 0=Flop, 1=Niche, 2=Breakout
      ├── "user_features"    : ndarray (N, 4)
      └── "user_labels"      : ndarray (N,)   — 0=Spam, 1=Casual, 2=Expert

Usage:
    python generate_mock_data.py
"""

import pickle
import os
import numpy as np

np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "02_unsupervised_ml")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  FCM Centroids  (what Member 2 saves after DBSCAN + Fuzzy C-Means)
# ──────────────────────────────────────────────────────────────────────────────
# Game features order:
#   [addictiveness, tech_polish, aesthetic, narrative, replayability, viral_mom]
GAME_CENTROIDS = np.array([
    [0.15, 0.20, 0.18, 0.10, 0.12, 0.05],   # Cluster 0 — Flop
    [0.50, 0.55, 0.52, 0.48, 0.45, 0.30],   # Cluster 1 — Niche
    [0.85, 0.88, 0.90, 0.80, 0.82, 0.75],   # Cluster 2 — Breakout
])

# User features order:
#   [insight_depth, toxicity_level, genre_expertise, sentiment_consistency]
USER_CENTROIDS = np.array([
    [0.10, 0.85, 0.12, 0.20],   # Cluster 0 — Spam
    [0.50, 0.30, 0.50, 0.65],   # Cluster 1 — Casual
    [0.88, 0.05, 0.90, 0.92],   # Cluster 2 — Expert
])


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Synthetic training data  (so we can train the SVMs)
# ──────────────────────────────────────────────────────────────────────────────
SAMPLES_PER_CLUSTER = 80   # 240 total per pipeline

def _generate_cluster_samples(centroid: np.ndarray, n: int, noise: float = 0.08):
    """Generate samples around a centroid, clipped to [0, 1]."""
    samples = centroid + np.random.randn(n, len(centroid)) * noise
    return np.clip(samples, 0.0, 1.0)


game_features_list, game_labels_list = [], []
for label, centroid in enumerate(GAME_CENTROIDS):
    samples = _generate_cluster_samples(centroid, SAMPLES_PER_CLUSTER)
    game_features_list.append(samples)
    game_labels_list.append(np.full(SAMPLES_PER_CLUSTER, label))

game_features = np.vstack(game_features_list)
game_labels   = np.concatenate(game_labels_list)

user_features_list, user_labels_list = [], []
for label, centroid in enumerate(USER_CENTROIDS):
    samples = _generate_cluster_samples(centroid, SAMPLES_PER_CLUSTER)
    user_features_list.append(samples)
    user_labels_list.append(np.full(SAMPLES_PER_CLUSTER, label))

user_features = np.vstack(user_features_list)
user_labels   = np.concatenate(user_labels_list)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Cluster Mean SHAP Vectors  (Implicit RAG — used for cosine similarity)
# ──────────────────────────────────────────────────────────────────────────────
# In practice, Member 2 computes these from real SHAP values across the cluster.
# Here we create plausible directional vectors.
GAME_SHAP_MEANS = {
    "Flop":     np.array([-0.15, -0.20, -0.10, -0.05, -0.12, -0.18]),
    "Niche":    np.array([ 0.10,  0.08,  0.12,  0.15,  0.05,  0.02]),
    "Breakout": np.array([ 0.25,  0.22,  0.28,  0.20,  0.24,  0.30]),
}

USER_SHAP_MEANS = {
    "Spam":   np.array([-0.20, 0.35, -0.15, -0.25]),
    "Casual": np.array([ 0.05, 0.00,  0.08,  0.10]),
    "Expert": np.array([ 0.30, -0.20,  0.28,  0.25]),
}


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Save everything
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

centroids_path   = os.path.join(OUTPUT_DIR, "fcm_centroids.pkl")
shap_path        = os.path.join(OUTPUT_DIR, "cluster_mean_shap.pkl")
training_path    = os.path.join(OUTPUT_DIR, "training_data.pkl")

with open(centroids_path, "wb") as f:
    pickle.dump({
        "game_centroids": GAME_CENTROIDS,
        "user_centroids": USER_CENTROIDS,
    }, f)

with open(shap_path, "wb") as f:
    pickle.dump({
        "game_shap_means": GAME_SHAP_MEANS,
        "user_shap_means": USER_SHAP_MEANS,
    }, f)

with open(training_path, "wb") as f:
    pickle.dump({
        "game_features": game_features,
        "game_labels":   game_labels,
        "user_features": user_features,
        "user_labels":   user_labels,
    }, f)

print("[OK] Mock data generated successfully!")
print(f"   -> {centroids_path}")
print(f"   -> {shap_path}")
print(f"   -> {training_path}")
print(f"   Game training samples : {len(game_features)}")
print(f"   User training samples : {len(user_features)}")
