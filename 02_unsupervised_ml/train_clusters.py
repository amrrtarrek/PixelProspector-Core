"""
PixelProspector -- Step 2: Unsupervised Clustering & System Memory
===================================================================
Member 2 (The Structurer)

Builds the ground-truth labels and the system's memory via:
  1. Pipeline A (Games):  DBSCAN outlier filter -> FCM (3 clusters: Flop, Niche, Breakout)
  2. Pipeline B (Users):  DBSCAN outlier filter -> FCM (3 clusters: Spam, Casual, Expert)
  3. Implicit RAG:        Cluster Mean SHAP Vectors (document store for cosine similarity)
  4. Explicit RAG:        FAISS index for majority voting on borderline scores

Outputs saved:
  fcm_centroids.pkl       -> Game + User centroids (consumed by Member 3 & 4)
  cluster_mean_shap.pkl   -> Mean SHAP vectors per cluster (consumed by Member 3)
  training_data.pkl       -> Features + labels for SVM training (consumed by Member 3)
  pixel_prospector.index  -> FAISS index (consumed by Member 5)

Usage:
    python train_clusters.py
"""

import numpy as np
import pandas as pd
import pickle
import faiss
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# --- PIXELPROSPECTOR V4.0 CONFIGURATION ---
N_CLUSTERS = 3  # Flop, Niche, Breakout / Spam, Casual, Expert
DBSCAN_EPS = 0.3
MIN_SAMPLES = 5

# Label maps (index -> human-readable name)
GAME_LABEL_MAP = {0: "Flop", 1: "Niche", 2: "Breakout"}
USER_LABEL_MAP = {0: "Spam", 1: "Casual", 2: "Expert"}


class PixelProspectorStructurer:
    def __init__(self):
        self.game_centroids = None
        self.user_centroids = None
        self.mean_shap_vectors = {}

    def run_clustering_pipeline(self, df):
        """
        Executes Step 2: Unsupervised Clustering & System Memory.
        Returns cleaned data, membership matrices, and hard labels for
        both game and user pipelines.
        """
        # =============================================================
        # 1. Pipeline A: Games (Extracting game_ml_features)
        # =============================================================
        game_cols = [
            'gameplay_addictiveness', 'technical_polish', 'aesthetic_appeal',
            'narrative_depth', 'replayability', 'viral_momentum'
        ]
        game_data = df[game_cols].values

        # DBSCAN Outlier Filtering
        dbscan_games = DBSCAN(eps=DBSCAN_EPS, min_samples=MIN_SAMPLES).fit(game_data)
        game_mask = dbscan_games.labels_ != -1
        game_clean = game_data[game_mask]

        # Fuzzy C-Means (FCM) for 3 Game Clusters
        cntr_games, u_games, _, _, _, _, _ = fuzz.cluster.cmeans(
            game_clean.T, c=N_CLUSTERS, m=2, error=0.005, maxiter=1000
        )
        self.game_centroids = cntr_games

        # Hard labels from FCM membership matrix
        game_labels = np.argmax(u_games, axis=0)

        # =============================================================
        # 2. Pipeline B: Users (Extracting user_review_features)
        # =============================================================
        user_cols = [
            'insight_depth', 'toxicity_level', 'genre_expertise', 'sentiment_consistency'
        ]
        user_data = df[user_cols].values

        # DBSCAN + FCM for 3 User Clusters
        dbscan_users = DBSCAN(eps=DBSCAN_EPS, min_samples=MIN_SAMPLES).fit(user_data)
        user_mask = dbscan_users.labels_ != -1
        user_clean = user_data[user_mask]

        cntr_users, u_users, _, _, _, _, _ = fuzz.cluster.cmeans(
            user_clean.T, c=N_CLUSTERS, m=2, error=0.005, maxiter=1000
        )
        self.user_centroids = cntr_users

        # Hard labels from FCM membership matrix
        user_labels = np.argmax(u_users, axis=0)

        # =============================================================
        # 3. Export centroids to fcm_centroids.pkl
        # =============================================================
        self._save_centroids()

        # =============================================================
        # 4. Export training data for Member 3's SVM training
        # =============================================================
        self._save_training_data(game_clean, game_labels, user_clean, user_labels)

        return game_clean, u_games, game_labels, user_clean, u_users, user_labels

    def _save_centroids(self):
        """Save FCM centroids for Member 3 (live_inference) and Member 4 (drift_monitor)."""
        centroids_payload = {
            'game_centroids': self.game_centroids,
            'user_centroids': self.user_centroids
        }
        with open('fcm_centroids.pkl', 'wb') as f:
            pickle.dump(centroids_payload, f)
        print("[OK] fcm_centroids.pkl exported for Member 3 & 4.")

    def _save_training_data(self, game_features, game_labels, user_features, user_labels):
        """
        Save labelled training data so Member 3 can train the Game SVM
        and User SVM via `python live_inference.py --train`.
        """
        training_payload = {
            'game_features': game_features,
            'game_labels':   game_labels,
            'user_features': user_features,
            'user_labels':   user_labels,
        }
        with open('training_data.pkl', 'wb') as f:
            pickle.dump(training_payload, f)
        print(f"[OK] training_data.pkl exported "
              f"(games={len(game_features)}, users={len(user_features)}).")

    def build_implicit_rag_store(self, game_clean, u_matrix, shap_values):
        """
        Implicit RAG: Calculate Cluster Mean SHAP Vectors.
        Uses string labels ("Flop", "Niche", "Breakout") as keys so
        Member 3's cosine similarity lookup works directly.
        """
        # Assign hard labels from FCM membership matrix
        labels = np.argmax(u_matrix, axis=0)

        game_shap_means = {}
        for cluster_idx in range(N_CLUSTERS):
            indices = np.where(labels == cluster_idx)[0]
            if len(indices) > 0:
                cluster_shaps = shap_values[indices]
                label_name = GAME_LABEL_MAP[cluster_idx]
                game_shap_means[label_name] = np.mean(cluster_shaps, axis=0)

        self.mean_shap_vectors = game_shap_means

        shap_payload = {
            'game_shap_means': game_shap_means,
            'user_shap_means': {},  # Populated separately if user SHAP is computed
        }
        with open('cluster_mean_shap.pkl', 'wb') as f:
            pickle.dump(shap_payload, f)
        print("[OK] Implicit RAG Store (cluster_mean_shap.pkl) created.")

    def build_explicit_rag_index(self, game_data, shap_vectors):
        """
        Explicit RAG: Build FAISS index for majority voting.
        Groups combined feature vectors + SHAP vectors together.
        """
        combined = np.hstack([game_data, shap_vectors]).astype('float32')

        dimension = combined.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(combined)

        faiss.write_index(index, "pixel_prospector.index")
        print(f"[OK] Explicit RAG FAISS Index built (dim={dimension}, "
              f"vectors={combined.shape[0]}).")


# --- EXECUTION FLOW ---
if __name__ == "__main__":
    print("Starting Member 2: Unsupervised Pipeline...")

    # Placeholder for training data (would come from Member 1's PostgreSQL)
    # structurer = PixelProspectorStructurer()
    # clean_games, u_matrix, g_labels, clean_users, u_users, u_labels = (
    #     structurer.run_clustering_pipeline(raw_df)
    # )
