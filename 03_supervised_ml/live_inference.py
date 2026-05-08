"""
PixelProspector -- Step 3: Supervised State Detector & Explainability
======================================================================
Member 3 (The Reasoner)

Trains TWO independent SVM classifiers (Game SVM + User SVM) and computes
four of the five Intelligent Score signals for every live prediction:

    Signal 1  S_dynamic            = P_game * Trust_user
    Signal 2  Gap(SVM)             = P_top  - P_second
    Signal 3  Mu (geometric)       = membership to nearest FCM centroid
    Signal 5  SHAP cosine sim.     = cos_sim(live_shap, cluster_mean_shap)

(Signal 4 -- ARIMA trend multiplier -- is handled by Member 4.)

All outputs are mapped to the V4.0 intelligent_score_signals keys so that
Member 5 (FastAPI Agent) can consume them directly.

Usage:
    # Train models from Member 2's labelled data:
    python live_inference.py --train

    # Run a quick self-test with a sample payload:
    python live_inference.py --test
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.svm import SVC
import joblib

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("live_inference")

# ---------------------------------------------------------------------------
# Paths  (relative to THIS file so it works from any working directory)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_UNSUPERVISED_DIR = _HERE / ".." / "02_unsupervised_ml"
_MODELS_DIR       = _HERE / "models"

CENTROIDS_PATH   = _UNSUPERVISED_DIR / "fcm_centroids.pkl"
SHAP_MEANS_PATH  = _UNSUPERVISED_DIR / "cluster_mean_shap.pkl"
TRAINING_PATH    = _UNSUPERVISED_DIR / "training_data.pkl"

GAME_SVM_PATH    = _MODELS_DIR / "game_svm.joblib"
USER_SVM_PATH    = _MODELS_DIR / "user_svm.joblib"

# ---------------------------------------------------------------------------
# Label maps  (index -> human name)
# ---------------------------------------------------------------------------
GAME_LABELS = {0: "Flop", 1: "Niche", 2: "Breakout"}
USER_LABELS = {0: "Spam", 1: "Casual", 2: "Expert"}

# Trust multiplier per user class  (used for Dynamic Severity bonus)
TRUST_MULTIPLIER = {
    0: 0.2,    # Spam    -- heavily dampened
    1: 1.0,    # Casual  -- neutral
    2: 1.5,    # Expert  -- amplified
}

# Feature order (must match Member 2's training data)
GAME_FEATURE_KEYS = [
    "gameplay_addictiveness",
    "technical_polish",
    "aesthetic_appeal",
    "narrative_depth",
    "replayability",
    "viral_momentum",
]
USER_FEATURE_KEYS = [
    "insight_depth",
    "toxicity_level",
    "genre_expertise",
    "sentiment_consistency",
]


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_svms() -> None:
    """
    Train two independent SVM classifiers using Member 2's labelled data.
    Models are saved to 03_supervised_ml/models/ via joblib.
    Called once offline, or again by Member 4's autonomous outer loop when
    drift is detected.
    """
    log.info("Loading training data from %s", TRAINING_PATH)
    with open(TRAINING_PATH, "rb") as f:
        data = pickle.load(f)

    X_game = data["game_features"]
    y_game = data["game_labels"]
    X_user = data["user_features"]
    y_user = data["user_labels"]

    log.info("Training Game SVM  (%d samples, %d features) ...",
             X_game.shape[0], X_game.shape[1])
    game_svm = SVC(kernel="rbf", probability=True, random_state=42)
    game_svm.fit(X_game, y_game)

    log.info("Training User SVM  (%d samples, %d features) ...",
             X_user.shape[0], X_user.shape[1])
    user_svm = SVC(kernel="rbf", probability=True, random_state=42)
    user_svm.fit(X_user, y_user)

    # Persist
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(game_svm, GAME_SVM_PATH)
    joblib.dump(user_svm, USER_SVM_PATH)
    log.info("Models saved to %s", _MODELS_DIR)


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Stateless inference engine.  Loads pre-trained SVMs, FCM centroids,
    and cluster SHAP means once, then computes signals for any live payload.
    """

    def __init__(self):
        log.info("Initializing InferenceEngine ...")

        # ── Load SVMs ─────────────────────────────────────────────────────
        if not GAME_SVM_PATH.exists() or not USER_SVM_PATH.exists():
            raise FileNotFoundError(
                f"Trained SVM models not found at {_MODELS_DIR}. "
                "Run `python live_inference.py --train` first."
            )
        self.game_svm: SVC = joblib.load(GAME_SVM_PATH)
        self.user_svm: SVC = joblib.load(USER_SVM_PATH)
        log.info("  Game SVM loaded  (classes=%s)", list(self.game_svm.classes_))
        log.info("  User SVM loaded  (classes=%s)", list(self.user_svm.classes_))

        # ── Load FCM centroids ────────────────────────────────────────────
        with open(CENTROIDS_PATH, "rb") as f:
            centroids = pickle.load(f)
        self.game_centroids: np.ndarray = centroids["game_centroids"]
        self.user_centroids: np.ndarray = centroids["user_centroids"]
        log.info("  FCM centroids loaded  (game=%s, user=%s)",
                 self.game_centroids.shape, self.user_centroids.shape)

        # ── Load cluster mean SHAP vectors ────────────────────────────────
        with open(SHAP_MEANS_PATH, "rb") as f:
            shap_data = pickle.load(f)
        self.game_shap_means: Dict[str, np.ndarray] = shap_data["game_shap_means"]
        self.user_shap_means: Dict[str, np.ndarray] = shap_data["user_shap_means"]
        log.info("  Cluster mean SHAP vectors loaded")

        # ── SHAP Explainer (lazy init on first call) ──────────────────────
        self._game_explainer = None
        self._user_explainer = None

        log.info("InferenceEngine ready.")

    # ------------------------------------------------------------------
    # SHAP explainer helpers
    # ------------------------------------------------------------------
    def _get_game_explainer(self):
        """Lazy-load the SHAP KernelExplainer for the Game SVM."""
        if self._game_explainer is None:
            import shap
            # Use centroids as a lightweight background dataset
            background = self.game_centroids
            self._game_explainer = shap.KernelExplainer(
                self.game_svm.predict_proba, background
            )
            log.info("  SHAP KernelExplainer (Game) initialized")
        return self._game_explainer

    def _get_user_explainer(self):
        """Lazy-load the SHAP KernelExplainer for the User SVM."""
        if self._user_explainer is None:
            import shap
            background = self.user_centroids
            self._user_explainer = shap.KernelExplainer(
                self.user_svm.predict_proba, background
            )
            log.info("  SHAP KernelExplainer (User) initialized")
        return self._user_explainer

    # ------------------------------------------------------------------
    # Feature extraction from V4.0 payload
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_game_vector(payload: Dict[str, Any]) -> np.ndarray:
        gf = payload["game_ml_features"]
        return np.array([gf[k] for k in GAME_FEATURE_KEYS]).reshape(1, -1)

    @staticmethod
    def _extract_user_vector(payload: Dict[str, Any]) -> np.ndarray:
        uf = payload["user_review_features"]
        return np.array([uf[k] for k in USER_FEATURE_KEYS]).reshape(1, -1)

    # ------------------------------------------------------------------
    # Signal computations
    # ------------------------------------------------------------------
    def _signal_1_dynamic_severity(
        self, game_proba: np.ndarray, user_pred_class: int
    ) -> float:
        """
        S_dynamic = P_game(top) * Trust_user
        BONUS: Dynamic severity -- the user's predicted class acts as a
        trust multiplier that scales the game probability.
        """
        p_game_top = float(np.max(game_proba))
        trust = TRUST_MULTIPLIER.get(user_pred_class, 1.0)
        return round(p_game_top * trust, 4)

    @staticmethod
    def _signal_2_gap_svm(game_proba: np.ndarray) -> float:
        """
        Gap(SVM) = P_top - P_second
        Measures how decisive the classifier is.
        """
        sorted_proba = np.sort(game_proba.flatten())[::-1]
        gap = float(sorted_proba[0] - sorted_proba[1])
        return round(gap, 4)

    def _signal_3_mu_geometric(self, game_vector: np.ndarray) -> float:
        """
        Mu = 1 / (1 + d_min)
        Geometric membership: how close the live data point is to the
        nearest FCM centroid.  Returns a value in (0, 1] where 1 means
        the point sits exactly on a centroid.
        """
        distances = np.linalg.norm(self.game_centroids - game_vector, axis=1)
        d_min = float(np.min(distances))
        mu = 1.0 / (1.0 + d_min)
        return round(mu, 4)

    def _signal_5_shap_cosine(
        self, game_vector: np.ndarray, game_pred_label: str
    ) -> float:
        """
        SHAP Cosine Similarity = 1 - cosine_distance(live_shap, cluster_mean_shap)
        Measures prediction reliability by comparing the live data's SHAP
        vector against the expected SHAP direction of the predicted cluster.
        """
        explainer = self._get_game_explainer()
        shap_values = explainer.shap_values(game_vector, silent=True)

        # Handle both old and new SHAP API return formats
        pred_class_idx = list(GAME_LABELS.values()).index(game_pred_label)

        if isinstance(shap_values, list):
            # Old API: list of arrays, one per class
            live_shap = shap_values[pred_class_idx].flatten()
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # New API: shape (n_samples, n_features, n_classes)
            live_shap = shap_values[0, :, pred_class_idx].flatten()
        else:
            # Single output or (1, n_features) array
            live_shap = np.asarray(shap_values).flatten()

        # Retrieve the cluster mean SHAP vector from Member 2's store
        cluster_mean = self.game_shap_means.get(game_pred_label)
        if cluster_mean is None:
            log.warning("No cluster mean SHAP for '%s'; defaulting to 0.0",
                        game_pred_label)
            return 0.0

        # Guard against zero-norm vectors
        if np.linalg.norm(live_shap) < 1e-10 or np.linalg.norm(cluster_mean) < 1e-10:
            return 0.0

        similarity = 1.0 - cosine_distance(live_shap, cluster_mean)
        return round(float(similarity), 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_signals(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point.  Takes a V4.0 JSON payload and returns all
        Member-3 signals plus diagnostic metadata.

        Returns
        -------
        dict with keys:
            "intelligent_score_signals" : { S, Gap, Mu, SHAP }
            "diagnostics"               : { game_class, user_class, ... }
        """
        # Extract feature vectors
        game_vec = self._extract_game_vector(payload)
        user_vec = self._extract_user_vector(payload)

        # SVM predictions
        game_pred_class = int(self.game_svm.predict(game_vec)[0])
        game_proba      = self.game_svm.predict_proba(game_vec)[0]
        user_pred_class = int(self.user_svm.predict(user_vec)[0])
        user_proba      = self.user_svm.predict_proba(user_vec)[0]

        game_label = GAME_LABELS[game_pred_class]
        user_label = USER_LABELS[user_pred_class]

        # Compute the four signals
        s_dynamic = self._signal_1_dynamic_severity(game_proba, user_pred_class)
        gap_svm   = self._signal_2_gap_svm(game_proba)
        mu_geo    = self._signal_3_mu_geometric(game_vec.flatten())
        shap_cos  = self._signal_5_shap_cosine(game_vec, game_label)

        return {
            "intelligent_score_signals": {
                "S_class_severity":        s_dynamic,
                "Gap_SVM_confidence":      gap_svm,
                "mu_geometric_membership": mu_geo,
                "SHAP_cosine_similarity":  shap_cos,
                # Signal 4 is filled by Member 4:
                "ARIMA_trend_multiplier":  None,
            },
            "diagnostics": {
                "game_predicted_class":  game_label,
                "game_probabilities":    {GAME_LABELS[i]: round(float(p), 4)
                                          for i, p in enumerate(game_proba)},
                "user_predicted_class":  user_label,
                "user_probabilities":    {USER_LABELS[i]: round(float(p), 4)
                                          for i, p in enumerate(user_proba)},
                "trust_multiplier_used": TRUST_MULTIPLIER[user_pred_class],
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
#  CLI & SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    """Quick smoke test with a sample V4.0 payload."""
    engine = InferenceEngine()

    # ----- Test 1: High-quality game + Expert user -----
    payload_good = {
        "interaction_metadata": {
            "user_id": "test_user_01",
            "game_id": "st_12345",
            "timestamp": "2026-05-08T12:00:00Z",
            "developer_email": "dev@studio.com",
            "primary_genre": "RPG",
            "triage_status": "Pass",
        },
        "game_ml_features": {
            "gameplay_addictiveness": 0.90,
            "technical_polish": 0.85,
            "aesthetic_appeal": 0.88,
            "narrative_depth": 0.82,
            "replayability": 0.80,
            "viral_momentum": 0.70,
        },
        "user_review_features": {
            "insight_depth": 0.85,
            "toxicity_level": 0.05,
            "genre_expertise": 0.90,
            "sentiment_consistency": 0.88,
        },
    }

    # ----- Test 2: Low-quality game + Spam user -----
    payload_bad = {
        "interaction_metadata": {
            "user_id": "test_user_02",
            "game_id": "st_99999",
            "timestamp": "2026-05-08T12:00:00Z",
            "developer_email": "dev@studio.com",
            "primary_genre": "Puzzle",
            "triage_status": "Pass",
        },
        "game_ml_features": {
            "gameplay_addictiveness": 0.12,
            "technical_polish": 0.18,
            "aesthetic_appeal": 0.15,
            "narrative_depth": 0.08,
            "replayability": 0.10,
            "viral_momentum": 0.05,
        },
        "user_review_features": {
            "insight_depth": 0.08,
            "toxicity_level": 0.80,
            "genre_expertise": 0.10,
            "sentiment_consistency": 0.15,
        },
    }

    for label, payload in [("HIGH-QUALITY GAME + EXPERT", payload_good),
                           ("LOW-QUALITY GAME + SPAM",    payload_bad)]:
        print("\n" + "=" * 60)
        print(f"  TEST: {label}")
        print("=" * 60)
        result = engine.compute_signals(payload)

        signals = result["intelligent_score_signals"]
        diag    = result["diagnostics"]

        print(f"  Game Class  : {diag['game_predicted_class']}")
        print(f"  User Class  : {diag['user_predicted_class']}")
        print(f"  Trust Mult. : {diag['trust_multiplier_used']}x")
        print(f"  ---")
        print(f"  Signal 1 (S_dynamic)  : {signals['S_class_severity']}")
        print(f"  Signal 2 (Gap SVM)    : {signals['Gap_SVM_confidence']}")
        print(f"  Signal 3 (Mu)         : {signals['mu_geometric_membership']}")
        print(f"  Signal 5 (SHAP cos)   : {signals['SHAP_cosine_similarity']}")
        print(f"  Signal 4 (ARIMA)      : {signals['ARIMA_trend_multiplier']}  "
              "(filled by Member 4)")

    print("\n[OK] All signals computed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PixelProspector Step 3 -- Supervised ML & Explainability",
    )
    parser.add_argument("--train", action="store_true",
                        help="Train Game SVM + User SVM from Member 2's data")
    parser.add_argument("--test",  action="store_true",
                        help="Run self-test with sample payloads")
    args = parser.parse_args()

    if args.train:
        train_svms()
    elif args.test:
        _run_self_test()
    else:
        print("Usage: python live_inference.py --train | --test")
        print("  --train   Train models from Member 2's labelled data")
        print("  --test    Run self-test with sample V4.0 payloads")
