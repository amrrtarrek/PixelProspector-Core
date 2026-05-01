import json
import pickle
import numpy as np
import pmdarima as pm
from scipy.spatial.distance import cdist
from typing import Dict, Any, List

class DriftMonitor:
    def __init__(self, fcm_centroids_path: str = "fcm_centroids.pkl", drift_threshold: float = 2.5):
        """
        Initializes the Step 4 Forecasting and Drift Detection module.
        """
        self.drift_threshold = drift_threshold
        self.centroids = self._load_centroids(fcm_centroids_path)
        
    def _load_centroids(self, path: str) -> np.ndarray:
        """Loads the pre-trained DBSCAN/FCM centroids from Step 2."""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"[WARNING] SYSTEM WARNING: {path} not found. Drift detection will be bypassed.")
            return None

    def calculate_market_multiplier(self, historical_viral_momentum: List[float]) -> float:
        """
        Uses S-ARIMA to forecast the weekly trajectory of viral_momentum.
        Outputs a Market Multiplier to scale the Game SVM's probability.
        """
        # Fallback if insufficient historical data exists for the time series
        if not historical_viral_momentum or len(historical_viral_momentum) < 10:
            return 1.0 

        try:
            # Fit Auto-ARIMA (S-ARIMA) on the weekly trajectory
            # m=4 assumes a roughly monthly seasonality cycle over weekly data
            model = pm.auto_arima(
                historical_viral_momentum, 
                seasonal=True, 
                m=4,
                suppress_warnings=True, 
                error_action="ignore"
            )
            
            # Forecast the next immediate time step
            forecasted_momentum = model.predict(n_periods=1)[0]
            
            # Calculate the multiplier based on forecasted momentum vs recent baseline
            recent_baseline = np.mean(historical_viral_momentum[-4:])
            
            if recent_baseline <= 0.01: # Prevent division by zero / extreme spikes
                return 1.0
                
            multiplier = forecasted_momentum / recent_baseline
            
            # Clamp the multiplier between 0.5 (heavy decay) and 2.0 (massive virality)
            market_multiplier = max(0.5, min(multiplier, 2.0))
            return round(market_multiplier, 3)
            
        except Exception as e:
            print(f"[WARNING] S-ARIMA Forecasting Error: {e}. Defaulting Market Multiplier to 1.0")
            return 1.0

    def detect_drift(self, v3_1_payload: Dict[str, Any]) -> bool:
        """
        Parses the V3.1 JSON contract and calculates geometric distance 
        against the saved FCM centroids to trigger a Drift Alert.
        """
        if self.centroids is None:
            return False

        try:
            # Extract features mapped exactly to the Data Contract (V3.1)
            game_features = v3_1_payload["game_ml_features"]
            user_features = v3_1_payload["user_review_features"]
            
            # Vectorize the incoming continuous features for spatial comparison
            # Order must match the exact feature order used during Step 2 training
            live_vector = np.array([
                game_features["gameplay_addictiveness"],
                game_features["technical_polish"],
                game_features["aesthetic_appeal"],
                game_features["narrative_depth"],
                game_features["replayability"],
                game_features["viral_momentum"],
                user_features["insight_depth"],
                user_features["toxicity_level"],
                user_features["genre_expertise"],
                user_features["sentiment_consistency"]
            ]).reshape(1, -1)
            
        except KeyError as e:
            print(f"❌ DATA CONTRACT VIOLATION: Missing expected key {e} in payload.")
            return False

        # Calculate geometric distance (Euclidean) to all stored centroids
        distances = cdist(live_vector, self.centroids, metric='euclidean')
        
        # Find the distance to the nearest cluster centroid
        min_distance = np.min(distances)

        # Trigger alert if the incoming data point sits too far outside known cluster boundaries
        if min_distance > self.drift_threshold:
            print("=" * 50)
            print("🚨 DRIFT ALERT TRIGGERED 🚨")
            print(f"Live telemetry deviated past strict threshold!")
            print(f"Geometric Distance: {min_distance:.3f} (Threshold: {self.drift_threshold})")
            print(f"Game ID: {v3_1_payload['interaction_metadata'].get('game_id', 'UNKNOWN')}")
            print("=" * 50)
            return True
            
        return False

# ==========================================
# Example Pipeline Execution
# ==========================================
if __name__ == "__main__":
    # Initialize the Step 4 Monitor with a highly sensitive threshold
    monitor = DriftMonitor(drift_threshold=0.7)
    
    # 1. Market Multiplier Forecasting
    # Mocking an upward trend in weekly viral_momentum (e.g., 24 weeks of data)
    historical_momentum = [
        0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.06, 0.08,
        0.10, 0.12, 0.12, 0.14, 0.15, 0.18, 0.20, 0.22,
        0.25, 0.28, 0.35, 0.40, 0.45, 0.50, 0.60, 0.85
    ]
    market_mult = monitor.calculate_market_multiplier(historical_momentum)
    print(f"Generated Market Multiplier: {market_mult}x")
    
    # 2. V3.1 Contract Parsing & Drift Detection
    live_stream_payload = {
        "interaction_metadata": {
            "user_id": "usr_992x",
            "game_id": "game_alpha_protocol",
            "timestamp": "2026-05-01T17:57:32Z",
            "developer_email": "dev@studio.com",
            "primary_genre": "RPG"
        },
        "game_ml_features": {
            "gameplay_addictiveness": 0.88,
            "technical_polish": 0.20,     # Noticeable drop in polish
            "aesthetic_appeal": 0.95,
            "narrative_depth": 0.90,
            "replayability": 0.85,
            "viral_momentum": 0.92
        },
        "user_review_features": {
            "insight_depth": 0.75,
            "toxicity_level": 0.80,       # Spike in toxicity
            "genre_expertise": 0.90,
            "sentiment_consistency": 0.40
        }
    }
    
    # Check for spatial deviation against the FCM centroids
    is_drifting = monitor.detect_drift(live_stream_payload)
    print(f"Pipeline Action - Is Drifting: {is_drifting}")