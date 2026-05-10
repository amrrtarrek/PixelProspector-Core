import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_api_health():
    print("\n[TEST] Checking API Health...")
    try:
        r = requests.get(f"{BASE_URL}/")
        if r.status_code == 200:
            print("[SUCCESS] API is Online")
        else:
            print(f"[ERROR] API returned status {r.status_code}")
    except Exception as e:
        print(f"[ERROR] Connection Failed: {e}")

def test_triage_rejection():
    print("\n[TEST] Testing Pre-Scoring Triage (Toxicity)...")
    payload = {
        "interaction_metadata": {"user_id": "tester", "game_id": "game_1", "developer_email": "test@test.com"},
        "game_ml_features": {"gameplay_addictiveness": 0.5, "technical_polish": 0.5, "aesthetic_appeal": 0.5, "narrative_depth": 0.5, "replayability": 0.5, "viral_momentum": 0.5},
        "user_review_features": {"insight_depth": 0.5, "toxicity_level": 0.95, "genre_expertise": 0.5, "sentiment_consistency": 0.5},
        "intelligent_score_signals": {"S_class_severity": 0.5, "Gap_SVM_confidence": 0.5, "mu_geometric_membership": 0.5, "ARIMA_trend_multiplier": 1.0, "SHAP_cosine_similarity": 0.5}
    }
    r = requests.post(f"{BASE_URL}/v1/predict", json=payload)
    res = r.json()
    if res.get("status") == "Rejected":
        print("[SUCCESS] Triage correctly REJECTED toxic input")
    else:
        print("[ERROR] Triage FAILED to reject toxic input")

def test_full_pipeline():
    print("\n[TEST] Testing Full AI Pipeline (Success Path)...")
    payload = {
        "interaction_metadata": {"user_id": "tester", "game_id": "game_1", "developer_email": "test@test.com", "primary_genre": "RPG"},
        "game_ml_features": {"gameplay_addictiveness": 0.9, "technical_polish": 0.9, "aesthetic_appeal": 0.9, "narrative_depth": 0.9, "replayability": 0.9, "viral_momentum": 0.9},
        "user_review_features": {"insight_depth": 0.9, "toxicity_level": 0.05, "genre_expertise": 0.9, "sentiment_consistency": 0.9},
        "intelligent_score_signals": {"S_class_severity": 0.8, "Gap_SVM_confidence": 0.8, "mu_geometric_membership": 0.8, "ARIMA_trend_multiplier": 1.1, "SHAP_cosine_similarity": 0.9}
    }
    r = requests.post(f"{BASE_URL}/v1/predict", json=payload)
    if r.status_code == 200:
        res = r.json()
        print(f"[SUCCESS] Pipeline Success! Score: {res.get('intelligent_score')}")
        print(f"[SUCCESS] Agent Response: {res.get('agents', {}).get('investor')[:100]}...")
    else:
        print(f"[ERROR] Pipeline Failed: {r.text}")

if __name__ == "__main__":
    print("PixelProspector V4.1 Automated Test Suite")
    print("=========================================")
    test_api_health()
    test_triage_rejection()
    test_full_pipeline()
    print("\n=========================================")
    print("Testing Complete.")
