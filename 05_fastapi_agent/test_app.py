import requests
import json

url = "http://localhost:8000/v1/predict"

payload = {
    "interaction_metadata": {
        "user_id": "test_user_99",
        "game_id": "game_valheim_001",
        "primary_genre": "Survival",
        "developer_email": "dev@studio.com"
    },
    "game_ml_features": {
        "gameplay_addictiveness": 0.88,
        "technical_polish": 0.77,
        "aesthetic_appeal": 0.80,
        "narrative_depth": 0.40,
        "replayability": 0.88,
        "viral_momentum": 0.82
    },
    "user_review_features": {
        "insight_depth": 0.72,
        "toxicity_level": 0.04,
        "genre_expertise": 0.70,
        "sentiment_consistency": 0.90
    }
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
