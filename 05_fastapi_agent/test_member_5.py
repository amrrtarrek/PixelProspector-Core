import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def run_test(name, payload):
    print(f"\n[TEST] {name}...")
    try:
        r = requests.post(f"{BASE_URL}/v1/predict", json=payload)
        if r.status_code == 200:
            res = r.json()
            path = res.get('decision_path', 'N/A')
            score = res.get('intelligent_score', 'N/A')
            print(f"  [SUCCESS] Status: {res.get('status', 'OK')}")
            print(f"  [SUCCESS] Path taken: {path}")
            print(f"  [SUCCESS] Intelligent Score: {score}")
            return res
        else:
            print(f"  [ERROR] Status {r.status_code}: {r.text}")
            return None
    except Exception as e:
        print(f"  [ERROR] Connection Failed: {e}")
        return None

def test_member_5_suite():
    # 1. Triage Rejection (Toxicity)
    payload_toxic = {
        "user_review_features": {"toxicity_level": 0.95, "insight_depth": 0.5},
        "interaction_metadata": {"game_id": "T1"}
    }
    res = run_test("Triage Rejection (Toxicity)", payload_toxic)
    if res and res.get("status") == "Rejected":
        print("  [PASSED] Triage caught high toxicity.")

    # 2. Triage Rejection (Low Insight)
    payload_low_insight = {
        "user_review_features": {"toxicity_level": 0.05, "insight_depth": 0.05},
        "interaction_metadata": {"game_id": "T2"}
    }
    res = run_test("Triage Rejection (Low Insight)", payload_low_insight)
    if res and res.get("status") == "Rejected":
        print("  [PASSED] Triage caught low insight.")

    # 3. Path 1: Direct Dispatch (High signals)
    payload_high = {
        "user_review_features": {"toxicity_level": 0.05, "insight_depth": 0.9},
        "intelligent_score_signals": {"S_dynamic": 0.9, "Gap_SVM": 0.9, "Mu_geometric": 0.9, "ARIMA_multiplier": 1.1, "SHAP_cosine": 0.9},
        "game_ml_features": {"narrative": 0.9},
        "interaction_metadata": {"game_id": "P1"}
    }
    res = run_test("Path 1: Direct Dispatch", payload_high)
    if res and res.get("decision_path") == "Direct Dispatch":
        print("  [PASSED] Correctly dispatched.")

    # 4. Path 2: SHAP Re-check (Low cosine)
    payload_shap_low = {
        "user_review_features": {"toxicity_level": 0.05, "insight_depth": 0.9},
        "intelligent_score_signals": {"S_dynamic": 0.9, "Gap_SVM": 0.9, "Mu_geometric": 0.9, "ARIMA_multiplier": 1.1, "SHAP_cosine": 0.4},
        "game_ml_features": {"narrative": 0.9},
        "interaction_metadata": {"game_id": "P2"}
    }
    res = run_test("Path 2: SHAP Re-check", payload_shap_low)
    if res and res.get("decision_path") == "SHAP Re-check":
        print("  [PASSED] Correctly flagged for re-check.")

    # 5. Path 7: Below Minimum Threshold (Low score)
    # Calculation: (0.05*0.4) + (0.05*0.2) + (0.05*0.1) + (0.8*0.2) + (0.1*0.1) 
    # = 0.02 + 0.01 + 0.005 + 0.16 + 0.01 = 0.205 (< 0.3)
    payload_low_score = {
        "user_review_features": {"toxicity_level": 0.05, "insight_depth": 0.9},
        "intelligent_score_signals": {"S_dynamic": 0.05, "Gap_SVM": 0.05, "Mu_geometric": 0.05, "ARIMA_multiplier": 0.8, "SHAP_cosine": 0.1},
        "game_ml_features": {"narrative": 0.1},
        "interaction_metadata": {"game_id": "P7"}
    }
    res = run_test("Path 7: Below Minimum Threshold", payload_low_score)
    if res and res.get("decision_path") == "Below Minimum Threshold":
        print("  [PASSED] Correctly ignored flop.")

    # 6. Path 5: RAG Tie (Simulated 50/50)
    payload_rag_tie = {
        "user_review_features": {"toxicity_level": 0.05, "insight_depth": 0.9},
        "intelligent_score_signals": {"S_dynamic": 0.5, "Gap_SVM": 0.5, "Mu_geometric": 0.5, "ARIMA_multiplier": 1.0, "SHAP_cosine": 0.8, "force_rag_tie": True},
        "game_ml_features": {"narrative": 0.5},
        "interaction_metadata": {"game_id": "P5"}
    }
    res = run_test("Path 5: RAG Tie", payload_rag_tie)
    if res and res.get("decision_path") == "RAG Tie":
        print("  [PASSED] Correctly identified RAG tie.")

    # 7. Path 6: RAG Zero-Success (Simulated 0 neighbors)
    payload_rag_zero = {
        "user_review_features": {"toxicity_level": 0.05, "insight_depth": 0.9},
        "intelligent_score_signals": {"S_dynamic": 0.5, "Gap_SVM": 0.5, "Mu_geometric": 0.5, "ARIMA_multiplier": 1.0, "SHAP_cosine": 0.8, "force_rag_zero": True},
        "game_ml_features": {"narrative": 0.5},
        "interaction_metadata": {"game_id": "P6"}
    }
    res = run_test("Path 6: RAG Zero-Success", payload_rag_zero)
    if res and res.get("decision_path") == "RAG Zero-Success":
        print("  [PASSED] Correctly identified zero success.")

    # 8. Path 4: Human Review (High score, low confidence fallback)
    # Since shap_cos < 0.5 is SHAP Re-check, and score > 0.8 & shap > 0.8 is Direct Dispatch,
    # Human Review is the fallback for everything else that isn't RAG Retrieval (0.3-0.8).
    # So if score > 0.8 but shap is 0.6, it should be Human Review.
    payload_human = {
        "user_review_features": {"toxicity_level": 0.05, "insight_depth": 0.9},
        "intelligent_score_signals": {"S_dynamic": 0.9, "Gap_SVM": 0.9, "Mu_geometric": 0.9, "ARIMA_multiplier": 1.1, "SHAP_cosine": 0.6},
        "game_ml_features": {"narrative": 0.9},
        "interaction_metadata": {"game_id": "P4"}
    }
    res = run_test("Path 4: Human Review", payload_human)
    if res and res.get("decision_path") == "Human Review":
        print("  [PASSED] Correctly flagged for human audit.")

    # 9. Bonus Check: LangChain Dynamic Action & Agents
    print("\n[BONUS CHECK] Verifying Generative Content...")
    if res:
        agents = res.get("agents", {})
        if agents.get("community") and agents.get("investor"):
            print("  [SUCCESS] Multi-Agent roles populated.")
        if res.get("action_plan"):
            print(f"  [SUCCESS] Dynamic Action Plan generated: {res.get('action_plan')[:60]}...")

if __name__ == "__main__":
    print("PixelProspector Member 5 - Role Validation Suite")
    print("===============================================")
    test_member_5_suite()
    print("\n===============================================")
    print("Validation Complete.")
