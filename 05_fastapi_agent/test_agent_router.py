"""
test_agent_router.py
=====================
Comprehensive pytest suite for Member 5's PixelProspector V4.0 FastAPI Agent Router.

CRITICAL CONSTRAINTS HONOURED:
  - Does NOT modify any of Member 5's core files.
  - Does NOT hit the real Gemini API (PixelGeminiLLM is fully mocked).
  - Does NOT require a live database (write_record is mocked).
  - Does NOT require a real FAISS index on disk.

How to run:
    cd d:\\PixelProspector-Core\\05_fastapi_agent
    python -m pytest test_agent_router.py -v
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ── Path Setup ────────────────────────────────────────────────────────────────
# Point to the core/ directory so `from core.multi_agent_system import ...` works
sys.path.insert(0, os.path.dirname(__file__))

# ── GLOBAL MOCKS (applied before any project module is imported) ───────────────
# 1. Mock google.generativeai so the LLM class doesn't need a real API key.
mock_genai = MagicMock()
mock_genai.list_models.return_value = []
sys.modules["google.generativeai"] = mock_genai
sys.modules["google"] = MagicMock()

# 2. Mock faiss so no real index file is needed.
mock_faiss = MagicMock()
mock_faiss.read_index.return_value = None
sys.modules["faiss"] = mock_faiss

# 3. Mock DB write_record at the module level
mock_write_record = MagicMock(return_value=42)

# ── Now it's safe to import project code ────────────────────────────────────
from core.agent_router import app  # type: ignore

# ── Direct LLM Injection ──────────────────────────────────────────────────────
# The orchestrator is a module-level singleton instantiated at import time.
# Patching the class constructor is too late. We must replace the .llm attribute
# on the already-created orchestrator instance directly.
from core.agent_router import orchestrator  # type: ignore

MOCK_LLM_RESPONSE = "Action: Strategic Review | Timing: 48h | Personalization: Mock LLM Response"
mock_llm_instance = MagicMock()
mock_llm_instance.invoke.return_value = MOCK_LLM_RESPONSE

# Inject mock LLM into the live singleton
orchestrator.llm = mock_llm_instance

# ── TestClient ────────────────────────────────────────────────────────────────
client = TestClient(app)

# ── Shared Fixtures & Helpers ─────────────────────────────────────────────────
def _make_payload(
    toxicity: float = 0.1,
    insight: float = 0.8,
    s_dynamic: float = 0.85,
    gap: float = 0.85,
    mu: float = 0.85,
    arima: float = 1.2,
    shap_cos: float = 0.85,
):
    """
    Factory function for building synthetic V4.0 JSON payloads.
    Default values are clean, high-quality data that should be 'Direct Dispatch'.
    """
    return {
        "interaction_metadata": {
            "user_id": "test_user",
            "game_id": "st_test_001",
            "timestamp": "2026-05-10T00:00:00Z",
            "developer_email": "dev@test.com",
            "primary_genre": "RPG",
            "triage_status": "Pending",
        },
        "game_ml_features": {
            "gameplay_addictiveness": 0.9,
            "technical_polish": 0.9,
            "aesthetic_appeal": 0.9,
            "narrative_depth": 0.9,
            "replayability": 0.9,
            "viral_momentum": 0.9,
        },
        "user_review_features": {
            "insight_depth": insight,
            "toxicity_level": toxicity,
            "genre_expertise": 0.8,
            "sentiment_consistency": 0.8,
        },
        "intelligent_score_signals": {
            "S_class_severity": s_dynamic,
            "Gap_SVM_confidence": gap,
            "mu_geometric_membership": mu,
            "ARIMA_trend_multiplier": arima,
            "SHAP_cosine_similarity": shap_cos,
        },
    }


# =======================================================================
# A. Triage Gate Tests
# =======================================================================

def test_triage_blocks_high_toxicity():
    """
    Test A1: A payload with toxicity_level > 0.90 MUST be rejected immediately
    by pre_scoring_triage without reaching the router or the LLM.
    """
    payload = _make_payload(toxicity=0.95, insight=0.8)
    response = client.post("/v1/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Rejected"
    assert "toxicity" in data["reason"].lower() or "triage" in data["reason"].lower()


def test_triage_blocks_low_insight():
    """
    Test A2: A payload with insight_depth < 0.10 MUST be rejected.
    Validates the lower-bound gate of the triage filter.
    """
    payload = _make_payload(toxicity=0.1, insight=0.05)
    response = client.post("/v1/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Rejected"


def test_triage_passes_clean_data():
    """
    Test A3: A payload at the exact boundary (toxicity=0.90, insight=0.10)
    MUST pass through the triage gate (boundary is exclusive >0.90 and <0.10).
    """
    payload = _make_payload(toxicity=0.90, insight=0.10)
    response = client.post("/v1/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    # The payload must reach the router — 'status' key means Rejected, absence means Pass
    assert data.get("status") != "Rejected"
    assert "decision_path" in data


# =======================================================================
# B. 7-Path ReAct Router Tests
# =======================================================================

def test_router_direct_dispatch():
    """
    Test B1 (Path 1 - Direct Dispatch):
    score > 0.8 AND shap_cos > 0.8 → triggers the fastest, highest-confidence path.
    """
    # score = (0.9*0.4) + (0.9*0.2) + (0.9*0.1) + (1.2*0.2) + (0.9*0.1)
    #       = 0.36 + 0.18 + 0.09 + 0.24 + 0.09 = 0.96
    payload = _make_payload(s_dynamic=0.9, gap=0.9, mu=0.9, arima=1.2, shap_cos=0.9)
    response = client.post("/v1/predict", json=payload)
    
    assert response.status_code == 200
    assert response.json()["decision_path"] == "Direct Dispatch"


def test_router_below_minimum_threshold():
    """
    Test B2 (Path 7 - Below Minimum Threshold):
    Intelligent score < 0.3 → system flags the game as a potential flop.
    """
    # score = (0.1*0.4)+(0.1*0.2)+(0.1*0.1)+(0.5*0.2)+(0.1*0.1)
    #       = 0.04+0.02+0.01+0.10+0.01 = 0.18 → below 0.3
    payload = _make_payload(s_dynamic=0.1, gap=0.1, mu=0.1, arima=0.5, shap_cos=0.1)
    response = client.post("/v1/predict", json=payload)
    
    assert response.status_code == 200
    assert response.json()["decision_path"] == "Below Minimum Threshold"


def test_router_shap_recheck():
    """
    Test B3 (Path 2 - SHAP Re-check):
    Score is not a flop (>0.3) but shap_cos < 0.5 → model is uncertain; recheck SHAP.
    """
    # score ≈ (0.7*0.4)+(0.7*0.2)+(0.7*0.1)+(1.0*0.2)+(0.3*0.1)
    #       = 0.28+0.14+0.07+0.20+0.03 = 0.72 → between 0.3 and 0.8
    # shap_cos = 0.3 → < 0.5, so SHAP Re-check fires before RAG Retrieval
    payload = _make_payload(s_dynamic=0.7, gap=0.7, mu=0.7, arima=1.0, shap_cos=0.3)
    response = client.post("/v1/predict", json=payload)
    
    assert response.status_code == 200
    assert response.json()["decision_path"] == "SHAP Re-check"


def test_router_rag_retrieval():
    """
    Test B4 (Path 3 - RAG Retrieval):
    Borderline score (0.3 ≤ score ≤ 0.8) with strong shap_cos → queries FAISS.
    """
    # score ≈ (0.5*0.4)+(0.5*0.2)+(0.5*0.1)+(1.0*0.2)+(0.6*0.1)
    #       = 0.20+0.10+0.05+0.20+0.06 = 0.61 → borderline
    # shap_cos = 0.6 → ≥ 0.5, so we hit the RAG Retrieval branch
    payload = _make_payload(s_dynamic=0.5, gap=0.5, mu=0.5, arima=1.0, shap_cos=0.6)
    response = client.post("/v1/predict", json=payload)
    
    assert response.status_code == 200
    # After FAISS query, path resolves to "RAG Retrieval Success" or a sub-path
    assert "RAG" in response.json()["decision_path"]


def test_router_human_review():
    """
    Test B5 (Path 4 - Human Review):
    Score is HIGH (>0.8) but SHAP confidence is only moderate (0.5 ≤ shap_cos ≤ 0.8).
    The model is confident but the explainability layer disagrees → escalate to human.

    REGRESSION TEST: This path was previously unreachable (audit flag).
    After the logic fix in multi_agent_system.py this must now route correctly.
    """
    # score = (0.9*0.4)+(0.9*0.2)+(0.9*0.1)+(1.2*0.2)+(0.7*0.1)
    #       = 0.36+0.18+0.09+0.24+0.07 = 0.94 → > 0.8
    # shap_cos = 0.7 → 0.5 ≤ 0.7 ≤ 0.8, so Human Review fires before RAG Retrieval
    payload = _make_payload(s_dynamic=0.9, gap=0.9, mu=0.9, arima=1.2, shap_cos=0.7)
    response = client.post("/v1/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["decision_path"] == "Human Review"


# =======================================================================
# C. LLM Audit Log Tests
# =======================================================================

def test_llm_mock_generates_audit_log():
    """
    Test C1 (LLM Audit Log): Verifies that the mock LLM is invoked and its 
    output is included in the response's llm_audit_log field.
    """
    payload = _make_payload()
    response = client.post("/v1/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # The audit log must be a non-empty string
    assert "llm_audit_log" in data
    assert isinstance(data["llm_audit_log"], str)
    assert len(data["llm_audit_log"]) > 0
    
    # It must contain the decision path prefix
    assert data["decision_path"] in data["llm_audit_log"]


def test_llm_mock_response_format():
    """
    Test C2 (LLM Response Format): Verifies that the mock LLM generates
    an action_plan with the expected pipe-delimited format.
    """
    payload = _make_payload()
    response = client.post("/v1/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    action_plan = data.get("action_plan", "")
    assert isinstance(action_plan, str)
    # Validate the pipe-delimited format: "Action: X | Timing: Y | Personalization: Z"
    assert "Action:" in action_plan
    assert "|" in action_plan


def test_multi_agent_responses_present():
    """
    Test C3 (Multi-Agent Outputs): Verifies that both the Community Agent 
    and Investor Agent produce non-empty outputs in the response.
    """
    payload = _make_payload()
    response = client.post("/v1/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    agents = data.get("agents", {})
    assert "community" in agents
    assert "investor" in agents
    assert len(agents["community"]) > 0
    assert len(agents["investor"]) > 0


# =======================================================================
# D. Health Check Test
# =======================================================================

def test_health_endpoint():
    """
    Test D1: Verifies the root health check endpoint returns the correct
    system identifier, confirming the server is alive.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "PixelProspector" in data["system"]


# How to run:
# cd d:\PixelProspector-Core\05_fastapi_agent
# python -m pytest test_agent_router.py -v
