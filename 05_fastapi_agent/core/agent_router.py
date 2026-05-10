import os
import logging
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from typing import Dict

import sys
import os
import importlib

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import DB using importlib
try:
    db_module = importlib.import_module("01_data_ingestion.db")
    write_record = db_module.write_record
except ImportError:
    # Fallback to a no-op or simple print if DB is missing
    def write_record(data):
        print(f"[MOCK DB] Record written for {data.get('interaction_metadata', {}).get('game_id')}")
        return 999

from core.multi_agent_system import PixelProspectorOrchestrator

# --- CONFIGURATION ---
# SECURITY FIX: API key is now read from the environment.
# Set GEMINI_API_KEY in your local .env file (never commit it to git).
API_KEY = os.environ.get("GEMINI_API_KEY")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "02_unsupervised_ml", "pixel_prospector.index")
RELIABILITY_THRESHOLD = 0.75 

app = FastAPI(title="PixelProspector V4.0 Orchestrator")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Member5-Main")

# Initialize the Brain (Member 5 Orchestrator)
orchestrator = PixelProspectorOrchestrator(api_key=API_KEY, faiss_index_path=FAISS_INDEX_PATH)

# --- LOAD FAISS (From Member 2) ---
def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    return None

index = load_faiss_index()

@app.get("/")
async def health():
    return {"status": "online", "system": "PixelProspector V4.0"}

@app.post("/v1/predict")
async def predict_game_success(payload: dict):
    """
    Member 5: The final integration hub for PixelProspector.
    """
    # 1. [BONUS] Pre-scoring Triage (Absolute First Step)
    triage_result = orchestrator.pre_scoring_triage(payload)
    if triage_result == "Rejected":
        return {
            "status": "Rejected",
            "reason": "Triage: High toxicity or low insight",
            "db_id": payload.get("interaction_metadata", {}).get("db_id", "Logged")
        }

    # 2. Receive & Calculate 5-Signal Intelligent Score
    signals = payload.get("intelligent_score_signals", {})
    intelligent_score = orchestrator.get_intelligent_score(signals)
    shap_cosine = signals.get("SHAP_cosine") or signals.get("SHAP_cosine_similarity", 0.0)

    # 3. The 7-Path ReAct Router
    decision_path = orchestrator.react_router(intelligent_score, shap_cosine)
    
    rag_results = None
    if decision_path == "RAG Retrieval":
        # Triggers a query to Member 2's FAISS index
        rag_results = orchestrator.query_faiss(signals)
        # Re-evaluate with RAG results (Paths 5 & 6 handled here)
        decision_path = orchestrator.react_router(intelligent_score, shap_cosine, rag_results)

    # 4. [BONUS] Generative Explainability (SHAP to Narrative)
    raw_shap = payload.get("game_ml_features", {})
    audit_explanation = orchestrator.explain_shap(raw_shap)
    
    # Save to llm_audit_log as required
    payload["llm_audit_log"] = f"[{decision_path}] {audit_explanation}"

    # 5. [BONUS] Dynamic Action Generation (LangChain)
    final_action = orchestrator.generate_dynamic_action(decision_path, intelligent_score)

    # 6. [BONUS] Multi-Agent Architecture (LangChain)
    community_profile = orchestrator.community_agent(raw_shap)
    investor_pitch = orchestrator.investor_agent(intelligent_score, audit_explanation)

    # 7. Final DB Sync
    payload["action_plan"] = final_action
    payload["community_insight"] = community_profile
    payload["investor_pitch"] = investor_pitch
    row_id = write_record(payload)

    return {
        "db_id": row_id,
        "intelligent_score": intelligent_score,
        "decision_path": decision_path,
        "llm_audit_log": payload["llm_audit_log"],
        "agents": {
            "community": community_profile,
            "investor": investor_pitch
        },
        "action_plan": final_action
    }