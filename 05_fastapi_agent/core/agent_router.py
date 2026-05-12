import os
import sys
import logging
import faiss
import numpy as np
from fastapi import FastAPI
from pathlib import Path
from datetime import datetime, timezone

# ── Auto-load .env from the 05_fastapi_agent directory ──────────────────────
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=True)
        print(f"[startup] Loaded .env from {_env_path}")
except ImportError:
    pass  # python-dotenv not installed; fall back to OS env vars

# ── Resolve project root & add member directories to path ──────────────────
_HERE         = Path(__file__).resolve().parent          # 05_fastapi_agent/core/
_PROJECT_ROOT = _HERE.parent.parent                      # PixelProspector-Core/

for _p in ["01_data_ingestion", "03_supervised_ml", "04_forecasting"]:
    _full = str(_PROJECT_ROOT / _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)

# ── Import Member 1 DB write helper ───────────────────────────────────
try:
    from db import write_record  # type: ignore
except ImportError:
    def write_record(data, engine=None):
        print(f"[MOCK DB] write_record called for {data.get('interaction_metadata', {}).get('game_id')}")
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

# In-memory ring buffer for Zone 3 dashboard (last 50 decisions)
_recent_actions: list = []

# Initialize the Brain (Member 5 Orchestrator)
orchestrator = PixelProspectorOrchestrator(api_key=API_KEY, faiss_index_path=FAISS_INDEX_PATH)

# --- LOAD FAISS (From Member 2) ---
def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    return None

index = load_faiss_index()

@app.get("/")
async def root():
    return {"status": "online", "system": "PixelProspector V4.0"}

@app.get("/health")
async def health():
    return {"status": "online", "system": "PixelProspector V4.0", "version": "4.1"}

@app.get("/recent_actions")
async def recent_actions():
    """Return the last N ReAct router decisions for the dashboard Zone 3."""
    return _recent_actions


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

    # 2. Compute all 5 signals via Members 3 (SVM/SHAP) and 4 (ARIMA)
    signals = orchestrator.compute_live_signals(payload)
    intelligent_score = orchestrator.get_intelligent_score(signals)
    shap_cosine = signals.get("SHAP_cosine_similarity", 0.0)

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

    result = {
        "db_id": row_id,
        "intelligent_score": intelligent_score,
        "decision_path": decision_path,
        "llm_audit_log": payload["llm_audit_log"],
        "signals": signals,   # ← real computed signals for dashboard display
        "agents": {
            "community": community_profile,
            "investor": investor_pitch
        },
        "action_plan": final_action
    }

    # Append to Zone 3 ring buffer (keep last 50)
    _recent_actions.append({
        "timestamp":        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "game_id":          payload.get("interaction_metadata", {}).get("game_id", "?"),
        "decision_path":    decision_path,
        "intelligent_score": round(intelligent_score, 4),
        "shap_cosine":      round(shap_cosine, 4),
        "action_plan":      str(final_action)[:120],   # truncate for table display
        "db_id":            row_id,
    })
    if len(_recent_actions) > 50:
        _recent_actions.pop(0)

    return result