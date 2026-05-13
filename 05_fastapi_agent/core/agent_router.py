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
    from db import write_record, get_engine, get_session, InteractionLog, DriftEvent  # type: ignore
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

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hashlib

app = FastAPI(title="PixelProspector V4.0 Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
    arima_val = signals.get("ARIMA_trend_multiplier", 1.0)
    
    shap_feature = "Unknown"
    if raw_shap:
        numeric_feats = {k: v for k, v in raw_shap.items() if isinstance(v, (int, float))}
        if numeric_feats:
            shap_feature = max(numeric_feats, key=lambda k: abs(numeric_feats[k]))
            
    rag_vote = rag_results.get("split", "N/A") if rag_results else "N/A"

    community_profile = orchestrator.community_agent(intelligent_score, arima_val, shap_feature, rag_vote, raw_shap)
    
    game_name = payload.get("game_name", payload.get("interaction_metadata", {}).get("game_id", "Unknown Game"))
    investor_name = os.environ.get("INVESTOR_NAME", "Valued Investor")
    investor_pitch = orchestrator.investor_agent(intelligent_score, arima_val, shap_feature, rag_vote, raw_shap, game_name, investor_name)

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

# ── New Endpoints for React Dashboard ───────────────────────────────────────

@app.get("/v1/logs")
async def get_logs(limit: int = 50, triage_filter: str = "All"):
    try:
        engine = get_engine()
        with get_session(engine) as session:
            q = session.query(InteractionLog).order_by(InteractionLog.created_at.desc())
            if triage_filter != "All":
                q = q.filter(InteractionLog.triage_status == triage_filter)
            rows = q.limit(limit).all()
            return [
                {**r.to_v40_dict(), "id": r.id, "created_at": str(r.created_at)}
                for r in rows
            ]
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return []

@app.get("/v1/drift_events")
async def get_drift_events(limit: int = 20):
    try:
        engine = get_engine()
        with get_session(engine) as session:
            rows = session.query(DriftEvent).order_by(DriftEvent.detected_at.desc()).limit(limit).all()
            return [
                {
                    "id": r.id,
                    "detected_at": str(r.detected_at),
                    "centroid_drift": r.centroid_drift,
                    "gap_svm_trend": r.gap_svm_trend,
                    "auto_healed": r.auto_healed,
                    "notes": r.notes,
                }
                for r in rows
            ]
    except Exception as e:
        logger.error(f"Error fetching drift events: {e}")
        return []

@app.get("/v1/cluster_health")
async def get_cluster_health():
    try:
        engine = get_engine()
        with get_session(engine) as session:
            # Simple avg calculation by loading recent rows (or could be pure SQL)
            rows = session.query(InteractionLog).order_by(InteractionLog.created_at.desc()).limit(100).all()
            if not rows:
                return {"game_features": {}, "user_features": {}}
            
            game_features = ["gameplay_addictiveness", "technical_polish", "aesthetic_appeal", "narrative_depth", "replayability", "viral_momentum"]
            user_features = ["insight_depth", "toxicity_level", "genre_expertise", "sentiment_consistency"]
            
            avg_game = {f: sum(getattr(r, f) for r in rows)/len(rows) for f in game_features}
            avg_user = {f: sum(getattr(r, f) for r in rows)/len(rows) for f in user_features}
            
            return {
                "game_features": {k: round(v, 3) for k, v in avg_game.items()},
                "user_features": {k: round(v, 3) for k, v in avg_user.items()}
            }
    except Exception as e:
        logger.error(f"Error computing cluster health: {e}")
        return {"game_features": {}, "user_features": {}}

class ReviewSubmission(BaseModel):
    review_text: str
    game_name: str
    user_id: str
    genre: str
    recommended: str

@app.post("/v1/submit_review")
async def submit_review(payload: ReviewSubmission):
    try:
        import sys as _sys
        import os as _os
        # Ensure 01_data_ingestion is in path
        _sys.path.insert(0, str(_PROJECT_ROOT / "01_data_ingestion"))
        from ingest import parse_user_review
        
        gen_game_id = "st_" + hashlib.md5(payload.game_name.strip().lower().encode()).hexdigest()[:8]
        gemini_live = bool(os.environ.get("GEMINI_API_KEY", ""))
        
        result = parse_user_review(
            review_text=payload.review_text.strip(),
            game_name=payload.game_name.strip(),
            game_id=gen_game_id,
            user_id=payload.user_id.strip(),
            recommended=(payload.recommended == "Yes"),
            genre=payload.genre.strip() or "Uncategorized",
            dry_run=not gemini_live,
        )
        
        if result is None:
            return {"status": "error", "message": "Analysis failed to produce a valid contract."}
            
        result["game_name"] = payload.game_name.strip()
        
        # Manually call our own predict function
        predict_res = await predict_game_success(result)
        return {"status": "success", "result": result, "predict": predict_res}
        
    except Exception as e:
        logger.error(f"Submit review failed: {e}")
        return {"status": "error", "message": str(e)}