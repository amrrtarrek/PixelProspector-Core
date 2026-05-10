"""
test_master_e2e.py
==================
Master End-to-End Integration Test for the PixelProspector SRA Agentic Flywheel.

This test validates the FULL vertical slice of the V4.0 architecture:
  Member 1 (DB) → Member 5 (FastAPI Router) → DB Persistence Verification

WHAT IS REAL vs MOCKED:
  REAL:  - TEST PostgreSQL database (pixelprospector_TEST)
         - Member 5's FastAPI application (TestClient, no live server)
         - Member 5's pre_scoring_triage(), react_router(), get_intelligent_score()
         - Member 1's write_record() and SQLAlchemy ORM persistence
  MOCKED: - PixelGeminiLLM.invoke() → prevents live Google Gemini API calls
           - FAISS index load → prevents dependency on a .index file on disk

CRITICAL SAFETY:
  DATABASE_URL is overridden to pixelprospector_TEST BEFORE any imports.
  The production database is NEVER touched.

How to run:
    cd d:\\PixelProspector-Core
    python -m pytest test_master_e2e.py -v -s
"""

import os
import sys
import pytest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

# =======================================================================
# CRITICAL SAFETY: Override DB URL BEFORE importing any project code
# =======================================================================
# This environment variable is consumed by db.get_database_url() which is
# called lazily, so setting it here guarantees the TEST db is always used.
os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/pixelprospector_TEST"

# =======================================================================
# Path Setup — make all member packages importable
# =======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "01_data_ingestion"))
sys.path.insert(0, str(PROJECT_ROOT / "05_fastapi_agent"))

# =======================================================================
# Pre-import System Mocks (must happen before Member 5 modules are loaded)
# =======================================================================

# 1. Stub out google.generativeai — prevents the LLM __init__ from
#    trying to connect to Google's servers at import time.
mock_genai = MagicMock()
mock_genai.list_models.return_value = []
sys.modules["google.generativeai"] = mock_genai
sys.modules["google"] = MagicMock()

# 2. Stub out faiss — prevents dependency on a compiled .index file on disk.
mock_faiss = MagicMock()
mock_faiss.read_index.return_value = None
sys.modules["faiss"] = mock_faiss

# =======================================================================
# Safe Imports (after env vars and stubs are in place)
# =======================================================================
from db import Base, get_engine, InteractionLog, get_session  # type: ignore
from sqlalchemy.orm import sessionmaker

# Boot Member 5's FastAPI app via TestClient
from core.agent_router import app, orchestrator  # type: ignore
from fastapi.testclient import TestClient

# =======================================================================
# Inject Mock LLM into the Live Orchestrator Singleton
# =======================================================================
# The orchestrator is created at module-load time. We cannot patch the
# constructor retroactively, so we directly replace its .llm attribute
# with a MagicMock that returns a realistic formatted string.
MOCK_LLM_TEXT = (
    "Action: Fast-Track Publishing | Timing: 24h | "
    "Personalization: Exceptional title — prioritise storefront placement."
)
mock_llm = MagicMock()
mock_llm.invoke.return_value = MOCK_LLM_TEXT
orchestrator.llm = mock_llm  # ← only the LLM is mocked; all routing math is REAL

# =======================================================================
# Database & TestClient Setup
# =======================================================================
test_engine = get_engine()
TestSession = sessionmaker(bind=test_engine)
client = TestClient(app)

# Unique game_id stamped with the test run timestamp so DB assertions are
# isolated even if the test table is not reset between runs.
E2E_GAME_ID = f"e2e_direct_dispatch_{int(datetime.now(timezone.utc).timestamp())}"

# =======================================================================
# Phase 1: System Boot & Data Prep
# =======================================================================
@pytest.fixture(scope="module", autouse=True)
def ensure_test_db_schema():
    """
    Phase 1 — Schema Guard:
    Ensures the TEST database has the required tables. Does NOT drop/recreate
    existing data, preserving any previously trained model artefacts (centroids,
    SVM models) that the real routing math depends on.
    """
    # Safety gate: refuse to run against the production DB
    db_url = str(test_engine.url).upper()
    assert "TEST" in db_url, (
        f"CRITICAL: ENGINE IS NOT POINTED AT THE TEST DB!\n"
        f"Current URL: {test_engine.url}\n"
        f"Refusing to execute the E2E test."
    )

    # Create tables if they don't yet exist (idempotent)
    Base.metadata.create_all(bind=test_engine)
    yield
    # No teardown — leave data for post-test inspection


# =======================================================================
# The Canonical E2E Test Payload
# =======================================================================
def build_direct_dispatch_payload() -> dict:
    """
    Constructs a realistic Steam review payload engineered to guarantee
    a 'Direct Dispatch' routing decision.

    ReAct Router condition: score > 0.8 AND shap_cos > 0.8
    Weighted score formula:
        score = S*0.4 + Gap*0.2 + Mu*0.1 + ARIMA*0.2 + SHAP*0.1

    With the values below:
        score = (0.95*0.4) + (0.90*0.2) + (0.88*0.1) + (1.30*0.2) + (0.92*0.1)
              = 0.380 + 0.180 + 0.088 + 0.260 + 0.092
              = 1.000  → clamped in practice but decisively > 0.8

    SHAP cosine = 0.92 → > 0.8, so Direct Dispatch fires.
    """
    return {
        "interaction_metadata": {
            "user_id": "e2e_test_user_master",
            "game_id": E2E_GAME_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "developer_email": "dev@e2e-studio.com",
            "primary_genre": "RPG",
            "triage_status": "Pending",
        },
        # Highly polished AAA-tier game metrics
        "game_ml_features": {
            "gameplay_addictiveness": 0.97,
            "technical_polish":       0.95,
            "aesthetic_appeal":       0.93,
            "narrative_depth":        0.96,
            "replayability":          0.94,
            "viral_momentum":         0.91,
        },
        # Expert reviewer with near-zero toxicity
        "user_review_features": {
            "insight_depth":         0.92,
            "toxicity_level":        0.02,  # well below 0.90 triage gate
            "genre_expertise":       0.95,
            "sentiment_consistency": 0.96,
        },
        # Pre-computed 5-Signal payload from Members 3 & 4
        "intelligent_score_signals": {
            "S_class_severity":        0.95,   # Signal 1
            "Gap_SVM_confidence":      0.90,   # Signal 2
            "mu_geometric_membership": 0.88,   # Signal 3
            "ARIMA_trend_multiplier":  1.30,   # Signal 4 (rising trend)
            "SHAP_cosine_similarity":  0.92,   # Signal 5
        },
    }


# =======================================================================
# The Master E2E Test
# =======================================================================
def test_master_e2e_flywheel():
    """
    THE MASTER E2E TEST — Validates the complete V4.0 architecture vertical slice.

    Lifecycle:
      Phase 2: POST a live payload to Member 5's /v1/predict endpoint.
      Phase 3: Assert all four contract obligations are met.
               Assert 1 — HTTP 200 OK
               Assert 2 — intelligent_score is present and numeric
               Assert 3 — ReAct router chose "Direct Dispatch"
               Assert 4 — Record was durably persisted in the TEST database
    """
    payload = build_direct_dispatch_payload()

    # ── Phase 2: Live API Request ─────────────────────────────────────────
    print(f"\n[E2E] Sending payload for game_id={E2E_GAME_ID} ...")
    response = client.post("/v1/predict", json=payload)
    data = response.json()
    print(f"[E2E] Response: {data}")

    # ── Assert 1: HTTP 200 OK ─────────────────────────────────────────────
    assert response.status_code == 200, (
        f"Expected 200 OK but got {response.status_code}.\n"
        f"Response body: {data}"
    )
    print("[E2E] ✅ Assert 1 PASSED — HTTP 200 OK")

    # ── Assert 2: intelligent_score is present and is a valid float ────────
    assert "intelligent_score" in data, (
        "Response is missing the 'intelligent_score' key."
    )
    score = data["intelligent_score"]
    assert isinstance(score, (int, float)), (
        f"intelligent_score is not numeric: {score!r}"
    )
    assert score > 0.0, (
        f"intelligent_score should be positive for a high-quality payload, got {score}"
    )
    print(f"[E2E] ✅ Assert 2 PASSED — intelligent_score={score}")

    # ── Assert 3: ReAct Router chose 'Direct Dispatch' ────────────────────
    assert "decision_path" in data, (
        "Response is missing the 'decision_path' key."
    )
    assert data["decision_path"] == "Direct Dispatch", (
        f"Expected 'Direct Dispatch' but router chose '{data['decision_path']}'.\n"
        f"Hint: Check that the intelligent_score ({score}) > 0.8 AND "
        f"SHAP_cosine_similarity > 0.8 in the payload."
    )
    print(f"[E2E] ✅ Assert 3 PASSED — decision_path='{data['decision_path']}'")

    # ── Assert 4: Durable persistence in the TEST database ───────────────
    # Member 5's router calls write_record(payload) as the last step.
    # We use Member 1's ORM to independently verify the row exists.
    db_id = data.get("db_id")
    assert db_id is not None, (
        "Response is missing 'db_id'. write_record() may have failed."
    )

    with get_session(test_engine) as session:
        record = session.query(InteractionLog).filter_by(game_id=E2E_GAME_ID).first()

    assert record is not None, (
        f"No row found in interaction_logs for game_id='{E2E_GAME_ID}'.\n"
        f"write_record() may have silently failed or used the wrong DB URL."
    )
    # Verify key fields were persisted correctly
    assert record.game_id == E2E_GAME_ID
    assert record.user_id == "e2e_test_user_master"
    assert record.gameplay_addictiveness == pytest.approx(0.97, abs=1e-4)
    assert record.toxicity_level == pytest.approx(0.02, abs=1e-4)

    print(
        f"[E2E] ✅ Assert 4 PASSED — row persisted in DB "
        f"(id={record.id}, game_id='{record.game_id}')"
    )

    print("\n[E2E] 🎉 MASTER E2E TEST PASSED — All 4 assertions satisfied.")
    print(f"       Flywheel lifecycle: Ingest → Triage → Score → Route → Persist")
    print(f"       Score:    {score}")
    print(f"       Path:     {data['decision_path']}")
    print(f"       DB Row:   id={record.id}")
    print(f"       LLM Log:  {data.get('llm_audit_log', '')[:80]}...")
