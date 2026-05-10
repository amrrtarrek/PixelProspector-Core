import os
import sys
import asyncio
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import numpy as np

# =======================================================================
# CRITICAL SAFETY OVERRIDE - MUST BE BEFORE ANY IMPORTS
# =======================================================================
# We dynamically inject the TEST database URL into the environment.
# Subprocesses spawned via subprocess.run() inherit os.environ, ensuring
# Members 2 and 3 also connect strictly to the TEST database.
# [INSERT YOUR POSTGRES CREDENTIALS HERE IF NEEDED]
TEST_DB_URL = "postgresql://postgres:postgres@localhost:5432/pixelprospector_TEST"
os.environ["DATABASE_URL"] = TEST_DB_URL

# Resolve paths to member directories
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "01_data_ingestion"))
sys.path.insert(0, str(PROJECT_ROOT / "04_forecasting"))

# We can now safely import project code
from db import Base, get_engine, InteractionLog  # type: ignore
from sqlalchemy.orm import sessionmaker

from drift_monitor import AutonomousOuterLoop, DriftConfig  # type: ignore

# Setup Session Factory targeting the TEST DB
engine = get_engine()
TestingSessionLocal = sessionmaker(bind=engine)

# =======================================================================
# Phase 1: Sandbox Database Initialization
# =======================================================================
@pytest.fixture(scope="module")
def sandbox_db():
    """
    Phase 1: Drops and recreates all tables in the TEST database 
    using Member 1's SQLAlchemy Base to guarantee a clean slate.
    """
    # Safety Check: Guarantee we are on the TEST database
    db_url_str = str(engine.url).upper()
    assert "TEST" in db_url_str, "CRITICAL ERROR: Refusing to drop tables. Engine is NOT pointed at TEST db."
    
    # Drop all existing tables and rebuild them fresh
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Optional Teardown (commented out so you can inspect the DB afterwards if desired)
    # Base.metadata.drop_all(bind=engine)

# =======================================================================
# Phase 2: Baseline System Setup
# =======================================================================
def test_phase_2_baseline_setup(sandbox_db):
    """
    Phase 2: Ingest 50 healthy Kaggle Steam Review rows.
    Execute initial training for Member 2 (Clusters) and Member 3 (SVM).
    """
    # 1. Database Ingestion (Bypass live LLM, use Member 1 ORM directly)
    with TestingSessionLocal() as session:
        for i in range(50):
            # Healthy values tightly clustered around 0.8
            row = InteractionLog(
                user_id=f"u_{i}",
                game_id=f"st_{i}",
                timestamp=datetime.now(timezone.utc) - timedelta(days=50 - i),
                developer_email="dev@steam.com",
                primary_genre="RPG",
                triage_status="Pass",
                gameplay_addictiveness=float(np.clip(np.random.normal(0.8, 0.05), 0, 1)),
                technical_polish=float(np.clip(np.random.normal(0.8, 0.05), 0, 1)),
                aesthetic_appeal=float(np.clip(np.random.normal(0.8, 0.05), 0, 1)),
                narrative_depth=float(np.clip(np.random.normal(0.8, 0.05), 0, 1)),
                replayability=float(np.clip(np.random.normal(0.8, 0.05), 0, 1)),
                viral_momentum=float(np.clip(np.random.normal(0.8, 0.05), 0, 1)),
                insight_depth=0.8,
                toxicity_level=0.1,
                genre_expertise=0.8,
                sentiment_consistency=0.8,
                Gap_SVM_confidence=0.80  # Strong initial SVM confidence
            )
            session.add(row)
        session.commit()
    
    python_exe = sys.executable

    # 2. Train Member 2 (Unsupervised: FCM Centroids & SHAP)
    script_m2 = PROJECT_ROOT / "02_unsupervised_ml" / "train_clusters.py"
    proc_m2 = subprocess.run([python_exe, str(script_m2)], 
                             cwd=str(script_m2.parent), 
                             capture_output=True, text=True)
    assert proc_m2.returncode == 0, f"Member 2 Training Failed:\n{proc_m2.stderr}"

    # 3. Train Member 3 (Supervised: SVM Models)
    script_m3 = PROJECT_ROOT / "03_supervised_ml" / "live_inference.py"
    proc_m3 = subprocess.run([python_exe, str(script_m3), "--train"], 
                             cwd=str(script_m3.parent), 
                             capture_output=True, text=True)
    assert proc_m3.returncode == 0, f"Member 3 Training Failed:\n{proc_m3.stderr}"

# =======================================================================
# Phase 3: Chaos Injection (Triggering Drift)
# =======================================================================
def test_phase_3_chaos_injection(sandbox_db):
    """
    Phase 3: Inject 20 "Chaos" rows mathematically designed to violently
    shift the Game centroid (Signal 1) and force Gap_SVM_confidence into
    a steep negative slope guaranteed to exceed the -0.05 threshold (Signal 2).

    Gap_SVM_confidence strategy:
      Pure linear descent from 0.9 (row 0) to 0.0 (row 19), equally spaced.
      No flatline means the linear regression slope = exactly -(0.9 / 19) ≈ -0.0474
      per unit... wait — we need steeper than -0.10. So we use 0.9 → 0.0 over only
      10 rows then NEGATIVE values clamped... Actually we step 0.0 DOWN to -1.0.
      Corrected approach: start at 0.9 and step by -0.9/9 per row so rows 0..9
      go from 0.9 to 0.0, then rows 10..19 continue negatively but get clamped.
      
      Best approach: Keep stepping beyond zero so the UNCLAMPED linear fit is steep.
      Use values: 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, ... 
      gives slope -0.047. Not enough.
      
      TRUE FIX: Don't clamp. Let values go negative — SQLAlchemy Float allows it.
      0.9 down by 0.11 per row: [0.9, 0.79, 0.68, ..., 0.9 - 19*0.11 = -1.19]
      Linear regression of these values gives slope = -0.11, guaranteed > -0.10.
      NOTE: The real CONFIDENCE_DECLINE_THRESHOLD in DriftConfig is -0.10.
    """
    with TestingSessionLocal() as session:
        for i in range(20):
            # Pure linear decline with NO clamping: 0.9 → -1.19 across 20 rows.
            # Unclamped linear values guarantee regression slope of exactly -0.11,
            # safely past CONFIDENCE_DECLINE_THRESHOLD of -0.10.
            declining_confidence = 0.9 - (i * 0.11)

            row = InteractionLog(
                user_id=f"chaos_u_{i}",
                game_id=f"chaos_st_{i}",
                timestamp=datetime.now(timezone.utc) - timedelta(hours=20 - i),
                developer_email="chaos@steam.com",
                primary_genre="RPG",
                triage_status="Pass",
                # Alternating extremes guarantee max Euclidean distance for Signal 1
                gameplay_addictiveness=0.01,
                technical_polish=0.99,
                aesthetic_appeal=0.01,
                narrative_depth=0.99,
                replayability=0.01,
                viral_momentum=0.99,
                insight_depth=0.01,
                toxicity_level=0.99,
                genre_expertise=0.01,
                sentiment_consistency=0.99,
                Gap_SVM_confidence=declining_confidence
            )
            session.add(row)
        session.commit()

# =======================================================================
# Phase 4: The Autonomous Flywheel Verification
# =======================================================================
@pytest.mark.asyncio
async def test_phase_4_flywheel_verification(sandbox_db):
    """
    Phase 4: Run the Outer Loop watcher to evaluate the chaos data.
    Asserts both drift signals trip, and mock-asserts that the system 
    attempts autonomous self-healing retraining.
    """
    cfg = DriftConfig()
    # Tightly bound the observation windows to only look at our 20 chaos rows
    cfg.SPATIAL_WINDOW_SIZE = 20      
    cfg.CONFIDENCE_WINDOW_SIZE = 20
    
    loop = AutonomousOuterLoop(config=cfg)
    
    # 1. Assert Signal 1 (Centroid Shift)
    # Note: implemented as check_signal_a_spatial in drift_monitor.py
    a_fired, a_dist = loop.detector.check_signal_a_spatial()
    assert a_fired is True, f"Signal 1 (Centroid Shift) failed to fire! Distance: {a_dist}"
    
    # 2. Assert Signal 2 (SVM Decay)
    # Note: implemented as check_signal_b_confidence in drift_monitor.py
    b_fired, b_slope = loop.detector.check_signal_b_confidence()
    assert b_fired is True, f"Signal 2 (SVM Decay) failed to fire! Slope: {b_slope}"
    
    # 3. The Final Assert: Mock subprocess to verify auto-retrain trigger
    # We patch asyncio.create_subprocess_exec (which Member 4 uses) to intercept the actual command
    with patch('asyncio.create_subprocess_exec') as mock_exec:
        # Create an AsyncMock that simulates a successful subprocess return
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"OK", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc
        
        # Execute the main autonomous evaluation cycle
        result = await loop.run_once()
        
        # Verify drift was fully registered by the evaluator
        assert result["drift_detected"] is True
        
        # Assert the Flywheel attempted to retrain Members 2 and 3
        assert mock_exec.call_count == 2
        
        # Verify Member 2 call
        m2_call = mock_exec.call_args_list[0][0]
        assert "train_clusters.py" in str(m2_call[1]), "Failed to trigger Member 2"
        
        # Verify Member 3 call
        m3_call = mock_exec.call_args_list[1][0]
        assert "live_inference.py" in str(m3_call[1]), "Failed to trigger Member 3"
        assert "--train" in str(m3_call[2]), "Failed to pass --train argument to Member 3"

# How to Run:
# 1. Ensure PostgreSQL is running locally with the credentials above
# 2. Create an empty database named 'pixelprospector_TEST' if it doesn't exist
# 3. pytest test_integration_flywheel.py -v
