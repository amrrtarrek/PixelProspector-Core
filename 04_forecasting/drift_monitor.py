"""
PixelProspector -- Step 4: Forecasting & The Outer Loop (V4.0)
===============================================================
Member 4 (The Temporal Monitor)

Implements three core responsibilities for the Agentic Flywheel:

  1. ARIMA Trend Multiplier (Signal 4)
     S-ARIMA on historical viral_momentum → multiplier > 1.0 (rising)
     or < 1.0 (declining).

  2. Dual-Signal Drift Detection
     Signal A (Spatial)   : centroid movement vs saved fcm_centroids.pkl
     Signal B (Confidence): declining Gap_SVM_confidence trend over time

  3. Autonomous Outer Loop
     Async watcher that fires ONLY when BOTH signals trip simultaneously,
     triggering subprocess retraining of Member 2 (clusters) and Member 3
     (SVMs), with a cooldown to prevent infinite trigger loops.

Interfaces (read-only — Members 1-3 are NOT modified):
  Member 1  →  InteractionLog / DriftEvent ORM (01_data_ingestion/db.py)
  Member 2  →  02_unsupervised_ml/fcm_centroids.pkl, train_clusters.py
  Member 3  →  03_supervised_ml/live_inference.py --train

Usage:
    python drift_monitor.py                # start the async outer loop
    python drift_monitor.py --once         # run a single check and exit
    python drift_monitor.py --arima-only   # compute Signal 4 and exit
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pmdarima as pm
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Resolve paths relative to THIS file so it works from any working directory
# ---------------------------------------------------------------------------
_HERE              = Path(__file__).resolve().parent
_PROJECT_ROOT      = _HERE.parent
_DATA_INGESTION    = _PROJECT_ROOT / "01_data_ingestion"
_UNSUPERVISED_DIR  = _PROJECT_ROOT / "02_unsupervised_ml"
_SUPERVISED_DIR    = _PROJECT_ROOT / "03_supervised_ml"

CENTROIDS_PATH     = _UNSUPERVISED_DIR / "fcm_centroids.pkl"
TRAIN_CLUSTERS_PY  = _UNSUPERVISED_DIR / "train_clusters.py"
LIVE_INFERENCE_PY  = _SUPERVISED_DIR   / "live_inference.py"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("drift_monitor")

# ---------------------------------------------------------------------------
# Database connection  (re-uses Member 1's setup; import kept lazy so the
# module can still be unit-tested without a live Postgres instance)
# ---------------------------------------------------------------------------

_engine = None
_SessionLocal = None


def _init_db():
    """Lazy-initialise the SQLAlchemy engine and session factory."""
    global _engine, _SessionLocal

    if _engine is not None:
        return

    # Add Member 1's package to sys.path so we can import their ORM models
    sys.path.insert(0, str(_DATA_INGESTION))
    from db import get_engine, InteractionLog, DriftEvent  # type: ignore
    from sqlalchemy.orm import sessionmaker

    _engine = get_engine()
    _SessionLocal = sessionmaker(bind=_engine)
    log.info("Database connection initialised.")


def _get_session():
    """Return a new DB session (caller must close)."""
    _init_db()
    return _SessionLocal()


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — tuneable thresholds
# ═══════════════════════════════════════════════════════════════════════════

class DriftConfig:
    """Central place for every tuneable knob in the outer loop."""

    # --- Signal A: Spatial centroid drift ---
    CENTROID_DRIFT_THRESHOLD: float = 0.35
    # Maximum allowable Euclidean distance between the live centroid
    # (computed from the most recent N records) and the saved FCM centroid.

    # --- Signal B: Confidence decline ---
    CONFIDENCE_DECLINE_THRESHOLD: float = -0.10
    # If the linear slope of Gap_SVM_confidence over the observation window
    # is more negative than this, the signal fires.

    # --- Observation windows ---
    SPATIAL_WINDOW_SIZE: int = 50
    # Number of most-recent records used to compute the live centroid.

    CONFIDENCE_WINDOW_SIZE: int = 30
    # Number of most-recent records used to fit the confidence trend line.

    ARIMA_MIN_HISTORY: int = 10
    # Minimum number of viral_momentum observations before ARIMA will run.

    # --- Outer-loop timing ---
    POLL_INTERVAL_SECONDS: int = 300       # 5 minutes between checks
    COOLDOWN_SECONDS: int = 1800           # 30-min cooldown after retrain

    # --- ARIMA ---
    ARIMA_SEASONAL_PERIOD: int = 4         # m=4 (monthly seasonality on weekly data)
    ARIMA_MULTIPLIER_FLOOR: float = 0.50
    ARIMA_MULTIPLIER_CEILING: float = 2.00


# ═══════════════════════════════════════════════════════════════════════════
#  1.  ARIMA TREND MULTIPLIER  (Signal 4)
# ═══════════════════════════════════════════════════════════════════════════

class ARIMAForecaster:
    """
    Fits a Seasonal ARIMA model on historical viral_momentum and returns
    Signal 4 — the ARIMA Trend Multiplier.

    * > 1.0  →  cluster is *rising*   (trusted / amplified)
    * < 1.0  →  cluster is *declining* (dampened)
    * = 1.0  →  neutral / insufficient data
    """

    def __init__(self, config: DriftConfig | None = None):
        self.cfg = config or DriftConfig()
        self._model: Optional[pm.ARIMA] = None

    # ── query historical data from PostgreSQL ─────────────────────────────

    def _fetch_viral_momentum(self) -> List[float]:
        """
        Query viral_momentum from interaction_logs ordered by timestamp.
        Returns a plain list of floats suitable for time-series modelling.
        """
        sys.path.insert(0, str(_DATA_INGESTION))
        from db import InteractionLog  # type: ignore

        session = _get_session()
        try:
            rows = (
                session.query(InteractionLog.viral_momentum)
                .order_by(InteractionLog.timestamp.asc())
                .all()
            )
            return [float(r[0]) for r in rows]
        finally:
            session.close()

    # ── core ARIMA logic ──────────────────────────────────────────────────

    def fit_and_forecast(
        self, history: List[float] | None = None
    ) -> float:
        """
        Fit S-ARIMA and return the trend multiplier.

        Parameters
        ----------
        history : optional pre-supplied list of viral_momentum values.
                  If None the method queries the database automatically.

        Returns
        -------
        float   The clamped ARIMA trend multiplier.
        """
        if history is None:
            history = self._fetch_viral_momentum()

        if len(history) < self.cfg.ARIMA_MIN_HISTORY:
            log.warning(
                "Insufficient history for ARIMA (%d < %d). Returning 1.0.",
                len(history), self.cfg.ARIMA_MIN_HISTORY,
            )
            return 1.0

        try:
            m_period = self.cfg.ARIMA_SEASONAL_PERIOD
            # Defensive check: auto_arima requires at least 2*m points (often more) for seasonal differencing
            use_seasonal = len(history) >= (2 * m_period)
            
            if not use_seasonal:
                log.info(
                    "ARIMA: Data length (%d) too small for seasonal m=%d. Disabling seasonal.", 
                    len(history), m_period
                )

            self._model = pm.auto_arima(
                history,
                seasonal=use_seasonal,
                m=m_period if use_seasonal else 1,
                suppress_warnings=True,
                error_action="ignore",
                stepwise=True,
                trace=False,
            )

            forecast = self._model.predict(n_periods=1)[0]

            # Baseline = mean of the last `m` observations
            baseline_window = history[-self.cfg.ARIMA_SEASONAL_PERIOD:]
            baseline = float(np.mean(baseline_window))

            if baseline <= 0.01:
                log.warning("Baseline viral_momentum ≈ 0; returning 1.0.")
                return 1.0

            raw_multiplier = forecast / baseline
            multiplier = float(np.clip(
                raw_multiplier,
                self.cfg.ARIMA_MULTIPLIER_FLOOR,
                self.cfg.ARIMA_MULTIPLIER_CEILING,
            ))

            log.info(
                "ARIMA Signal 4 → forecast=%.4f  baseline=%.4f  "
                "multiplier=%.3f (clamped [%.2f, %.2f])",
                forecast, baseline, multiplier,
                self.cfg.ARIMA_MULTIPLIER_FLOOR,
                self.cfg.ARIMA_MULTIPLIER_CEILING,
            )
            return round(multiplier, 4)

        except Exception as exc:
            log.error("S-ARIMA fitting failed: %s. Returning 1.0.", exc)
            return 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  2.  DUAL-SIGNAL DRIFT DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class DualSignalDriftDetector:
    """
    Monitors two independent drift signals and fires ONLY when both
    are active simultaneously.

    Signal A (Spatial)    — live centroid vs saved FCM centroids
    Signal B (Confidence) — declining Gap_SVM_confidence trend
    """

    def __init__(self, config: DriftConfig | None = None):
        self.cfg = config or DriftConfig()
        self._saved_centroids: Optional[Dict[str, np.ndarray]] = None

    # ── load saved centroids from Member 2 ────────────────────────────────

    def load_centroids(self) -> bool:
        """Load fcm_centroids.pkl. Returns True on success."""
        if not CENTROIDS_PATH.exists():
            log.warning("fcm_centroids.pkl not found at %s", CENTROIDS_PATH)
            return False
        try:
            with open(CENTROIDS_PATH, "rb") as f:
                self._saved_centroids = pickle.load(f)
            log.info(
                "FCM centroids loaded  (game=%s, user=%s)",
                self._saved_centroids["game_centroids"].shape,
                self._saved_centroids["user_centroids"].shape,
            )
            return True
        except Exception as exc:
            log.error("Failed to load centroids: %s", exc)
            return False

    # ── Signal A: Spatial centroid drift ───────────────────────────────────

    def _compute_live_centroid(self) -> Optional[np.ndarray]:
        """
        Compute the mean feature vector of the most recent N game records
        from the database. This is compared against the saved FCM centroids
        to measure spatial drift.
        """
        sys.path.insert(0, str(_DATA_INGESTION))
        from db import InteractionLog  # type: ignore

        session = _get_session()
        try:
            rows = (
                session.query(
                    InteractionLog.gameplay_addictiveness,
                    InteractionLog.technical_polish,
                    InteractionLog.aesthetic_appeal,
                    InteractionLog.narrative_depth,
                    InteractionLog.replayability,
                    InteractionLog.viral_momentum,
                )
                .order_by(InteractionLog.id.desc())
                .limit(self.cfg.SPATIAL_WINDOW_SIZE)
                .all()
            )
            if not rows:
                return None
            matrix = np.array(rows, dtype=np.float64)
            return matrix.mean(axis=0).reshape(1, -1)
        finally:
            session.close()

    def check_signal_a_spatial(self) -> Tuple[bool, float]:
        """
        Signal A — spatial centroid drift.

        Returns
        -------
        (fired: bool, min_distance: float)
        """
        if self._saved_centroids is None:
            if not self.load_centroids():
                return False, 0.0

        live_centroid = self._compute_live_centroid()
        if live_centroid is None:
            log.warning("Signal A: no live data available.")
            return False, 0.0

        game_centroids = self._saved_centroids["game_centroids"]
        distances = cdist(live_centroid, game_centroids, metric="euclidean")
        min_dist = float(np.min(distances))

        fired = min_dist > self.cfg.CENTROID_DRIFT_THRESHOLD
        level = "WARNING" if fired else "INFO"
        log.log(
            logging.getLevelName(level),
            "Signal A (Spatial) → min_distance=%.4f  threshold=%.4f  fired=%s",
            min_dist, self.cfg.CENTROID_DRIFT_THRESHOLD, fired,
        )
        return fired, min_dist

    # ── Signal B: Confidence decline ──────────────────────────────────────

    def _fetch_recent_confidence(self) -> List[float]:
        """
        Query the most recent Gap_SVM_confidence values ordered by time
        for trend analysis.
        """
        sys.path.insert(0, str(_DATA_INGESTION))
        from db import InteractionLog  # type: ignore

        session = _get_session()
        try:
            rows = (
                session.query(InteractionLog.Gap_SVM_confidence)
                .order_by(InteractionLog.id.desc())
                .limit(self.cfg.CONFIDENCE_WINDOW_SIZE)
                .all()
            )
            # Reverse so index 0 = oldest in the window
            return [float(r[0]) for r in reversed(rows)]
        finally:
            session.close()

    def check_signal_b_confidence(self) -> Tuple[bool, float]:
        """
        Signal B — declining Gap_SVM_confidence trend.
        Fits a simple least-squares line; if the slope is more negative
        than the threshold, the signal fires.

        Returns
        -------
        (fired: bool, slope: float)
        """
        values = self._fetch_recent_confidence()
        if len(values) < 5:
            log.warning("Signal B: insufficient data (%d points).", len(values))
            return False, 0.0

        x = np.arange(len(values), dtype=np.float64)
        y = np.array(values, dtype=np.float64)

        # Least-squares linear fit: y = slope * x + intercept
        slope, _ = np.polyfit(x, y, 1)
        slope = float(slope)

        fired = slope < self.cfg.CONFIDENCE_DECLINE_THRESHOLD
        level = "WARNING" if fired else "INFO"
        log.log(
            logging.getLevelName(level),
            "Signal B (Confidence) → slope=%.6f  threshold=%.6f  fired=%s",
            slope, self.cfg.CONFIDENCE_DECLINE_THRESHOLD, fired,
        )
        return fired, slope

    # ── combined dual-signal check ────────────────────────────────────────

    def evaluate(self) -> Dict[str, object]:
        """
        Run both signals and determine if the dual-signal condition is met.

        Returns
        -------
        dict  {
            "drift_detected": bool,
            "signal_a_fired": bool,  "centroid_distance": float,
            "signal_b_fired": bool,  "confidence_slope": float,
        }
        """
        a_fired, a_dist  = self.check_signal_a_spatial()
        b_fired, b_slope = self.check_signal_b_confidence()

        both = a_fired and b_fired

        if both:
            log.warning(
                "🚨 DUAL-SIGNAL DRIFT DETECTED — both spatial and "
                "confidence signals have fired simultaneously."
            )

        return {
            "drift_detected":    both,
            "signal_a_fired":    a_fired,
            "centroid_distance": a_dist,
            "signal_b_fired":    b_fired,
            "confidence_slope":  b_slope,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  3.  AUTONOMOUS OUTER LOOP  (self-healing retrain + cooldown)
# ═══════════════════════════════════════════════════════════════════════════

class AutonomousOuterLoop:
    """
    Asynchronous execution loop that:
      1. Polls the drift detector on a fixed interval.
      2. When dual-signal drift fires, triggers retraining via subprocess.
      3. Enforces a cooldown window to prevent infinite retrain loops.
      4. Logs DriftEvent records to PostgreSQL for the dashboard.
    """

    def __init__(self, config: DriftConfig | None = None):
        self.cfg = config or DriftConfig()
        self.detector = DualSignalDriftDetector(config=self.cfg)
        self.forecaster = ARIMAForecaster(config=self.cfg)
        self._last_retrain_ts: float = 0.0  # epoch seconds

    # ── cooldown gate ─────────────────────────────────────────────────────

    @property
    def is_cooling_down(self) -> bool:
        elapsed = time.time() - self._last_retrain_ts
        return elapsed < self.cfg.COOLDOWN_SECONDS

    @property
    def cooldown_remaining(self) -> int:
        remaining = self.cfg.COOLDOWN_SECONDS - (time.time() - self._last_retrain_ts)
        return max(0, int(remaining))

    # ── subprocess retraining ─────────────────────────────────────────────

    async def _retrain_pipeline(self) -> bool:
        """
        Trigger the full retrain sequence via subprocess calls:
          Step 1 → Member 2: python train_clusters.py   (re-cluster + FAISS)
          Step 2 → Member 3: python live_inference.py --train  (retrain SVMs)

        Returns True if both steps succeed.
        """
        python = sys.executable
        success = True

        # ── Step 1: Re-cluster (Member 2) ─────────────────────────────────
        log.info("🔄 [RETRAIN 1/2] Invoking Member 2: %s", TRAIN_CLUSTERS_PY)
        try:
            proc = await asyncio.create_subprocess_exec(
                python, str(TRAIN_CLUSTERS_PY),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_UNSUPERVISED_DIR),
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                log.error(
                    "Member 2 retrain FAILED (rc=%d):\n%s",
                    proc.returncode, stderr.decode(errors="replace"),
                )
                success = False
            else:
                log.info("Member 2 retrain OK:\n%s", stdout.decode(errors="replace"))
        except Exception as exc:
            log.error("Failed to invoke Member 2: %s", exc)
            success = False

        # ── Step 2: Retrain SVMs (Member 3) ───────────────────────────────
        log.info("🔄 [RETRAIN 2/2] Invoking Member 3: %s --train", LIVE_INFERENCE_PY)
        try:
            proc = await asyncio.create_subprocess_exec(
                python, str(LIVE_INFERENCE_PY), "--train",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_SUPERVISED_DIR),
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                log.error(
                    "Member 3 retrain FAILED (rc=%d):\n%s",
                    proc.returncode, stderr.decode(errors="replace"),
                )
                success = False
            else:
                log.info("Member 3 retrain OK:\n%s", stdout.decode(errors="replace"))
        except Exception as exc:
            log.error("Failed to invoke Member 3: %s", exc)
            success = False

        if success:
            # Reload centroids so subsequent drift checks use the fresh ones
            self.detector.load_centroids()
            log.info("✅ Full retrain sequence completed successfully.")
        else:
            log.error("⚠️  Retrain sequence completed with errors.")

        return success

    # ── log drift event to database ───────────────────────────────────────

    def _log_drift_event(
        self,
        centroid_dist: float,
        confidence_slope: float,
        healed: bool,
    ) -> None:
        """Write a row to the drift_events table for dashboard visibility."""
        try:
            sys.path.insert(0, str(_DATA_INGESTION))
            from db import DriftEvent  # type: ignore

            session = _get_session()
            event = DriftEvent(
                detected_at=datetime.now(timezone.utc),
                centroid_drift=centroid_dist,
                gap_svm_trend=confidence_slope,
                auto_healed=healed,
                notes=(
                    "Autonomous retrain triggered and completed."
                    if healed
                    else "Drift detected; retrain failed or on cooldown."
                ),
            )
            session.add(event)
            session.commit()
            session.close()
            log.info("DriftEvent logged (healed=%s).", healed)
        except Exception as exc:
            log.error("Could not log DriftEvent to DB: %s", exc)

    # ── single iteration ──────────────────────────────────────────────────

    async def run_once(self) -> Dict[str, object]:
        """
        Execute a single drift-check + ARIMA cycle.
        Returns a summary dict of all computed signals.
        """
        log.info("=" * 60)
        log.info("  OUTER LOOP — drift check at %s", datetime.now(timezone.utc).isoformat())
        log.info("=" * 60)

        # 1. Compute Signal 4 (ARIMA trend multiplier)
        arima_multiplier = self.forecaster.fit_and_forecast()

        # 2. Evaluate dual-signal drift
        result = self.detector.evaluate()
        result["ARIMA_trend_multiplier"] = arima_multiplier

        # 3. Act on drift
        if result["drift_detected"]:
            if self.is_cooling_down:
                log.warning(
                    "🛑 Drift detected but COOLDOWN active (%ds remaining). "
                    "Skipping retrain.",
                    self.cooldown_remaining,
                )
                self._log_drift_event(
                    result["centroid_distance"],
                    result["confidence_slope"],
                    healed=False,
                )
            else:
                log.warning("🚀 Initiating autonomous retrain sequence…")
                healed = await self._retrain_pipeline()
                self._last_retrain_ts = time.time()  # start cooldown
                self._log_drift_event(
                    result["centroid_distance"],
                    result["confidence_slope"],
                    healed=healed,
                )
                # Reset ARIMA baseline after retrain
                if healed:
                    arima_multiplier = self.forecaster.fit_and_forecast()
                    result["ARIMA_trend_multiplier"] = arima_multiplier
                    log.info("ARIMA baseline reset after retrain.")
        else:
            log.info("✅ No dual-signal drift — system nominal.")

        return result

    # ── continuous async loop ─────────────────────────────────────────────

    async def run_forever(self) -> None:
        """
        Continuously poll for drift on a fixed interval.
        This is the main entry point for production deployment.
        """
        log.info(
            "Autonomous Outer Loop starting  "
            "(poll=%ds, cooldown=%ds, centroid_thresh=%.3f, "
            "confidence_thresh=%.4f)",
            self.cfg.POLL_INTERVAL_SECONDS,
            self.cfg.COOLDOWN_SECONDS,
            self.cfg.CENTROID_DRIFT_THRESHOLD,
            self.cfg.CONFIDENCE_DECLINE_THRESHOLD,
        )

        # Pre-load centroids once at startup
        self.detector.load_centroids()

        while True:
            try:
                await self.run_once()
            except Exception as exc:
                log.error("Outer loop iteration failed: %s", exc, exc_info=True)

            log.info(
                "Next check in %d seconds…", self.cfg.POLL_INTERVAL_SECONDS
            )
            await asyncio.sleep(self.cfg.POLL_INTERVAL_SECONDS)


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API  — convenience functions for other modules
# ═══════════════════════════════════════════════════════════════════════════

def compute_arima_multiplier(history: List[float] | None = None) -> float:
    """
    Standalone helper to get the ARIMA trend multiplier.
    If `history` is None, queries the database automatically.
    """
    return ARIMAForecaster().fit_and_forecast(history=history)


def check_drift() -> Dict[str, object]:
    """Standalone helper — run a single dual-signal check."""
    detector = DualSignalDriftDetector()
    detector.load_centroids()
    return detector.evaluate()


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="PixelProspector Step 4 — Forecasting & Drift Outer Loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--once", action="store_true",
        help="Run a single drift check + ARIMA cycle then exit.",
    )
    p.add_argument(
        "--arima-only", action="store_true",
        help="Compute Signal 4 (ARIMA trend multiplier) and exit.",
    )
    p.add_argument(
        "--poll-interval", type=int, default=DriftConfig.POLL_INTERVAL_SECONDS,
        help="Seconds between drift checks in continuous mode.",
    )
    p.add_argument(
        "--cooldown", type=int, default=DriftConfig.COOLDOWN_SECONDS,
        help="Seconds of cooldown after a retrain before allowing another.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    cfg = DriftConfig()
    cfg.POLL_INTERVAL_SECONDS = args.poll_interval
    cfg.COOLDOWN_SECONDS = args.cooldown

    if args.arima_only:
        # ── ARIMA-only mode ───────────────────────────────────────────────
        multiplier = ARIMAForecaster(config=cfg).fit_and_forecast()
        print(f"\nSignal 4 (ARIMA Trend Multiplier): {multiplier}")

    elif args.once:
        # ── Single-shot mode ──────────────────────────────────────────────
        loop = AutonomousOuterLoop(config=cfg)
        result = asyncio.run(loop.run_once())
        print("\n── Drift Check Summary ──")
        for k, v in result.items():
            print(f"  {k}: {v}")

    else:
        # ── Continuous outer loop (production) ────────────────────────────
        loop = AutonomousOuterLoop(config=cfg)
        try:
            asyncio.run(loop.run_forever())
        except KeyboardInterrupt:
            log.info("Outer loop stopped by user (Ctrl+C).")