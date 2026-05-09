"""
PixelProspector — Step 1: Database Architecture  (V4.0)
=======================================================
Sets up the local PostgreSQL database using SQLAlchemy.
Defines the interaction_logs table mapping all V4.0 contract fields.
Seeds the database with a variety of real game scenarios to ensure
model versatility during training (narrative-heavy, high-replayability, etc.)

Usage:
    python db.py                      # init tables + seed
    python db.py --no-seed            # init tables only
    python db.py --seed-only          # seed into existing tables
    python db.py --reset              # drop, recreate, and seed

Environment variables:
    DATABASE_URL   postgresql://user:password@localhost:5432/pixelprospector
                   (defaults to the value below if not set)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer,
    String, Text, create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("db")

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

DEFAULT_DATABASE_URL = "postgresql://postgres:password@localhost:5432/pixelprospector"

def get_database_url() -> str:
    """Read DATABASE_URL from environment or fall back to the default."""
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine():
    url = get_database_url()
    log.info("Connecting to: %s", url.split("@")[-1])  # hide credentials
    return create_engine(url, echo=False, pool_pre_ping=True)


# ---------------------------------------------------------------------------
# ORM Models — V4.0 contract mapped to flat columns
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class InteractionLog(Base):
    """
    Central table. Every ingested review and every live inference result
    lands here. Columns mirror the V4.0 JSON contract field-by-field.
    """
    __tablename__ = "interaction_logs"

    # Primary key
    id: int = Column(Integer, primary_key=True, autoincrement=True)

    # --- interaction_metadata ---
    user_id:          str   = Column(String(64),  nullable=False, index=True)
    game_id:          str   = Column(String(64),  nullable=False, index=True)
    timestamp:        str   = Column(String(40),  nullable=False)
    developer_email:  str   = Column(String(128), nullable=False)
    primary_genre:    str   = Column(String(64),  nullable=True)
    triage_status:    str   = Column(String(16),  nullable=False, default="Pass")

    # --- game_ml_features ---
    gameplay_addictiveness: float = Column(Float, nullable=False)
    technical_polish:       float = Column(Float, nullable=False)
    aesthetic_appeal:       float = Column(Float, nullable=False)
    narrative_depth:        float = Column(Float, nullable=False)
    replayability:          float = Column(Float, nullable=False)
    viral_momentum:         float = Column(Float, nullable=False)

    # --- user_review_features ---
    insight_depth:          float = Column(Float, nullable=False)
    toxicity_level:         float = Column(Float, nullable=False)
    genre_expertise:        float = Column(Float, nullable=False)
    sentiment_consistency:  float = Column(Float, nullable=False)

    # --- intelligent_score_signals (zeroed at ingest; filled at inference) ---
    S_class_severity:        float = Column(Float, nullable=False, default=0.0)
    Gap_SVM_confidence:      float = Column(Float, nullable=False, default=0.0)
    mu_geometric_membership: float = Column(Float, nullable=False, default=0.0)
    ARIMA_trend_multiplier:  float = Column(Float, nullable=False, default=0.0)
    SHAP_cosine_similarity:  float = Column(Float, nullable=False, default=0.0)

    # --- llm_audit_log ---
    llm_audit_log: str = Column(Text, nullable=True, default="")

    # --- housekeeping ---
    created_at: datetime = Column(DateTime(timezone=True),
                                  default=lambda: datetime.now(timezone.utc))

    def to_v40_dict(self) -> dict:
        """Serialise back to the V4.0 JSON contract shape."""
        return {
            "interaction_metadata": {
                "user_id":        self.user_id,
                "game_id":        self.game_id,
                "timestamp":      self.timestamp,
                "developer_email": self.developer_email,
                "primary_genre":  self.primary_genre,
                "triage_status":  self.triage_status,
            },
            "game_ml_features": {
                "gameplay_addictiveness": self.gameplay_addictiveness,
                "technical_polish":       self.technical_polish,
                "aesthetic_appeal":       self.aesthetic_appeal,
                "narrative_depth":        self.narrative_depth,
                "replayability":          self.replayability,
                "viral_momentum":         self.viral_momentum,
            },
            "user_review_features": {
                "insight_depth":          self.insight_depth,
                "toxicity_level":         self.toxicity_level,
                "genre_expertise":        self.genre_expertise,
                "sentiment_consistency":  self.sentiment_consistency,
            },
            "intelligent_score_signals": {
                "S_class_severity":        self.S_class_severity,
                "Gap_SVM_confidence":      self.Gap_SVM_confidence,
                "mu_geometric_membership": self.mu_geometric_membership,
                "ARIMA_trend_multiplier":  self.ARIMA_trend_multiplier,
                "SHAP_cosine_similarity":  self.SHAP_cosine_similarity,
            },
            "llm_audit_log": self.llm_audit_log or "",
        }


class DriftEvent(Base):
    """
    Populated by Member 4 (drift_monitor.py) when dual-signal drift fires.
    Visible in dashboard Zone 6 (System Alerts).
    """
    __tablename__ = "drift_events"

    id: int                 = Column(Integer, primary_key=True, autoincrement=True)
    detected_at: datetime   = Column(DateTime(timezone=True),
                                     default=lambda: datetime.now(timezone.utc))
    centroid_drift: float   = Column(Float, nullable=False)
    gap_svm_trend:  float   = Column(Float, nullable=False)
    auto_healed:    bool    = Column(Boolean, nullable=False, default=False)
    notes:          str     = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db(engine) -> None:
    """Create all tables if they don't already exist."""
    log.info("Creating tables (if not exist)…")
    Base.metadata.create_all(engine)
    log.info("Tables ready: %s", list(Base.metadata.tables.keys()))


def drop_db(engine) -> None:
    """Drop all managed tables."""
    log.warning("Dropping all PixelProspector tables…")
    Base.metadata.drop_all(engine)
    log.info("Tables dropped.")


# ---------------------------------------------------------------------------
# Seed data — diverse real game scenarios to test model versatility
# ---------------------------------------------------------------------------

SEED_GAMES: list[dict] = [
    # ── Narrative-heavy (The Witcher 3) ─────────────────────────────────────
    {
        "user_id": "seed_u001", "game_id": "st_292030",
        "timestamp": "2024-06-01T10:00:00+00:00",
        "developer_email": "dev-contact@steampublisher.com",
        "primary_genre": "RPG", "triage_status": "Pass",
        "gameplay_addictiveness": 0.85, "technical_polish": 0.82,
        "aesthetic_appeal": 0.90, "narrative_depth": 0.97,
        "replayability": 0.80, "viral_momentum": 0.88,
        "insight_depth": 0.80, "toxicity_level": 0.02,
        "genre_expertise": 0.85, "sentiment_consistency": 0.95,
    },
    # ── High-replayability loop (Stardew Valley) ────────────────────────────
    {
        "user_id": "seed_u002", "game_id": "st_413150",
        "timestamp": "2024-06-02T11:00:00+00:00",
        "developer_email": "dev-contact@steampublisher.com",
        "primary_genre": "Simulation", "triage_status": "Pass",
        "gameplay_addictiveness": 0.92, "technical_polish": 0.88,
        "aesthetic_appeal": 0.78, "narrative_depth": 0.55,
        "replayability": 0.97, "viral_momentum": 0.85,
        "insight_depth": 0.70, "toxicity_level": 0.01,
        "genre_expertise": 0.75, "sentiment_consistency": 0.96,
    },
    # ── Competitive / viral (CS2) ────────────────────────────────────────────
    {
        "user_id": "seed_u003", "game_id": "st_730",
        "timestamp": "2024-06-03T12:00:00+00:00",
        "developer_email": "dev-contact@steampublisher.com",
        "primary_genre": "FPS", "triage_status": "Pass",
        "gameplay_addictiveness": 0.90, "technical_polish": 0.65,
        "aesthetic_appeal": 0.72, "narrative_depth": 0.10,
        "replayability": 0.95, "viral_momentum": 0.93,
        "insight_depth": 0.60, "toxicity_level": 0.35,
        "genre_expertise": 0.80, "sentiment_consistency": 0.70,
    },
    # ── Indie gem / niche (Hollow Knight) ───────────────────────────────────
    {
        "user_id": "seed_u004", "game_id": "st_367520",
        "timestamp": "2024-06-04T09:00:00+00:00",
        "developer_email": "dev-contact@steampublisher.com",
        "primary_genre": "Metroidvania", "triage_status": "Pass",
        "gameplay_addictiveness": 0.88, "technical_polish": 0.91,
        "aesthetic_appeal": 0.94, "narrative_depth": 0.80,
        "replayability": 0.82, "viral_momentum": 0.75,
        "insight_depth": 0.85, "toxicity_level": 0.01,
        "genre_expertise": 0.90, "sentiment_consistency": 0.97,
    },
    # ── Low-quality / flop (generic shovelware) ──────────────────────────────
    {
        "user_id": "seed_u005", "game_id": "st_999991",
        "timestamp": "2024-06-05T14:00:00+00:00",
        "developer_email": "dev-contact@steampublisher.com",
        "primary_genre": "Casual", "triage_status": "Pass",
        "gameplay_addictiveness": 0.15, "technical_polish": 0.20,
        "aesthetic_appeal": 0.18, "narrative_depth": 0.05,
        "replayability": 0.12, "viral_momentum": 0.08,
        "insight_depth": 0.20, "toxicity_level": 0.05,
        "genre_expertise": 0.15, "sentiment_consistency": 0.30,
    },
    # ── Toxic reviewer (triage candidate) ───────────────────────────────────
    {
        "user_id": "seed_u006", "game_id": "st_570",
        "timestamp": "2024-06-06T16:00:00+00:00",
        "developer_email": "dev-contact@steampublisher.com",
        "primary_genre": "MOBA", "triage_status": "Rejected",
        "gameplay_addictiveness": 0.60, "technical_polish": 0.55,
        "aesthetic_appeal": 0.50, "narrative_depth": 0.20,
        "replayability": 0.70, "viral_momentum": 0.65,
        "insight_depth": 0.05,  # below 0.10 → rejected
        "toxicity_level": 0.95, # above 0.90 → rejected
        "genre_expertise": 0.30, "sentiment_consistency": 0.25,
    },
    # ── Open-world survival (Valheim) ────────────────────────────────────────
    {
        "user_id": "seed_u007", "game_id": "st_892970",
        "timestamp": "2024-06-07T08:00:00+00:00",
        "developer_email": "dev-contact@steampublisher.com",
        "primary_genre": "Survival", "triage_status": "Pass",
        "gameplay_addictiveness": 0.87, "technical_polish": 0.77,
        "aesthetic_appeal": 0.80, "narrative_depth": 0.40,
        "replayability": 0.88, "viral_momentum": 0.82,
        "insight_depth": 0.72, "toxicity_level": 0.04,
        "genre_expertise": 0.70, "sentiment_consistency": 0.90,
    },
    # ── Story-driven adventure (Disco Elysium) ───────────────────────────────
    {
        "user_id": "seed_u008", "game_id": "st_632470",
        "timestamp": "2024-06-08T15:00:00+00:00",
        "developer_email": "dev-contact@steampublisher.com",
        "primary_genre": "RPG", "triage_status": "Pass",
        "gameplay_addictiveness": 0.78, "technical_polish": 0.83,
        "aesthetic_appeal": 0.88, "narrative_depth": 0.99,
        "replayability": 0.65, "viral_momentum": 0.72,
        "insight_depth": 0.92, "toxicity_level": 0.01,
        "genre_expertise": 0.93, "sentiment_consistency": 0.97,
    },
]


def seed_db(session: Session) -> None:
    """Insert seed records if the table is empty."""
    existing = session.execute(text("SELECT COUNT(*) FROM interaction_logs")).scalar()
    if existing and existing > 0:
        log.info("Seed skipped — table already has %d rows.", existing)
        return

    log.info("Seeding %d game scenarios…", len(SEED_GAMES))
    for seed in SEED_GAMES:
        record = InteractionLog(**seed)
        session.add(record)
    session.commit()
    log.info("Seed complete.")


# ---------------------------------------------------------------------------
# Public API used by dashboard.py and other members
# ---------------------------------------------------------------------------

def get_session(engine=None) -> Session:
    """Return a new SQLAlchemy Session. Caller must close it."""
    if engine is None:
        engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def from_v40_dict(data: dict) -> InteractionLog:
    """
    Construct an InteractionLog ORM object from a V4.0 contract dict.
    Convenience for ingest pipeline writing to the DB.
    """
    meta   = data["interaction_metadata"]
    game_f = data["game_ml_features"]
    user_f = data["user_review_features"]
    sig    = data.get("intelligent_score_signals", {})

    return InteractionLog(
        user_id          = meta["user_id"],
        game_id          = meta["game_id"],
        timestamp        = meta["timestamp"],
        developer_email  = meta["developer_email"],
        primary_genre    = meta.get("primary_genre", "Uncategorized"),
        triage_status    = meta.get("triage_status", "Pass"),

        gameplay_addictiveness = game_f["gameplay_addictiveness"],
        technical_polish       = game_f["technical_polish"],
        aesthetic_appeal       = game_f["aesthetic_appeal"],
        narrative_depth        = game_f["narrative_depth"],
        replayability          = game_f["replayability"],
        viral_momentum         = game_f["viral_momentum"],

        insight_depth         = user_f["insight_depth"],
        toxicity_level        = user_f["toxicity_level"],
        genre_expertise       = user_f["genre_expertise"],
        sentiment_consistency = user_f["sentiment_consistency"],

        S_class_severity        = sig.get("S_class_severity", 0.0),
        Gap_SVM_confidence      = sig.get("Gap_SVM_confidence", 0.0),
        mu_geometric_membership = sig.get("mu_geometric_membership", 0.0),
        ARIMA_trend_multiplier  = sig.get("ARIMA_trend_multiplier", 0.0),
        SHAP_cosine_similarity  = sig.get("SHAP_cosine_similarity", 0.0),

        llm_audit_log = data.get("llm_audit_log", ""),
    )


def write_record(data: dict, engine=None) -> int:
    """
    Insert one V4.0 contract dict into interaction_logs and return the new row id.

    This is the single entry-point for Flow B (live user review submission).
    The dashboard calls this after parse_user_review() succeeds.

    Parameters
    ----------
    data   : dict   Validated V4.0 contract dict (from parse_user_review).
    engine :        Optional pre-built SQLAlchemy engine. Creates one if None.

    Returns
    -------
    int   The auto-generated primary key of the new row.
    """
    if engine is None:
        engine = get_engine()

    record = from_v40_dict(data)
    with get_session(engine) as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        row_id = record.id

    log.info(
        "write_record: inserted row id=%d | game=%s user=%s triage=%s",
        row_id,
        data["interaction_metadata"]["game_id"],
        data["interaction_metadata"]["user_id"],
        data["interaction_metadata"]["triage_status"],
    )
    return row_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PixelProspector Step 1 — Database Setup (V4.0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-seed",   action="store_true", help="Skip seed data")
    p.add_argument("--seed-only", action="store_true", help="Only run seed (tables must exist)")
    p.add_argument("--reset",     action="store_true", help="Drop and recreate all tables first")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        engine = get_engine()
    except Exception as exc:
        log.error("Could not create engine: %s", exc)
        sys.exit(1)

    if args.reset:
        drop_db(engine)

    if not args.seed_only:
        init_db(engine)

    if not args.no_seed:
        with get_session(engine) as session:
            seed_db(session)

    log.info("Database setup complete.")
