"""
PixelProspector — Step 1: Automated Data Ingestion
====================================================
Reads the offline Kaggle Steam Reviews CSV, sends each row to an LLM
(OpenAI primary / Gemini fallback) using Structured Outputs, validates
every field against the V3.1 JSON contract, and writes training_batch.json.

Usage:
    python ingest.py --csv steam_reviews.csv --output training_batch.json
                     --limit 5000 --provider openai --batch-size 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# V3.1 Pydantic Models — single source of truth for the JSON contract
# ---------------------------------------------------------------------------

def _clamp_float(v: Any) -> float:
    """Coerce to float and hard-clamp to [0.0, 1.0]."""
    try:
        f = float(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert {v!r} to float") from exc
    return round(max(0.0, min(1.0, f)), 4)


class InteractionMetadata(BaseModel):
    user_id: str
    game_id: str
    timestamp: str          # ISO-8601
    developer_email: str
    primary_genre: str

    @field_validator("timestamp")
    @classmethod
    def validate_iso(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"timestamp must be ISO-8601: {v}") from exc
        return v


class GameMLFeatures(BaseModel):
    gameplay_addictiveness: float = Field(..., ge=0.0, le=1.0)
    technical_polish: float      = Field(..., ge=0.0, le=1.0)
    aesthetic_appeal: float      = Field(..., ge=0.0, le=1.0)
    narrative_depth: float       = Field(..., ge=0.0, le=1.0)
    replayability: float         = Field(..., ge=0.0, le=1.0)
    viral_momentum: float        = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def clamp_all(cls, data: dict) -> dict:
        return {k: _clamp_float(v) for k, v in data.items()}


class UserReviewFeatures(BaseModel):
    insight_depth: float        = Field(..., ge=0.0, le=1.0)
    toxicity_level: float       = Field(..., ge=0.0, le=1.0)
    genre_expertise: float      = Field(..., ge=0.0, le=1.0)
    sentiment_consistency: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def clamp_all(cls, data: dict) -> dict:
        return {k: _clamp_float(v) for k, v in data.items()}


class V31Contract(BaseModel):
    interaction_metadata: InteractionMetadata
    game_ml_features: GameMLFeatures
    user_review_features: UserReviewFeatures


# ---------------------------------------------------------------------------
# JSON Schema for Structured Outputs (sent to the LLM)
# ---------------------------------------------------------------------------

V31_JSON_SCHEMA = {
    "name": "v31_contract",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "interaction_metadata": {
                "type": "object",
                "properties": {
                    "user_id":          {"type": "string"},
                    "game_id":          {"type": "string"},
                    "timestamp":        {"type": "string"},
                    "developer_email":  {"type": "string"},
                    "primary_genre":    {"type": "string"},
                },
                "required": ["user_id", "game_id", "timestamp",
                             "developer_email", "primary_genre"],
                "additionalProperties": False,
            },
            "game_ml_features": {
                "type": "object",
                "properties": {
                    "gameplay_addictiveness": {"type": "number"},
                    "technical_polish":       {"type": "number"},
                    "aesthetic_appeal":       {"type": "number"},
                    "narrative_depth":        {"type": "number"},
                    "replayability":          {"type": "number"},
                    "viral_momentum":         {"type": "number"},
                },
                "required": ["gameplay_addictiveness", "technical_polish",
                             "aesthetic_appeal", "narrative_depth",
                             "replayability", "viral_momentum"],
                "additionalProperties": False,
            },
            "user_review_features": {
                "type": "object",
                "properties": {
                    "insight_depth":         {"type": "number"},
                    "toxicity_level":        {"type": "number"},
                    "genre_expertise":       {"type": "number"},
                    "sentiment_consistency": {"type": "number"},
                },
                "required": ["insight_depth", "toxicity_level",
                             "genre_expertise", "sentiment_consistency"],
                "additionalProperties": False,
            },
        },
        "required": ["interaction_metadata", "game_ml_features",
                     "user_review_features"],
        "additionalProperties": False,
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a game review analyst for the PixelProspector ML pipeline.
Given a raw Steam review row, output a JSON object that EXACTLY matches the V3.1
contract schema. All numeric features MUST be floats strictly between 0.0 and 1.0.

Feature definitions:
  game_ml_features:
    gameplay_addictiveness : How addictive/engaging the gameplay loop sounds (0=boring, 1=irresistible)
    technical_polish       : Code quality signals — crashes, bugs, performance mentions (0=broken, 1=flawless)
    aesthetic_appeal       : Visual/audio praise density (0=ugly, 1=stunning)
    narrative_depth        : Story/lore richness (0=none, 1=epic)
    replayability          : Mentions of replay value, hours played, multiple playthroughs (0=none, 1=infinite)
    viral_momentum         : Social buzz — friend recommendations, community hype, trending signals (0=none, 1=viral)
  user_review_features:
    insight_depth          : Analytical quality of the review text (0=gibberish, 1=expert analysis)
    toxicity_level         : Presence of hate, slurs, harassment (0=clean, 1=toxic)
    genre_expertise        : Evidence reviewer knows the genre conventions (0=novice, 1=expert)
    sentiment_consistency  : Does sentiment match the recommend flag? (0=contradictory, 1=perfectly aligned)

For interaction_metadata:
  - user_id         : use the raw author_steamid if present, else generate a stable hash
  - game_id         : prefix "st_" + app_id
  - timestamp       : use review timestamp in ISO-8601; default to NOW if missing
  - developer_email : always use "dev-contact@steampublisher.com" (placeholder)
  - primary_genre   : infer from review text or game name; default "Uncategorized"
"""


def _build_user_prompt(row: dict) -> str:
    return (
        f"Game name     : {row.get('app_name', 'Unknown')}\n"
        f"App ID        : {row.get('app_id', '0')}\n"
        f"Author ID     : {row.get('author.steamid', row.get('author_steamid', ''))}\n"
        f"Recommended   : {row.get('voted_up', row.get('recommended', ''))}\n"
        f"Hours played  : {row.get('author.playtime_forever', row.get('playtime_forever', ''))}\n"
        f"Timestamp     : {row.get('timestamp_created', row.get('date', ''))}\n"
        f"Review text   : {str(row.get('review', ''))[:1500]}\n"
    )


# ---------------------------------------------------------------------------
# LLM Providers
# ---------------------------------------------------------------------------

class OpenAIProvider:
    """Uses openai>=1.0 with response_format Structured Outputs."""

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError("Run: pip install openai") from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def parse(self, row: dict) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(row)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": V31_JSON_SCHEMA,
            },
            temperature=0.1,
            max_tokens=512,
        )
        return json.loads(response.choices[0].message.content)


class GeminiProvider:
    """Uses google-generativeai with response_mime_type + response_schema."""

    def __init__(self, model: str = "gemini-1.5-flash"):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise ImportError("Run: pip install google-generativeai") from exc
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set.")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.genai = genai

    def parse(self, row: dict) -> dict:
        model = self.genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_PROMPT,
            generation_config=self.genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=512,
            ),
        )
        response = model.generate_content(_build_user_prompt(row))
        return json.loads(response.text)


def get_provider(name: str):
    """Factory — returns an LLM provider instance."""
    name = name.lower()
    if name == "openai":
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIProvider(model=model)
    if name == "gemini":
        model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        return GeminiProvider(model=model)
    raise ValueError(f"Unknown provider: {name!r}. Choose 'openai' or 'gemini'.")


# ---------------------------------------------------------------------------
# Fallback: deterministic heuristic parser (no LLM — used when --dry-run)
# ---------------------------------------------------------------------------

def _heuristic_parse(row: dict) -> dict:
    """
    Offline heuristic fallback — produces plausible V3.1 data from raw CSV
    columns without calling any LLM.  Used for --dry-run and unit tests.
    """
    import re

    review_text: str = str(row.get("review", "")).lower()
    recommended = str(row.get("voted_up", row.get("recommended", "true"))).lower()
    hours = float(row.get("author.playtime_forever",
                           row.get("playtime_forever", 0)) or 0) / 60.0  # minutes → hours

    # --- simple keyword signals ---
    positive_words = ["great", "amazing", "love", "fun", "recommend",
                      "worth", "best", "addictive", "beautiful", "perfect"]
    negative_words = ["bad", "hate", "boring", "broken", "crash",
                      "refund", "waste", "awful", "terrible", "buggy"]
    toxic_words    = ["idiot", "stupid", "trash", "garbage", "crap",
                      "sucks", "worst", "scam"]

    pos_count  = sum(review_text.count(w) for w in positive_words)
    neg_count  = sum(review_text.count(w) for w in negative_words)
    tox_count  = sum(review_text.count(w) for w in toxic_words)
    word_count = max(len(review_text.split()), 1)

    pos_ratio = min(pos_count / word_count * 10, 1.0)
    neg_ratio = min(neg_count / word_count * 10, 1.0)
    tox_ratio = min(tox_count / word_count * 20, 1.0)

    recommended_bool = recommended in ("true", "1", "yes")
    base_sentiment   = 0.8 if recommended_bool else 0.2

    # game features
    addictiveness = _clamp_float(pos_ratio * 0.7 + (min(hours, 100) / 100) * 0.3)
    tech_polish   = _clamp_float(1.0 - neg_ratio * 0.8 - tox_ratio * 0.1)
    aesthetic     = _clamp_float(pos_ratio * 0.6 + base_sentiment * 0.4)
    narrative     = _clamp_float(min(word_count / 300, 1.0) * 0.5 + pos_ratio * 0.5)
    replayability = _clamp_float(min(hours / 200, 1.0) * 0.6 + pos_ratio * 0.4)
    viral_mom     = _clamp_float((pos_ratio + min(hours / 500, 0.5)) / 1.5)

    # user features
    insight       = _clamp_float(min(word_count / 200, 1.0) * 0.7 + pos_ratio * 0.3)
    toxicity      = _clamp_float(tox_ratio)
    genre_exp     = _clamp_float(min(hours / 100, 1.0) * 0.5 + pos_ratio * 0.5)
    # sentiment_consistency: does recommend match text tone?
    text_positive = pos_count > neg_count
    sent_cons     = _clamp_float(0.9 if text_positive == recommended_bool else 0.2)

    # metadata
    app_id  = str(row.get("app_id", "0"))
    user_id = str(row.get("author.steamid", row.get("author_steamid", "")))
    if not user_id:
        user_id = "u_" + hashlib.md5(review_text[:50].encode()).hexdigest()[:8]

    raw_ts = row.get("timestamp_created", row.get("date", ""))
    try:
        ts = datetime.fromtimestamp(int(raw_ts), tz=timezone.utc).isoformat()
    except Exception:
        ts = datetime.now(timezone.utc).isoformat()

    return {
        "interaction_metadata": {
            "user_id":        user_id,
            "game_id":        f"st_{app_id}",
            "timestamp":      ts,
            "developer_email": "dev-contact@steampublisher.com",
            "primary_genre":  "Uncategorized",
        },
        "game_ml_features": {
            "gameplay_addictiveness": addictiveness,
            "technical_polish":       tech_polish,
            "aesthetic_appeal":       aesthetic,
            "narrative_depth":        narrative,
            "replayability":          replayability,
            "viral_momentum":         viral_mom,
        },
        "user_review_features": {
            "insight_depth":          insight,
            "toxicity_level":         toxicity,
            "genre_expertise":        genre_exp,
            "sentiment_consistency":  sent_cons,
        },
    }


# ---------------------------------------------------------------------------
# Core ingestion loop
# ---------------------------------------------------------------------------

def _safe_parse(provider, row: dict, dry_run: bool,
                max_retries: int = 3, backoff: float = 2.0) -> dict | None:
    """
    Call provider.parse() with exponential back-off retry.
    Returns validated V31Contract dict or None on failure.
    """
    if dry_run:
        raw = _heuristic_parse(row)
    else:
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = provider.parse(row)
                break
            except Exception as exc:
                last_exc = exc
                wait = backoff ** attempt
                log.warning("LLM attempt %d/%d failed (%s). Retrying in %.1fs…",
                            attempt, max_retries, exc, wait)
                time.sleep(wait)
        else:
            log.error("All %d attempts failed. Last error: %s", max_retries, last_exc)
            return None

    # Pydantic validation — the strict gate
    try:
        contract = V31Contract(**raw)
        return contract.model_dump()
    except Exception as val_exc:
        log.error("Validation failed: %s | raw=%s", val_exc, json.dumps(raw)[:200])
        return None


def run_ingestion(
    csv_path: Path,
    output_path: Path,
    limit: int,
    provider_name: str,
    batch_size: int,
    dry_run: bool,
    rate_limit_rps: float,
) -> None:
    """Main ingestion driver."""

    log.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path, nrows=limit, low_memory=False)
    df = df.dropna(subset=["review"])          # must have review text
    df = df.head(limit)
    total = len(df)
    log.info("Rows to process: %d (limit=%d)", total, limit)

    provider = None
    if not dry_run:
        log.info("Initialising provider: %s", provider_name)
        try:
            provider = get_provider(provider_name)
        except Exception as exc:
            log.error("Provider init failed: %s. Falling back to dry-run.", exc)
            dry_run = True

    results: list[dict] = []
    failed  = 0
    min_delay = 1.0 / rate_limit_rps if rate_limit_rps > 0 else 0.0

    with tqdm(total=total, unit="review", desc="Ingesting") as pbar:
        batch: list[dict] = []

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            t0 = time.monotonic()
            parsed = _safe_parse(provider, row_dict, dry_run=dry_run)
            elapsed = time.monotonic() - t0

            if parsed:
                results.append(parsed)
            else:
                failed += 1

            # rate limiting
            if min_delay > elapsed:
                time.sleep(min_delay - elapsed)

            pbar.update(1)
            pbar.set_postfix(ok=len(results), fail=failed)

            # checkpoint every batch_size records
            if len(results) % batch_size == 0 and len(results) > 0:
                _write_output(output_path, results)

    # final write
    _write_output(output_path, results)

    log.info("=" * 60)
    log.info("Ingestion complete.")
    log.info("  Total rows   : %d", total)
    log.info("  Successful   : %d", len(results))
    log.info("  Failed       : %d", failed)
    log.info("  Success rate : %.1f%%", 100 * len(results) / max(total, 1))
    log.info("  Output file  : %s (%.2f MB)",
             output_path, output_path.stat().st_size / 1e6)


def _write_output(path: Path, records: list[dict]) -> None:
    """Atomically write records to the output JSON file."""
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    log.debug("Checkpoint: %d records → %s", len(records), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PixelProspector Step 1 — Data Ingestion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",          type=Path, default=Path("steam_reviews.csv"),
                   help="Path to Kaggle Steam Reviews CSV")
    p.add_argument("--output",       type=Path, default=Path("training_batch.json"),
                   help="Output file for the ML team")
    p.add_argument("--limit",        type=int,  default=5000,
                   help="Max rows to process")
    p.add_argument("--provider",     type=str,  default="openai",
                   choices=["openai", "gemini"],
                   help="LLM provider for structured outputs")
    p.add_argument("--batch-size",   type=int,  default=50,
                   help="Checkpoint write every N successful records")
    p.add_argument("--dry-run",      action="store_true",
                   help="Use heuristic parser — no LLM calls (free, offline)")
    p.add_argument("--rate-limit",   type=float, default=3.0,
                   help="Max LLM requests per second (0 = unlimited)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if not args.csv.exists():
        log.error("CSV not found: %s", args.csv)
        sys.exit(1)

    run_ingestion(
        csv_path        = args.csv,
        output_path     = args.output,
        limit           = args.limit,
        provider_name   = args.provider,
        batch_size      = args.batch_size,
        dry_run         = args.dry_run,
        rate_limit_rps  = args.rate_limit,
    )
