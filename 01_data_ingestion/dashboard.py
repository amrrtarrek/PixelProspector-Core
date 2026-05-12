"""
PixelProspector — Step 1: Live Dashboard  (V4.0)
=================================================
Streamlit frontend. Strictly read-only — pulls data from the PostgreSQL
database (via SQLAlchemy) and the FastAPI agent endpoint (Member 5).
Contains ZERO math or ML logic.

Exactly 6 required UI Zones:
  Zone 1 — Cluster Health
  Zone 2 — Live Scoring Feed
  Zone 3 — Action Dispatch
  Zone 4 — SHAP Reliability
  Zone 5 — Outcome Tracking
  Zone 6 — System Alerts

Usage:
    streamlit run dashboard.py

Environment variables:
    DATABASE_URL    postgresql://user:password@localhost:5432/pixelprospector
    FASTAPI_URL     http://localhost:8000  (Member 5's agent_router.py)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

# Force-override OS environment if Streamlit secrets contains the key
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PixelProspector Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports from our db module
# ---------------------------------------------------------------------------
try:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from db import get_engine, get_session, InteractionLog, DriftEvent
    DB_AVAILABLE = True
except Exception as _db_err:
    DB_AVAILABLE = False
    _DB_ERROR = str(_db_err)

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Custom CSS — premium dark aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0a0f1e 50%, #0d0d1a 100%);
    color: #e2e8f0;
}

/* Zone cards */
.zone-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99, 179, 237, 0.18);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}

/* Zone titles */
.zone-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 10px;
}

/* KPI metric boxes */
.metric-box {
    background: rgba(99,179,237,0.07);
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
}

/* Alert badge */
.badge-pass     { color: #68d391; font-weight: 600; }
.badge-rejected { color: #fc8181; font-weight: 600; }
.badge-alert    { color: #f6e05e; font-weight: 600; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 30, 0.85);
    border-right: 1px solid rgba(99,179,237,0.15);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/controller.png", width=60)
    st.title("PixelProspector")
    st.caption("Agentic Flywheel Dashboard · V4.0")
    st.divider()

    auto_refresh = st.toggle("Auto-refresh (10 s)", value=False)
    limit = st.slider("Records to display", min_value=10, max_value=500, value=50, step=10)
    filter_triage = st.selectbox("Triage filter", ["All", "Pass", "Rejected"])
    st.divider()
    st.caption(f"DB: `{os.environ.get('DATABASE_URL', 'localhost/pixelprospector').split('@')[-1]}`")
    st.caption(f"API: `{FASTAPI_URL}`")

# ---------------------------------------------------------------------------
# Data fetchers — pure read operations
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10, show_spinner=False)
def fetch_logs(limit: int, triage_filter: str) -> pd.DataFrame:
    """Pull interaction_logs from PostgreSQL."""
    if not DB_AVAILABLE:
        return pd.DataFrame()
    try:
        engine = get_engine()
        with get_session(engine) as session:
            q = session.query(InteractionLog).order_by(
                InteractionLog.created_at.desc()
            )
            if triage_filter != "All":
                q = q.filter(InteractionLog.triage_status == triage_filter)
            rows = q.limit(limit).all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([r.to_v40_dict() | {"id": r.id,
                                                  "created_at": str(r.created_at)}
                              for r in rows])
    except Exception as exc:
        st.warning(f"DB read error: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=10, show_spinner=False)
def fetch_drift_events(limit: int = 20) -> pd.DataFrame:
    """Pull drift events from PostgreSQL."""
    if not DB_AVAILABLE:
        return pd.DataFrame()
    try:
        engine = get_engine()
        with get_session(engine) as session:
            rows = session.query(DriftEvent).order_by(
                DriftEvent.detected_at.desc()
            ).limit(limit).all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "id": r.id,
            "detected_at":   str(r.detected_at),
            "centroid_drift": r.centroid_drift,
            "gap_svm_trend":  r.gap_svm_trend,
            "auto_healed":    r.auto_healed,
            "notes":          r.notes,
        } for r in rows])
    except Exception as exc:
        st.warning(f"Drift read error: {exc}")
        return pd.DataFrame()


def fetch_fastapi_status() -> dict:
    """Ping Member 5 FastAPI health endpoint."""
    try:
        r = requests.get(f"{FASTAPI_URL}/health", timeout=2)
        return r.json() if r.status_code == 200 else {"status": "unreachable"}
    except Exception:
        return {"status": "offline"}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding: 24px 0 8px 0;">
    <h1 style="font-size:2rem; font-weight:700; color:#63b3ed; letter-spacing:0.04em;">
        🎮 PixelProspector Command Center
    </h1>
    <p style="color:#718096; font-size:0.9rem; margin:0;">
        Autonomous Agentic Flywheel · V4.0 · Live System Monitor
    </p>
</div>
""", unsafe_allow_html=True)

# Pull data once per render
raw_df = fetch_logs(limit, filter_triage)
drift_df = fetch_drift_events()
api_status = fetch_fastapi_status()

# ---------------------------------------------------------------------------
# ── ZONE 1: Cluster Health ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------
st.markdown('<div class="zone-card">', unsafe_allow_html=True)
st.markdown('<p class="zone-title">📊 Zone 1 · Cluster Health</p>', unsafe_allow_html=True)

if raw_df.empty:
    st.info("No data loaded. Make sure the database is seeded and connected.")
else:
    # Derive cluster-level feature averages from the flat dataframe
    game_features = ["gameplay_addictiveness", "technical_polish", "aesthetic_appeal",
                     "narrative_depth", "replayability", "viral_momentum"]
    user_features = ["insight_depth", "toxicity_level", "genre_expertise", "sentiment_consistency"]

    # Flatten nested dicts if needed
    try:
        game_df = raw_df["game_ml_features"].apply(pd.Series)
        user_df = raw_df["user_review_features"].apply(pd.Series)
    except Exception:
        game_df = raw_df.reindex(columns=game_features, fill_value=0)
        user_df = raw_df.reindex(columns=user_features, fill_value=0)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("🎮 Game Cluster — Feature Averages")
        avg_game = game_df.mean().rename("Average").round(3)
        st.bar_chart(avg_game)

    with col2:
        st.caption("👤 User Cluster — Feature Averages")
        avg_user = user_df.mean().rename("Average").round(3)
        st.bar_chart(avg_user)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ── ZONE 2: Live Scoring Feed ───────────────────────────────────────────────
# ---------------------------------------------------------------------------
st.markdown('<div class="zone-card">', unsafe_allow_html=True)
st.markdown('<p class="zone-title">📡 Zone 2 · Live Scoring Feed</p>', unsafe_allow_html=True)

tab_feed, tab_submit = st.tabs(["📋 Recent Records", "✍️ Submit New Review"])

with tab_feed:
    if raw_df.empty:
        st.info("Waiting for data — make sure the database is seeded and connected.")
    else:
        try:
            sig_df = raw_df["intelligent_score_signals"].apply(pd.Series)
            meta_df = raw_df["interaction_metadata"].apply(pd.Series)
            feed_df = pd.concat([
                meta_df[["user_id", "game_id", "triage_status", "primary_genre"]],
                sig_df
            ], axis=1).head(limit)
        except Exception:
            feed_df = raw_df.head(limit)

        st.dataframe(feed_df, use_container_width=True, height=260)

        kpi_cols = st.columns(5)
        signal_labels = [
            ("S Class",  "S_class_severity"),
            ("Gap SVM",  "Gap_SVM_confidence"),
            ("μ Member", "mu_geometric_membership"),
            ("ARIMA ×",  "ARIMA_trend_multiplier"),
            ("SHAP cos", "SHAP_cosine_similarity"),
        ]
        for col, (label, key) in zip(kpi_cols, signal_labels):
            with col:
                try:
                    val = sig_df[key].mean()
                except Exception:
                    val = 0.0
                st.metric(label=label, value=f"{val:.3f}")

# ── Tab 2: Submit a new review via Gemini ───────────────────────────────────
with tab_submit:
    gemini_live = bool(os.environ.get("GEMINI_API_KEY", ""))
    if gemini_live:
        st.success("🟢 Gemini API key detected — live analysis enabled.")
    else:
        st.warning(
            "⚠️ `GEMINI_API_KEY` not set. "
            "Reviews will be analysed using the heuristic fallback. "
            "Set the key in your environment to enable real Gemini analysis."
        )

    st.markdown("##### Submit a new game review for analysis")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        game_name  = st.text_input("Game name",   placeholder="e.g. The Witcher 3",  key="rv_game_name")
        user_id    = st.text_input("Your user ID",placeholder="e.g. user_xyz",       key="rv_user_id")
    with col_right:
        genre      = st.text_input("Genre",       placeholder="e.g. RPG",            key="rv_genre")
        recommended = st.radio("Do you recommend this game?", ["Yes", "No"],
                               horizontal=True, key="rv_recommend")
        st.write("")  # spacing

    review_text = st.text_area(
        "Your review",
        placeholder="Write your review here… (minimum 20 characters)",
        height=130,
        key="rv_text",
    )

    analyse_btn = st.button("🚀 Analyse with Gemini", type="primary", key="rv_submit")

    if analyse_btn:
        # Input validation
        errors = []
        if not game_name.strip():
            errors.append("Game name is required.")
        if not user_id.strip():
            errors.append("User ID is required.")
        if len(review_text.strip()) < 20:
            errors.append("Review must be at least 20 characters.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner("Analysing with Gemini…" if gemini_live else "Running heuristic analysis…"):
                try:
                    # Generate deterministic Game ID from Game Name
                    import hashlib
                    gen_game_id = "st_" + hashlib.md5(game_name.strip().lower().encode()).hexdigest()[:8]

                    # Import parse_user_review and write_record from our own modules
                    import sys as _sys
                    import os as _os
                    _sys.path.insert(0, _os.path.dirname(__file__))
                    from ingest import parse_user_review
                    from db    import write_record, get_engine

                    result = parse_user_review(
                        review_text = review_text.strip(),
                        game_name   = game_name.strip(),
                        game_id     = gen_game_id,
                        user_id     = user_id.strip(),
                        recommended = (recommended == "Yes"),
                        genre       = genre.strip() or "Uncategorized",
                        dry_run     = not gemini_live,
                    )

                    if result is None:
                        st.error("❌ Analysis failed — could not produce a valid V4.0 contract. Check logs.")
                    else:
                        # Add game_name to payload so the orchestrator can use it for the investor email
                        result["game_name"] = game_name.strip()
                        
                        # Forward payload to FastAPI Agent so it calculates the 5 signals
                        db_msg = ""
                        res_data = {}
                        try:
                            import requests
                            res = requests.post("http://localhost:8000/v1/predict", json=result, timeout=120)

                            if res.status_code == 200:
                                res_data = res.json()
                                new_id = res_data.get("db_id", "?")
                                score  = res_data.get("intelligent_score", 0.0)
                                path   = res_data.get("decision_path", "Unknown")
                                db_msg = f"✅ **Processed & Saved!** (Row #{new_id}) | Score: **{score:.3f}** | Path: **{path}**"

                                # Merge real signals back into the displayed contract
                                if "intelligent_score_signals" in result:
                                    # Re-fetch the live record from DB to get real signals
                                    # (FastAPI updated the DB; we patch the display dict here)
                                    result["intelligent_score_signals"] = {
                                        "S_class_severity":        res_data.get("signals", {}).get("S_class_severity", result["intelligent_score_signals"].get("S_class_severity", 0)),
                                        "Gap_SVM_confidence":      res_data.get("signals", {}).get("Gap_SVM_confidence", result["intelligent_score_signals"].get("Gap_SVM_confidence", 0)),
                                        "mu_geometric_membership": res_data.get("signals", {}).get("mu_geometric_membership", result["intelligent_score_signals"].get("mu_geometric_membership", 0)),
                                        "ARIMA_trend_multiplier":  res_data.get("signals", {}).get("ARIMA_trend_multiplier", result["intelligent_score_signals"].get("ARIMA_trend_multiplier", 0)),
                                        "SHAP_cosine_similarity":  res_data.get("signals", {}).get("SHAP_cosine_similarity", result["intelligent_score_signals"].get("SHAP_cosine_similarity", 0)),
                                    }
                                result["llm_audit_log"] = res_data.get("llm_audit_log", "")
                            else:
                                db_msg = f"⚠️ Inference failed (HTTP {res.status_code}: {res.text})."

                            st.cache_data.clear()   # force Zone 2 Tab 1 to reload
                        except Exception as req_err:
                            db_msg = f"⚠️ API call failed ({req_err}). Is the FastAPI server running on port 8000?"

                        st.success(db_msg if db_msg else "✅ Analysis complete")

                        # Show inference results panel if we got a real response
                        if res_data and res_data.get("intelligent_score") is not None:
                            st.markdown("---")
                            st.markdown("#### 🧠 Inference Results")
                            r1, r2, r3 = st.columns(3)
                            r1.metric("Intelligent Score", f"{res_data.get('intelligent_score', 0):.4f}")
                            r2.metric("Decision Path",     res_data.get("decision_path", "—"))
                            r3.metric("DB Row",            f"#{res_data.get('db_id', '?')}")

                            if res_data.get("agents"):
                                with st.expander("🤖 Agent Outputs", expanded=True):
                                    st.markdown(f"**Community Agent:** {res_data['agents'].get('community', '—')}")
                                    st.markdown("**Investor Agent (Email Draft):**")
                                    st.info(res_data['agents'].get('investor', '—'))

                        # Show the V4.0 contract with real signals merged in
                        with st.expander("📄 View V4.0 JSON output (with real signals)", expanded=True):
                            import json as _json
                            st.json(_json.dumps(result, indent=2))

                        # Feature bar chart for quick visual
                        gf = result.get("game_ml_features", {})
                        uf = result.get("user_review_features", {})
                        all_scores = {**gf, **uf}
                        if all_scores:
                            score_df = pd.DataFrame(
                                list(all_scores.values()),
                                index=list(all_scores.keys()),
                                columns=["Score"],
                            )
                            st.caption("Feature scores (all should be in [0.0 – 1.0])")
                            st.bar_chart(score_df)
                        if all_scores:
                            score_df = pd.DataFrame(
                                list(all_scores.values()),
                                index=list(all_scores.keys()),
                                columns=["Score"],
                            )
                            st.caption("Feature scores (all should be in [0.0 – 1.0])")
                            st.bar_chart(score_df)

                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")

st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ── ZONE 3: Action Dispatch ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------
st.markdown('<div class="zone-card">', unsafe_allow_html=True)
st.markdown('<p class="zone-title">⚡ Zone 3 · Action Dispatch</p>', unsafe_allow_html=True)

api_col, status_col = st.columns([3, 1])
with api_col:
    st.caption("Recent ReAct router decisions (from Member 5 FastAPI)")
    _api_online = api_status.get("status") not in ("offline", "unreachable")
    try:
        recent_resp = requests.get(f"{FASTAPI_URL}/recent_actions", timeout=3)
        recent_actions = recent_resp.json() if recent_resp.status_code == 200 else []
        if isinstance(recent_actions, list) and recent_actions:
            action_df = pd.DataFrame(recent_actions)
            # Reorder columns for readability if they exist
            preferred_cols = ["timestamp", "game_id", "decision_path",
                              "intelligent_score", "shap_cosine", "action_plan", "db_id"]
            action_df = action_df[[c for c in preferred_cols if c in action_df.columns]]
            st.dataframe(action_df, use_container_width=True, height=200)
            st.caption(f"Showing {len(recent_actions)} most recent router decisions.")
        elif _api_online:
            st.info(
                "🟡 FastAPI is **online** but no predictions have been routed yet. "
                "Submit a review in Zone 2 → Tab 'Submit New Review' to trigger the ReAct router."
            )
        else:
            st.warning("Agent router is offline. Start the FastAPI server first.")
    except Exception as _z3_err:
        if _api_online:
            st.info(
                "🟡 FastAPI is **online** but `/recent_actions` returned an error. "
                f"Details: `{_z3_err}`"
            )
        else:
            st.warning("Agent router is offline or not yet implemented.")

with status_col:
    st.caption("FastAPI Status")
    if api_status.get("status") not in ("offline", "unreachable"):
        st.success(f"🟢 Online\n\n`{api_status.get('status', 'ok')}`")
    else:
        st.error(f"🔴 {api_status.get('status', 'offline').title()}")


st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ── ZONE 4: SHAP Reliability ────────────────────────────────────────────────
# ---------------------------------------------------------------------------
st.markdown('<div class="zone-card">', unsafe_allow_html=True)
st.markdown('<p class="zone-title">🔬 Zone 4 · SHAP Reliability</p>', unsafe_allow_html=True)

if not raw_df.empty:
    try:
        shap_series = raw_df["intelligent_score_signals"].apply(
            lambda x: x.get("SHAP_cosine_similarity", 0.0) if isinstance(x, dict) else 0.0
        )
        shap_df = pd.DataFrame({
            "SHAP cosine similarity": shap_series,
            "record #": range(len(shap_series)),
        }).set_index("record #")

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.caption("SHAP cosine similarity over recent records (higher = more reliable)")
            st.line_chart(shap_df)
        with col_b:
            st.caption("Reliability breakdown")
            high_rel   = (shap_series >= 0.75).sum()
            mid_rel    = ((shap_series >= 0.40) & (shap_series < 0.75)).sum()
            low_rel    = (shap_series < 0.40).sum()
            st.metric("✅ High reliability  (≥0.75)", high_rel)
            st.metric("⚠️ Medium (0.40–0.75)", mid_rel)
            st.metric("❌ Low reliability  (<0.40)", low_rel)
    except Exception as exc:
        st.info(f"SHAP data not yet available: {exc}")
else:
    st.info("Waiting for scored records…")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ── ZONE 5: Outcome Tracking ────────────────────────────────────────────────
# ---------------------------------------------------------------------------
st.markdown('<div class="zone-card">', unsafe_allow_html=True)
st.markdown('<p class="zone-title">🏆 Zone 5 · Outcome Tracking</p>', unsafe_allow_html=True)

if not raw_df.empty:
    try:
        meta_df2 = raw_df["interaction_metadata"].apply(pd.Series)
        triage_counts = meta_df2["triage_status"].value_counts()
        genre_counts  = meta_df2["primary_genre"].value_counts().head(8)

        col_t, col_g = st.columns(2)
        with col_t:
            st.caption("Triage outcomes")
            pass_count = triage_counts.get("Pass", 0)
            rej_count  = triage_counts.get("Rejected", 0)
            total = pass_count + rej_count or 1
            st.progress(pass_count / total, text=f"Pass rate: {pass_count/total:.1%}")
            st.metric("✅ Passed",   pass_count)
            st.metric("❌ Rejected", rej_count)

        with col_g:
            st.caption("Genre distribution")
            st.bar_chart(genre_counts)

    except Exception as exc:
        st.info(f"Outcome data unavailable: {exc}")
else:
    st.info("No records yet.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ── ZONE 6: System Alerts ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------
st.markdown('<div class="zone-card">', unsafe_allow_html=True)
st.markdown('<p class="zone-title">🚨 Zone 6 · System Alerts</p>', unsafe_allow_html=True)

alert_col, meta_col = st.columns([3, 1])
with alert_col:
    if drift_df.empty:
        st.success("✅ No drift events detected. System is healthy.")
    else:
        # Show any unhealed events as warnings
        unhealed = drift_df[drift_df["auto_healed"] == False]
        healed   = drift_df[drift_df["auto_healed"] == True]

        if not unhealed.empty:
            st.error(f"⚠️ **{len(unhealed)} active drift event(s)** require attention!")
        if not healed.empty:
            st.success(f"🔄 **{len(healed)} event(s)** auto-healed by the outer loop.")

        st.dataframe(drift_df, use_container_width=True, height=200)

with meta_col:
    st.caption("System Info")
    st.metric("Records loaded", len(raw_df) if not raw_df.empty else 0)
    st.metric("Drift events", len(drift_df) if not drift_df.empty else 0)
    st.caption(f"Last refresh\n{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if auto_refresh:
    time.sleep(10)
    st.cache_data.clear()
    st.rerun()
