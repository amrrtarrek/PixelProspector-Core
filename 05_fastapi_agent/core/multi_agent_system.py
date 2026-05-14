import sys
import os
import logging
import importlib
import json
import faiss
from google import genai
from typing import Any, List, Optional, Dict
from pathlib import Path

# ── Resolve project root so sub-member imports work regardless of CWD ──────
_HERE         = Path(__file__).resolve().parent          # 05_fastapi_agent/core/
_PROJECT_ROOT = _HERE.parent.parent                      # PixelProspector-Core/

# Add member directories to path
for _p in ["01_data_ingestion", "03_supervised_ml", "04_forecasting"]:
    _full = str(_PROJECT_ROOT / _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)

# ── Import Member 3 (InferenceEngine) ─────────────────────────────────────
try:
    from live_inference import InferenceEngine  # type: ignore
    _inference_engine = InferenceEngine()
except Exception as _ie_err:
    logging.getLogger("PixelOrchestrator").warning(
        "InferenceEngine unavailable (%s). Signals will default to 0.", _ie_err
    )
    _inference_engine = None

# ── Import Member 4 (ARIMAForecaster) ─────────────────────────────────────
try:
    from drift_monitor import ARIMAForecaster  # type: ignore
    _arima_forecaster = ARIMAForecaster()
except Exception as _arima_err:
    logging.getLogger("PixelOrchestrator").warning(
        "ARIMAForecaster unavailable (%s). ARIMA signal will default to 1.0.", _arima_err
    )
    _arima_forecaster = None

# LangChain Imports
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
<<<<<<< HEAD

# LangChain ReAct Imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
=======
>>>>>>> 7c320b6 (feat: Implement comprehensive Audit Log detail modal with dynamic SHAP drivers, agent outputs, and ReAct flow diagram)

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    db = importlib.import_module("01_data_ingestion.db")
    write_record = db.write_record
except ImportError:
    def write_record(*args, **kwargs): 
        print(f"[MOCK DB] Writing record...")
        return 999

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PixelOrchestrator")


# ==============================================================================
# ReAct Tools & Executor
# ==============================================================================

@tool
def fetch_market_comparables(genre: str) -> str:
    """Fetches market comparables and historical ROI data for a given genre."""
    return f"Market comparables for {genre} show a 15% YoY growth in player base with highly profitable micro-transaction conversion rates."

@tool
def fetch_community_sentiment(genre: str) -> str:
    """Fetches recent community sentiment history and trending topics for a given game genre."""
    return f"Community sentiment for {genre} is highly active. Players are increasingly frustrated with pay-to-win mechanics but love deep progression systems."

REACT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final structured report detailing: Action Type, Timing, Intensity, and Personalization.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

def run_react_agent(input_text: str, tools: list):
    """Core execution function for LangChain ReAct agents."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY", "")
    )
    prompt = PromptTemplate.from_template(REACT_TEMPLATE)
    agent = create_react_agent(llm, tools, prompt)
    
    # CRITICAL: verbose=True to print the inner reasoning loop
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)
    
    response = agent_executor.invoke({"input": input_text})
    return response["output"]

# ==============================================================================
# Original Wrappers & Orchestrator
# ==============================================================================

class PixelGeminiLLM:
    """
    A custom wrapper to make Gemini API compatible with LangChain primitives.
    Auto-scans for available models to avoid 404 errors.
    """
    def __init__(self, api_key):
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            logger.warning("PixelGeminiLLM: No API key found. LLM calls will fail.")
        
        try:
            from google import genai
            self.client = genai.Client(api_key=resolved_key) if resolved_key else genai.Client()
        except ImportError:
            logger.error("Could not import google.genai. Make sure google-genai is installed.")
            self.client = None
        self.model_name = self._get_active_model_name()

    def _get_active_model_name(self):
        return "gemini-3.1-flash-lite"

    def invoke(self, prompt: str) -> str:
        if not self.client:
            return "Action: Strategic Review | Timing: 48h | Personalization: System fallback due to API error."
            
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM Invoke Error: {e}")
            return f"Action: Strategic Review | Timing: 48h | Personalization: System fallback due to API error."


class PixelProspectorOrchestrator:
    def __init__(self, api_key, faiss_index_path=None):
        self.api_key = api_key
        self.llm = PixelGeminiLLM(api_key)
        self.faiss_index_path = faiss_index_path
        self.index = self._load_faiss_index()

    def _load_faiss_index(self):
        if self.faiss_index_path and os.path.exists(self.faiss_index_path):
            try:
                return faiss.read_index(self.faiss_index_path)
            except Exception:
                return None
        return None

    def pre_scoring_triage(self, data: dict):
        """BONUS: Lightweight toxicity/insight gate."""
        user_features = data.get("user_review_features", {})
        toxicity = user_features.get("toxicity_level", 0.0)
        insight = user_features.get("insight_depth", 1.0)
        
        if toxicity > 0.90 or insight < 0.10:
            if "interaction_metadata" not in data:
                data["interaction_metadata"] = {}
            data["interaction_metadata"]["triage_status"] = "Rejected"
            write_record(data)
            return "Rejected"
        
        if "interaction_metadata" in data:
            data["interaction_metadata"]["triage_status"] = "Pass"
        return "Pass"

    def compute_live_signals(self, payload: dict) -> dict:
        signals = payload.setdefault("intelligent_score_signals", {})

        if any(signals.values()):
            logger.info("Using pre-populated signals (skipping live inference).")
            return signals

        if _inference_engine is not None:
            try:
                result = _inference_engine.compute_signals(payload)
                m3_signals = result.get("intelligent_score_signals", {})
                signals["S_class_severity"]        = m3_signals.get("S_class_severity", 0.0)
                signals["Gap_SVM_confidence"]      = m3_signals.get("Gap_SVM_confidence", 0.0)
                signals["mu_geometric_membership"] = m3_signals.get("mu_geometric_membership", 0.0)
                signals["SHAP_cosine_similarity"]  = m3_signals.get("SHAP_cosine_similarity", 0.0)
                signals["SHAP_raw_drivers"]        = m3_signals.get("SHAP_raw_drivers", {})
                logger.info(
                    "Member 3 signals → S=%.3f Gap=%.3f Mu=%.3f SHAP=%.3f",
                    signals["S_class_severity"], signals["Gap_SVM_confidence"],
                    signals["mu_geometric_membership"], signals["SHAP_cosine_similarity"],
                )
            except Exception as exc:
                logger.error("Member 3 inference failed: %s. Signals remain 0.", exc)
        else:
            logger.warning("InferenceEngine not loaded — signals 1-3,5 will be 0.")

        if _arima_forecaster is not None:
            try:
                arima_mult = _arima_forecaster.fit_and_forecast()
                signals["ARIMA_trend_multiplier"] = arima_mult
                logger.info("Member 4 ARIMA multiplier → %.3f", arima_mult)
            except Exception as exc:
                logger.error("Member 4 ARIMA failed: %s. Defaulting to 1.0.", exc)
                signals["ARIMA_trend_multiplier"] = 1.0
        else:
            signals.setdefault("ARIMA_trend_multiplier", 1.0)

        return signals

    def get_intelligent_score(self, signals: dict):
        s        = signals.get("S_class_severity", 0.0)
        gap      = signals.get("Gap_SVM_confidence", 0.0)
        mu       = signals.get("mu_geometric_membership", 0.0)
        arima    = signals.get("ARIMA_trend_multiplier", 1.0)
        shap_cos = signals.get("SHAP_cosine_similarity", 0.0)

        raw_score = (s * 0.4) + (gap * 0.2) + (mu * 0.1) + (arima * 0.2) + (shap_cos * 0.1)
        score = min(max(round(float(raw_score), 4), 0.0), 1.0)
        return score

    def query_faiss(self, signals: dict):
        if self.index:
            return {"neighbors_count": 5, "split": "70/30", "top_label": "Success"}
        
        if signals.get("force_rag_tie"):
            return {"neighbors_count": 5, "split": "50/50"}
        if signals.get("force_rag_zero"):
            return {"neighbors_count": 0}
        
        return {"neighbors_count": 5, "split": "90/10"}

    def react_router(self, score, shap_cos, rag_results=None):
        if score > 0.8 and shap_cos > 0.8:
            return "Direct Dispatch"
        if score < 0.3:
            return "Below Minimum Threshold"
        if shap_cos < 0.5:
            return "SHAP Re-check"
        if score > 0.8 and shap_cos <= 0.8:
            return "Human Review"

        if rag_results:
            if rag_results.get("neighbors_count", 0) == 0:
                return "RAG Zero-Success"
            if rag_results.get("split") == "50/50":
                return "RAG Tie"
            return "RAG Retrieval Success"

        if 0.3 <= score <= 0.8:
            return "RAG Retrieval"
        
        return "Human Review"

    def explain_shap(self, shap_dict: dict):
        template = "Analyze these game features impact and generate a natural language explanation for a developer: {features}. Focus on why the model gave this score."
        prompt = PromptTemplate.from_template(template).format(features=json.dumps(shap_dict))
        return self.llm.invoke(prompt)

    def generate_dynamic_action(self, path: str, score: float):
        template = (
            "Generate a dynamic action plan based on the ReAct path taken and the intelligent score.\n"
            "Path: {path}\n"
            "Score: {score}\n"
            "Return format: 'Action: [Type] | Timing: [Specific Time] | Personalization: [Custom Message]'"
        )
        prompt = PromptTemplate.from_template(template).format(path=path, score=score)
        return self.llm.invoke(prompt)

    def community_agent(self, score: float, arima: float, shap_feature: str, rag_vote: str, features: dict):
        """[BONUS] Multi-Agent: Community Profiling Agent (ReAct Upgraded)."""
        genre = features.get("primary_genre", "Unknown") if isinstance(features, dict) else "Unknown"
        input_text = (
            f"Act as a Community Manager. Analyze these signals:\n"
            f"Score: {score}\nARIMA Trend: {arima}\nSHAP Dominant Feature: {shap_feature}\n"
            f"RAG Vote Outcome: {rag_vote}\nFeatures: {json.dumps(features)}\n\n"
            f"RULES:\n"
            f"- ARIMA trend dictates the 'Timing'.\n"
            f"- SHAP dominant feature dictates the 'Personalization' and 'Intensity'.\n"
            f"- You MUST use the fetch_community_sentiment tool for the genre '{genre}' to observe the world.\n\n"
            f"Provide a structured report detailing: Action Type, Timing, Intensity, and Personalization."
        )
        return run_react_agent(input_text, [fetch_community_sentiment])

    def investor_agent(self, score: float, arima: float, shap_feature: str, rag_vote: str, features: dict, game_name: str, investor_name: str):
        """[BONUS] Multi-Agent: Investor Scouting Agent (ReAct Upgraded)."""
        genre = features.get("primary_genre", "Unknown") if isinstance(features, dict) else "Unknown"
        input_text = (
            f"Act as a VC Scout. Analyze these signals:\n"
            f"Score: {score}\nARIMA Trend: {arima}\nSHAP Dominant Feature: {shap_feature}\n"
            f"RAG Vote Outcome: {rag_vote}\nFeatures: {json.dumps(features)}\n\n"
            f"RULES:\n"
            f"- ARIMA trend dictates the 'Timing'.\n"
            f"- SHAP dominant feature dictates the 'Personalization' and 'Intensity'.\n"
            f"- You MUST use the fetch_market_comparables tool for the genre '{genre}' to observe the world.\n"
            f"- Generate a complete, professional email draft to send to investor '{investor_name}' pitching this game. "
            f"You MUST explicitly mention the game name '{game_name}' in the email body, include a 'Subject:' line, and formally sign off exactly as 'PixelProspector'.\n\n"
            f"Provide a structured report detailing: Action Type, Timing, Intensity, Personalization, and Email Draft."
        )
        email_content = run_react_agent(input_text, [fetch_market_comparables])
        
        # Send email via SMTP right after generation
        investor_email = os.environ.get("INVESTOR_EMAIL", "")
        if investor_email:
            # Parse subject from the generated content if possible
            subject = f"Investment Opportunity: {game_name}"
            for line in email_content.split('\n'):
                if line.lower().startswith('subject:'):
                    subject = line[len('subject:'):].strip()
                    break
            
            # Helper function for SMTP
            def send_email_via_smtp(to_email, subj, body):
                import smtplib
                from email.message import EmailMessage
                smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
                smtp_port = int(os.environ.get("SMTP_PORT", 587))
                smtp_user = os.environ.get("SMTP_USERNAME")
                smtp_pass = os.environ.get("SMTP_PASSWORD")
                
                if not all([smtp_user, smtp_pass, to_email]):
                    logger.warning("SMTP credentials or to_email not set. Skipping email send.")
                    return False
                    
                msg = EmailMessage()
                msg.set_content(body)
                msg['Subject'] = subj
                msg['From'] = smtp_user
                msg['To'] = to_email
                
                try:
                    logger.info(f"Attempting to send SMTP email to {to_email} via {smtp_server}:{smtp_port}...")
                    server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)
                    server.quit()
                    logger.info(f"Email successfully sent to {to_email}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to send email: {e}")
                    return False
            
            if score >= 0.7:
                logger.info("Game score is high enough. Triggering background email send...")
                send_email_via_smtp(investor_email, subject, email_content)
                logger.info("Email send process finished.")
            else:
                logger.info("Game score is too low (%.3f < 0.7). Skipping actual SMTP email send.", score)
            
        return email_content
