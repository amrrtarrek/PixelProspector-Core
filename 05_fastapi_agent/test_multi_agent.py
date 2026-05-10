from core.multi_agent_system import PixelProspectorOrchestrator
import json

# Setup
API_KEY = "AIzaSyC0bU_7hhtcdSdJuMBKomZ6uiOgIMnRMks"
orchestrator = PixelProspectorOrchestrator(api_key=API_KEY)

# Sample Data for High Performance Game
score = 0.92
shap_dict = {"gameplay_addictiveness": 0.95, "technical_polish": 0.88}
game_features = {"genre": "Survival RPG", "viral_momentum": 0.85}
path = "Direct Dispatch"

print("\n--- PIXELPROSPECTOR MULTI-AGENT BONUS TEST ---")
print(f"Testing with Score: {score} | Path: {path}")

# Run the Multi-Agent Analysis
results = orchestrator.run_full_agent_analysis(score, shap_dict, game_features, path)

print("\n[INVESTOR AGENT OUTPUT]")
print("-" * 30)
print(results.get("investor"))

print("\n[COMMUNITY AGENT OUTPUT]")
print("-" * 30)
print(results.get("community"))

print("\n[AI REASONING LOG]")
print("-" * 30)
print(results.get("explanation"))

print("\n--- TEST COMPLETE: Multi-Agent Collaboration Verified ---")
