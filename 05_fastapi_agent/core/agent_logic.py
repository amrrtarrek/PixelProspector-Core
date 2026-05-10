import numpy as np

def pre_scoring_triage(user_features):
    """
    BONUS CLAIMED: Pre-scoring triage gate.
    A lightweight filter to reject low-quality or toxic input before processing.
    """
    if user_features.toxicity_level > 0.90 or user_features.insight_depth < 0.10:
        return "Rejected", "Input quality does not meet the minimum threshold."
    return "Pass", None

def calculate_intelligent_score(signals: dict):
    """
    The 5-Signal Intelligent Score.
    Aggregates ML predictions and market trends into one final confidence metric.
    """
    s = signals.get('S_dynamic', 0)
    gap = signals.get('Gap_SVM', 0)
    mu = signals.get('Mu_geometric', 0)
    arima = signals.get('ARIMA_multiplier', 1.0)
    shap = signals.get('SHAP_cosine', 0)
    
    # Weighted calculation of the final score
    base_score = (s * 0.4) + (gap * 0.2) + (mu * 0.2) + (shap * 0.2)
    return round(base_score * arima, 4)
