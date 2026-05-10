from pydantic import BaseModel
from typing import Optional

class GameMLFeatures(BaseModel):
    gameplay_addictiveness: float
    technical_polish: float
    aesthetic_appeal: float
    narrative_depth: float
    replayability: float
    viral_momentum: float

class UserReviewFeatures(BaseModel):
    insight_depth: float
    toxicity_level: float
    genre_expertise: float
    sentiment_consistency: float

class InteractionMetadata(BaseModel):
    user_id: str
    game_id: str
    primary_genre: str
    timestamp: Optional[str] = None
    developer_email: Optional[str] = None
    triage_status: str = "Pending"

class IntelligentScoreSignals(BaseModel):
    S_class_severity: float
    Gap_SVM_confidence: float
    mu_geometric_membership: float
    ARIMA_trend_multiplier: float
    SHAP_cosine_similarity: float

class PixelProspectorRequest(BaseModel):
    interaction_metadata: InteractionMetadata
    game_ml_features: GameMLFeatures
    user_review_features: UserReviewFeatures
    intelligent_score_signals: Optional[IntelligentScoreSignals] = None
    llm_audit_log: Optional[str] = ""