"""
Recommendation models and reports.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from .types import ServiceTypeUnion, RecommendationType, RiskLevel


class Recommendation(BaseModel):
    """Recommendation model"""

    id: str = Field(..., description="Unique recommendation identifier")
    resource_id: str
    service: ServiceTypeUnion
    recommendation_type: RecommendationType

    # Current configuration
    current_config: Dict[str, Any]

    # Recommended configuration
    recommended_config: Dict[str, Any]

    # Cost impact
    current_monthly_cost: float
    estimated_monthly_cost: float
    estimated_monthly_savings: float
    annual_savings: float

    # Risk and impact
    risk_level: RiskLevel
    impact_description: str
    rollback_plan: str

    # Detailed reasoning
    rationale: str
    evidence: Dict[str, Any] = Field(default_factory=dict)

    # Implementation details
    implementation_steps: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)

    # Metadata
    confidence_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str

    # Business context
    business_hours_impact: bool = False
    downtime_required: bool = False
    sla_impact: Optional[str] = None


class RecommendationReport(BaseModel):
    """Complete recommendation report"""

    id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Summary metrics
    total_monthly_savings: float
    total_annual_savings: float
    total_recommendations: int

    # Recommendations by category
    recommendations: List[Recommendation]

    # Risk distribution
    low_risk_count: int = 0
    medium_risk_count: int = 0
    high_risk_count: int = 0

    # Service breakdown
    savings_by_service: Dict[str, float] = Field(default_factory=dict)

    # Implementation timeline
    quick_wins: List[str] = Field(
        default_factory=list
    )  # Can be implemented immediately
    medium_term: List[str] = Field(default_factory=list)  # Require planning
    long_term: List[str] = Field(default_factory=list)  # Require significant changes

    # Quality metrics
    coverage: Dict[str, Any] = Field(default_factory=dict)  # What was analyzed
    data_quality_issues: List[str] = Field(default_factory=list)
