"""
API request/response models for the LLM cost recommendation system.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

from .types import ServiceTypeUnion, RecommendationType, RiskLevel
from .resources import Resource
from .metrics import BillingData, Metrics
from .recommendations import Recommendation, RecommendationReport


class APIError(BaseModel):
    """Standard API error response"""

    error: str
    code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RecommendationConversionError(BaseModel):
    """Error in converting LLM recommendation to model"""

    resource_id: str
    error_type: str  # e.g., "missing_fields", "validation_error", "conversion_error"
    error_message: str
    missing_fields: Optional[List[str]] = None
    original_data: Optional[Dict[str, Any]] = None
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RecommendationConversionResult(BaseModel):
    """Result of attempting to convert LLM recommendations"""

    recommendations: List[Recommendation] = Field(default_factory=list)
    errors: List[RecommendationConversionError] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    resources_processed: int
    successful_conversions: int
    failed_conversions: int


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    components: Dict[str, str] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    """System status response"""

    config: Dict[str, Any]
    agents: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AnalysisRequest(BaseModel):
    """Complete analysis request with all data - uses existing models"""

    resources: List[Resource] = Field(
        min_length=1, description="List of cloud resources to analyze"
    )
    billing: List[BillingData] = Field(
        default_factory=list, description="Billing data for resources"
    )
    metrics: List[Metrics] = Field(
        default_factory=list, description="Performance metrics for resources"
    )

    # Analysis options
    individual_processing: bool = Field(
        default=False,
        description="Process resources individually (slower but more precise)",
    )
    max_recommendations: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Maximum number of recommendations to return",
    )
    min_savings_threshold: Optional[float] = Field(
        default=None, ge=0, description="Minimum monthly savings threshold"
    )
    risk_levels: List[RiskLevel] = Field(
        default_factory=lambda: [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH],
        description="Risk levels to include in analysis",
    )

    @field_validator("resources")
    @classmethod
    def validate_unique_resource_ids(cls, v):
        resource_ids = [r.resource_id for r in v]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("Resource IDs must be unique")
        return v


class AnalysisResponse(BaseModel):
    """Analysis response with recommendations"""

    request_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Processing info
    resources_analyzed: int
    processing_time_seconds: float
    individual_processing: bool

    # Reuse existing RecommendationReport structure
    report: RecommendationReport

    @property
    def total_recommendations(self) -> int:
        """Convenience property"""
        return self.report.total_recommendations

    @property
    def total_monthly_savings(self) -> float:
        """Convenience property"""
        return self.report.total_monthly_savings

    @property
    def total_annual_savings(self) -> float:
        """Convenience property"""
        return self.report.total_annual_savings


# Pagination and filtering
class PaginationParams(BaseModel):
    """Pagination parameters"""

    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)


class FilterParams(BaseModel):
    """Filtering parameters for recommendations"""

    service: Optional[str] = None
    recommendation_type: Optional[RecommendationType] = None
    risk_level: Optional[RiskLevel] = None
    min_savings: Optional[float] = Field(None, ge=0)
    max_savings: Optional[float] = Field(None, ge=0)
    resource_ids: List[str] = Field(default_factory=list)

    @field_validator("max_savings")
    @classmethod
    def max_greater_than_min(cls, v, values):
        if (
            v is not None
            and "min_savings" in values.data
            and values.data["min_savings"] is not None
        ):
            if v <= values.data["min_savings"]:
                raise ValueError("max_savings must be greater than min_savings")
        return v


class SystemMetrics(BaseModel):
    """System performance metrics"""

    cpu_percent: float
    memory_percent: float
    disk_percent: float
    load_average: float
    python_version: str
    platform: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
