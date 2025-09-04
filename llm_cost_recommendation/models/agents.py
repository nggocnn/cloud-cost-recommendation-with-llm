"""
Agent configuration and capability models.
"""

from typing import Dict, List, Any
from pydantic import BaseModel, Field

from .types import ServiceTypeUnion, RecommendationType
from .conditions import ConditionalRule


class AgentCapability(BaseModel):
    """Service agent capability definition"""

    service: ServiceTypeUnion
    supported_recommendation_types: List[RecommendationType]
    required_metrics: List[str]
    optional_metrics: List[str] = Field(default_factory=list)

    # Configuration
    thresholds: Dict[str, float] = Field(default_factory=dict)
    analysis_window_days: int = 30


class ServiceAgentConfig(BaseModel):
    """Configuration for service agents"""

    agent_id: str
    service: ServiceTypeUnion
    enabled: bool = True

    # Capability definition
    capability: AgentCapability

    # Prompt configuration
    base_prompt: str
    service_specific_prompt: str

    # LLM settings
    temperature: float = 0.1
    max_tokens: int = 2000

    # Analysis settings
    min_cost_threshold: float = 1.0  # Don't recommend for resources < $1/month
    confidence_threshold: float = 0.7  # Minimum confidence to include recommendation

    # Custom conditional rules
    custom_rules: List[ConditionalRule] = Field(default_factory=list)


class GlobalConfig(BaseModel):
    """Global configuration shared across all agents and coordinator"""

    enabled_services: List[ServiceTypeUnion]

    # Deduplication settings
    similarity_threshold: float = 0.8

    # Ranking weights
    savings_weight: float = 0.4
    risk_weight: float = 0.3
    confidence_weight: float = 0.2
    implementation_ease_weight: float = 0.1

    # Report settings
    max_recommendations_per_service: int = 50
    include_low_impact: bool = False  # Include recommendations < $10/month savings
    
    # Global cost tier configuration for resource batching strategy
    cost_tiers: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
        "minimal_cost": {"min": 0, "max": 10, "batch_adjustment": 2},
        "low_cost": {"min": 10, "max": 100, "batch_adjustment": 0},
        "medium_cost": {"min": 100, "max": 1000, "batch_adjustment": -1},
        "high_cost": {"min": 1000, "max": float('inf'), "batch_adjustment": -2}
    })
    
    # Global complexity tier configuration for resource batching strategy
    complexity_tiers: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "simple": {"metric_threshold": 3, "base_batch_size": 6},
        "moderate": {"metric_threshold": 8, "base_batch_size": 4},
        "complex": {"metric_threshold": float('inf'), "base_batch_size": 2}
    })
    
    # Global batch size configuration
    batch_config: Dict[str, Any] = Field(default_factory=lambda: {
        "min_batch_size": 1,
        "max_batch_size": 10,
        "default_batch_size": 4,
        "single_resource_threshold_cost": 5000  # Resources above this cost get individual analysis
    })
