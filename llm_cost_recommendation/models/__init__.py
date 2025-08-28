"""
Core data models for the LLM cost recommendation system.

This module provides a centralized import point for all model classes.
The models are organized into separate modules for better maintainability:

- types: Core enums and type definitions
- resources: Resource models  
- metrics: Metrics and billing data models
- recommendations: Recommendation and report models
- agents: Agent configuration models
- conditions: Custom conditions and rules
- pricing: Pricing models and calculations
"""

# Core types and enums
from .types import (
    CloudProvider,
    ServiceType,
    ServiceTypeUnion,
    RecommendationType,
    RiskLevel,
    ConditionOperator,
    ConditionField,
)

# Resource models
from .resources import Resource

# Metrics and billing
from .metrics import Metrics, BillingData

# Conditions and rules
from .conditions import CustomCondition, ConditionalRule

# Recommendations
from .recommendations import Recommendation, RecommendationReport

# Agent configuration
from .agents import AgentCapability, ServiceAgentConfig, CoordinatorConfig

# Pricing models
from .pricing import (
    PricingModel,
    ReservationTerm,
    PaymentOption,
    Currency,
    PricingUnit,
    ReservationPricing,
    ServicePricing,
    PricingRule,
    CostCalculation,
    PricingCache,
    PricingConfig,
)

# For backwards compatibility, export all models at package level
__all__ = [
    # Types
    "CloudProvider",
    "ServiceType", 
    "ServiceTypeUnion",
    "RecommendationType",
    "RiskLevel",
    "ConditionOperator",
    "ConditionField",
    
    # Resources
    "Resource",
    
    # Metrics
    "Metrics",
    "BillingData",
    
    # Conditions
    "CustomCondition",
    "ConditionalRule",
    
    # Recommendations
    "Recommendation",
    "RecommendationReport",
    
    # Agents
    "AgentCapability",
    "ServiceAgentConfig", 
    "CoordinatorConfig",
    
    # Pricing
    "PricingModel",
    "ReservationTerm",
    "PaymentOption",
    "Currency",
    "PricingUnit",
    "ReservationPricing",
    "ServicePricing",
    "PricingRule",
    "CostCalculation",
    "PricingCache",
    "PricingConfig",
]
