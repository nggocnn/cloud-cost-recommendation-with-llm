"""
Conditions and rules for custom agent behavior.
"""

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field

from .types import ConditionOperator, RecommendationType


class CustomCondition(BaseModel):
    """Custom condition for agent decision making"""

    field: str  # Field to evaluate (can be ConditionField or custom like "tag.Environment")
    operator: ConditionOperator
    value: Union[str, int, float, bool, List[Union[str, int, float]]]
    description: Optional[str] = None


class ConditionalRule(BaseModel):
    """A rule with conditions and actions"""

    name: str
    description: str
    conditions: List[CustomCondition]
    logic: Literal["AND", "OR"] = "AND"  # How to combine conditions

    # Actions when conditions are met
    actions: Dict[str, Any] = Field(default_factory=dict)

    # Override thresholds when conditions are met
    threshold_overrides: Dict[str, float] = Field(default_factory=dict)

    # Skip certain recommendation types
    skip_recommendation_types: List[RecommendationType] = Field(default_factory=list)

    # Force certain recommendation types
    force_recommendation_types: List[RecommendationType] = Field(default_factory=list)

    # Custom prompt additions
    custom_prompt: Optional[str] = None

    # Risk level adjustment
    risk_adjustment: Optional[Literal["increase", "decrease"]] = None

    # Priority for this rule (higher = more important)
    priority: int = 0

    # Enable/disable the rule
    enabled: bool = True
