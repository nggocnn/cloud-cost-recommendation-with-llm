"""
Metrics and billing data models.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Metrics(BaseModel):
    """Resource metrics model"""

    resource_id: str
    timestamp: datetime
    metrics: Dict[str, float] = Field(default_factory=dict)
    period_days: int = Field(default=30)

    # Common metrics patterns
    cpu_utilization_p50: Optional[float] = None
    cpu_utilization_p90: Optional[float] = None
    cpu_utilization_p95: Optional[float] = None
    memory_utilization_p50: Optional[float] = None
    memory_utilization_p90: Optional[float] = None
    memory_utilization_p95: Optional[float] = None

    # Storage metrics
    iops_read: Optional[float] = None
    iops_write: Optional[float] = None
    throughput_read: Optional[float] = None
    throughput_write: Optional[float] = None

    # Network metrics
    network_in: Optional[float] = None
    network_out: Optional[float] = None

    # Usage patterns
    is_idle: bool = False
    peak_usage_hours: List[int] = Field(default_factory=list)


class BillingData(BaseModel):
    """Billing data model"""

    service: str
    resource_id: Optional[str] = None
    region: str
    usage_type: str
    usage_amount: float
    usage_unit: str
    unblended_cost: float
    amortized_cost: float
    credit: float = 0.0
    bill_period_start: datetime
    bill_period_end: datetime
    tags: Dict[str, str] = Field(default_factory=dict)
